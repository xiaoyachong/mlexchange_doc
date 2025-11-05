import asyncio
import json
import logging
import msgpack
import numpy as np
import os
import websockets
from arroyopy.listener import Listener
from arroyopy.operator import Operator
from arroyosas.schemas import RawFrameEvent, SerializableNumpyArrayModel

logger = logging.getLogger("arroyo_reduction.xps_websocket_listener")


class XPSWebSocketListener(Listener):
    """Listen to XPS websocket and process shot_mean data"""
    
    def __init__(self, operator: Operator, websocket_url: str):
        self.operator = operator
        self.websocket_url = websocket_url
        self.should_stop = False
        self.current_scan_name = None
        self.current_tiled_url = None
        self.frame_counter = 0

    async def start(self):
        """Connect to XPS websocket and listen for messages"""
        logger.info(f"XPS WebSocket listener starting on {self.websocket_url}")
        
        while not self.should_stop:
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    logger.info("Connected to XPS websocket")
                    
                    async for message in websocket:
                        if self.should_stop:
                            break
                        try:
                            await self._handle_message(message)
                        except Exception as e:
                            logger.exception(f"Error processing message: {e}")
                        
            except websockets.ConnectionClosed:
                logger.warning("XPS websocket connection closed, reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in XPS websocket connection: {e}")
                await asyncio.sleep(5)
    
    async def _handle_message(self, message):
        """Parse XPS message and extract shot_mean for processing"""
        # First message is JSON metadata
        if isinstance(message, str):
            data = json.loads(message)
            logger.debug(f"Received XPS metadata: {data}")
            
            # Handle start message
            if data.get('msg_type') == 'start':
                self.current_scan_name = data.get('scan_name', '')
                self.current_tiled_url = data.get('tiled_url', '')
                self.frame_counter = 0
                logger.info(f"Starting new XPS run: {self.current_scan_name}")
                logger.info(f"Tiled URL: {self.current_tiled_url}")
            
            # Handle regular frame metadata
            elif 'frame_number' in data:
                # Update metadata from frame message
                self.current_tiled_url = data.get('tiled_url', self.current_tiled_url)
                self.current_scan_name = data.get('scan_name', self.current_scan_name)
            
            return
        
        # Second message is msgpack with images
        data = msgpack.unpackb(message)
        
        shot_mean_bytes = data.get('shot_mean')
        width = data.get('width')
        height = data.get('height')
        shot_num = data.get('shot_num', 0)
        
        if not shot_mean_bytes or not width or not height:
            logger.warning("Received XPS message without shot_mean data")
            return
        
        # Convert bytes to numpy array
        shot_mean = np.frombuffer(shot_mean_bytes, dtype=np.uint8).reshape(width, height)
        
        logger.debug(f"Received shot_mean for shot {shot_num}: shape {shot_mean.shape}")
        
        # Construct tiled_url for this specific frame
        # Format: {tiled_url}/runs/{scan_name}/shot_mean?slice={frame}:{frame+1},0:{height},0:{width}
        tiled_url = (
            f"{self.current_tiled_url}/api/v1/array/full/runs/{self.current_scan_name}/shot_mean"
            f"?slice={self.frame_counter}:{self.frame_counter+1},0:{height},0:{width}"
        )
        
        logger.debug(f"Constructed tiled_url: {tiled_url}")
        
        # Create RawFrameEvent
        frame_event = RawFrameEvent(
            image=SerializableNumpyArrayModel(array=shot_mean),
            frame_number=self.frame_counter,
            tiled_url=tiled_url
        )
        
        # Increment frame counter
        self.frame_counter += 1
        
        # Process through operator
        await self.operator.process(frame_event)
    
    async def stop(self):
        """Stop the listener"""
        logger.info("Stopping XPS websocket listener")
        self.should_stop = True
    
    @classmethod
    def from_settings(cls, settings: dict, operator: Operator) -> "XPSWebSocketListener":
        """Create listener from settings"""
        websocket_url = settings.websocket_url
        logger.info(f"Listening for XPS frames on {websocket_url}")
        return cls(operator, websocket_url)
