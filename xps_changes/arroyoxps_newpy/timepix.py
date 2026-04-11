import logging

import numpy as np
import msgpack
import zmq.asyncio

from arroyopy.listener import Listener

from .schemas import NumpyArrayModel, XPSImageInfo, XPSRawEvent


logger = logging.getLogger(__name__)


class XPSTimepixZMQListener(Listener):
    stop_signal = False

    def __init__(
        self,
        operator,
        zmq_pub_address: str = "tcp://localhost",
        zmq_pub_port: int = 5657,
    ):
        self.operator = operator
        self.stop_signal = False
        ctx = zmq.asyncio.Context()
        self.zmq_socket = ctx.socket(zmq.SUB)
        self.zmq_socket.setsockopt(zmq.RCVHWM, 100000)
        logger.info(f"binding to: {zmq_pub_address}:{zmq_pub_port}")
        self.zmq_socket.connect(f"{zmq_pub_address}:{zmq_pub_port}")
        self.zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")

    async def start(self):
        logger.info("Listener started")
        while True:
            try:
                if self.stop_signal:
                    logger.info("Stopping listener.")
                    break
                metadata_msg_packed = await self.zmq_socket.recv()
                raw_message = await self.zmq_socket.recv()
                # print(raw_message[0:300])
                try:
                    metadata = msgpack.unpackb(metadata_msg_packed)
                except Exception as e:
                    logger.error(f"Error unpacking message: {e}")
                    continue

                # Must be an event with an image
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug(f"event: {metadata.keys()}")

                await self.operator.process(
                    self._build_event(raw_message, metadata)
                )
                logger.debug("event processed")
            except Exception as e:
                logger.error(e)

    async def stop(self):
        self.stop_signal = True

    @staticmethod
    def _build_event(
        image: bytes,
        metadata: dict,
    ) -> XPSRawEvent:
        shape = tuple(metadata["shape"])
        dtype = metadata["dtype"]

        image_info = XPSImageInfo(
            frame_number=0,
            width=shape[0],
            height=shape[1],
            data_type=dtype
        )

        array_received = np.frombuffer(image, dtype=dtype).reshape(shape)
        image_info.frame_number = metadata.get("flush_number")
        return XPSRawEvent(
            image=NumpyArrayModel(array=array_received), image_info=image_info
        )
