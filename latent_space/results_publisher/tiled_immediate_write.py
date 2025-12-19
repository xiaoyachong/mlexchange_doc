import asyncio
import logging
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
from arroyopy.publisher import Publisher
from arroyosas.schemas import SASStop
from tiled.client import from_uri

from .schemas import LatentSpaceEvent

logger = logging.getLogger("arroyo_reduction.tiled_results_publisher")

# Environment variables for Tiled connections
RESULTS_TILED_URI = os.getenv("RESULTS_TILED_URI", "http://tiled:8000")
RESULTS_TILED_API_KEY = os.getenv("RESULTS_TILED_API_KEY", "")
# Constants
# Timezone for log timestamps
CALIFORNIA_TZ = pytz.timezone("US/Pacific")
# Regex pattern to extract UUID from tiled_url
UUID_PATTERN = r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})"


class TiledResultsPublisher(Publisher):
    """Publisher that saves latent space vectors to a Tiled server with immediate writes."""

    def __init__(
        self, tiled_uri=None, tiled_api_key=None, root_segments=None, tiled_prefix=None
    ):
        super().__init__()
        self.tiled_uri = tiled_uri or RESULTS_TILED_URI
        self.tiled_api_key = tiled_api_key or RESULTS_TILED_API_KEY
        self.tiled_prefix = tiled_prefix
        self.root_segments = root_segments or ["lse_live_results"]
        self.client = None
        self.root_container = None
        self.year_container = None
        self.month_container = None
        self.day_container = None

        # CHANGED: Dictionary to store array clients (like image publisher)
        self.array_clients = {}  # {uuid: ArrayClient}
        
        # Set to track UUIDs that already exist in Tiled
        self.existing_uuids = set()
        # Default table name if no UUID is available
        self.default_table_name = "feature_vectors"
        # Keep track of the current UUID
        self.current_uuid = None

        # Track current experiment name (will be set from message)
        self.current_experiment_name = "default_experiment"

        logger.info(f"Initialized publisher with immediate write using patch")

    async def start(self):
        """Connect to Tiled server and initialize containers."""
        try:
            await asyncio.to_thread(self._start_sync)
        except Exception as e:
            logger.error(f"Failed to initialize Tiled client: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _start_sync(self):
        """Synchronous implementation of start() to be run in a thread."""
        try:
            self.client = from_uri(self.tiled_uri, api_key=self.tiled_api_key)

            # Navigate to prefix first if specified
            container = self.client
            if self.tiled_prefix:
                prefix_segments = self.tiled_prefix.split("/")
                for segment in prefix_segments:
                    if segment:
                        if segment in container:
                            logger.info(f"Using existing prefix container: {segment}")
                            container = container[segment]
                        else:
                            logger.info(f"Creating prefix container: {segment}")
                            container = container.create_container(segment)

            # Navigate to the root container and create the hierarchy
            self._setup_containers_sync(container)

            # List all existing tables in the day container
            if self.day_container is not None:
                table_keys = list(self.day_container)
                logger.info(f"Found {len(table_keys)} existing tables in day container")
                self.existing_uuids.update(table_keys)
                logger.info(f"Tracking {len(self.existing_uuids)} existing UUIDs")

            logger.info(f"Connected to Tiled server at {self.tiled_uri}")
            prefix_path = f"{self.tiled_prefix}/" if self.tiled_prefix else ""
            now = datetime.now(CALIFORNIA_TZ)
            logger.info(
                f"Using container path: {prefix_path}{'/'.join(self.root_segments)}/{now.year}/{now.month:02d}/{now.day:02d}"
            )
        except Exception as e:
            logger.error(f"Error in _start_sync: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _extract_uuid_from_url(self, url):
        """Extract UUID from tiled_url."""
        if not url:
            return self.default_table_name

        logger.debug(f"Extracting UUID from URL: {url}")
        match = re.search(UUID_PATTERN, url)
        if match:
            uuid = match.group(1)
            logger.debug(f"Extracted UUID: {uuid}")
            return uuid

        logger.debug(f"No UUID found in URL, using default: {self.default_table_name}")
        return self.default_table_name

    def _setup_containers_sync(self, starting_container=None):
        """Set up the container structure without USER level (synchronous version)."""
        try:
            container = (
                starting_container if starting_container is not None else self.client
            )

            # Navigate through root_segments
            for segment in self.root_segments:
                if segment in container:
                    logger.info(f"Using existing container: {segment}")
                    container = container[segment]
                else:
                    logger.info(f"Creating container: {segment}")
                    container = container.create_container(segment)

            self.root_container = container

            # Create Year/Month/Day hierarchy
            now = datetime.now(CALIFORNIA_TZ)
            year_str = str(now.year)
            month_str = f"{now.month:02d}"
            day_str = f"{now.day:02d}"

            # Create Year container
            if year_str not in self.root_container:
                logger.info(f"Creating year container: {year_str}")
                self.root_container.create_container(year_str)
            else:
                logger.info(f"Using existing year container: {year_str}")
            self.year_container = self.root_container[year_str]

            # Create Month container
            if month_str not in self.year_container:
                logger.info(f"Creating month container: {month_str}")
                self.year_container.create_container(month_str)
            else:
                logger.info(f"Using existing month container: {month_str}")
            self.month_container = self.year_container[month_str]

            # Create Day container
            if day_str not in self.month_container:
                logger.info(f"Creating day container: {day_str}")
                self.month_container.create_container(day_str)
            else:
                logger.info(f"Using existing day container: {day_str}")
            self.day_container = self.month_container[day_str]

        except Exception as e:
            logger.error(f"Error setting up containers: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _get_experiment_container(self, experiment_name=None):
        """Get or create the experiment container based on experiment name"""
        try:
            exp_name = (
                experiment_name or self.current_experiment_name or "default_experiment"
            )

            if exp_name not in self.day_container:
                logger.info(f"Creating experiment container: {exp_name}")
                self.day_container.create_container(exp_name)

            return self.day_container[exp_name]
        except Exception as e:
            logger.error(f"Error getting experiment container: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self.day_container

    # CHANGED: New method to get or create array client (like image publisher)
    def _get_or_create_array_client(self, uuid, experiment_name, first_record):
        """Get or create an array client for the UUID."""
        if uuid in self.array_clients:
            return self.array_clients[uuid]

        try:
            experiment_container = self._get_experiment_container(experiment_name)

            # Create UUID container if it doesn't exist
            if uuid not in experiment_container:
                logger.info(f"Creating UUID container: {uuid}")
                experiment_container.create_container(uuid)

            uuid_container = experiment_container[uuid]

            # Check if feature_vectors already exists
            if "feature_vectors" in uuid_container:
                logger.info(f"Loading existing array for UUID: {uuid}")
                array_client = uuid_container["feature_vectors"]
                self.array_clients[uuid] = array_client
                return array_client

            # Create initial DataFrame with first record
            df = pd.DataFrame([first_record])
            
            logger.info(
                f"Creating new table with {len(df.columns)} columns for UUID: {uuid}"
            )

            # Write initial DataFrame
            array_client = uuid_container.write_dataframe(df, key="feature_vectors")
            
            # Cache the array client
            self.array_clients[uuid] = array_client
            self.existing_uuids.add(uuid)
            
            logger.info(f"Created new table for UUID: {uuid}")

            return array_client

        except Exception as e:
            logger.error(f"Error creating array client: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    # CHANGED: New method to append row using patch (like image publisher)
    def _patch_tiled_dataframe(self, array_client, record):
        """Append new row to existing Tiled DataFrame using patch."""
        try:
            # Convert record to DataFrame
            new_row = pd.DataFrame([record])
            
            # Get current number of rows
            current_length = len(array_client.read())
            
            logger.debug(f"[PATCH] Current rows: {current_length}, appending 1 row")

            # Patch at the end (offset is current_length)
            array_client.patch_dataframe(new_row, offset=current_length)

            logger.debug(f"[PATCH SUCCESS] Row appended at index {current_length}")

        except Exception as e:
            logger.error(f"[PATCH ERROR] Error patching Tiled dataframe: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def publish(self, message):
        """Publish a message to Tiled server."""

        # Check for flush signal
        if isinstance(message, LatentSpaceEvent):
            if message.tiled_url == "FLUSH_SIGNAL":
                logger.info("Received flush signal - clearing array clients cache")
                self.array_clients.clear()
                return

        if isinstance(message, SASStop):
            logger.info("Received Stop message")
            self.array_clients.clear()
            return

        if not isinstance(message, LatentSpaceEvent):
            return

        try:
            # CHANGED: Write immediately instead of batching
            await asyncio.to_thread(self._publish_sync, message)

        except Exception as e:
            logger.error(f"Error publishing to Tiled: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _publish_sync(self, message):
        """Synchronous implementation of publish() to be run in a thread."""
        try:
            if self.day_container is None:
                logger.error("Day container not initialized, cannot publish")
                return

            # Format vector and metadata
            vector = np.array(message.feature_vector, dtype=np.float32)
            if vector.ndim == 1:
                tiled_url = getattr(message, "tiled_url", None)
                uuid = self._extract_uuid_from_url(tiled_url)

                # Get experiment name from message
                experiment_name = getattr(message, "experiment_name", None)
                if experiment_name:
                    self.current_experiment_name = experiment_name

                # Create record with metadata
                record = {
                    "tiled_url": tiled_url,
                    "autoencoder_model": getattr(message, "autoencoder_model", None),
                    "dimred_model": getattr(message, "dimred_model", None),
                    "timestamp": getattr(message, "timestamp", time.time()),
                    "total_processing_time": getattr(
                        message, "total_processing_time", None
                    ),
                    "autoencoder_time": getattr(message, "autoencoder_time", None),
                    "dimred_time": getattr(message, "dimred_time", None),
                }

                # Add vector elements as columns
                for i, val in enumerate(vector[:20]):
                    record[f"feature_{i}"] = float(val)

                # CHANGED: Get or create array client
                array_client = self._get_or_create_array_client(
                    uuid, experiment_name, record
                )

                if not array_client:
                    logger.warning(f"Failed to get array client for UUID {uuid}")
                    return

                # CHANGED: If this is not the first row, patch immediately
                current_length = len(array_client.read())
                if current_length > 0:  # Not the first row
                    self._patch_tiled_dataframe(array_client, record)
                    logger.debug(f"Patched vector to table '{uuid}'")
                else:
                    logger.debug(f"First vector already written for '{uuid}'")

            else:
                logger.warning(
                    f"Received vector with unexpected dimensions: {vector.shape}"
                )

        except Exception as e:
            logger.error(f"Error in _publish_sync: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # REMOVED: write_table_to_tiled methods (no longer needed)

    async def stop(self):
        """Clear array clients cache before stopping."""
        try:
            logger.info("Publisher stopping, clearing array clients cache")
            self.array_clients.clear()
            logger.info("Publisher stopped")
        except Exception as e:
            logger.error(f"Error stopping publisher: {e}")
            import traceback
            logger.error(traceback.format_exc())

    @classmethod
    def from_settings(cls, settings):
        """Create a TiledResultsPublisher from settings."""
        return cls(
            root_segments=settings.get("root_segments"),
            tiled_prefix=settings.get("tiled_prefix"),
        )
