# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from typing import Any

import aiohttp
from aiohttp import ClientError
from aiohttp import ClientSession
from aiohttp import ClientTimeout

logger = logging.getLogger(__name__)


class SimpleHttpMixin:
    """Mixin for HTTP-based telemetry exporters with connection pooling and format support.

    This mixin provides HTTP functionality for telemetry exporters, featuring:
    - Connection pooling with configurable limits for optimal performance
    - Automatic Content-Type handling based on format (JSON/JSONLines)
    - Case-insensitive header management
    - Lazy session initialization and proper cleanup
    - Support for both JSON and JSONLines (NDJSON) serialization formats

    The mixin uses aiohttp for sending telemetry data to HTTP endpoints and is designed
    to be mixed with SpanExporter classes or used directly for custom HTTP operations.
    """

    def __init__(
        self,
        *args,
        endpoint: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        method: str = "POST",
        limit: int = 100,
        limit_per_host: int = 20,
        keepalive_timeout: float = 30,
        jsonlines: bool = False,
        **kwargs,
    ):
        """Initialize the HTTP mixin with connection pooling and format options.

        Args:
            endpoint (str): The HTTP endpoint URL.
            headers (dict[str, str] | None): HTTP headers to include with requests. Defaults to None.
            timeout (float): Request timeout in seconds. Defaults to 30.0.
            method (str): HTTP method to use. Defaults to "POST".
            limit (int): Total connection limit for the session. Defaults to 100.
            limit_per_host (int): Per-host connection limit for the session. Defaults to 20.
            keepalive_timeout (float): Keep-alive timeout in seconds for connection reuse. Defaults to 30.
            jsonlines (bool): Use JSONLines (NDJSON) format instead of standard JSON. Defaults to False.
        """
        if headers is None:
            headers = {}

        # Set Content-Type if not already present (case-insensitive check)
        if not any(key.lower() == "content-type" for key in headers.keys()):
            headers["Content-Type"] = "application/x-ndjson" if jsonlines else "application/json"

        self._endpoint = endpoint
        self._headers = headers
        self._timeout = ClientTimeout(total=timeout)
        self._method = method.upper()
        self._session: ClientSession | None = None
        self._limit = limit
        self._limit_per_host = limit_per_host
        self._keepalive_timeout = keepalive_timeout
        self._jsonlines = jsonlines
        super().__init__(*args, **kwargs)

    async def _ensure_session(self) -> ClientSession:
        """Create session with connection pooling if needed."""
        if self._session is None or self._session.closed:
            # Create session with connection pooling
            connector = aiohttp.TCPConnector(
                limit=self._limit,
                limit_per_host=self._limit_per_host,
                keepalive_timeout=self._keepalive_timeout,
            )
            self._session = ClientSession(connector=connector, headers=self._headers, timeout=self._timeout)
        return self._session

    def _serialize_http_payload(self, item: dict | list[dict]) -> str:
        """Serialize payload based on configured format.

        Args:
            item (dict | list[dict]): Data to serialize

        Returns:
            str: Serialized payload string
        """
        try:
            if self._jsonlines and isinstance(item, list):
                # JSONLines format: each item on a separate line
                return '\n'.join(json.dumps(record) for record in item)
            # Standard JSON format (works for both dict and list)
            return json.dumps(item)
        except Exception as e:
            logger.error("Failed to serialize payload: %s", e)
            raise

    async def export_http(self, item: dict | list[dict] | Any) -> None:
        """Export data via HTTP request with automatic serialization and connection pooling.

        This method handles the complete HTTP export process:
        - Ensures HTTP session exists (creates with connection pooling if needed)
        - Serializes data to JSON or JSONLines format based on configuration
        - Sends HTTP request with proper headers and error handling
        - Logs success/failure appropriately

        Args:
            item (dict | list[dict] | Any): Data to export via HTTP.

        Raises:
            ClientError: For HTTP client-related errors (network, DNS, etc.)
            ValueError/TypeError: For data serialization errors
            Exception: For other unexpected errors during export
        """

        session = await self._ensure_session()

        try:
            # Prepare the payload
            if isinstance(item, (dict, list)):
                payload = self._serialize_http_payload(item)
            else:
                # For non-JSON serializable objects, convert to string
                payload = str(item)

            # Safety check - ensure payload is always a string
            if not isinstance(payload, str):
                logger.warning("Payload is not a string (type: %s), converting to string", type(payload).__name__)
                payload = str(payload)

            # Make the HTTP request
            async with session.request(method=self._method,
                                       url=self._endpoint,
                                       data=payload.encode('utf-8'),
                                       headers=self._headers) as response:
                if response.status >= 400:
                    logger.warning("HTTP request failed with status %d: %s", response.status, await response.text())
                else:
                    logger.debug("Successfully exported data to %s with status %d", self._endpoint, response.status)

        except ClientError as e:
            logger.error("HTTP client error during export: %s", e)
        except (TypeError, ValueError) as e:
            logger.error("Failed to serialize data for HTTP export: %s", e)
        except Exception as e:
            logger.error("Error during HTTP export: %s", e, exc_info=True)

    async def close_session(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
