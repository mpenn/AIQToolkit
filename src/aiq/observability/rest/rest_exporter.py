# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from collections.abc import Callable

import httpx
from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel

from aiq.observability.async_exporter_base import AbstractAsyncTelemetryExporter
from aiq.observability.async_exporter_base import ExportStatus

logger = logging.getLogger(__name__)


class RestTelemetryExporter(AbstractAsyncTelemetryExporter):
    """A REST exporter to transmit traces to a consuming service."""

    def __init__(self, endpoint: str, timeout: int, translator: Callable):
        """Initialize the REST exporter.

        Args:
            endpoint (str): The endpoint to send the traces to.
            timeout (int): The timeout for the REST request.
            translator (Callable): The translator to translate the span to the appropriate format.
        """

        super().__init__(translator=translator)
        self._endpoint = endpoint
        self._timeout = timeout
        self._translator = translator
        self._headers = {"Content-Type": "application/json"}

    async def _export_span(self, span: ReadableSpan) -> None:
        """Export the span to the consuming service.

        Args:
            span (ReadableSpan): The span to export.
        """

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            translated_payload: BaseModel | None = self._translate_span(span=span)

            if translated_payload is not None:

                try:
                    serialized_payload: str = translated_payload.model_dump_json()
                except Exception as e:
                    logger.exception("Error occured when exporting telemetry traces: %s", e, exc_info=e)
                    return

                try:
                    response = await client.post(self._endpoint, headers=self._headers, json=serialized_payload)
                    response.raise_for_status()
                    logger.info("Trace export status: %s", ExportStatus.SUCCESS.name)
                except httpx.HTTPStatusError as e:
                    logger.exception("Error occured when exporting telemetry traces: %s", e, exc_info=e)
                    logger.info("Trace export status: %s", ExportStatus.FAILURE.name)
