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

import asyncio
import logging
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Sequence
from enum import Enum

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ExportStatus(Enum):
    """The status of the export."""

    SUCCESS = "success"
    FAILURE = "failure"
    SCHEDULED = "scheduled"


class AbstractAsyncTelemetryExporter(SpanExporter, ABC):
    """Base class to support exporting telemetry data to a consuming service."""

    def __init__(self, translator: Callable):
        self._loop = asyncio.get_running_loop()
        self._translator = translator

    def _translate_span(self, span: ReadableSpan) -> BaseModel | None:
        """Translate the span to the appropriate format the consuming service.

        Args:
            span (ReadableSpan): The span to translate.

        Returns:
            BaseModel | None: The translated span.
        """

        return self._translator(span)

    @abstractmethod
    async def _export_span(self, span: ReadableSpan) -> None:
        """Export the span to the consuming service."""
        pass

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Exports a batch of telemetry data.

        Args:
            spans: The list of `opentelemetry.trace.Span` objects to be exported

        Returns:
            SpanExportResult: The result of the export, always returns SpanExportResult.SUCCESS.
        """

        try:
            for span in spans:
                self._loop.create_task(self._export_span(span=span))
                logger.info("Trace export status: %s", ExportStatus.SCHEDULED.name)
        except Exception as e:
            logger.exception("Error exporting telemetry traces: %s", e, exc_info=e)

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shuts down the exporter.

        Called when the SDK is shut down.
        """

        logger.info("Shutting down trace exporter...")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Nothing is buffered in this exporter, so this method does nothing."""

        return True
