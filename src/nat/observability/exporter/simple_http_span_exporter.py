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

from nat.builder.context import ContextState
from nat.data_models.span import Span
from nat.observability.exporter.span_exporter import SpanExporter
from nat.observability.mixin.simple_http_mixin import SimpleHttpMixin
from nat.observability.processor.batching_processor import DictBatchingProcessor
from nat.observability.processor.pydantic_to_dict_processor import SpanToDictProcessor

logger = logging.getLogger(__name__)


class SimpleHttpSpanExporter(SpanExporter[Span, dict], SimpleHttpMixin):
    """HTTP span exporter with connection pooling and configurable serialization formats.

    This exporter processes Span objects through a pipeline and exports them via HTTP:
    1. SpanToDictProcessor: Converts Span objects to dictionaries
    2. DictBatchingProcessor: Batches dictionaries for efficient bulk transmission
    3. HTTP export: Sends data using connection pooling and format-aware serialization

    Supports both JSON and JSONLines formats, configurable connection pooling,
    and integrates with the NAT lifecycle management system.
    """

    def __init__(self,
                 context_state: ContextState | None = None,
                 batch_size: int = 100,
                 flush_interval: float = 5.0,
                 max_queue_size: int = 1000,
                 drop_on_overflow: bool = False,
                 shutdown_timeout: float = 10.0,
                 **simple_http_kwargs):
        # Initialize each parent class explicitly to handle multiple inheritance
        SpanExporter.__init__(self, context_state=context_state)
        SimpleHttpMixin.__init__(self, **simple_http_kwargs)

        self.add_processor(SpanToDictProcessor(), name="span_to_dict")
        self.add_processor(DictBatchingProcessor(batch_size=batch_size,
                                                 flush_interval=flush_interval,
                                                 max_queue_size=max_queue_size,
                                                 drop_on_overflow=drop_on_overflow,
                                                 shutdown_timeout=shutdown_timeout),
                           name="dict_batching")

    async def export_processed(self, item: dict | list[dict]) -> None:
        """Export processed spans to the HTTP endpoint.

        This method implements the SpanExporter abstract method by delegating to
        SimpleHttpMixin.export_http() which handles HTTP serialization, connection
        pooling, and request transmission. Supports both individual span dictionaries
        and batched lists of span dictionaries.

        Args:
            item (dict | list[dict]): Processed span data to export via HTTP.
        """
        await self.export_http(item)

    async def _cleanup(self) -> None:
        """Clean up HTTP session and other resources."""
        try:
            await self.close_session()
        except Exception as e:
            logger.warning("Error closing HTTP session during cleanup: %s", e)

        # Continue with parent cleanup chain
        await super()._cleanup()
