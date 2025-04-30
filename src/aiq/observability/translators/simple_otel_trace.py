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

import json
import logging

from opentelemetry.sdk.trace import ReadableSpan

from aiq.observability.schemas.simple_otel_trace import SimpleOtelTrace
from aiq.observability.schemas.simple_otel_trace import SimpleOtelTraceContext
from aiq.observability.schemas.simple_otel_trace import SimpleOtelTraceEvent
from aiq.observability.schemas.simple_otel_trace import SimpleOtelTraceLink
from aiq.observability.schemas.simple_otel_trace import SimpleOtelTraceResource
from aiq.observability.schemas.simple_otel_trace import SimpleOtelTraceResourceAttribute
from aiq.observability.schemas.simple_otel_trace import SimpleOtelTraceStatus

logger = logging.getLogger(__name__)


def translate_span_to_simple_trace_model(span: ReadableSpan) -> SimpleOtelTrace | None:
    """Translate the span to the appropriate format the consuming service.

    Args:
        span (ReadableSpan): The span to translate.

    Returns:
        SimpleOtelTrace | None: The translated span.
    """

    try:

        # Extract span data
        span_context = span.get_span_context()
        if span_context is None:
            return None

        root_context = SimpleOtelTraceContext(trace_id=span_context.trace_id,
                                              span_id=span_context.span_id,
                                              is_remote=span_context.is_remote,
                                              trace_flags=span_context.trace_flags,
                                              trace_state=str(span_context.trace_state))

        root_attributes = dict(span.attributes) if span.attributes else None

        # Validate trace events
        events = [
            SimpleOtelTraceEvent(name=event.name,
                                 timestamp=event.timestamp,
                                 attributes=dict(event.attributes) if event.attributes else None)
            for event in span.events
        ]

        # Validate trace status
        status = SimpleOtelTraceStatus(status_code=span.status.status_code.value,
                                       description=span.status.description,
                                       is_ok=span.status.is_ok,
                                       is_unset=span.status.is_unset)

        # Validate trace links
        links = [
            SimpleOtelTraceLink(context=SimpleOtelTraceContext(trace_id=link.context.trace_id,
                                                               span_id=link.context.span_id,
                                                               is_remote=link.context.is_remote,
                                                               trace_flags=link.context.trace_flags,
                                                               trace_state=str(link.context.trace_state)),
                                attributes=dict(link.attributes) if link.attributes else None) for link in span.links
        ]

        # Validate trace resource using the otel to json method to extract serializable data
        raw_resource: dict = json.loads(span.resource.to_json())
        resource = SimpleOtelTraceResource(**raw_resource)

        # Validate simple trace model
        otel_trace_model = SimpleOtelTrace(name=span.name,
                                           context=root_context,
                                           parent_span_id=str(span.parent.span_id) if span.parent else None,
                                           start_time=span.start_time,
                                           end_time=span.end_time,
                                           attributes=root_attributes,
                                           events=events,
                                           links=links,
                                           status=status,
                                           resource=resource)

        return otel_trace_model

    except Exception as e:
        logger.exception("Error translating span: %s", e, exc_info=e)
        return None
