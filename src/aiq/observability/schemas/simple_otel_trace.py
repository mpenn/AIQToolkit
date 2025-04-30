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

from pydantic import BaseModel
from pydantic import Field


class SimpleOtelTraceContext(BaseModel):
    """A simple OTLP trace context."""

    trace_id: int
    span_id: int
    is_remote: bool
    trace_flags: int
    trace_state: str  # dict


class SimpleOtelTraceEvent(BaseModel):
    """A simple OTLP trace event."""

    name: str
    timestamp: float
    attributes: dict | None = None


class SimpleOtelTraceLink(BaseModel):
    """A simple OTLP trace link."""

    context: SimpleOtelTraceContext
    attributes: dict | None = None


class SimpleOtelTraceStatus(BaseModel):
    """A simple OTLP trace status."""

    status_code: int | None = None
    description: str | None = None
    is_ok: bool | None = None
    is_unset: bool | None = None


class SimpleOtelTraceResourceAttribute(BaseModel):
    """A simple OTLP trace resource attribute."""

    telemetry_sdk_language: str | None = Field(default=None, alias="telemetry.sdk.language")
    telemetry_sdk_name: str | None = Field(default=None, alias="telemetry.sdk.name")
    telemetry_sdk_version: str | None = Field(default=None, alias="telemetry.sdk.version")
    service_name: str | None = Field(default=None, alias="service.name")


class SimpleOtelTraceResource(BaseModel):
    """A simple OTLP trace resource."""

    attributes: SimpleOtelTraceResourceAttribute | None = None
    schema_url: str | None = None


class SimpleOtelTrace(BaseModel):
    """A simple OTLP trace."""

    name: str
    context: SimpleOtelTraceContext
    parent_span_id: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    attributes: dict | None = None
    events: list[SimpleOtelTraceEvent] = []
    links: list[SimpleOtelTraceLink] = []
    status: SimpleOtelTraceStatus | None = None
    resource: SimpleOtelTraceResource | None = None
