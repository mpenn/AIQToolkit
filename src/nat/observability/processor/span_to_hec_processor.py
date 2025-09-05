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

from nat.data_models.span import Span
from nat.observability.processor.processor import Processor
from nat.utils.type_utils import override


class HECSpan(BaseModel):
    """A HECSpan object for use in HTTP Event Collector (HEC) exports."""

    event: Span = Field(..., description="The Span event to convert to a HECSpan.")
    timezone: str = Field(default="Z", description="The timezone of the event.")


class SpanToHECProcessor(Processor[Span, HECSpan]):
    """Processor that converts a Span to a HECSpan for use in HTTP Event Collector (HEC) exports."""

    def __init__(self, timezone: str = "Z"):
        self._timezone = timezone

    @override
    async def process(self, item: Span) -> HECSpan:
        return HECSpan(event=item, timezone=self._timezone)
