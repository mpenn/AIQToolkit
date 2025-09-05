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
from typing import TypeVar

from pydantic import BaseModel

from nat.data_models.span import Span
from nat.observability.processor.processor import Processor
from nat.observability.processor.span_to_hec_processor import HECSpan

logger = logging.getLogger(__name__)

PydanticModelT = TypeVar("PydanticModelT", bound=BaseModel)


class PydanticToDictProcessor(Processor[PydanticModelT, dict]):
    """Base class for processors that convert Pydantic models to dictionaries.

    This base class provides common functionality for converting Pydantic models
    to dictionaries with configurable serialization options.

    Key Features:
    - Configurable serialization options (by_alias, exclude, include, etc.)
    - Support for both model_dump() and model_dump_json() approaches
    - Proper handling of None values
    - Type-safe generic implementation

    Example:
        .. code-block:: python

            class MyModelToDictProcessor(PydanticToDictProcessor[MyModel]):
                def __init__(self, **kwargs):
                    super().__init__(by_alias=True, exclude_none=True, **kwargs)
    """

    def __init__(
        self,
        by_alias: bool | None = None,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude: set[str] | None = None,
        include: set[str] | None = None,
        use_json_serialization: bool = False,
        **kwargs,
    ):
        """Initialize the Pydantic to dict processor.

        Args:
            by_alias (bool): Whether to use field aliases in output. Defaults to False.
            exclude_none (bool): Whether to exclude None values. Defaults to False.
            exclude_unset (bool): Whether to exclude unset fields. Defaults to False.
            exclude_defaults (bool): Whether to exclude default values. Defaults to False.
            exclude (set[str] | None): Fields to exclude from output. Defaults to None.
            include (set[str] | None): Fields to include in output. Defaults to None.
            use_json_serialization (bool): Whether to use JSON serialization. Defaults to False.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._by_alias = by_alias
        self._exclude_none = exclude_none
        self._exclude_unset = exclude_unset
        self._exclude_defaults = exclude_defaults
        self._exclude = exclude
        self._include = include
        self._use_json_serialization = use_json_serialization

    async def process(self, item: PydanticModelT | None) -> dict:
        """Convert a Pydantic model to a dictionary.

        Args:
            item (PydanticModelT | None): The Pydantic model to convert.

        Returns:
            dict: The converted dictionary.
        """
        if item is None:
            logger.debug("Cannot process 'None' item, returning empty dict")
            return {}

        try:
            if self._use_json_serialization:
                return self._convert_via_json(item)
            return self._convert_direct(item)
        except Exception as e:
            logger.error("Failed to convert Pydantic model to dict: %s", e)
            raise

    def _convert_direct(self, item: PydanticModelT) -> dict:
        """Convert using model_dump() directly."""
        return item.model_dump(
            by_alias=self._by_alias,
            exclude_none=self._exclude_none,
            exclude_unset=self._exclude_unset,
            exclude_defaults=self._exclude_defaults,
            exclude=self._exclude,
            include=self._include,
        )

    def _convert_via_json(self, item: PydanticModelT) -> dict:
        """Convert using model_dump_json() for better field alias handling."""
        json_str = item.model_dump_json(
            by_alias=self._by_alias,
            exclude_none=self._exclude_none,
            exclude_unset=self._exclude_unset,
            exclude_defaults=self._exclude_defaults,
            exclude=self._exclude,
            include=self._include,
        )
        return json.loads(json_str)


class SpanToDictProcessor(PydanticToDictProcessor[Span]):
    """Concrete processor that converts Span objects to dictionaries.

    This processor is optimized for Span objects and includes common
    serialization options for telemetry data.

    Example:
        .. code-block:: python

            processor = SpanToDictProcessor(
                by_alias=True,
                exclude_none=True,
                exclude_unset=True
            )
    """

    def __init__(self, **kwargs):
        """Initialize with Span-optimized defaults."""
        # Set sensible defaults for telemetry data
        kwargs.setdefault("by_alias", True)
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("exclude_unset", True)
        super().__init__(**kwargs)


class HECToDictProcessor(PydanticToDictProcessor[HECSpan]):
    """Concrete processor that converts HECSpan objects to dictionaries.

    This processor is optimized for HECSpan objects and includes common
    serialization options for telemetry data.

    Example:
        .. code-block:: python

            processor = SpanToDictProcessor(
                by_alias=True,
                exclude_none=True,
                exclude_unset=True
            )
    """

    def __init__(self, **kwargs):
        """Initialize with Span-optimized defaults."""
        # Set sensible defaults for telemetry data
        kwargs.setdefault("by_alias", True)
        kwargs.setdefault("exclude_none", True)
        kwargs.setdefault("exclude_unset", True)
        super().__init__(**kwargs)
