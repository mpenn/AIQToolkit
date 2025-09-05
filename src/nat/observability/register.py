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
from typing import Literal

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_logging_method
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.logging import LoggingBaseConfig
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.mixin.batch_config_mixin import BatchConfigMixin
from nat.observability.mixin.file_mode import FileMode

logger = logging.getLogger(__name__)


class FileTelemetryExporterConfig(TelemetryExporterBaseConfig, name="file"):
    """A telemetry exporter that writes runtime traces to local files with optional rolling."""

    output_path: str = Field(description="Output path for logs. When rolling is disabled: exact file path. "
                             "When rolling is enabled: directory path or file path (directory + base name).")
    project: str = Field(description="Name to affiliate with this application.")
    mode: FileMode = Field(
        default=FileMode.APPEND,
        description="File write mode: 'append' to add to existing file or 'overwrite' to start fresh.")
    enable_rolling: bool = Field(default=False, description="Enable rolling log files based on size limits.")
    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum file size in bytes before rolling to a new file.")
    max_files: int = Field(default=5, description="Maximum number of rolled files to keep.")
    cleanup_on_init: bool = Field(default=False, description="Clean up old files during initialization.")


@register_telemetry_exporter(config_type=FileTelemetryExporterConfig)
async def file_telemetry_exporter(config: FileTelemetryExporterConfig, builder: Builder):
    """
    Build and return a FileExporter for file-based telemetry export with optional rolling.
    """

    from nat.observability.exporter.file_exporter import FileExporter

    yield FileExporter(output_path=config.output_path,
                       project=config.project,
                       mode=config.mode,
                       enable_rolling=config.enable_rolling,
                       max_file_size=config.max_file_size,
                       max_files=config.max_files,
                       cleanup_on_init=config.cleanup_on_init)


class SimpleHttpSpanTelemetryExporterConfig(TelemetryExporterBaseConfig, BatchConfigMixin, name="simple_http_span"):
    """A telemetry exporter that writes runtime traces to a HTTP endpoint."""

    endpoint: str = Field(description="The HTTP endpoint URL.")
    headers: dict[str, str] | None = Field(default=None, description="HTTP headers to include with requests.")
    timeout: float = Field(default=30.0, description="Request timeout in seconds.")
    method: Literal["POST"] = Field(default="POST", description="HTTP method to use.")
    timezone: str = Field(default="Z", description="The timezone of the event.")


@register_telemetry_exporter(config_type=SimpleHttpSpanTelemetryExporterConfig)
async def simple_http_telemetry_exporter(config: SimpleHttpSpanTelemetryExporterConfig, _builder: Builder):
    """
    Build and return a SimpleHttpSpanExporter for simple span telemetry export to a HTTP endpoint.
    """

    from nat.observability.exporter.simple_http_span_exporter import SimpleHttpSpanExporter

    yield SimpleHttpSpanExporter(endpoint=config.endpoint,
                                 headers=config.headers,
                                 timeout=config.timeout,
                                 method=config.method,
                                 timezone=config.timezone,
                                 batch_size=config.batch_size,
                                 flush_interval=config.flush_interval,
                                 max_queue_size=config.max_queue_size,
                                 drop_on_overflow=config.drop_on_overflow,
                                 shutdown_timeout=config.shutdown_timeout)


class HECSpanTelemetryExporterConfig(SimpleHttpSpanTelemetryExporterConfig, name="hec_span"):
    """A telemetry exporter that writes runtime traces to a HTTP Event Collector (HEC) endpoint."""
    pass


@register_telemetry_exporter(config_type=HECSpanTelemetryExporterConfig)
async def hec_span_telemetry_exporter(config: HECSpanTelemetryExporterConfig, _builder: Builder):
    """
    Build and return a HECSpanExporter for to an HTTP Event Collector (HEC) endpoint.
    """

    from nat.observability.exporter.hec_span_exporter import HECSpanExporter

    yield HECSpanExporter(endpoint=config.endpoint,
                          headers=config.headers,
                          timeout=config.timeout,
                          method=config.method,
                          batch_size=config.batch_size,
                          flush_interval=config.flush_interval,
                          max_queue_size=config.max_queue_size,
                          drop_on_overflow=config.drop_on_overflow,
                          shutdown_timeout=config.shutdown_timeout)


class ConsoleLoggingMethodConfig(LoggingBaseConfig, name="console"):
    """A logger to write runtime logs to the console."""

    level: str = Field(description="The logging level of console logger.")


@register_logging_method(config_type=ConsoleLoggingMethodConfig)
async def console_logging_method(config: ConsoleLoggingMethodConfig, builder: Builder):
    """
    Build and return a StreamHandler for console-based logging.
    """
    import sys

    level = getattr(logging, config.level.upper(), logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    yield handler


class FileLoggingMethod(LoggingBaseConfig, name="file"):
    """A logger to write runtime logs to a file."""

    path: str = Field(description="The file path to save the logging output.")
    level: str = Field(description="The logging level of file logger.")


@register_logging_method(config_type=FileLoggingMethod)
async def file_logging_method(config: FileLoggingMethod, builder: Builder):
    """
    Build and return a FileHandler for file-based logging.
    """
    level = getattr(logging, config.level.upper(), logging.INFO)
    handler = logging.FileHandler(filename=config.path, mode="a", encoding="utf-8")
    handler.setLevel(level)
    yield handler
