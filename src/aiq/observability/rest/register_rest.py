import logging

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_telemetry_exporter
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig

logger = logging.getLogger(__name__)


class RestTelemetryExporterConfig(TelemetryExporterBaseConfig, name="rest"):
    """A telemetry exporter to transmit traces to a REST endpoint."""

    endpoint: str = Field(description="The REST service endpoint to export telemetry traces.")
    timeout: int = Field(default=60, description="The timeout for the REST request.")
    project: str = Field(description="The project name to group the telemetry traces.")


@register_telemetry_exporter(config_type=RestTelemetryExporterConfig)
async def rest_telemetry_exporter(config: RestTelemetryExporterConfig, builder: Builder):

    from aiq.observability.rest.rest_exporter import RestTelemetryExporter
    from aiq.observability.translators.simple_otel_trace import translate_span_to_simple_trace_model

    try:
        yield RestTelemetryExporter(endpoint=config.endpoint,
                                    timeout=config.timeout,
                                    translator=translate_span_to_simple_trace_model)
    except ConnectionError as ex:
        logger.warning("Unable to connect to REST service. Are you sure the service is running?\n %s",
                       ex,
                       exc_info=True)
    except Exception as ex:
        logger.error("Error in REST telemetry Exporter\n %s", ex, exc_info=True)
