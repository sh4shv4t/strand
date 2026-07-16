"""Basic observability: structured logging + OpenTelemetry tracing.

Uses OpenTelemetry's ConsoleSpanExporter -- spans print to stdout next to
the request logs. No collector, agent, or external service required, so
there's nothing to stand up or maintain; swapping to a real backend later
(Jaeger, Honeycomb, etc.) is a one-line change to the exporter, not a
rewrite of the instrumentation.
"""

import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("strand")

_provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "strand-api"}))
_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(_provider)

tracer = trace.get_tracer("strand")
