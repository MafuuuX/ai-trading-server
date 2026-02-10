"""
Prometheus Metrics Collector
=============================
Exposes ``/metrics`` in Prometheus text format and tracks:
- HTTP request latency & count
- WebSocket connections
- Training duration & queue depth
- Model prediction latency
- Cache hit/miss ratios
- System resource usage
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class Counter:
    """Simple monotonically increasing counter."""

    def __init__(self, name: str, help_text: str = "", labels: Optional[List[str]] = None):
        self.name = name
        self.help_text = help_text
        self.labels = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)

    def inc(self, amount: float = 1.0, **label_values) -> None:
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] += amount

    def get(self, **label_values) -> float:
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._values.get(key, 0.0)

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} counter"]
        if not self._values:
            lines.append(f"{self.name} 0")
        else:
            for key, val in self._values.items():
                if self.labels:
                    label_str = ",".join(
                        f'{l}="{v}"' for l, v in zip(self.labels, key)
                    )
                    lines.append(f"{self.name}{{{label_str}}} {val}")
                else:
                    lines.append(f"{self.name} {val}")
        return "\n".join(lines)


class Gauge:
    """Gauge that can go up or down."""

    def __init__(self, name: str, help_text: str = "", labels: Optional[List[str]] = None):
        self.name = name
        self.help_text = help_text
        self.labels = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)

    def set(self, value: float, **label_values) -> None:
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] = value

    def inc(self, amount: float = 1.0, **label_values) -> None:
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] += amount

    def dec(self, amount: float = 1.0, **label_values) -> None:
        key = tuple(label_values.get(l, "") for l in self.labels)
        self._values[key] -= amount

    def get(self, **label_values) -> float:
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._values.get(key, 0.0)

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} gauge"]
        if not self._values:
            lines.append(f"{self.name} 0")
        else:
            for key, val in self._values.items():
                if self.labels:
                    label_str = ",".join(
                        f'{l}="{v}"' for l, v in zip(self.labels, key)
                    )
                    lines.append(f"{self.name}{{{label_str}}} {val}")
                else:
                    lines.append(f"{self.name} {val}")
        return "\n".join(lines)


class Histogram:
    """Simple histogram using fixed buckets."""

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        help_text: str = "",
        buckets: Optional[tuple] = None,
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.help_text = help_text
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self.labels = labels or []
        # key → {bucket_bound: count, "_sum": total, "_count": n}
        self._data: Dict[tuple, Dict] = defaultdict(
            lambda: {b: 0 for b in self.buckets} | {"_sum": 0.0, "_count": 0}
        )

    def observe(self, value: float, **label_values) -> None:
        key = tuple(label_values.get(l, "") for l in self.labels)
        d = self._data[key]
        d["_count"] += 1
        d["_sum"] += value
        for b in self.buckets:
            if value <= b:
                d[b] += 1

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} histogram"]
        for key, d in self._data.items():
            label_str = ""
            if self.labels:
                label_str = ",".join(f'{l}="{v}"' for l, v in zip(self.labels, key))

            cumulative = 0
            for b in self.buckets:
                cumulative += d[b]
                le_label = f'le="{b}"'
                if label_str:
                    lines.append(f'{self.name}_bucket{{{label_str},{le_label}}} {cumulative}')
                else:
                    lines.append(f'{self.name}_bucket{{{le_label}}} {cumulative}')

            inf_label = 'le="+Inf"'
            if label_str:
                lines.append(f'{self.name}_bucket{{{label_str},{inf_label}}} {d["_count"]}')
                lines.append(f'{self.name}_sum{{{label_str}}} {d["_sum"]:.6f}')
                lines.append(f'{self.name}_count{{{label_str}}} {d["_count"]}')
            else:
                lines.append(f'{self.name}_bucket{{{inf_label}}} {d["_count"]}')
                lines.append(f'{self.name}_sum {d["_sum"]:.6f}')
                lines.append(f'{self.name}_count {d["_count"]}')
        return "\n".join(lines)


class MetricsCollector:
    """Central metrics registry.

    Collects counters, gauges and histograms; renders them all in
    Prometheus exposition format via :meth:`render`.
    """

    def __init__(self) -> None:
        # ------- HTTP -------
        self.http_requests_total = Counter(
            "trading_http_requests_total",
            "Total HTTP requests",
            labels=["method", "path", "status"],
        )
        self.http_request_duration = Histogram(
            "trading_http_request_duration_seconds",
            "HTTP request latency",
            labels=["method", "path"],
        )

        # ------- WebSocket -------
        self.ws_connections = Gauge(
            "trading_ws_connections",
            "Current WebSocket connections",
        )
        self.ws_messages_total = Counter(
            "trading_ws_messages_total",
            "Total WebSocket messages",
            labels=["direction"],  # "inbound" / "outbound"
        )

        # ------- Training -------
        self.training_duration = Histogram(
            "trading_training_duration_seconds",
            "Model training duration",
            labels=["ticker", "model_type"],
            buckets=(10, 30, 60, 120, 300, 600, 1200, 3600),
        )
        self.training_queue_depth = Gauge(
            "trading_training_queue_depth",
            "Number of tickers waiting in training queue",
        )
        self.trainings_total = Counter(
            "trading_trainings_total",
            "Total training runs",
            labels=["ticker", "status"],  # "completed" / "failed"
        )

        # ------- Predictions -------
        self.prediction_latency = Histogram(
            "trading_prediction_latency_seconds",
            "Model prediction latency",
            labels=["model_type"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )
        self.predictions_total = Counter(
            "trading_predictions_total",
            "Total predictions made",
            labels=["ticker", "signal"],
        )

        # ------- Cache -------
        self.cache_hits = Counter("trading_cache_hits_total", "Cache hits")
        self.cache_misses = Counter("trading_cache_misses_total", "Cache misses")

        # ------- System -------
        self.uptime_seconds = Gauge(
            "trading_uptime_seconds", "Server uptime in seconds"
        )
        self.cpu_percent = Gauge("trading_cpu_percent", "CPU usage percent")
        self.memory_percent = Gauge("trading_memory_percent", "Memory usage percent")
        self.active_models = Gauge(
            "trading_active_models", "Number of active models"
        )

        # ------- Ensemble / A-B -------
        self.ensemble_predictions_total = Counter(
            "trading_ensemble_predictions_total",
            "Total ensemble predictions",
            labels=["strategy"],
        )
        self.ab_assignments_total = Counter(
            "trading_ab_assignments_total",
            "A/B test group assignments",
            labels=["experiment", "group"],
        )

        self._all = [
            self.http_requests_total,
            self.http_request_duration,
            self.ws_connections,
            self.ws_messages_total,
            self.training_duration,
            self.training_queue_depth,
            self.trainings_total,
            self.prediction_latency,
            self.predictions_total,
            self.cache_hits,
            self.cache_misses,
            self.uptime_seconds,
            self.cpu_percent,
            self.memory_percent,
            self.active_models,
            self.ensemble_predictions_total,
            self.ab_assignments_total,
        ]

        logger.info("MetricsCollector initialised with %d metrics", len(self._all))

    def render(self) -> str:
        """Render all metrics in Prometheus exposition format."""
        sections = [m.render() for m in self._all]
        return "\n\n".join(sections) + "\n"

    def update_system_metrics(self, uptime_s: float) -> None:
        """Periodically called to refresh system-level gauges."""
        try:
            import psutil
            self.cpu_percent.set(psutil.cpu_percent(interval=0))
            self.memory_percent.set(psutil.virtual_memory().percent)
        except Exception:
            pass
        self.uptime_seconds.set(uptime_s)


# Singleton
metrics = MetricsCollector()
