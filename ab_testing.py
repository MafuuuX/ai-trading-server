"""
A/B Testing Framework for Models
=================================
Allows serving different model versions to different clients and tracking
which version performs better in production.

Features:
- Experiment creation with control/treatment groups
- Deterministic client assignment (hash-based)
- Metric collection per group
- Statistical significance testing
"""

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ExperimentGroup:
    """One arm of an A/B experiment."""
    name: str  # "control" or "treatment"
    model_version: str  # e.g. "v3" or "v5"
    model_type: str = "per_ticker"  # "per_ticker" or "universal"
    traffic_pct: float = 50.0  # percentage of traffic
    # Accumulated metrics
    predictions: int = 0
    correct_predictions: int = 0
    total_pnl: float = 0.0
    total_confidence: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0


@dataclass
class Experiment:
    """A/B test experiment comparing two model versions."""
    experiment_id: str
    name: str
    description: str = ""
    ticker: str = ""  # empty = applies to all tickers
    status: str = "active"  # active, paused, completed
    created_at: str = ""
    completed_at: str = ""
    control: ExperimentGroup = field(default_factory=lambda: ExperimentGroup("control", "v1"))
    treatment: ExperimentGroup = field(default_factory=lambda: ExperimentGroup("treatment", "v2"))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        control_data = data.pop("control", {})
        treatment_data = data.pop("treatment", {})
        # Filter out unknown fields
        known = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known}
        exp = cls(**filtered)
        if control_data:
            exp.control = ExperimentGroup(**{
                k: v for k, v in control_data.items()
                if k in ExperimentGroup.__dataclass_fields__
            })
        if treatment_data:
            exp.treatment = ExperimentGroup(**{
                k: v for k, v in treatment_data.items()
                if k in ExperimentGroup.__dataclass_fields__
            })
        return exp


class ABTestingManager:
    """Manages A/B testing experiments for model comparison.

    Assigns clients to groups deterministically based on a hash of
    their client ID + experiment ID, so the same client always sees
    the same model version within one experiment.
    """

    def __init__(self, experiments_file: str = "./data/ab_experiments.json") -> None:
        self.experiments_file = Path(experiments_file)
        self.experiments_file.parent.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, Experiment] = {}
        self._load()
        logger.info("ABTestingManager loaded — %d experiments", len(self.experiments))

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        try:
            if self.experiments_file.exists():
                with open(self.experiments_file, "r") as fh:
                    data = json.load(fh)
                for exp_data in data.get("experiments", []):
                    try:
                        exp = Experiment.from_dict(exp_data)
                        self.experiments[exp.experiment_id] = exp
                    except Exception as exc:
                        logger.warning("Could not parse experiment: %s", exc)
        except Exception as exc:
            logger.warning("Could not load AB experiments: %s", exc)

    def save(self) -> None:
        try:
            payload = {
                "experiments": [e.to_dict() for e in self.experiments.values()],
                "updated_at": datetime.utcnow().isoformat(),
            }
            with open(self.experiments_file, "w") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:
            logger.error("Could not save AB experiments: %s", exc)

    # ------------------------------------------------------------------ #
    #  Experiment lifecycle
    # ------------------------------------------------------------------ #

    def create_experiment(
        self,
        name: str,
        control_version: str,
        treatment_version: str,
        ticker: str = "",
        description: str = "",
        control_traffic_pct: float = 50.0,
        model_type: str = "per_ticker",
    ) -> Experiment:
        """Create a new A/B experiment.

        Args:
            name: Human-readable experiment name.
            control_version: Model version for the control group.
            treatment_version: Model version for the treatment group.
            ticker: Restrict to one ticker (empty = all).
            description: Optional description.
            control_traffic_pct: Percentage of traffic for control (rest goes to treatment).
            model_type: ``per_ticker`` or ``universal``.

        Returns:
            The created :class:`Experiment`.
        """
        exp_id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:12]

        exp = Experiment(
            experiment_id=exp_id,
            name=name,
            description=description,
            ticker=ticker.upper() if ticker else "",
            status="active",
            created_at=datetime.utcnow().isoformat(),
            control=ExperimentGroup(
                name="control",
                model_version=control_version,
                model_type=model_type,
                traffic_pct=control_traffic_pct,
            ),
            treatment=ExperimentGroup(
                name="treatment",
                model_version=treatment_version,
                model_type=model_type,
                traffic_pct=100.0 - control_traffic_pct,
            ),
        )

        self.experiments[exp_id] = exp
        self.save()
        logger.info(
            "Created experiment %s: %s vs %s for %s",
            exp_id,
            control_version,
            treatment_version,
            ticker or "all tickers",
        )
        return exp

    def complete_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Mark an experiment as completed and return results.

        Returns:
            Results dict with winner, metrics and significance, or ``None``
            if the experiment was not found.
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return None

        exp.status = "completed"
        exp.completed_at = datetime.utcnow().isoformat()
        self.save()

        return self.get_results(experiment_id)

    def pause_experiment(self, experiment_id: str) -> bool:
        """Pause an active experiment."""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return False
        exp.status = "paused"
        self.save()
        return True

    def resume_experiment(self, experiment_id: str) -> bool:
        """Resume a paused experiment."""
        exp = self.experiments.get(experiment_id)
        if not exp or exp.status != "paused":
            return False
        exp.status = "active"
        self.save()
        return True

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        if experiment_id in self.experiments:
            del self.experiments[experiment_id]
            self.save()
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Client assignment
    # ------------------------------------------------------------------ #

    def assign_group(
        self, experiment_id: str, client_id: str
    ) -> Tuple[str, ExperimentGroup]:
        """Deterministically assign a client to a group.

        Uses a hash of ``client_id + experiment_id`` mapped to [0, 100)
        to ensure the same client always lands in the same group.

        Args:
            experiment_id: The experiment to assign for.
            client_id: Client identifier (IP, API key, …).

        Returns:
            ``(group_name, ExperimentGroup)`` tuple.

        Raises:
            KeyError: If experiment not found.
            ValueError: If experiment is not ``active``.
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise KeyError(f"Experiment {experiment_id} not found")
        if exp.status != "active":
            raise ValueError(f"Experiment {experiment_id} is {exp.status}")

        # Deterministic hash → [0, 100)
        h = hashlib.md5(f"{client_id}:{experiment_id}".encode()).hexdigest()
        bucket = int(h[:8], 16) % 100

        if bucket < exp.control.traffic_pct:
            return "control", exp.control
        return "treatment", exp.treatment

    def get_active_experiment(self, ticker: str = "") -> Optional[Experiment]:
        """Return the first active experiment matching *ticker*.

        If *ticker* is given, looks for experiments scoped to that ticker
        first, then falls back to experiments with no ticker restriction.
        """
        ticker = ticker.upper() if ticker else ""

        # Exact ticker match first
        for exp in self.experiments.values():
            if exp.status == "active" and exp.ticker == ticker and ticker:
                return exp

        # Global experiments
        for exp in self.experiments.values():
            if exp.status == "active" and not exp.ticker:
                return exp

        return None

    # ------------------------------------------------------------------ #
    #  Metric recording
    # ------------------------------------------------------------------ #

    def record_prediction(
        self,
        experiment_id: str,
        group_name: str,
        correct: bool,
        confidence: float = 0.0,
    ) -> None:
        """Record a prediction outcome for a group."""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return

        group = exp.control if group_name == "control" else exp.treatment
        group.predictions += 1
        group.total_confidence += confidence
        if correct:
            group.correct_predictions += 1

        # Save periodically (every 50 predictions)
        total = exp.control.predictions + exp.treatment.predictions
        if total % 50 == 0:
            self.save()

    def record_trade(
        self,
        experiment_id: str,
        group_name: str,
        pnl: float,
        outcome: str,
    ) -> None:
        """Record a trade outcome for a group.

        Args:
            experiment_id: Experiment ID.
            group_name: ``control`` or ``treatment``.
            pnl: Realized P&L.
            outcome: ``WIN``, ``LOSS``, or ``BREAKEVEN``.
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return

        group = exp.control if group_name == "control" else exp.treatment
        group.trades += 1
        group.total_pnl += pnl
        if outcome == "WIN":
            group.wins += 1
        elif outcome == "LOSS":
            group.losses += 1

        if (exp.control.trades + exp.treatment.trades) % 20 == 0:
            self.save()

    # ------------------------------------------------------------------ #
    #  Results & significance
    # ------------------------------------------------------------------ #

    def get_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Compute results for an experiment.

        Returns accuracy, win rate, average P&L and a simple z-test
        for statistical significance.
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return None

        def _group_metrics(g: ExperimentGroup) -> Dict[str, Any]:
            accuracy = (
                g.correct_predictions / g.predictions * 100 if g.predictions > 0 else 0
            )
            win_rate = g.wins / g.trades * 100 if g.trades > 0 else 0
            avg_pnl = g.total_pnl / g.trades if g.trades > 0 else 0
            avg_conf = g.total_confidence / g.predictions if g.predictions > 0 else 0
            return {
                "name": g.name,
                "model_version": g.model_version,
                "predictions": g.predictions,
                "accuracy": round(accuracy, 2),
                "trades": g.trades,
                "wins": g.wins,
                "losses": g.losses,
                "win_rate": round(win_rate, 2),
                "total_pnl": round(g.total_pnl, 2),
                "avg_pnl": round(avg_pnl, 2),
                "avg_confidence": round(avg_conf, 4),
            }

        control_m = _group_metrics(exp.control)
        treatment_m = _group_metrics(exp.treatment)

        # Determine winner
        winner = "inconclusive"
        if control_m["accuracy"] > treatment_m["accuracy"] + 2:
            winner = "control"
        elif treatment_m["accuracy"] > control_m["accuracy"] + 2:
            winner = "treatment"

        # Simple z-test for proportions
        significance = self._z_test_proportions(
            exp.control.correct_predictions,
            exp.control.predictions,
            exp.treatment.correct_predictions,
            exp.treatment.predictions,
        )

        return {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "status": exp.status,
            "ticker": exp.ticker,
            "control": control_m,
            "treatment": treatment_m,
            "winner": winner,
            "significant": significance.get("significant", False),
            "p_value": significance.get("p_value"),
            "z_score": significance.get("z_score"),
        }

    @staticmethod
    def _z_test_proportions(
        successes_a: int, n_a: int, successes_b: int, n_b: int
    ) -> Dict[str, Any]:
        """Two-proportion z-test for statistical significance.

        Returns:
            Dict with ``z_score``, ``p_value``, ``significant`` keys.
        """
        if n_a < 10 or n_b < 10:
            return {"z_score": None, "p_value": None, "significant": False,
                    "reason": "Insufficient samples (need ≥10 per group)"}

        p_a = successes_a / n_a
        p_b = successes_b / n_b
        p_pool = (successes_a + successes_b) / (n_a + n_b)

        # Avoid division by zero
        denom = p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)
        if denom <= 0:
            return {"z_score": 0.0, "p_value": 1.0, "significant": False}

        z = (p_a - p_b) / math.sqrt(denom)

        # Approximate p-value (two-tailed) using error function
        p_value = 2 * (1 - _norm_cdf(abs(z)))

        return {
            "z_score": round(z, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
        }

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

    def list_experiments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments, optionally filtered by status."""
        result = []
        for exp in self.experiments.values():
            if status and exp.status != status:
                continue
            result.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "status": exp.status,
                "ticker": exp.ticker,
                "control_version": exp.control.model_version,
                "treatment_version": exp.treatment.model_version,
                "total_predictions": exp.control.predictions + exp.treatment.predictions,
                "total_trades": exp.control.trades + exp.treatment.trades,
                "created_at": exp.created_at,
            })
        return result


def _norm_cdf(x: float) -> float:
    """Approximation of the standard normal CDF (no scipy needed)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
