"""
Ensemble Prediction Engine
===========================
Combines predictions from multiple model versions/types for more robust signals.

Strategies:
- Weighted averaging based on historical accuracy
- Majority voting for classification (UP/DOWN/NEUTRAL)
- Confidence-weighted blending for regression predictions
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Single prediction from one model."""
    model_id: str
    model_type: str  # 'per_ticker' or 'universal'
    signal: str  # UP, DOWN, NEUTRAL
    confidence: float
    predicted_change: float
    version: str = "v1"
    latency_ms: float = 0.0


@dataclass
class EnsemblePrediction:
    """Aggregated ensemble prediction."""
    ticker: str
    signal: str
    confidence: float
    predicted_change: float
    agreement_ratio: float  # How many models agree on the signal
    individual_predictions: List[Dict[str, Any]] = field(default_factory=list)
    strategy: str = "weighted_average"
    model_count: int = 0


class EnsemblePredictor:
    """Combines predictions from multiple models using configurable strategies.

    Supported strategies:
        - ``weighted_average``: weight each model by its historical accuracy
        - ``majority_vote``: pick the signal most models agree on
        - ``confidence_weighted``: blend by raw confidence scores
    """

    STRATEGIES = ("weighted_average", "majority_vote", "confidence_weighted")

    def __init__(
        self,
        strategy: str = "weighted_average",
        weights_file: str = "./data/ensemble_weights.json",
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}")

        self.strategy = strategy
        self.weights_file = Path(weights_file)
        self.weights_file.parent.mkdir(parents=True, exist_ok=True)

        # model_id → accuracy weight (default 1.0)
        self.model_weights: Dict[str, float] = {}
        self._load_weights()

        # Track prediction history for weight updates
        self.prediction_history: List[Dict[str, Any]] = []
        self.max_history = 500

        logger.info(
            "EnsemblePredictor initialised (strategy=%s, models=%d)",
            self.strategy,
            len(self.model_weights),
        )

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def _load_weights(self) -> None:
        """Load model weights from disk."""
        try:
            if self.weights_file.exists():
                with open(self.weights_file, "r") as fh:
                    data = json.load(fh)
                self.model_weights = data.get("weights", {})
                self.prediction_history = data.get("history", [])
                logger.info("Loaded ensemble weights for %d models", len(self.model_weights))
        except Exception as exc:
            logger.warning("Could not load ensemble weights: %s", exc)
            self.model_weights = {}

    def save_weights(self) -> None:
        """Persist model weights to disk."""
        try:
            with open(self.weights_file, "w") as fh:
                json.dump(
                    {
                        "weights": self.model_weights,
                        "history": self.prediction_history[-self.max_history:],
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    fh,
                    indent=2,
                )
        except Exception as exc:
            logger.error("Could not save ensemble weights: %s", exc)

    # ------------------------------------------------------------------ #
    #  Core prediction logic
    # ------------------------------------------------------------------ #

    def predict(
        self,
        ticker: str,
        predictions: List[ModelPrediction],
    ) -> EnsemblePrediction:
        """Combine *predictions* into a single ensemble prediction.

        Args:
            ticker: Stock ticker symbol.
            predictions: List of :class:`ModelPrediction` from various models.

        Returns:
            An :class:`EnsemblePrediction`.
        """
        if not predictions:
            return EnsemblePrediction(
                ticker=ticker,
                signal="NEUTRAL",
                confidence=0.0,
                predicted_change=0.0,
                agreement_ratio=0.0,
                strategy=self.strategy,
            )

        if len(predictions) == 1:
            p = predictions[0]
            return EnsemblePrediction(
                ticker=ticker,
                signal=p.signal,
                confidence=p.confidence,
                predicted_change=p.predicted_change,
                agreement_ratio=1.0,
                individual_predictions=[self._pred_to_dict(p)],
                strategy="single_model",
                model_count=1,
            )

        if self.strategy == "majority_vote":
            return self._majority_vote(ticker, predictions)
        elif self.strategy == "confidence_weighted":
            return self._confidence_weighted(ticker, predictions)
        else:
            return self._weighted_average(ticker, predictions)

    # ------------------------------------------------------------------ #
    #  Strategy implementations
    # ------------------------------------------------------------------ #

    def _weighted_average(
        self, ticker: str, preds: List[ModelPrediction]
    ) -> EnsemblePrediction:
        """Weight each model's prediction by its historical accuracy."""
        weights: List[float] = []
        for p in preds:
            w = self.model_weights.get(p.model_id, 1.0)
            weights.append(max(w, 0.1))  # floor at 0.1

        total_w = sum(weights)
        norm = [w / total_w for w in weights]

        # Weighted confidence and predicted change
        conf = sum(p.confidence * n for p, n in zip(preds, norm))
        change = sum(p.predicted_change * n for p, n in zip(preds, norm))

        # Weighted signal vote
        signal_scores: Dict[str, float] = {"UP": 0.0, "DOWN": 0.0, "NEUTRAL": 0.0}
        for p, n in zip(preds, norm):
            signal_scores[p.signal] += n

        signal = max(signal_scores, key=signal_scores.get)  # type: ignore[arg-type]
        agreement = signal_scores[signal]

        return EnsemblePrediction(
            ticker=ticker,
            signal=signal,
            confidence=round(conf, 4),
            predicted_change=round(change, 4),
            agreement_ratio=round(agreement, 4),
            individual_predictions=[self._pred_to_dict(p) for p in preds],
            strategy="weighted_average",
            model_count=len(preds),
        )

    def _majority_vote(
        self, ticker: str, preds: List[ModelPrediction]
    ) -> EnsemblePrediction:
        """Pick the signal that the majority of models agree on."""
        vote_count: Dict[str, int] = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}
        for p in preds:
            vote_count[p.signal] = vote_count.get(p.signal, 0) + 1

        signal = max(vote_count, key=vote_count.get)  # type: ignore[arg-type]
        agreement = vote_count[signal] / len(preds)

        # Average confidence from models that voted for the winning signal
        winning = [p for p in preds if p.signal == signal]
        conf = sum(p.confidence for p in winning) / len(winning) if winning else 0.0
        change = sum(p.predicted_change for p in winning) / len(winning) if winning else 0.0

        return EnsemblePrediction(
            ticker=ticker,
            signal=signal,
            confidence=round(conf, 4),
            predicted_change=round(change, 4),
            agreement_ratio=round(agreement, 4),
            individual_predictions=[self._pred_to_dict(p) for p in preds],
            strategy="majority_vote",
            model_count=len(preds),
        )

    def _confidence_weighted(
        self, ticker: str, preds: List[ModelPrediction]
    ) -> EnsemblePrediction:
        """Blend predictions weighted by each model's raw confidence."""
        total_conf = sum(p.confidence for p in preds)
        if total_conf == 0:
            total_conf = 1.0

        norm = [p.confidence / total_conf for p in preds]

        signal_scores: Dict[str, float] = {"UP": 0.0, "DOWN": 0.0, "NEUTRAL": 0.0}
        change = 0.0
        for p, n in zip(preds, norm):
            signal_scores[p.signal] += n
            change += p.predicted_change * n

        signal = max(signal_scores, key=signal_scores.get)  # type: ignore[arg-type]
        agreement = signal_scores[signal]
        conf = sum(p.confidence * n for p, n in zip(preds, norm))

        return EnsemblePrediction(
            ticker=ticker,
            signal=signal,
            confidence=round(conf, 4),
            predicted_change=round(change, 4),
            agreement_ratio=round(agreement, 4),
            individual_predictions=[self._pred_to_dict(p) for p in preds],
            strategy="confidence_weighted",
            model_count=len(preds),
        )

    # ------------------------------------------------------------------ #
    #  Feedback / weight update
    # ------------------------------------------------------------------ #

    def record_outcome(
        self,
        ticker: str,
        actual_change: float,
        predictions: List[ModelPrediction],
    ) -> None:
        """Update model weights based on actual market outcome.

        Args:
            ticker: Stock ticker.
            actual_change: Actual price-change percentage observed.
            predictions: The predictions that were made.
        """
        actual_signal = "UP" if actual_change > 0.5 else ("DOWN" if actual_change < -0.5 else "NEUTRAL")

        for p in predictions:
            correct = p.signal == actual_signal
            direction_error = abs(p.predicted_change - actual_change)

            # Exponential moving average update
            alpha = 0.1
            old_w = self.model_weights.get(p.model_id, 1.0)

            if correct:
                reward = 1.0 + min(p.confidence, 0.5)
            else:
                reward = max(0.3, 1.0 - direction_error / 10.0)

            new_w = old_w * (1 - alpha) + reward * alpha
            self.model_weights[p.model_id] = round(max(0.1, min(3.0, new_w)), 4)

        self.prediction_history.append({
            "ticker": ticker,
            "actual_change": actual_change,
            "actual_signal": actual_signal,
            "model_count": len(predictions),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        if len(self.prediction_history) > self.max_history:
            self.prediction_history = self.prediction_history[-self.max_history:]

        self.save_weights()

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pred_to_dict(p: ModelPrediction) -> Dict[str, Any]:
        return {
            "model_id": p.model_id,
            "model_type": p.model_type,
            "signal": p.signal,
            "confidence": p.confidence,
            "predicted_change": p.predicted_change,
            "version": p.version,
            "latency_ms": p.latency_ms,
        }

    def get_model_weights(self) -> Dict[str, float]:
        """Return current model weights."""
        return dict(self.model_weights)

    def set_strategy(self, strategy: str) -> None:
        """Change the ensemble strategy at runtime."""
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}")
        self.strategy = strategy
        logger.info("Ensemble strategy changed to: %s", strategy)
