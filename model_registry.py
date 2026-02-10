"""
Model Registry — Versioning, Metadata & Rollback
=================================================
Stores model artifacts with semantic versions, training metrics,
and supports rollback to any previous version.
"""

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Metadata for a single model version."""
    ticker: str
    version: str  # "v1", "v2", …
    created_at: str = ""
    file_hash: str = ""
    file_size: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    training_duration_s: float = 0.0
    is_active: bool = False
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModelRegistry:
    """Keeps track of every model version ever trained.

    Each ticker may have multiple versions.  Only one is *active* at any time.
    Older versions are stored in ``backups/`` and can be promoted back.
    """

    def __init__(
        self,
        models_dir: str = "./models",
        registry_file: str = "./data/model_registry.json",
        max_versions_per_ticker: int = 10,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.backup_dir = self.models_dir / "backups"
        self.registry_file = Path(registry_file)
        self.max_versions = max_versions_per_ticker

        # ticker → [ModelVersion, …] (newest last)
        self.versions: Dict[str, List[ModelVersion]] = {}

        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        self._load()
        logger.info(
            "ModelRegistry loaded — %d tickers, %d total versions",
            len(self.versions),
            sum(len(v) for v in self.versions.values()),
        )

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Load registry from JSON."""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, "r") as fh:
                    data = json.load(fh)
                for ticker, ver_list in data.get("versions", {}).items():
                    self.versions[ticker] = [ModelVersion.from_dict(v) for v in ver_list]
        except Exception as exc:
            logger.warning("Could not load model registry: %s", exc)

    def save(self) -> None:
        """Persist registry to JSON."""
        try:
            payload = {
                "versions": {
                    t: [v.to_dict() for v in vs] for t, vs in self.versions.items()
                },
                "updated_at": datetime.utcnow().isoformat(),
            }
            with open(self.registry_file, "w") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:
            logger.error("Could not save model registry: %s", exc)

    # ------------------------------------------------------------------ #
    #  Register a new version
    # ------------------------------------------------------------------ #

    def register(
        self,
        ticker: str,
        metrics: Optional[Dict[str, Any]] = None,
        training_duration_s: float = 0.0,
        tags: Optional[List[str]] = None,
    ) -> ModelVersion:
        """Register the current model files on disk as a new version.

        - Archives the previous active version to ``backups/{ticker}/``.
        - Computes the file hash for integrity checks.
        - Enforces ``max_versions_per_ticker`` by pruning the oldest backups.

        Returns:
            The created :class:`ModelVersion`.
        """
        ticker = ticker.upper()
        model_path = self.models_dir / f"{ticker}_model.h5"
        scaler_path = self.models_dir / f"{ticker}_scaler.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Compute next version number
        existing = self.versions.get(ticker, [])
        next_num = 1
        if existing:
            try:
                nums = [int(v.version.lstrip("v")) for v in existing]
                next_num = max(nums) + 1
            except ValueError:
                next_num = len(existing) + 1

        version_str = f"v{next_num}"

        # Hash the model file
        file_hash = self._hash_file(model_path)
        file_size = model_path.stat().st_size

        # Archive previous active version
        self._archive_active(ticker)

        # Create version record
        mv = ModelVersion(
            ticker=ticker,
            version=version_str,
            created_at=datetime.utcnow().isoformat(),
            file_hash=file_hash,
            file_size=file_size,
            metrics=metrics or {},
            training_duration_s=training_duration_s,
            is_active=True,
            tags=tags or [],
        )

        # Deactivate previous versions
        for v in existing:
            v.is_active = False

        if ticker not in self.versions:
            self.versions[ticker] = []
        self.versions[ticker].append(mv)

        # Prune old backups
        self._prune(ticker)

        self.save()
        logger.info("Registered %s %s (hash=%s)", ticker, version_str, file_hash[:8])
        return mv

    # ------------------------------------------------------------------ #
    #  Rollback
    # ------------------------------------------------------------------ #

    def rollback(self, ticker: str, target_version: Optional[str] = None) -> ModelVersion:
        """Rollback *ticker* to *target_version* (defaults to previous).

        Copies the backup files back to the models directory and updates the
        active pointer.

        Returns:
            The now-active :class:`ModelVersion`.

        Raises:
            FileNotFoundError: If no backup is available.
        """
        ticker = ticker.upper()
        versions = self.versions.get(ticker, [])
        if not versions:
            raise FileNotFoundError(f"No versions found for {ticker}")

        # Find target
        if target_version:
            target = next((v for v in versions if v.version == target_version), None)
            if target is None:
                raise FileNotFoundError(
                    f"Version {target_version} not found for {ticker}"
                )
        else:
            # Previous version (second-to-last that isn't already active)
            inactive = [v for v in versions if not v.is_active]
            if not inactive:
                raise FileNotFoundError(f"No previous versions to rollback for {ticker}")
            target = inactive[-1]

        # Locate backup files
        backup_dir = self.backup_dir / ticker
        model_backup = backup_dir / f"{ticker}_model_{target.version}.h5"
        scaler_backup = backup_dir / f"{ticker}_scaler_{target.version}.pkl"

        if not model_backup.exists():
            raise FileNotFoundError(f"Backup model not found: {model_backup}")

        # Archive current active first
        self._archive_active(ticker, suffix="_pre_rollback")

        # Copy backup to active location
        model_path = self.models_dir / f"{ticker}_model.h5"
        scaler_path = self.models_dir / f"{ticker}_scaler.pkl"
        shutil.copy2(model_backup, model_path)
        if scaler_backup.exists():
            shutil.copy2(scaler_backup, scaler_path)

        # Update active flags
        for v in versions:
            v.is_active = v.version == target.version
        target.is_active = True

        self.save()
        logger.info("Rolled back %s to %s", ticker, target.version)
        return target

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

    def get_active_version(self, ticker: str) -> Optional[ModelVersion]:
        """Return the currently active version for *ticker*, or ``None``."""
        ticker = ticker.upper()
        for v in reversed(self.versions.get(ticker, [])):
            if v.is_active:
                return v
        return None

    def get_all_versions(self, ticker: str) -> List[ModelVersion]:
        """Return all versions for *ticker*, newest last."""
        return self.versions.get(ticker.upper(), [])

    def get_version_count(self, ticker: str) -> int:
        """Return the number of versions for *ticker*."""
        return len(self.versions.get(ticker.upper(), []))

    def list_tickers(self) -> List[str]:
        """Return all tickers that have at least one version."""
        return list(self.versions.keys())

    def get_summary(self) -> Dict[str, Any]:
        """High-level registry summary."""
        total_versions = sum(len(vs) for vs in self.versions.values())
        return {
            "tickers": len(self.versions),
            "total_versions": total_versions,
            "max_versions_per_ticker": self.max_versions,
            "tickers_list": self.list_tickers(),
        }

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _archive_active(self, ticker: str, suffix: str = "") -> None:
        """Copy the current active model files to the backup directory."""
        model_path = self.models_dir / f"{ticker}_model.h5"
        scaler_path = self.models_dir / f"{ticker}_scaler.pkl"
        if not model_path.exists():
            return

        active = self.get_active_version(ticker)
        ver_label = active.version if active else "v0"

        backup_dir = self.backup_dir / ticker
        backup_dir.mkdir(parents=True, exist_ok=True)

        dest_model = backup_dir / f"{ticker}_model_{ver_label}{suffix}.h5"
        dest_scaler = backup_dir / f"{ticker}_scaler_{ver_label}{suffix}.pkl"

        try:
            shutil.copy2(model_path, dest_model)
            if scaler_path.exists():
                shutil.copy2(scaler_path, dest_scaler)
        except Exception as exc:
            logger.warning("Could not archive %s %s: %s", ticker, ver_label, exc)

    def _prune(self, ticker: str) -> None:
        """Remove the oldest versions if we exceed ``max_versions``."""
        versions = self.versions.get(ticker, [])
        if len(versions) <= self.max_versions:
            return

        to_remove = versions[: len(versions) - self.max_versions]
        self.versions[ticker] = versions[len(to_remove):]

        # Remove backup files
        backup_dir = self.backup_dir / ticker
        for v in to_remove:
            for ext in (".h5", ".pkl"):
                path = backup_dir / f"{ticker}_model_{v.version}{ext}"
                if path.exists():
                    try:
                        path.unlink()
                    except Exception as exc:
                        logger.warning("Could not remove %s: %s", path, exc)

        logger.info("Pruned %d old versions for %s", len(to_remove), ticker)

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Compute MD5 hash of a file."""
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
