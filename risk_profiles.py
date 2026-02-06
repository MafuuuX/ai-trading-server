"""
Risk Profile Management for AI Trading Server

Provides predefined risk profiles (Conservative, Balanced, Aggressive)
and allows dynamic switching between them. Also supports custom risk
parameters with a conservativeness slider (1-10).
"""
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskProfile:
    """Risk profile configuration"""
    name: str
    level: int  # 1-10, 1=most conservative, 10=most aggressive
    
    # Position sizing (% of portfolio)
    position_size_min: float
    position_size_max: float
    position_size_default: float
    
    # Stop-loss settings (% loss to trigger)
    stop_loss_tight: float
    stop_loss_default: float
    stop_loss_wide: float
    
    # Take-profit settings (% gain to trigger)
    take_profit_min: float
    take_profit_default: float
    take_profit_max: float
    
    # Risk-reward ratio
    risk_reward_ratio: float
    
    # Concurrent trades
    max_concurrent_trades: int
    
    # Entry thresholds (min expected change to enter trade)
    long_entry_threshold: float
    short_entry_threshold: float
    
    # Confidence requirements
    min_confidence: float
    
    # Description for UI
    description: str
    color: str  # For UI display
    icon: str


# Predefined profiles
CONSERVATIVE_PROFILE = RiskProfile(
    name="Conservative",
    level=3,
    position_size_min=0.5,
    position_size_max=2.0,
    position_size_default=1.0,
    stop_loss_tight=0.5,
    stop_loss_default=1.0,
    stop_loss_wide=1.5,
    take_profit_min=0.5,
    take_profit_default=1.0,
    take_profit_max=2.0,
    risk_reward_ratio=1.5,
    max_concurrent_trades=3,
    long_entry_threshold=2.0,
    short_entry_threshold=-1.5,
    min_confidence=0.75,
    description="Kleine Positionen, enge Stop-Loss, höhere Gewinnwahrscheinlichkeit",
    color="#22c55e",  # Green
    icon="🛡️"
)

BALANCED_PROFILE = RiskProfile(
    name="Balanced",
    level=5,
    position_size_min=1.0,
    position_size_max=5.0,
    position_size_default=2.5,
    stop_loss_tight=1.0,
    stop_loss_default=2.0,
    stop_loss_wide=3.0,
    take_profit_min=1.0,
    take_profit_default=2.0,
    take_profit_max=4.0,
    risk_reward_ratio=2.0,
    max_concurrent_trades=5,
    long_entry_threshold=1.5,
    short_entry_threshold=-1.0,
    min_confidence=0.65,
    description="Ausgewogenes Risk-Reward-Verhältnis, moderate Positionen",
    color="#3b82f6",  # Blue
    icon="⚖️"
)

AGGRESSIVE_PROFILE = RiskProfile(
    name="Aggressive",
    level=8,
    position_size_min=3.0,
    position_size_max=10.0,
    position_size_default=5.0,
    stop_loss_tight=2.0,
    stop_loss_default=3.0,
    stop_loss_wide=5.0,
    take_profit_min=2.0,
    take_profit_default=4.0,
    take_profit_max=8.0,
    risk_reward_ratio=2.5,
    max_concurrent_trades=10,
    long_entry_threshold=1.0,
    short_entry_threshold=-0.5,
    min_confidence=0.55,
    description="Größere Positionen, höheres Gewinnpotenzial bei höherem Risiko",
    color="#ef4444",  # Red
    icon="🔥"
)

PROFILES: Dict[str, RiskProfile] = {
    "conservative": CONSERVATIVE_PROFILE,
    "balanced": BALANCED_PROFILE,
    "aggressive": AGGRESSIVE_PROFILE,
}


def interpolate_profile(level: int) -> RiskProfile:
    """
    Create a custom profile by interpolating between predefined ones
    based on conservativeness level (1-10)
    """
    level = max(1, min(10, level))
    
    if level <= 3:
        # Conservative range (1-3)
        base = CONSERVATIVE_PROFILE
        factor = level / 3.0
    elif level <= 7:
        # Balanced range (4-7)
        if level <= 5:
            base = CONSERVATIVE_PROFILE
            target = BALANCED_PROFILE
            factor = (level - 3) / 2.0
        else:
            base = BALANCED_PROFILE
            target = AGGRESSIVE_PROFILE
            factor = (level - 5) / 2.0
        
        # Interpolate between base and target
        return RiskProfile(
            name=f"Custom (Level {level})",
            level=level,
            position_size_min=_lerp(base.position_size_min, target.position_size_min, factor),
            position_size_max=_lerp(base.position_size_max, target.position_size_max, factor),
            position_size_default=_lerp(base.position_size_default, target.position_size_default, factor),
            stop_loss_tight=_lerp(base.stop_loss_tight, target.stop_loss_tight, factor),
            stop_loss_default=_lerp(base.stop_loss_default, target.stop_loss_default, factor),
            stop_loss_wide=_lerp(base.stop_loss_wide, target.stop_loss_wide, factor),
            take_profit_min=_lerp(base.take_profit_min, target.take_profit_min, factor),
            take_profit_default=_lerp(base.take_profit_default, target.take_profit_default, factor),
            take_profit_max=_lerp(base.take_profit_max, target.take_profit_max, factor),
            risk_reward_ratio=_lerp(base.risk_reward_ratio, target.risk_reward_ratio, factor),
            max_concurrent_trades=int(_lerp(base.max_concurrent_trades, target.max_concurrent_trades, factor)),
            long_entry_threshold=_lerp(base.long_entry_threshold, target.long_entry_threshold, factor),
            short_entry_threshold=_lerp(base.short_entry_threshold, target.short_entry_threshold, factor),
            min_confidence=_lerp(base.min_confidence, target.min_confidence, factor),
            description=f"Benutzerdefiniert - Risiko-Level {level}/10",
            color=_lerp_color(base.color, target.color, factor),
            icon="⚙️"
        )
    else:
        # Aggressive range (8-10)
        base = AGGRESSIVE_PROFILE
        factor = (level - 7) / 3.0
        # Scale up aggressive profile
        return RiskProfile(
            name=f"Custom (Level {level})",
            level=level,
            position_size_min=base.position_size_min * (1 + factor * 0.3),
            position_size_max=min(15.0, base.position_size_max * (1 + factor * 0.3)),
            position_size_default=base.position_size_default * (1 + factor * 0.3),
            stop_loss_tight=base.stop_loss_tight * (1 + factor * 0.2),
            stop_loss_default=base.stop_loss_default * (1 + factor * 0.2),
            stop_loss_wide=base.stop_loss_wide * (1 + factor * 0.2),
            take_profit_min=base.take_profit_min * (1 + factor * 0.3),
            take_profit_default=base.take_profit_default * (1 + factor * 0.3),
            take_profit_max=base.take_profit_max * (1 + factor * 0.3),
            risk_reward_ratio=base.risk_reward_ratio * (1 + factor * 0.2),
            max_concurrent_trades=min(20, int(base.max_concurrent_trades * (1 + factor * 0.5))),
            long_entry_threshold=max(0.5, base.long_entry_threshold * (1 - factor * 0.3)),
            short_entry_threshold=min(-0.3, base.short_entry_threshold * (1 - factor * 0.3)),
            min_confidence=max(0.45, base.min_confidence * (1 - factor * 0.15)),
            description=f"Benutzerdefiniert - Risiko-Level {level}/10 (Hoch-Aggressiv)",
            color="#dc2626",  # Darker red
            icon="⚠️"
        )
    
    return base


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b"""
    return a + (b - a) * t


def _lerp_color(c1: str, c2: str, t: float) -> str:
    """Interpolate between two hex colors"""
    try:
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        r = int(_lerp(r1, r2, t))
        g = int(_lerp(g1, g2, t))
        b = int(_lerp(b1, b2, t))
        return f"#{r:02x}{g:02x}{b:02x}"
    except:
        return c1


class RiskManager:
    """
    Manages risk profiles and provides trading rules based on active profile.
    Persists configuration to disk for server restarts.
    """
    
    def __init__(self, config_path: str = "./data/risk_config.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Default state
        self.active_profile_name: str = "balanced"
        self.custom_level: int = 5
        self.use_custom: bool = False
        self.custom_overrides: Dict[str, Any] = {}
        self.last_changed: Optional[str] = None
        self.profile_performance: Dict[str, Dict] = {}  # profile_name -> metrics
        
        self._load_config()
    
    def _load_config(self):
        """Load configuration from disk"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    self.active_profile_name = data.get("active_profile", "balanced")
                    self.custom_level = data.get("custom_level", 5)
                    self.use_custom = data.get("use_custom", False)
                    self.custom_overrides = data.get("custom_overrides", {})
                    self.last_changed = data.get("last_changed")
                    self.profile_performance = data.get("profile_performance", {})
                    logger.info(f"Loaded risk config: {self.active_profile_name} (custom={self.use_custom})")
        except Exception as e:
            logger.warning(f"Could not load risk config: {e}")
    
    def save_config(self):
        """Save configuration to disk"""
        try:
            data = {
                "active_profile": self.active_profile_name,
                "custom_level": self.custom_level,
                "use_custom": self.use_custom,
                "custom_overrides": self.custom_overrides,
                "last_changed": self.last_changed,
                "profile_performance": self.profile_performance
            }
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save risk config: {e}")
    
    def get_active_profile(self) -> RiskProfile:
        """Get the currently active risk profile"""
        if self.use_custom:
            profile = interpolate_profile(self.custom_level)
        else:
            profile = PROFILES.get(self.active_profile_name, BALANCED_PROFILE)
        
        # Apply any custom overrides
        if self.custom_overrides:
            profile_dict = asdict(profile)
            profile_dict.update(self.custom_overrides)
            profile = RiskProfile(**profile_dict)
        
        return profile
    
    def set_profile(self, profile_name: str) -> RiskProfile:
        """Switch to a predefined profile"""
        if profile_name not in PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}")
        
        self.active_profile_name = profile_name
        self.use_custom = False
        self.last_changed = datetime.now().isoformat()
        self.save_config()
        
        logger.info(f"Switched to risk profile: {profile_name}")
        return self.get_active_profile()
    
    def set_custom_level(self, level: int) -> RiskProfile:
        """Switch to a custom conservativeness level"""
        level = max(1, min(10, level))
        self.custom_level = level
        self.use_custom = True
        self.last_changed = datetime.now().isoformat()
        self.save_config()
        
        logger.info(f"Switched to custom risk level: {level}")
        return self.get_active_profile()
    
    def set_override(self, key: str, value: Any):
        """Set a custom override for a specific parameter"""
        valid_keys = [f.name for f in RiskProfile.__dataclass_fields__.values()]
        if key not in valid_keys:
            raise ValueError(f"Invalid parameter: {key}")
        
        self.custom_overrides[key] = value
        self.last_changed = datetime.now().isoformat()
        self.save_config()
        
        logger.info(f"Set risk override: {key} = {value}")
    
    def clear_overrides(self):
        """Clear all custom overrides"""
        self.custom_overrides = {}
        self.last_changed = datetime.now().isoformat()
        self.save_config()
    
    def record_performance(self, profile_name: str, metrics: Dict):
        """Record performance metrics for a profile"""
        if profile_name not in self.profile_performance:
            self.profile_performance[profile_name] = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "last_updated": None
            }
        
        perf = self.profile_performance[profile_name]
        perf["trades"] = metrics.get("trades", perf["trades"])
        perf["wins"] = metrics.get("wins", perf["wins"])
        perf["losses"] = metrics.get("losses", perf["losses"])
        perf["total_pnl"] = metrics.get("total_pnl", perf["total_pnl"])
        perf["avg_pnl"] = metrics.get("avg_pnl", perf["avg_pnl"])
        perf["sharpe"] = metrics.get("sharpe", perf.get("sharpe", 0.0))
        perf["max_drawdown"] = metrics.get("max_drawdown", perf.get("max_drawdown", 0.0))
        perf["last_updated"] = datetime.now().isoformat()
        
        self.save_config()
    
    def get_state(self) -> Dict:
        """Get full state for API response"""
        profile = self.get_active_profile()
        return {
            "active_profile_name": self.active_profile_name if not self.use_custom else "custom",
            "use_custom": self.use_custom,
            "custom_level": self.custom_level,
            "profile": asdict(profile),
            "custom_overrides": self.custom_overrides,
            "last_changed": self.last_changed,
            "available_profiles": {
                name: {
                    "name": p.name,
                    "level": p.level,
                    "description": p.description,
                    "color": p.color,
                    "icon": p.icon
                }
                for name, p in PROFILES.items()
            },
            "profile_performance": self.profile_performance
        }
    
    def calculate_position_size(self, portfolio_value: float, confidence: float) -> float:
        """
        Calculate position size based on active profile and confidence.
        Returns dollar amount to invest.
        """
        profile = self.get_active_profile()
        
        # Base size is default percentage
        base_pct = profile.position_size_default
        
        # Adjust based on confidence (higher confidence = larger position within limits)
        if confidence >= profile.min_confidence:
            confidence_factor = (confidence - profile.min_confidence) / (1.0 - profile.min_confidence)
            adjusted_pct = base_pct + (profile.position_size_max - base_pct) * confidence_factor * 0.5
        else:
            adjusted_pct = profile.position_size_min
        
        # Clamp to profile limits
        final_pct = max(profile.position_size_min, min(profile.position_size_max, adjusted_pct))
        
        return portfolio_value * (final_pct / 100.0)
    
    def get_stop_loss(self, entry_price: float, is_long: bool = True) -> float:
        """Calculate stop-loss price based on active profile"""
        profile = self.get_active_profile()
        sl_pct = profile.stop_loss_default / 100.0
        
        if is_long:
            return entry_price * (1 - sl_pct)
        else:
            return entry_price * (1 + sl_pct)
    
    def get_take_profit(self, entry_price: float, is_long: bool = True) -> float:
        """Calculate take-profit price based on active profile"""
        profile = self.get_active_profile()
        tp_pct = profile.take_profit_default / 100.0
        
        if is_long:
            return entry_price * (1 + tp_pct)
        else:
            return entry_price * (1 - tp_pct)
    
    def should_enter_trade(self, expected_change: float, confidence: float, 
                           current_open_trades: int, is_long: bool = True) -> tuple:
        """
        Determine if a trade should be entered based on profile rules.
        Returns (should_enter: bool, reason: str)
        """
        profile = self.get_active_profile()
        
        # Check concurrent trades limit
        if current_open_trades >= profile.max_concurrent_trades:
            return False, f"Max concurrent trades reached ({profile.max_concurrent_trades})"
        
        # Check confidence
        if confidence < profile.min_confidence:
            return False, f"Confidence too low ({confidence:.1%} < {profile.min_confidence:.1%})"
        
        # Check entry threshold
        if is_long:
            if expected_change < profile.long_entry_threshold:
                return False, f"Expected change too low for LONG ({expected_change:.2f}% < {profile.long_entry_threshold:.2f}%)"
        else:
            if expected_change > profile.short_entry_threshold:
                return False, f"Expected change too high for SHORT ({expected_change:.2f}% > {profile.short_entry_threshold:.2f}%)"
        
        return True, "Trade allowed by risk profile"


# Global instance
risk_manager = RiskManager()


def get_risk_manager() -> RiskManager:
    """Get the global risk manager instance"""
    return risk_manager
