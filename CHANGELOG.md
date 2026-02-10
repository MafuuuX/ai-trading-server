# Changelog

Alle erwähnenswerten Änderungen an diesem Projekt werden in dieser Datei dokumentiert.

## [2.1.0] - 2026-02-01

### Added
- ✅ **Ensemble Predictions** (`ensemble.py`) – Combine multiple model predictions with configurable strategies (weighted_average, majority_vote, confidence_weighted), adaptive weight learning from outcomes
- ✅ **Model Versioning** (`model_registry.py`) – Automatic versioning of trained models with rollback support, hash-based integrity checks, configurable pruning (max versions per ticker)
- ✅ **WebSocket Rate Limiting** (`rate_limiter.py`) – Token-bucket algorithm with per-IP connection limits, automatic banning after threshold violations, configurable burst/rate/ban duration
- ✅ **Prometheus Metrics** (`metrics_collector.py`) – Pure-Python metrics collector (no external dependency) with Counter/Gauge/Histogram types, `/metrics` endpoint in Prometheus exposition format
- ✅ **A/B Testing Framework** (`ab_testing.py`) – Full experiment lifecycle (create/pause/resume/complete/delete), traffic splitting, prediction & trade outcome tracking, statistical results
- ✅ New API endpoints: `/metrics`, `/api/ensemble/*`, `/api/registry/*`, `/api/rate-limit/*`, `/api/ab/*`
- ✅ Request middleware for HTTP metrics tracking (request count, latency by method/path/status)
- ✅ WebSocket integration with rate limiting and Prometheus counters
- ✅ Model registration with metrics tracking after training completion

### Changed
- `server.py`: Added imports and global instances for all 5 new modules
- `server.py`: Train task now registers model versions and records Prometheus metrics on completion
- `server.py`: WebSocket endpoint integrates rate limiting (check_connection, allow_message, cleanup)

## [2.0.0] - 2026-01-30

### Added
- ✅ Pandas-only technical indicators implementation
- ✅ Comprehensive API documentation in README
- ✅ Full production-ready systemd service configuration
- ✅ Client integration examples in README

### Changed
- **BREAKING**: Removed `ta` library dependency entirely
  - All technical indicators now use pure pandas calculations
  - Improved maintainability and compatibility
  
### Fixed
- Fixed deprecated `fillna(method=)` syntax (pandas 2.0 compatibility)
- Fixed NoneType arithmetic errors in feature engineering
- Fixed Stooq Polish column mapping (Data→Date, Otwarcie→Open, etc.)

### Technical Details

#### Technical Indicators (Pandas-based)
1. **RSI** - Manual gain/loss calculation with 14-period rolling mean
2. **MACD** - EMA12 - EMA26 using pandas ewm()
3. **Bollinger Bands** - SMA ± 2×std using rolling statistics
4. **ATR** - Max of price ranges with 14-period rolling mean
5. **SMA** - 20 and 50-period simple moving averages
6. **Returns & Volatility** - pct_change() and rolling std()

#### Data Pipeline Improvements
- YFinance primary source with aggressive error detection
- Stooq fallback with Polish→English column mapping
- Configurable timeouts (ping 3s, fetch 5s, 2 retries)
- Cache validation and integrity checks

#### Service Configuration
- Systemd service: TimeoutStartSec=120s, TimeoutStopSec=30s
- Auto-restart on failure
- Full traceback logging in error handler

---

## [1.5.0] - 2026-01-25

### Added
- Data fetcher with intelligent YFinance → Stooq fallback
- Cache validation for corrupted dataframes
- Detailed Stooq logging for debugging

### Fixed
- YFinance rate limiting (HTTP 429) handling
- Service startup timeout issues

---

## [1.0.0] - 2025-12-01

### Initial Release
- Dual-head LSTM model architecture
- FastAPI server with web dashboard
- Support for 135+ stocks
- Systemd service integration
- Model distribution with hash-based change detection
