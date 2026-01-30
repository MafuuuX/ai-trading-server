# Contributing Guide

Danke für dein Interesse an diesem Projekt! Hier sind einige Richtlinien für Contributions.

## Entwicklungs-Setup

```bash
# Clone und Virtual Environment
git clone https://github.com/MafuuuX/ai-trading-server.git
cd ai-trading-server
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# oder
venv\Scripts\activate     # Windows

# Dependencies
pip install -r requirements.txt
```

## Code Style

- Python: Follow PEP 8 (max 100 chars per line)
- Variable names: snake_case
- Classes: PascalCase
- Constants: UPPER_CASE
- Type hints: Use where practical

## Testing

```bash
# Trainer test
python test_trainer.py

# Syntax check
python -m py_compile server/trainer.py
python -m py_compile server/data_fetcher.py
python -m py_compile server/server.py
```

## Commit Messages

Format: `[Category] Short description`

Categories:
- `[Feature]` - Neue Funktionalität
- `[Fix]` - Bugfix
- `[Improve]` - Verbesserung bestehenden Codes
- `[Docs]` - Dokumentation
- `[Refactor]` - Code Umstrukturierung

Example:
```
[Fix] Remove ta library import - use pandas-only for technical indicators
[Feature] Add ATR calculation with pandas
[Improve] Optimize data fetcher timeout handling
```

## Pull Request Process

1. Fork das Repository
2. Erstelle einen Feature Branch (`git checkout -b feature/amazing-feature`)
3. Commit deine Änderungen (`git commit -m '[Feature] Add amazing feature'`)
4. Push zum Branch (`git push origin feature/amazing-feature`)
5. Öffne einen Pull Request

## Reporting Issues

Bitte berichte Bugs mit:
- Klarer Beschreibung des Problems
- Steps to reproduce
- Expected vs. actual behavior
- Python version und OS
- Relevant logs/tracebacks

## Areas for Contribution

- [ ] More technical indicators (Stochastic, Williams %R, etc.)
- [ ] WebSocket live price streaming
- [ ] Model backtesting framework
- [ ] Client SDK in other languages (JavaScript, Go, Rust)
- [ ] Performance optimizations
- [ ] Additional data sources (Binance, Kraken, etc.)
- [ ] Documentation improvements
- [ ] UI/UX improvements for web dashboard

---

**Questions?** Open an issue or contact the maintainers.
