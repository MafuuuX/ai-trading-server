# Contributing Guide

Thank you for your interest in this project! Here are some guidelines for contributions.

## Development Setup

```bash
# Clone and Virtual Environment
git clone https://github.com/MafuuuX/ai-trading-server.git
cd ai-trading-server
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
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

### Unit Tests
```bash
# Trainer test
python test_trainer.py

# Syntax check
python -m py_compile server.py
python -m py_compile trainer.py
python -m py_compile data_fetcher.py
```

### Integration Tests
```bash
# Test API endpoints
curl http://localhost:8000/api/health
curl http://localhost:8000/api/models

# Test WebSocket
python -c 'import websocket; ws = websocket.WebSocket(); ws.connect("ws://localhost:8000/ws"); print(ws.recv()); ws.close()'

# Test live prices
curl "http://localhost:8000/api/prices?tickers=AAPL,MSFT"
```

### Before Submitting PR
- [ ] All tests pass
- [ ] No syntax errors
- [ ] Code follows PEP 8
- [ ] Added docstrings for new functions
- [ ] Updated README if needed
- [ ] No API keys or secrets in code

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code restructuring without behavior change
- `perf:` - Performance improvements
- `test:` - Adding or updating tests
- `chore:` - Build/tooling changes

**Examples:**
```
feat(data): add Alpaca API integration for real-time prices
fix(websocket): resolve connection timeout issues
docs: update README with WebSocket configuration
refactor(trainer): optimize LSTM architecture
perf(cache): implement parallel price fetching with ThreadPoolExecutor
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes using conventional commits format
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Reporting Issues

Please report bugs with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Python version and OS
- Relevant logs/tracebacks

## Areas for Contribution

### High Priority
- ✅ ~~Model ensemble voting strategies (combine multiple predictions)~~ (Completed)
- [ ] Backtesting framework with historical performance metrics
- [ ] Additional technical indicators (Stochastic, Williams %R, OBV, etc.)
- [ ] Support for cryptocurrency markets (Binance, Coinbase)
- [ ] Performance optimizations (GPU training, caching improvements)

### Features
- ✅ ~~WebSocket live price streaming~~ (Completed)
- ✅ ~~Google Drive automated backups~~ (Completed)
- ✅ ~~Multi-provider price fallback~~ (Completed)
- [ ] Client SDK in other languages (JavaScript, Go, Rust)
- [ ] Web dashboard improvements (charts, real-time updates)
- [ ] Mobile app support (REST API client)
- [ ] Automated model selection based on market conditions

### Infrastructure
- [ ] Docker containerization
- [ ] Kubernetes deployment configs
- [ ] CI/CD pipeline (GitHub Actions)
- ✅ ~~Prometheus metrics endpoint~~ (Completed)
- [ ] Grafana dashboard templates

### Documentation
- [ ] API documentation with OpenAPI/Swagger
- [ ] Tutorial videos
- [ ] Example client implementations
- [ ] Architecture diagrams

---

**Questions?** Open an issue or contact the maintainers.
