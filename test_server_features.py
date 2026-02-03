"""
Server-Side Test Suite
Tests new server endpoints and error handling

Run with: python test_server_features.py
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add server directory to path
SERVER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server')
sys.path.insert(0, SERVER_DIR)


class TestFeatureStatsEndpoints(unittest.TestCase):
    """Test Feature Drift Statistics API"""
    
    @classmethod
    def setUpClass(cls):
        """Set up FastAPI test client"""
        try:
            from fastapi.testclient import TestClient
            # Patch data directory before importing
            cls.temp_dir = tempfile.mkdtemp()
            os.environ['DATA_DIR'] = cls.temp_dir
            
            from server import app, state
            cls.client = TestClient(app)
            cls.state = state
            cls.state.feature_stats = {}
        except ImportError as e:
            print(f"Warning: Could not import server modules: {e}")
            cls.client = None
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temp directory"""
        if hasattr(cls, 'temp_dir'):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Reset state before each test"""
        if self.client and hasattr(self, 'state'):
            self.state.feature_stats = {}
            self.state.trade_journal = []
            self.state.error_log = []
    
    def test_record_feature_stats(self):
        """Test recording feature statistics"""
        if not self.client:
            self.skipTest("Server not available")
        
        response = self.client.post(
            "/api/drift/stats",
            json={
                "ticker": "AAPL",
                "features": {
                    "RSI": {"mean": 50.0, "std": 10.0, "min": 0, "max": 100, "p5": 20, "p95": 80},
                    "MACD": {"mean": 0.5, "std": 0.2, "min": -1, "max": 1, "p5": -0.3, "p95": 0.8}
                }
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["features"], 2)
    
    def test_get_feature_stats(self):
        """Test retrieving feature statistics"""
        if not self.client:
            self.skipTest("Server not available")
        
        # First record some stats
        self.state.feature_stats["AAPL"] = {
            "RSI": {"mean": 50.0, "std": 10.0}
        }
        
        response = self.client.get("/api/drift/stats/AAPL")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["ticker"], "AAPL")
        self.assertIn("RSI", data["stats"])
    
    def test_get_nonexistent_stats(self):
        """Test 404 for missing ticker stats"""
        if not self.client:
            self.skipTest("Server not available")
        
        response = self.client.get("/api/drift/stats/NONEXISTENT")
        
        self.assertEqual(response.status_code, 404)
    
    def test_drift_check(self):
        """Test drift checking endpoint"""
        if not self.client:
            self.skipTest("Server not available")
        
        # Record baseline stats
        self.state.feature_stats["AAPL"] = {
            "RSI": {"mean": 50.0, "std": 10.0}
        }
        
        # Check for drift with normal value
        response = self.client.post(
            "/api/drift/check",
            json={
                "ticker": "AAPL",
                "features": {"RSI": 55.0}  # Close to mean
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["critical_count"], 0)
    
    def test_drift_check_critical(self):
        """Test drift check with critical drift"""
        if not self.client:
            self.skipTest("Server not available")
        
        # Record baseline stats with low std
        self.state.feature_stats["AAPL"] = {
            "RSI": {"mean": 50.0, "std": 5.0}
        }
        
        # Check with extreme value
        response = self.client.post(
            "/api/drift/check",
            json={
                "ticker": "AAPL",
                "features": {"RSI": 100.0}  # Very far from mean
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(data["critical_count"], 0)


class TestTradeJournalEndpoints(unittest.TestCase):
    """Test Trade Journal API"""
    
    @classmethod
    def setUpClass(cls):
        """Set up FastAPI test client"""
        try:
            from fastapi.testclient import TestClient
            from server import app, state
            cls.client = TestClient(app)
            cls.state = state
        except ImportError as e:
            print(f"Warning: Could not import server modules: {e}")
            cls.client = None
    
    def setUp(self):
        """Reset state before each test"""
        if self.client and hasattr(self, 'state'):
            self.state.trade_journal = []
    
    def test_log_trade(self):
        """Test logging a trade"""
        if not self.client:
            self.skipTest("Server not available")
        
        response = self.client.post(
            "/api/trades",
            json={
                "ticker": "AAPL",
                "action": "BUY",
                "price": 150.0,
                "shares": 10,
                "signal_reason": "Strong uptrend",
                "confidence": 0.75,
                "position_multiplier": 1.0,
                "model_type": "hybrid",
                "volatility": 0.02
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["trade_id"], 0)
    
    def test_get_trades(self):
        """Test retrieving trades"""
        if not self.client:
            self.skipTest("Server not available")
        
        # Add a trade
        self.state.trade_journal.append({
            "timestamp": datetime.now().isoformat(),
            "ticker": "AAPL",
            "action": "BUY",
            "price": 150.0,
            "shares": 10
        })
        
        response = self.client.get("/api/trades")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 1)
    
    def test_get_trades_filtered(self):
        """Test filtering trades by ticker"""
        if not self.client:
            self.skipTest("Server not available")
        
        # Add trades for different tickers
        self.state.trade_journal = [
            {"ticker": "AAPL", "action": "BUY", "price": 150.0, "shares": 10},
            {"ticker": "MSFT", "action": "BUY", "price": 300.0, "shares": 5},
            {"ticker": "AAPL", "action": "SELL", "price": 155.0, "shares": 10}
        ]
        
        response = self.client.get("/api/trades?ticker=AAPL")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["trades"]), 2)
    
    def test_close_trade(self):
        """Test closing a trade"""
        if not self.client:
            self.skipTest("Server not available")
        
        # Add an open trade
        self.state.trade_journal.append({
            "ticker": "AAPL",
            "action": "BUY",
            "price": 150.0,
            "shares": 10,
            "exit_price": None
        })
        
        response = self.client.post(
            "/api/trades/close",
            json={
                "ticker": "AAPL",
                "exit_price": 160.0,
                "realized_pnl": 100.0,
                "outcome": "WIN"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
    
    def test_trade_analytics(self):
        """Test trade analytics endpoint"""
        if not self.client:
            self.skipTest("Server not available")
        
        # Add closed trades
        self.state.trade_journal = [
            {"ticker": "AAPL", "outcome": "WIN", "confidence": 0.75, "model_type": "hybrid", "realized_pnl": 100},
            {"ticker": "MSFT", "outcome": "LOSS", "confidence": 0.55, "model_type": "hybrid", "realized_pnl": -50},
            {"ticker": "GOOGL", "outcome": "WIN", "confidence": 0.90, "model_type": "server", "realized_pnl": 200}
        ]
        
        response = self.client.get("/api/trades/analytics")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        analytics = data["analytics"]
        self.assertEqual(analytics["total_trades"], 3)
        self.assertEqual(analytics["wins"], 2)
        self.assertEqual(analytics["losses"], 1)
        self.assertAlmostEqual(analytics["win_rate"], 66.7, places=0)


class TestHealthEndpoints(unittest.TestCase):
    """Test Health Check Endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up FastAPI test client"""
        try:
            from fastapi.testclient import TestClient
            from server import app
            cls.client = TestClient(app)
        except ImportError as e:
            print(f"Warning: Could not import server modules: {e}")
            cls.client = None
    
    def test_health_check(self):
        """Test basic health endpoint"""
        if not self.client:
            self.skipTest("Server not available")
        
        response = self.client.get("/api/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("active_models", data)
    
    def test_heartbeat(self):
        """Test heartbeat endpoint"""
        if not self.client:
            self.skipTest("Server not available")
        
        response = self.client.get("/api/heartbeat")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "alive")
        self.assertIn("timestamp", data)
        self.assertIn("server_time_ms", data)
    
    def test_metrics(self):
        """Test metrics endpoint"""
        if not self.client:
            self.skipTest("Server not available")
        
        response = self.client.get("/api/metrics")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("cpu_percent", data)
        self.assertIn("ram_percent", data)


class TestErrorHandling(unittest.TestCase):
    """Test Server Error Handling"""
    
    @classmethod
    def setUpClass(cls):
        """Set up FastAPI test client"""
        try:
            from fastapi.testclient import TestClient
            from server import app, state
            cls.client = TestClient(app)
            cls.state = state
        except ImportError as e:
            print(f"Warning: Could not import server modules: {e}")
            cls.client = None
    
    def setUp(self):
        """Reset error log before each test"""
        if self.client and hasattr(self, 'state'):
            self.state.error_log = []
    
    def test_invalid_json(self):
        """Test handling of invalid JSON"""
        if not self.client:
            self.skipTest("Server not available")
        
        response = self.client.post(
            "/api/trades",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        # Should return 422 for validation error
        self.assertIn(response.status_code, [400, 422])
    
    def test_missing_required_field(self):
        """Test handling of missing required fields"""
        if not self.client:
            self.skipTest("Server not available")
        
        response = self.client.post(
            "/api/trades",
            json={
                "ticker": "AAPL"
                # Missing action, price, shares
            }
        )
        
        self.assertEqual(response.status_code, 422)
    
    def test_get_error_log(self):
        """Test retrieving error log"""
        if not self.client:
            self.skipTest("Server not available")
        
        # Add some errors
        self.state.error_log = [
            {"timestamp": datetime.now().isoformat(), "type": "TEST", "message": "Test error"}
        ]
        
        response = self.client.get("/api/errors")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 1)
    
    def test_process_time_header(self):
        """Test that X-Process-Time header is added"""
        if not self.client:
            self.skipTest("Server not available")
        
        response = self.client.get("/api/health")
        
        self.assertIn("X-Process-Time", response.headers)


class TestChartCacheEndpoints(unittest.TestCase):
    """Test Chart Cache API (existing functionality)"""
    
    @classmethod
    def setUpClass(cls):
        """Set up FastAPI test client"""
        try:
            from fastapi.testclient import TestClient
            from server import app, state
            cls.client = TestClient(app)
            cls.state = state
        except ImportError as e:
            print(f"Warning: Could not import server modules: {e}")
            cls.client = None
    
    def setUp(self):
        """Reset state before each test"""
        if self.client and hasattr(self, 'state'):
            self.state.live_prices_cache = {}
    
    def test_add_chart_price(self):
        """Test adding a chart price point"""
        if not self.client:
            self.skipTest("Server not available")
        
        response = self.client.post(
            "/api/chart-cache/AAPL",
            json={"price": 150.0}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["price"], 150.0)
    
    def test_batch_chart_prices(self):
        """Test adding batch price updates"""
        if not self.client:
            self.skipTest("Server not available")
        
        response = self.client.post(
            "/api/chart-cache/batch",
            json={
                "AAPL": 150.0,
                "MSFT": 300.0,
                "GOOGL": 2800.0
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["added"], 3)
    
    def test_get_chart_cache(self):
        """Test retrieving chart cache"""
        if not self.client:
            self.skipTest("Server not available")
        
        # Add some data
        self.state.live_prices_cache = {
            "AAPL": [{"price": 150.0, "timestamp": datetime.now().isoformat()}]
        }
        
        response = self.client.get("/api/chart-cache")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("AAPL", data["prices"])


def run_tests():
    """Run all server tests"""
    print("=" * 70)
    print("AI Trading Server - Test Suite")
    print("=" * 70)
    print()
    
    # Check if server modules are available
    try:
        sys.path.insert(0, SERVER_DIR)
        from fastapi.testclient import TestClient
        print("✓ FastAPI TestClient available")
    except ImportError:
        print("✗ FastAPI TestClient not available - install with: pip install httpx")
        print("  Some tests will be skipped")
    
    try:
        from server import app
        print("✓ Server module importable")
    except ImportError as e:
        print(f"✗ Server module not importable: {e}")
        print("  Running from server directory...")
    
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureStatsEndpoints))
    suite.addTests(loader.loadTestsFromTestCase(TestTradeJournalEndpoints))
    suite.addTests(loader.loadTestsFromTestCase(TestHealthEndpoints))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestChartCacheEndpoints))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    return len(result.failures) + len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
