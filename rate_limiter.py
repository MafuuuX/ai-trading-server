"""
WebSocket Rate Limiter
=======================
Per-client connection limits and message rate limiting for the WebSocket API.

Features:
- Max connections per IP address
- Message rate limiting (token bucket algorithm)
- Configurable burst allowance
- Auto-ban for persistent violators
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ClientBucket:
    """Token-bucket state for a single client."""
    tokens: float = 10.0
    last_refill: float = field(default_factory=time.time)
    message_count: int = 0
    violation_count: int = 0
    banned_until: float = 0.0


class RateLimiter:
    """Token-bucket rate limiter for WebSocket connections.

    Each client (identified by IP or API key) has a bucket that refills
    at ``rate`` tokens/second up to ``burst`` tokens.  Every message costs
    one token.  If the bucket is empty the message is rejected.

    Args:
        rate: Tokens added per second (sustained message rate).
        burst: Maximum bucket size (burst allowance).
        max_connections_per_ip: Maximum concurrent WebSocket connections from one IP.
        ban_threshold: Number of violations before temporary ban.
        ban_duration_s: How long a ban lasts (seconds).
    """

    def __init__(
        self,
        rate: float = 5.0,
        burst: int = 20,
        max_connections_per_ip: int = 5,
        ban_threshold: int = 50,
        ban_duration_s: float = 300.0,
    ) -> None:
        self.rate = rate
        self.burst = burst
        self.max_connections_per_ip = max_connections_per_ip
        self.ban_threshold = ban_threshold
        self.ban_duration_s = ban_duration_s

        # client_id → ClientBucket
        self._buckets: Dict[str, ClientBucket] = {}
        # client_id → set of websocket ids (for connection counting)
        self._connections: Dict[str, Set[str]] = defaultdict(set)
        # Stats
        self._total_allowed: int = 0
        self._total_rejected: int = 0

        logger.info(
            "RateLimiter initialised: rate=%.1f/s, burst=%d, max_conn/ip=%d",
            rate,
            burst,
            max_connections_per_ip,
        )

    # ------------------------------------------------------------------ #
    #  Connection management
    # ------------------------------------------------------------------ #

    def check_connection(self, client_id: str, ws_id: str) -> Tuple[bool, str]:
        """Check whether a new WebSocket connection should be allowed.

        Args:
            client_id: IP address or API key identifying the client.
            ws_id: Unique identifier for this WebSocket instance.

        Returns:
            ``(allowed, reason)`` tuple.
        """
        # Check ban
        bucket = self._buckets.get(client_id)
        if bucket and bucket.banned_until > time.time():
            remaining = int(bucket.banned_until - time.time())
            return False, f"Banned for {remaining}s due to rate-limit violations"

        current = len(self._connections[client_id])
        if current >= self.max_connections_per_ip:
            return False, (
                f"Max connections ({self.max_connections_per_ip}) reached for this IP"
            )

        return True, "OK"

    def register_connection(self, client_id: str, ws_id: str) -> None:
        """Register a new WebSocket connection."""
        self._connections[client_id].add(ws_id)
        if client_id not in self._buckets:
            self._buckets[client_id] = ClientBucket()
        logger.debug(
            "Connection registered: %s (total=%d)",
            client_id,
            len(self._connections[client_id]),
        )

    def unregister_connection(self, client_id: str, ws_id: str) -> None:
        """Remove a WebSocket connection."""
        self._connections[client_id].discard(ws_id)
        if not self._connections[client_id]:
            del self._connections[client_id]

    # ------------------------------------------------------------------ #
    #  Message rate limiting
    # ------------------------------------------------------------------ #

    def allow_message(self, client_id: str) -> Tuple[bool, str]:
        """Check whether a message from *client_id* is allowed.

        Consumes one token from the client's bucket.  If no tokens are
        available the message is rejected and the violation count increases.

        Returns:
            ``(allowed, reason)`` tuple.
        """
        bucket = self._get_or_create(client_id)

        # Check ban
        if bucket.banned_until > time.time():
            remaining = int(bucket.banned_until - time.time())
            self._total_rejected += 1
            return False, f"Banned for {remaining}s"

        # Refill tokens
        now = time.time()
        elapsed = now - bucket.last_refill
        bucket.tokens = min(self.burst, bucket.tokens + elapsed * self.rate)
        bucket.last_refill = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            bucket.message_count += 1
            self._total_allowed += 1
            return True, "OK"

        # Rejected
        bucket.violation_count += 1
        self._total_rejected += 1

        if bucket.violation_count >= self.ban_threshold:
            bucket.banned_until = now + self.ban_duration_s
            logger.warning(
                "Client %s banned for %ds after %d violations",
                client_id,
                int(self.ban_duration_s),
                bucket.violation_count,
            )

        return False, "Rate limit exceeded"

    # ------------------------------------------------------------------ #
    #  Stats
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict:
        """Return rate limiter statistics."""
        active = sum(1 for c in self._connections.values() if c)
        return {
            "active_clients": active,
            "total_connections": sum(len(c) for c in self._connections.values()),
            "total_allowed": self._total_allowed,
            "total_rejected": self._total_rejected,
            "banned_clients": sum(
                1 for b in self._buckets.values() if b.banned_until > time.time()
            ),
            "config": {
                "rate": self.rate,
                "burst": self.burst,
                "max_connections_per_ip": self.max_connections_per_ip,
                "ban_threshold": self.ban_threshold,
                "ban_duration_s": self.ban_duration_s,
            },
        }

    def get_client_info(self, client_id: str) -> Optional[Dict]:
        """Return info about a specific client."""
        bucket = self._buckets.get(client_id)
        if not bucket:
            return None
        return {
            "client_id": client_id,
            "tokens": round(bucket.tokens, 2),
            "message_count": bucket.message_count,
            "violation_count": bucket.violation_count,
            "banned_until": bucket.banned_until if bucket.banned_until > time.time() else None,
            "connections": len(self._connections.get(client_id, set())),
        }

    def reset_client(self, client_id: str) -> None:
        """Reset rate-limit state for a client (admin action)."""
        if client_id in self._buckets:
            self._buckets[client_id] = ClientBucket()
            logger.info("Rate-limit state reset for client %s", client_id)

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _get_or_create(self, client_id: str) -> ClientBucket:
        if client_id not in self._buckets:
            self._buckets[client_id] = ClientBucket()
        return self._buckets[client_id]
