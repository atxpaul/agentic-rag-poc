import io
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .logging_utils import log_event


class EphemeralMemory:
    """Ephemeral turn buffer backed by Redis with NDJSON durable log on S3/MinIO.

    - Redis key: buffer:<conv_id> → list of JSON turns (max N), TTL (hours)
    - Seq key:   buffer:<conv_id>:seq → integer counter
    - S3 path:   <prefix>/<conv_id>/<yyyy-mm-dd>.ndjson (one line per turn)
    """

    def __init__(self) -> None:
        self._redis = None
        self._s3 = None
        # Lazy init to avoid hard dependency if disabled
        if config.REDIS_ENABLED:
            try:
                import redis  # type: ignore

                if config.REDIS_URL:
                    self._redis = redis.Redis.from_url(
                        config.REDIS_URL, decode_responses=True
                    )
                else:
                    self._redis = redis.Redis(
                        host=config.REDIS_HOST,
                        port=config.REDIS_PORT,
                        db=config.REDIS_DB,
                        decode_responses=True,
                    )
            except Exception:
                self._redis = None

        if config.S3_ENABLED:
            try:
                from minio import Minio  # type: ignore

                endpoint = config.S3_ENDPOINT.replace("http://", "").replace(
                    "https://", ""
                )
                self._s3 = Minio(
                    endpoint,
                    access_key=config.S3_ACCESS_KEY,
                    secret_key=config.S3_SECRET_KEY,
                    secure=config.S3_SECURE,
                )
            except Exception:
                self._s3 = None

        # Log backend availability
        try:
            log_event(
                "mem.init",
                {
                    "redis_enabled": bool(getattr(config, "REDIS_ENABLED", False)),
                    "redis_connected": bool(self._redis),
                    "redis_url": getattr(config, "REDIS_URL", ""),
                    "redis_host": getattr(config, "REDIS_HOST", ""),
                    "redis_port": getattr(config, "REDIS_PORT", ""),
                    "s3_enabled": bool(getattr(config, "S3_ENABLED", False)),
                    "s3_ready": bool(self._s3),
                    "s3_endpoint": getattr(config, "S3_ENDPOINT", ""),
                    "s3_bucket": getattr(config, "S3_BUCKET", ""),
                    "s3_prefix": getattr(config, "S3_MEMLOG_PREFIX", ""),
                },
            )
        except Exception:
            pass

    # -------------------- Public API --------------------
    def ensure_backfill(self, conv_id: str) -> None:
        """If Redis has no buffer for this conversation, backfill last M lines from S3."""
        if not conv_id:
            conv_id = "default"
        if not self._redis or not config.S3_ENABLED or not self._s3:
            try:
                log_event(
                    "mem.backfill.skip",
                    {
                        "conv_id": conv_id,
                        "reason": "missing_redis_or_s3",
                        "has_redis": bool(self._redis),
                        "s3_enabled": bool(getattr(config, "S3_ENABLED", False)),
                        "has_s3": bool(self._s3),
                    },
                )
            except Exception:
                pass
            return
        try:
            key = self._buffer_key(conv_id)
            if self._redis.llen(key) > 0:
                try:
                    log_event(
                        "mem.backfill.skip",
                        {"conv_id": conv_id, "reason": "buffer_not_empty"},
                    )
                except Exception:
                    pass
                return
            turns = self._read_recent_from_s3(
                conv_id, config.S3_BACKFILL_MAX_LINES)
            if not turns:
                try:
                    log_event(
                        "mem.backfill.empty",
                        {
                            "conv_id": conv_id,
                            "max_lines": int(getattr(config, "S3_BACKFILL_MAX_LINES", 0)),
                        },
                    )
                except Exception:
                    pass
                return
            pipe = self._redis.pipeline()
            for t in turns[-config.REDIS_BUFFER_MAX_TURNS:]:
                pipe.rpush(key, json.dumps(t, ensure_ascii=False))
            pipe.expire(key, int(config.REDIS_TTL_HOURS * 3600))
            # Set seq to last seen seq
            last_seq = max((t.get("seq", 0) for t in turns), default=0)
            pipe.set(self._seq_key(conv_id), last_seq)
            pipe.execute()
            try:
                log_event(
                    "mem.backfill.ok",
                    {
                        "conv_id": conv_id,
                        "added": min(len(turns), int(getattr(config, "REDIS_BUFFER_MAX_TURNS", 0))),
                        "last_seq": int(last_seq),
                    },
                )
            except Exception:
                pass
        except Exception:
            try:
                log_event("mem.backfill.error", {"conv_id": conv_id})
            except Exception:
                pass
            return

    def append_turn(
        self,
        conv_id: str,
        role: str,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
        ts: Optional[datetime] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Append one turn to Redis buffer and S3 NDJSON.

        Returns (seq, turn_dict).
        """
        if not conv_id:
            conv_id = "default"
        if ts is None:
            ts = datetime.utcnow()

        seq = self._next_seq(conv_id)
        turn = {
            "ts": ts.isoformat() + "Z",
            "conv_id": conv_id,
            "seq": seq,
            "role": role,
            "text": text,
            "meta": meta or {},
        }

        # Best-effort Redis buffer
        self._append_to_redis(conv_id, turn)

        # Best-effort NDJSON durable log
        self._append_to_s3(turn)

        return seq, turn

    def get_buffer(self, conv_id: str) -> List[Dict[str, Any]]:
        if not self._redis:
            return []
        try:
            items = self._redis.lrange(
                self._buffer_key(conv_id or "default"), 0, -1)
            out: List[Dict[str, Any]] = []
            for it in items:
                try:
                    out.append(json.loads(it))
                except Exception:
                    continue
            try:
                log_event(
                    "mem.redis.read",
                    {"conv_id": conv_id or "default", "count": len(out)},
                )
            except Exception:
                pass
            return out
        except Exception:
            return []

    # -------------------- Internals --------------------
    def _buffer_key(self, conv_id: str) -> str:
        return f"buffer:{conv_id}"

    def _seq_key(self, conv_id: str) -> str:
        return f"buffer:{conv_id}:seq"

    def _next_seq(self, conv_id: str) -> int:
        # If Redis unavailable, derive a time-based seq
        if not self._redis:
            try:
                log_event(
                    "mem.seq.fallback",
                    {"conv_id": conv_id, "reason": "no_redis"},
                )
            except Exception:
                pass
            return int(datetime.utcnow().timestamp())
        try:
            return int(self._redis.incr(self._seq_key(conv_id)))
        except Exception:
            try:
                log_event(
                    "mem.seq.fallback",
                    {"conv_id": conv_id, "reason": "redis_incr_error"},
                )
            except Exception:
                pass
            return int(datetime.utcnow().timestamp())

    def _append_to_redis(self, conv_id: str, turn: Dict[str, Any]) -> None:
        if not self._redis:
            return
        try:
            key = self._buffer_key(conv_id)
            self._redis.rpush(key, json.dumps(turn, ensure_ascii=False))
            self._redis.ltrim(key, -config.REDIS_BUFFER_MAX_TURNS, -1)
            self._redis.expire(key, int(config.REDIS_TTL_HOURS * 3600))
            try:
                log_event(
                    "mem.redis.append",
                    {
                        "conv_id": conv_id,
                        "role": turn.get("role"),
                        "seq": int(turn.get("seq", 0)),
                    },
                )
            except Exception:
                pass
        except Exception:
            try:
                log_event(
                    "mem.redis.append.error",
                    {"conv_id": conv_id},
                )
            except Exception:
                pass
            return

    def _append_to_s3(self, turn: Dict[str, Any]) -> None:
        if not config.S3_ENABLED or not self._s3:
            return
        try:
            bucket = config.S3_BUCKET
            if not bucket:
                return
            obj = self._daily_object_path(
                turn["conv_id"], turn["ts"])  # yyyy-mm-dd
            line = json.dumps(turn, ensure_ascii=False) + "\n"

            # Try read-append-write (simple and fine for small logs)
            existing = None
            try:
                resp = self._s3.get_object(bucket, obj)
                try:
                    existing = resp.read()
                finally:
                    resp.close()
                    resp.release_conn()
            except Exception:
                existing = None

            payload = (existing or b"") + line.encode("utf-8")
            data_stream = io.BytesIO(payload)
            self._s3.put_object(
                bucket_name=bucket,
                object_name=obj,
                data=data_stream,
                length=len(payload),
                content_type="application/x-ndjson",
            )
            try:
                log_event(
                    "mem.s3.append",
                    {
                        "conv_id": turn.get("conv_id"),
                        "object": obj,
                        "bytes": int(len(payload)),
                    },
                )
            except Exception:
                pass
        except Exception as exc:
            try:
                log_event(
                    "mem.s3.append.error",
                    {
                        "conv_id": turn.get("conv_id"),
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
            except Exception:
                pass
            return

    def _read_recent_from_s3(self, conv_id: str, max_lines: int) -> List[Dict[str, Any]]:
        if not self._s3 or not config.S3_BUCKET:
            return []
        lines: List[str] = []
        # Today and yesterday for backfill
        for delta in [0, 1, 2]:
            day = datetime.utcnow() - timedelta(days=delta)
            key = self._daily_object_path(conv_id, day.isoformat() + "Z")
            try:
                resp = self._s3.get_object(config.S3_BUCKET, key)
                try:
                    blob = resp.read().decode("utf-8", errors="ignore")
                    lines.extend(
                        [ln for ln in blob.splitlines() if ln.strip()])
                finally:
                    resp.close()
                    resp.release_conn()
            except Exception:
                continue
            if len(lines) >= max_lines:
                break
        # Keep only last max_lines
        lines = lines[-max_lines:]
        turns: List[Dict[str, Any]] = []
        for ln in lines:
            try:
                turns.append(json.loads(ln))
            except Exception:
                continue
        return turns

    def _daily_object_path(self, conv_id: str, ts_iso: str) -> str:
        # ts_iso may be ISO string ending with Z
        date_part = ts_iso[:10]
        prefix = config.S3_MEMLOG_PREFIX.strip("/")
        return f"{prefix}/{conv_id}/{date_part}.ndjson"
