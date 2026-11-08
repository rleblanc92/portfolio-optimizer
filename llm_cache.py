"""Persistent asynchronous SQLite cache for Pydantic-backed LLM calls."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar, get_type_hints
from weakref import WeakKeyDictionary

import aiosqlite
import numpy as np


P = ParamSpec("P")
R = TypeVar("R")

DB_PATH = Path(__file__).resolve().with_name("llm_cache.db")
_SQLITE_RETRY_ATTEMPTS = 5
_INITIALIZED_DB_PATHS: set[Path] = set()
_INIT_LOCK = asyncio.Lock()
_INIT_LOCKS = WeakKeyDictionary()
_INIT_LOCKS_GUARD = Lock()


def _get_init_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    with _INIT_LOCKS_GUARD:
        lock = _INIT_LOCKS.get(loop)
        if lock is None:
            lock = _INIT_LOCK if not _INIT_LOCKS else asyncio.Lock()
            _INIT_LOCKS[loop] = lock
        return lock


async def _configure_connection(
    connection: aiosqlite.Connection,
    *,
    set_wal: bool = False,
) -> None:
    """Configure SQLite connection pragmas with retry on transient locks."""
    for attempt in range(_SQLITE_RETRY_ATTEMPTS):
        try:
            await connection.execute("PRAGMA busy_timeout=5000;")
            if set_wal:
                await connection.execute("PRAGMA journal_mode=WAL;")
            await connection.execute("PRAGMA synchronous=NORMAL;")
            return
        except sqlite3.OperationalError as error:
            message = str(error).lower()
            is_lock_error = "locked" in message or "busy" in message
            if not is_lock_error or attempt == _SQLITE_RETRY_ATTEMPTS - 1:
                raise
            await asyncio.sleep(0.05 * (2 ** attempt))


async def init_db(db_path: str | Path | None = None) -> None:
    """Create and configure the persistent LLM response cache."""
    path = (Path(db_path) if db_path is not None else DB_PATH).resolve()
    if path in _INITIALIZED_DB_PATHS and path.exists():
        return

    async with _get_init_lock():
        if path in _INITIALIZED_DB_PATHS and path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(path)) as connection:
            await _configure_connection(connection, set_wal=True)
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_responses (
                    request_hash TEXT PRIMARY KEY,
                    response_payload TEXT NOT NULL
                )
                """
            )
            await connection.commit()
        _INITIALIZED_DB_PATHS.add(path)


def _json_default(value: Any) -> Any:
    """Serialize common structured values used by agent requests."""
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating):
            return np.round(value, 12).tolist()
        return value.tolist()
    if isinstance(value, np.floating):
        return round(float(value), 12)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_request(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return round(float(value), 12)
    if isinstance(value, dict):
        return {key: _normalize_request(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_request(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)) or callable(getattr(value, "model_dump", None)):
        return _normalize_request(_json_default(value))
    return value


def _request_payload(
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    """Bind arguments and serialize a deterministic, non-secret cache payload."""
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    parameters = dict(bound_arguments.arguments)
    parameters.pop("api_key", None)

    # Preserve the explicit api_key exclusion for functions accepting **kwargs.
    keyword_arguments = parameters.get("kwargs")
    if isinstance(keyword_arguments, dict):
        keyword_arguments = dict(keyword_arguments)
        keyword_arguments.pop("api_key", None)
        parameters["kwargs"] = keyword_arguments

    return json.dumps(
        _normalize_request(parameters),
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def cache_request_hash(func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """Return the persistent-cache hash for a function call."""
    payload = _request_payload(inspect.signature(func), args, kwargs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def cache_contains(func: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    """Check whether a function call already has a persisted response."""
    await init_db()
    request_hash = cache_request_hash(func, *args, **kwargs)
    async with aiosqlite.connect(str(DB_PATH)) as connection:
        await _configure_connection(connection, set_wal=False)
        async with connection.execute(
            "SELECT 1 FROM llm_responses WHERE request_hash = ?",
            (request_hash,),
        ) as cursor:
            return await cursor.fetchone() is not None


def _return_model(func: Callable[..., Any]) -> type[Any]:
    """Resolve and validate the Pydantic model declared by a coroutine."""
    return_annotation = inspect.signature(func).return_annotation
    if return_annotation is inspect.Signature.empty:
        raise TypeError(
            f"{func.__qualname__} must declare a Pydantic return annotation"
        )
    if isinstance(return_annotation, str):
        return_annotation = get_type_hints(func)["return"]

    if not callable(getattr(return_annotation, "model_validate_json", None)):
        raise TypeError(
            f"{func.__qualname__} must return a Pydantic model with "
            "model_validate_json"
        )
    return return_annotation


def sqlite_persistent_cache(
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """Persist Pydantic results from an async function in SQLite."""
    signature = inspect.signature(func)
    model = _return_model(func)

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        request_hash = cache_request_hash(func, *args, **kwargs)
        await init_db()

        async with aiosqlite.connect(str(DB_PATH)) as connection:
            await _configure_connection(connection, set_wal=False)
            async with connection.execute(
                "SELECT response_payload FROM llm_responses WHERE request_hash = ?",
                (request_hash,),
            ) as cursor:
                row = await cursor.fetchone()

        if row is not None:
            try:
                return model.model_validate_json(row[0])
            except Exception:
                pass

        result = await func(*args, **kwargs)
        if getattr(result, "_is_fallback", False):
            return result
        model_dump_json = getattr(result, "model_dump_json", None)
        if not callable(model_dump_json):
            raise TypeError(
                f"{func.__qualname__} returned an object without model_dump_json"
            )
        response_payload = model_dump_json()

        async with aiosqlite.connect(str(DB_PATH)) as connection:
            await _configure_connection(connection, set_wal=False)
            for attempt in range(_SQLITE_RETRY_ATTEMPTS):
                try:
                    await connection.execute(
                        """
                        INSERT OR REPLACE INTO llm_responses
                            (request_hash, response_payload)
                        VALUES (?, ?)
                        """,
                        (request_hash, response_payload),
                    )
                    await connection.commit()
                    break
                except sqlite3.OperationalError as error:
                    message = str(error).lower()
                    is_lock_error = "locked" in message or "busy" in message
                    if not is_lock_error or attempt == _SQLITE_RETRY_ATTEMPTS - 1:
                        raise
                    await connection.rollback()
                    await asyncio.sleep(0.05 * (2 ** attempt))

        return result

    return wrapper


__all__ = [
    "DB_PATH",
    "cache_contains",
    "cache_request_hash",
    "init_db",
    "sqlite_persistent_cache",
]
