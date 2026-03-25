"""
utils/logger.py
---------------
Structured JSON logger.  Every pipeline stage emits a log event with:
  stage, query_id, latency_ms, tokens_used, model, confidence, extra fields.

Usage:
    from utils.logger import get_logger, log_stage
    logger = get_logger(__name__)
    with log_stage(logger, "retrieval", query_id=qid) as ctx:
        results = retriever.retrieve(query)
        ctx["chunks_returned"] = len(results)
"""

import time
import uuid
import logging
import json
from contextlib import contextmanager
from typing import Any, Generator

import structlog
from pythonjsonlogger import jsonlogger


def setup_logging(log_level: str = "INFO", fmt: str = "json") -> None:
    """Call once at startup."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    if fmt == "json":
        handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        handler.setFormatter(formatter)
        logging.basicConfig(level=level, handlers=[handler])
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)


@contextmanager
def log_stage(
    logger: structlog.BoundLogger,
    stage: str,
    query_id: str = "",
    **extra: Any,
) -> Generator[dict, None, None]:
    """
    Context manager that measures latency and logs a structured event.

    with log_stage(logger, "ranking", query_id=qid) as ctx:
        ctx["chunks_scored"] = 10
    """
    ctx: dict = {"stage": stage, "query_id": query_id or str(uuid.uuid4()), **extra}
    t0 = time.perf_counter()
    try:
        yield ctx
        ctx["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        ctx["status"] = "ok"
        logger.info("stage_complete", **ctx)
    except Exception as exc:
        ctx["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        ctx["status"] = "error"
        ctx["error"] = str(exc)
        logger.error("stage_error", **ctx)
        raise
