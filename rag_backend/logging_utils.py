"""Logging helpers for persisting query metadata."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict


def log_query(log_path: str, row: Dict[str, Any]) -> None:
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "question",
                "top_k",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "duration_sec",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
