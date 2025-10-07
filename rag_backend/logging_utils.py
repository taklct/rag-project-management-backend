"""Logging helpers for persisting query metadata."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict


def log_query(log_path: str, row: Dict[str, Any]) -> None:
    fieldnames = [
        "timestamp",
        "question",
        "top_k",
        "temperature_requested",
        "temperature_sent",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "duration_sec",
    ]
    write_header = True
    if os.path.exists(log_path):
        with open(log_path, "r", newline="", encoding="utf-8") as existing:
            reader = csv.reader(existing)
            try:
                existing_header = next(reader)
            except StopIteration:
                pass
            else:
                write_header = list(existing_header) != fieldnames
        mode = "a"
    else:
        mode = "w"

    with open(log_path, mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
