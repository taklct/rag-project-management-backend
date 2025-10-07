"""Text chunking helpers."""

from __future__ import annotations

from typing import List


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 150) -> List[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n") if paragraph.strip()]
    chunks: List[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 1 <= max_chars:
            buffer = (buffer + "\n" + paragraph).strip()
        else:
            if buffer:
                chunks.append(buffer)
            if len(paragraph) > max_chars:
                start = 0
                while start < len(paragraph):
                    end = start + max_chars
                    chunks.append(paragraph[start:end])
                    start = max(0, end - overlap)
            else:
                buffer = paragraph
    if buffer:
        chunks.append(buffer)

    if overlap <= 0:
        return chunks

    overlapped: List[str] = []
    for index, chunk in enumerate(chunks):
        if index == 0:
            overlapped.append(chunk)
        else:
            previous = chunks[index - 1]
            tail = previous[-overlap:]
            overlapped.append((tail + "\n" + chunk).strip())
    return overlapped
