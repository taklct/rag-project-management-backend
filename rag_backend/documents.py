"""Helpers for loading and normalising document content."""

from __future__ import annotations

from typing import List


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def read_pdf(path: str) -> str:
    try:
        import PyPDF2
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyPDF2 not installed. Run: pip install PyPDF2") from exc

    text: List[str] = []
    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            try:
                text.append(page.extract_text() or "")
            except Exception:  # pragma: no cover - defensive against malformed PDFs
                text.append("")
    return "\n".join(text)


def load_document(path: str) -> str:
    lower_path = path.lower()
    if lower_path.endswith((".txt", ".md")):
        return read_text_file(path)
    if lower_path.endswith(".pdf"):
        return read_pdf(path)
    if lower_path.endswith((".xlsx", ".xls")):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "pandas required for Excel. Run: pip install pandas openpyxl xlrd"
            ) from exc
        dataframes = pd.read_excel(path, sheet_name=None)
        sections = []
        for sheet, dataframe in dataframes.items():
            header = f"--- Sheet: {sheet} ---"
            dataframe = dataframe.fillna("")
            rows = dataframe.to_dict(orient="records")
            row_lines = [
                " | ".join(f"{key}={value}" for key, value in row.items())
                for row in rows
            ]
            sections.append(header + "\n" + "\n".join(row_lines))
        return "\n\n".join(sections)
    if lower_path.endswith(".csv"):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("pandas required for CSV. Run: pip install pandas") from exc
        dataframe = pd.read_csv(path).fillna("")
        rows = dataframe.to_dict(orient="records")
        return "\n".join(
            " | ".join(f"{key}={value}" for key, value in row.items()) for row in rows
        )
    if lower_path.endswith(".docx"):
        try:
            import docx
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "python-docx required for Word files. Run: pip install python-docx"
            ) from exc
        document = docx.Document(path)
        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text and paragraph.text.strip()
        ]
        return "\n".join(paragraphs)

    try:
        return read_text_file(path)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Unsupported file type: {path}") from exc
