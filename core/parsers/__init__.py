"""
Parsers package: the study's slides are .pptx only.
"""
import os

SUPPORTED = {".pptx"}


def parse_document(filepath: str) -> dict:
    """Parse a .pptx file into {doc_id, source_file, format, title, sections, ...}."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pptx":
        from .pptx_parser import parse_pptx
        return parse_pptx(filepath)
    raise ValueError(f"Unsupported file type: {ext}. Supported: .pptx")
