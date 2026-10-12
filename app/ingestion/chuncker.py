"""Structure-aware chunker.

Strategy:
1. Split Markdown/plain text into sections by heading, keeping the heading path
   ("Handbook > Leave > Sick leave") as context.
2. Sections that fit in `max_words` become one chunk.
3. Longer sections are packed paragraph by paragraph; paragraphs that are
   themselves too long are split with a sliding word window and overlap.

Chunk IDs are deterministic (uuid5 of source + position + text hash) so
re-ingesting the same corpus is idempotent in Qdrant.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_NAMESPACE = uuid.UUID("6f1c2d3e-0000-4000-8000-00000000c0de")
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    title: str
    section: str
    text: str
    position: int

    @property
    def embed_text(self) -> str:
        """Text actually embedded: context prefix + body, so short chunks keep their meaning."""
        prefix = " > ".join(p for p in (self.title, self.section) if p)
        return f"{prefix}\n{self.text}" if prefix else self.text

    def to_dict(self) -> dict:
        return asdict(self)


def _words(text: str) -> list[str]:
    return text.split()


def _split_long(text: str, max_words: int, overlap: int) -> list[str]:
    words = _words(text)
    if len(words) <= max_words:
        return [text.strip()]
    step = max(1, max_words - overlap)
    out = []
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        out.append(" ".join(window))
        if start + max_words >= len(words):
            break
    return out


def _pack_paragraphs(body: str, max_words: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for para in paragraphs:
        n = len(_words(para))
        if n > max_words:
            if current:
                chunks.append("\n\n".join(current))
                current, count = [], 0
            chunks.extend(_split_long(para, max_words, overlap))
            continue
        if count + n > max_words and current:
            chunks.append("\n\n".join(current))
            # carry the last paragraph over as overlap if it is small enough
            last = current[-1]
            if len(_words(last)) <= overlap:
                current, count = [last], len(_words(last))
            else:
                current, count = [], 0
        current.append(para)
        count += n
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def split_sections(text: str, default_title: str = "") -> list[tuple[str, str]]:
    """Return [(heading_path, body)] for a Markdown-ish document."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_code = False

    def flush():
        body = "\n".join(buf).strip()
        if body:
            path = " > ".join(h for _, h in stack)
            sections.append((path, body))
        buf.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else _HEADING.match(line)
        if m:
            flush()
            level, heading = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            buf.append(line)
    flush()
    return sections


def chunk_text(
    text: str,
    source: str,
    max_words: int = 180,
    overlap: int = 40,
) -> list[Chunk]:
    title = Path(source).stem.replace("_", " ").replace("-", " ").title()
    doc_id = hashlib.sha1(source.encode()).hexdigest()[:12]
    chunks: list[Chunk] = []
    position = 0
    for section, body in split_sections(text, title):
        pieces = [body] if len(_words(body)) <= max_words else _pack_paragraphs(body, max_words, overlap)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            digest = hashlib.sha1(f"{source}|{position}|{piece}".encode()).hexdigest()
            chunks.append(
                Chunk(
                    id=str(uuid.uuid5(_NAMESPACE, digest)),
                    doc_id=doc_id,
                    source=source,
                    title=title,
                    section=section,
                    text=piece,
                    position=position,
                )
            )
            position += 1
    return chunks


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # lazy: only needed for PDFs

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_directory(docs_dir: Path, max_words: int = 180, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            rel = str(path.relative_to(docs_dir))
            chunks.extend(chunk_text(read_document(path), rel, max_words, overlap))
    return chunks