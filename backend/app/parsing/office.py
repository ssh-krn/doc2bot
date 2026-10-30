"""Office formats. DOC/DOCX/PPT/PPTX/XLS are converted with LibreOffice (headless), then reuse the
PDF parser or the spreadsheet parser. XLSX is read directly so each sheet stays a table."""

import io
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import openpyxl

from ..config import get_settings
from .pdf import parse_pdf
from .tables import table_to_markdown
from .types import ContentBlock, ParseError


def _libreoffice() -> str:
    binary = get_settings().libreoffice_bin
    if shutil.which(binary) is None:
        raise ParseError(
            "LibreOffice is required to read DOC, DOCX, PPT, PPTX and XLS files. "
            "Install it, or set LIBREOFFICE_BIN to its path."
        )
    return binary


def convert(data: bytes, src_ext: str, target_ext: str) -> bytes:
    """Convert a document with LibreOffice. Each call gets its own profile dir so conversions can run concurrently."""
    binary = _libreoffice()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / f"input{src_ext}"
        src.write_bytes(data)
        profile = tmp_path / f"profile-{uuid.uuid4().hex}"
        cmd = [binary, f"-env:UserInstallation=file://{profile}", "--headless",
               "--convert-to", target_ext, "--outdir", str(tmp_path), str(src)]
        try:
            subprocess.run(cmd, check=True, timeout=180, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise ParseError(f"LibreOffice could not convert this file: {e.stderr.decode(errors='ignore')[:200]}")
        except subprocess.TimeoutExpired:
            raise ParseError("Conversion timed out. The file may be too large or malformed.")
        out = tmp_path / f"input.{target_ext}"
        if not out.exists():
            raise ParseError("Conversion produced no output. The file may be corrupt.")
        return out.read_bytes()


def parse_xlsx(data: bytes) -> list[ContentBlock]:
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    blocks: list[ContentBlock] = []
    try:
        for index, ws in enumerate(wb.worksheets, start=1):
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            md = table_to_markdown(rows)
            if md:
                blocks.append(ContentBlock("table", f"Sheet: {ws.title}\n{md}", index,
                                           meta={"sheet": ws.title}))
    finally:
        wb.close()
    return blocks


def parse_office(filename_ext: str, data: bytes) -> list[ContentBlock]:
    ext = filename_ext.lower()
    if ext in (".doc", ".docx", ".ppt", ".pptx"):
        return parse_pdf(convert(data, ext, "pdf"))  # pages and slides become PDF pages
    if ext == ".xls":
        return parse_xlsx(convert(data, ext, "xlsx"))
    if ext in (".xlsx", ".xlsm"):
        return parse_xlsx(data)
    raise ParseError(f"Unsupported Office format: {ext}")
