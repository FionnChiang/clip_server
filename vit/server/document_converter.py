"""PDF / OFD 文档转图片模块。

将上传的文档按页渲染为 PNG 图片，使训练集上传和推理流程
对文档文件与原始图片走完全相同的下游逻辑。

- PDF：使用 PyMuPDF 直接渲染。
- OFD：使用 easyofd 解析并渲染为 PIL Image。

注意：OFD 文本渲染依赖中文字体。easyofd 绘制文本时硬编码使用字体名
"宋体"（失败时回退 Helvetica，中文会渲染为空白但不会报错），因此
模块在首次处理 OFD 前会把系统中文字体注册为 "宋体" 等别名（见
``_ensure_cjk_font``）。无中文字体的环境仅影响 OFD 文本内容，
图片类内容不受影响。
"""

import io
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_DOC_EXTENSIONS = {".pdf", ".ofd"}

DEFAULT_DPI = 150
MAX_PAGES = 50
MAX_DIMENSION = 2000  # 渲染结果长边上限（像素），超限等比缩小

# easyofd 绘制文本时使用的字体名（draw_pdf.py 硬编码 "宋体"，字体映射见
# easyofd/parser_ofd/__init__.py 的 font_map）
_CJK_FONT_ALIASES = ["宋体", "楷体", "黑体", "SimSun", "SimHei", "KaiTi"]

# 常见系统中文字体文件位置（按优先级）
_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
]

_cjk_font_ready = False

# easyofd 解析时会把 OFD 文件写盘到当前工作目录（损坏文件不会自清理），
# 因此转换在独立临时目录中进行并加锁串行化（os.chdir 是进程级操作）。
_ofd_lock = threading.Lock()


def is_document(filename: str) -> bool:
    """判断文件名是否为支持的文档类型（pdf/ofd）。"""
    return Path(filename or "").suffix.lower() in SUPPORTED_DOC_EXTENSIONS


def document_to_images(
    filename: str,
    data: bytes,
    dpi: int = DEFAULT_DPI,
    max_pages: int = MAX_PAGES,
) -> list[tuple[int, bytes]]:
    """将 PDF/OFD 文档按页转换为 PNG 图片。

    Args:
        filename: 原始文件名（用于判断文档类型）。
        data: 文件字节内容。
        dpi: PDF 渲染分辨率。
        max_pages: 最大转换页数，超出部分截断。

    Returns:
        [(page_no, png_bytes), ...]，page_no 从 1 开始。

    Raises:
        ValueError: 不支持的类型、加密/损坏/解析失败、或转换结果为空。
    """
    ext = Path(filename or "").suffix.lower()
    if ext == ".pdf":
        pages = _pdf_to_images(data, dpi=dpi, max_pages=max_pages)
    elif ext == ".ofd":
        pages = _ofd_to_images(data, max_pages=max_pages)
    else:
        raise ValueError(f"Unsupported document type: {ext or filename}")

    if not pages:
        raise ValueError(f"No pages could be rendered from {filename}")

    return [(page_no, _downscale(png)) for page_no, png in pages]


def _pdf_to_images(data: bytes, dpi: int, max_pages: int) -> list[tuple[int, bytes]]:
    """PyMuPDF 渲染 PDF 每页为 PNG。"""
    try:
        import pymupdf
    except ImportError:
        raise ValueError("PDF support requires pymupdf. Please install it first.")

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Failed to parse PDF file: {e}")

    try:
        if doc.needs_pass:
            raise ValueError("Encrypted PDF is not supported")

        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        pages = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                logger.warning(f"PDF has more than {max_pages} pages, truncated")
                break
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pages.append((i + 1, pix.tobytes("png")))
        return pages
    finally:
        doc.close()


def _ofd_to_images(data: bytes, max_pages: int) -> list[tuple[int, bytes]]:
    """easyofd 解析 OFD 并渲染每页为 PNG。"""
    try:
        _ensure_cjk_font()
        from easyofd.ofd import OFD
    except ImportError:
        raise ValueError("OFD support requires easyofd. Please install it first.")

    # easyofd 会在当前工作目录写盘（`{pid}_{uuid}.ofd`，解析失败时不清理），
    # 切换到临时目录执行并在结束后删除，避免污染服务工作目录。
    workdir = tempfile.mkdtemp(prefix="ofd_convert_")
    with _ofd_lock:
        orig_cwd = os.getcwd()
        os.chdir(workdir)
        try:
            ofd = OFD()
            ofd.read(data, fmt="binary")
            pil_images = ofd.to_jpg(format="jpg")
        except Exception as e:
            raise ValueError(f"Failed to parse OFD file: {e}")
        finally:
            os.chdir(orig_cwd)
    shutil.rmtree(workdir, ignore_errors=True)

    pages = []
    for i, pil_img in enumerate(pil_images or []):
        if i >= max_pages:
            logger.warning(f"OFD has more than {max_pages} pages, truncated")
            break
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="PNG")
        pages.append((i + 1, buf.getvalue()))
    return pages


def _downscale(png_bytes: bytes) -> bytes:
    """长边超过 MAX_DIMENSION 时等比缩小，控制存储与传输体积。"""
    img = Image.open(io.BytesIO(png_bytes))
    if max(img.size) <= MAX_DIMENSION:
        return png_bytes
    img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _ensure_cjk_font() -> None:
    """将系统中文字体注册为 easyofd 使用的字体名（"宋体" 等）。

    easyofd 绘制 OFD 文本时硬编码使用 "宋体"，无此字体时文本会以
    Helvetica 回退渲染导致中文空白。此处从常见系统字体路径中找到
    第一个 CJK 字体，用 reportlab 注册为多个别名。全程吞掉异常，
    失败仅记 warning（图片类 OFD 不受影响）。
    """
    global _cjk_font_ready
    if _cjk_font_ready:
        return

    font_path = next((p for p in _CJK_FONT_CANDIDATES if Path(p).is_file()), None)
    if font_path is None:
        logger.warning("No CJK font found for OFD text rendering. "
                       "Install fonts-noto-cjk (e.g. apt-get install fonts-noto-cjk).")
        _cjk_font_ready = True
        return

    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        registered = 0
        for name in _CJK_FONT_ALIASES:
            try:
                pdfmetrics.registerFont(TTFont(name, font_path, subfontIndex=0))
                registered += 1
            except Exception:
                continue
        if registered:
            logger.info(f"Registered CJK font '{font_path}' as {registered} alias(es) for OFD rendering")
    except Exception as e:
        logger.warning(f"Failed to register CJK font for OFD rendering: {e}")
    finally:
        _cjk_font_ready = True
