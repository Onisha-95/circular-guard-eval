"""
OCR Utilities — Tesseract-based image text extraction.

Used by pdf_parser.py  (scanned/image-only pages)
Used by pptx_parser.py (image shapes inside slides)

Requirements:
    pip install pytesseract Pillow pdf2image
    brew install tesseract          # Mac
    apt install tesseract-ocr       # Linux

What it does:
    1. Receives a PIL Image or a raw image path
    2. Pre-processes (grayscale, contrast boost, denoise) for better accuracy
    3. Runs Tesseract OCR
    4. Returns cleaned text + confidence score

Confidence < MIN_CONFIDENCE → returns empty string (garbage text rejected).
"""

import os
import re

# Minimum Tesseract confidence (0–100) to accept OCR output.
# Below this, the image is likely a diagram/chart with no readable text.
MIN_CONFIDENCE = 40

# Minimum number of real words for OCR output to be considered useful.
MIN_WORD_COUNT = 5


def _tesseract_available() -> bool:
    """Return True if pytesseract + tesseract binary are both available."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


TESSERACT_OK = _tesseract_available()


def ocr_image(image, label: str = "image") -> dict:
    """
    Run OCR on a PIL Image.

    Args:
        image  : PIL.Image.Image
        label  : human-readable label for logging (e.g. "page_3", "slide_2_img_1")

    Returns:
        {
            "text":       str,    # extracted text (empty if unusable)
            "confidence": float,  # 0–100
            "used_ocr":   bool,   # True if OCR actually ran
            "label":      str,
        }
    """
    if not TESSERACT_OK:
        return {"text": "", "confidence": 0.0, "used_ocr": False, "label": label}

    try:
        import pytesseract
        from PIL import ImageFilter, ImageEnhance, ImageOps

        # ── Pre-processing ──────────────────────────────────────────────
        img = image.convert("L")                          # grayscale
        img = ImageOps.autocontrast(img, cutoff=2)        # stretch contrast
        img = img.filter(ImageFilter.MedianFilter(size=3)) # denoise
        # Upscale small images — Tesseract works best at ~300 DPI
        w, h = img.size
        if w < 1000 or h < 1000:
            scale = max(1000 / w, 1000 / h)
            img = img.resize((int(w * scale), int(h * scale)))

        # ── OCR ─────────────────────────────────────────────────────────
        data = pytesseract.image_to_data(
            img,
            output_type=pytesseract.Output.DICT,
            config="--psm 6"   # assume uniform block of text
        )

        words = []
        confs = []
        for i, conf in enumerate(data["conf"]):
            try:
                c = float(conf)
            except (ValueError, TypeError):
                continue
            if c < 0:
                continue
            word = (data["text"][i] or "").strip()
            if word:
                words.append(word)
                confs.append(c)

        if not words:
            return {"text": "", "confidence": 0.0, "used_ocr": True, "label": label}

        avg_conf = sum(confs) / len(confs)
        raw_text = " ".join(words)

        # ── Quality filter ───────────────────────────────────────────────
        real_words = [w for w in words if re.match(r"[A-Za-z]{2,}", w)]
        if avg_conf < MIN_CONFIDENCE or len(real_words) < MIN_WORD_COUNT:
            return {"text": "", "confidence": avg_conf, "used_ocr": True, "label": label}

        # ── Clean text ───────────────────────────────────────────────────
        text = _clean(raw_text)
        return {"text": text, "confidence": round(avg_conf, 1), "used_ocr": True, "label": label}

    except Exception as e:
        return {"text": "", "confidence": 0.0, "used_ocr": False, "label": label,
                "error": str(e)}


def ocr_image_file(filepath: str, label: str = None) -> dict:
    """
    Run OCR on an image file (PNG, JPEG, BMP, TIFF…).
    """
    if not TESSERACT_OK:
        return {"text": "", "confidence": 0.0, "used_ocr": False,
                "label": label or os.path.basename(filepath)}
    try:
        from PIL import Image
        img = Image.open(filepath)
        return ocr_image(img, label or os.path.basename(filepath))
    except Exception as e:
        return {"text": "", "confidence": 0.0, "used_ocr": False,
                "label": label or filepath, "error": str(e)}


def ocr_pdf_page_image(page, page_num: int) -> dict:
    """
    OCR a pdfplumber page that has no extractable text.
    Converts the page to an image first, then runs OCR.

    Args:
        page     : pdfplumber.Page
        page_num : 1-based page number for labelling
    """
    if not TESSERACT_OK:
        return {"text": "", "confidence": 0.0, "used_ocr": False,
                "label": f"page_{page_num}"}
    try:
        # pdfplumber can render to PIL image (requires pdf2image + poppler)
        pil_img = page.to_image(resolution=300).original
        return ocr_image(pil_img, label=f"page_{page_num}")
    except Exception:
        # Fallback: use pdf2image directly
        try:
            import pdf2image, io
            from PIL import Image
            images = pdf2image.convert_from_path(
                page.pdf.stream.name,
                first_page=page_num,
                last_page=page_num,
                dpi=300
            )
            if images:
                return ocr_image(images[0], label=f"page_{page_num}")
        except Exception as e2:
            pass
        return {"text": "", "confidence": 0.0, "used_ocr": False,
                "label": f"page_{page_num}"}


def ocr_pptx_image_shape(shape, slide_num: int, shape_num: int) -> dict:
    """
    OCR an image shape from a python-pptx slide.

    Args:
        shape     : pptx.shapes.picture.Picture
        slide_num : 1-based slide number
        shape_num : shape index on the slide
    """
    if not TESSERACT_OK:
        return {"text": "", "confidence": 0.0, "used_ocr": False,
                "label": f"slide_{slide_num}_img_{shape_num}"}
    try:
        from PIL import Image
        import io
        image_blob = shape.image.blob
        img = Image.open(io.BytesIO(image_blob))
        return ocr_image(img, label=f"slide_{slide_num}_img_{shape_num}")
    except Exception as e:
        return {"text": "", "confidence": 0.0, "used_ocr": False,
                "label": f"slide_{slide_num}_img_{shape_num}", "error": str(e)}


def _clean(text: str) -> str:
    """Remove OCR noise: repeated spaces, stray symbols, short garbled tokens."""
    # Remove lines that are pure symbols / numbers (likely rulers/borders)
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Skip lines with >50% non-alphanumeric characters
        alpha = sum(1 for c in stripped if c.isalnum())
        if len(stripped) > 0 and alpha / len(stripped) < 0.4:
            continue
        lines.append(stripped)
    cleaned = " ".join(lines)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tesseract_status() -> str:
    """Return a human-readable status string for the startup log."""
    if TESSERACT_OK:
        try:
            import pytesseract
            ver = pytesseract.get_tesseract_version()
            return f"[ok] Tesseract {ver} — OCR enabled"
        except Exception:
            pass
    return "[!] Tesseract not found — OCR disabled (images/scanned pages will be skipped)"
