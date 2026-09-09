import io
import logging

import pymupdf
import pytesseract

from PIL import Image
from pypdf import PdfReader

from config import (
    MIN_PAGE_TEXT_CHARS,
    OCR_SCALE,
    MAX_OCR_PAGES,
    OCR_LANGUAGE,
)


logger = logging.getLogger(__name__)


# ==================================================
# IMAGE OCR
# ==================================================

def ocr_image(
    image: Image.Image,
):
    if image.mode not in (
        "RGB",
        "L",
    ):
        image = image.convert(
            "RGB"
        )

    text = pytesseract.image_to_string(
        image,
        lang=OCR_LANGUAGE,
    )

    return text.strip()


# ==================================================
# PDF PAGE OCR
# ==================================================

def ocr_pdf_page(
    page,
):
    matrix = pymupdf.Matrix(
        OCR_SCALE,
        OCR_SCALE,
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image_bytes = pixmap.tobytes(
        "png"
    )

    image = Image.open(
        io.BytesIO(
            image_bytes
        )
    )

    return ocr_image(
        image
    )


# ==================================================
# PDF EXTRACTION + OCR FALLBACK
# ==================================================

def extract_pdf_text_with_ocr(
    file_bytes: bytes,
):
    text_parts = []

    ocr_pages = 0
    skipped_ocr_pages = 0

    reader = PdfReader(
        io.BytesIO(
            file_bytes
        )
    )

    render_document = pymupdf.open(
        stream=file_bytes,
        filetype="pdf",
    )

    try:
        total_pages = len(
            reader.pages
        )

        for page_index in range(
            total_pages
        ):
            page_number = (
                page_index + 1
            )

            try:
                normal_text = (
                    reader.pages[
                        page_index
                    ].extract_text()
                    or ""
                ).strip()

            except Exception:
                logger.exception(
                    "Normal PDF extraction failed "
                    "page=%s",
                    page_number,
                )

                normal_text = ""

            if (
                len(normal_text)
                >= MIN_PAGE_TEXT_CHARS
            ):
                text_parts.append(
                    (
                        f"[Page {page_number}]\n"
                        f"{normal_text}"
                    )
                )

                continue

            if ocr_pages >= MAX_OCR_PAGES:
                skipped_ocr_pages += 1

                if normal_text:
                    text_parts.append(
                        (
                            f"[Page {page_number}]\n"
                            f"{normal_text}"
                        )
                    )

                continue

            logger.info(
                "Running OCR page=%s",
                page_number,
            )

            try:
                render_page = (
                    render_document[
                        page_index
                    ]
                )

                ocr_text = ocr_pdf_page(
                    render_page
                )

            except Exception:
                logger.exception(
                    "OCR failed page=%s",
                    page_number,
                )

                ocr_text = ""

            if ocr_text:
                text_parts.append(
                    (
                        f"[Page {page_number} - OCR]\n"
                        f"{ocr_text}"
                    )
                )

                ocr_pages += 1

            elif normal_text:
                text_parts.append(
                    (
                        f"[Page {page_number}]\n"
                        f"{normal_text}"
                    )
                )

        return (
            "\n\n".join(
                text_parts
            ),
            ocr_pages,
            skipped_ocr_pages,
        )

    finally:
        render_document.close()
