import io

from docx import Document


def extract_docx_text(
    file_bytes: bytes,
):
    document = Document(
        io.BytesIO(
            file_bytes
        )
    )

    paragraphs = []

    for paragraph in document.paragraphs:
        text = (
            paragraph.text.strip()
        )

        if text:
            paragraphs.append(
                text
            )

    return "\n\n".join(
        paragraphs
    )


def extract_txt_text(
    file_bytes: bytes,
):
    try:
        return file_bytes.decode(
            "utf-8"
        )

    except UnicodeDecodeError:
        return file_bytes.decode(
            "latin-1",
            errors="ignore",
        )
