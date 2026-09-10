import asyncio
import logging
import os

from telegram import Update
from telegram.ext import ContextTypes

from config import MAX_DOCUMENT_SIZE

from database.documents import save_document

from rag.embeddings import (
    generate_passage_embeddings,
)

from rag.retrieval import (
    chunk_text,
)

from services.ocr import (
    extract_pdf_text_with_ocr,
)

from services.document_parser import (
    extract_docx_text,
    extract_txt_text,
)

from services.vision import (
    IMAGE_EXTENSIONS,
    process_image_upload,
)


logger = logging.getLogger(__name__)


async def handle_document(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    telegram_user_id = (
        update.effective_user.id
    )

    document = (
        update.message.document
    )

    filename = (
        document.file_name
        or "document"
    )

    extension = (
        os.path.splitext(
            filename
        )[1]
        .lower()
    )

    # Images sent as uncompressed Telegram documents.
    if extension in IMAGE_EXTENSIONS:
        await process_image_upload(
            update=update,
            context=context,
            file_id=document.file_id,
            filename=filename,
            file_size=document.file_size,
            caption=update.message.caption,
        )
        return

    logger.info(
        "Document received "
        "user_id=%s filename=%s",
        telegram_user_id,
        filename,
    )

    try:
        if (
            document.file_size
            and document.file_size
            > MAX_DOCUMENT_SIZE
        ):
            await update.message.reply_text(
                "That file is too large."
            )
            return

        if extension not in (
            ".pdf",
            ".txt",
            ".docx",
        ):
            await update.message.reply_text(
                "I currently support PDF, TXT, DOCX, "
                "JPG, JPEG, PNG and WEBP files."
            )
            return

        await update.message.reply_text(
            "📄 Reading and indexing your file..."
        )

        telegram_file = (
            await context.bot.get_file(
                document.file_id
            )
        )

        file_data = (
            await telegram_file
            .download_as_bytearray()
        )

        if len(file_data) > MAX_DOCUMENT_SIZE:
            await update.message.reply_text("That file is too large.")
            return

        file_bytes = bytes(
            file_data
        )

        ocr_pages = 0
        skipped_ocr_pages = 0

        if extension == ".pdf":
            await update.message.reply_text(
                "🔎 Checking whether OCR is needed..."
            )

            (
                extracted_text,
                ocr_pages,
                skipped_ocr_pages,
            ) = await asyncio.to_thread(
                extract_pdf_text_with_ocr,
                file_bytes,
            )

            file_type = "pdf"

        elif extension == ".docx":
            extracted_text = await asyncio.to_thread(extract_docx_text, file_bytes)

            file_type = "docx"

        else:
            extracted_text = (
                extract_txt_text(
                    file_bytes
                )
            )

            file_type = "txt"

        extracted_text = (
            extracted_text.strip()
        )

        if not extracted_text:
            await update.message.reply_text(
                "I couldn't find readable text "
                "in this file."
            )
            return

        if len(extracted_text) > 1_000_000:
            await update.message.reply_text("This document has too much text to index at once. Please split it into smaller files.")
            return

        chunks = chunk_text(
            extracted_text
        )

        if not chunks:
            await update.message.reply_text(
                "I couldn't create searchable "
                "chunks from this file."
            )
            return

        embeddings = None

        try:
            embeddings = (
                await asyncio.to_thread(
                    generate_passage_embeddings,
                    chunks,
                )
            )

        except Exception:
            logger.exception(
                "Semantic indexing failed "
                "user_id=%s",
                telegram_user_id,
            )

        await save_document(
            telegram_user_id,
            filename,
            file_type,
            chunks,
            embeddings,
        )

        lines = [
            "✅ File processed successfully.",
            "",
            f"📄 File: {filename}",
            (
                f"📚 Searchable chunks: "
                f"{len(chunks)}"
            ),
        ]

        if embeddings:
            lines.append(
                "🧠 Semantic index: ready"
            )

        else:
            lines.append(
                "⚠️ Semantic index unavailable"
            )

        if extension == ".pdf":
            if ocr_pages > 0:
                lines.append(
                    (
                        f"👁️ OCR used on "
                        f"{ocr_pages} page(s)"
                    )
                )

            else:
                lines.append(
                    "👁️ OCR was not needed"
                )

            if skipped_ocr_pages > 0:
                lines.append(
                    (
                        f"⚠️ {skipped_ocr_pages} "
                        f"page(s) exceeded the "
                        f"OCR processing limit"
                    )
                )

        lines.extend(
            [
                "",
                "You can now ask:",
                "• Summarize this document",
                "• Find information in it",
                "• Explain a section",
                "• What are the main points?",
            ]
        )

        await update.message.reply_text(
            "\n".join(lines)
        )

    except Exception:
        logger.exception(
            "Document processing error "
            "user_id=%s",
            telegram_user_id,
        )

        await update.message.reply_text(
            "I couldn't process that file."
        )
