from telegram import Update
from telegram.ext import ContextTypes

from database.memory import (
    delete_memory,
)

from database.documents import (
    load_documents,
    delete_documents,
    get_document,
    delete_document,
)

from database.images import (
    delete_latest_image,
)

from database.tasks import (
    get_tasks,
    create_task,
    complete_task,
    delete_task,
    resolve_task_number,
)

from database.long_term_memory import (
    save_long_term_memory,
    list_long_term_memories,
    forget_long_term_memory,
)

from services.memory import (
    remember_user_memory,
)


async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "Hello! 👋\n\n"
        "I am Vruyr's custom AI assistant.\n\n"
        "Features:\n"
        "• AI chat 💬\n"
        "• Conversation memory 🧠\n"
        "• Voice messages 🎤\n"
        "• Semantic document search 🔎\n"
        "• Normal and scanned PDF OCR 📄\n"
        "• Photo and screenshot understanding 🖼️\n"
        "• Image text reading 👁️\n"
        "• DOCX and TXT files 📝\n"
        "• AI task management ✅\n\n"
        "Send a photo first, then send a separate instruction such as:\n"
        "• What is in this image?\n"
        "• Read the text in this screenshot\n"
        "• Explain this diagram\n"
        "• What objects do you see?\n\n"
        "Commands:\n"
        "/tasks - show tasks\n"
        "/addtask - add a task\n"
        "/donetask - complete a task\n"
        "/deletetask - delete a task\n"
        "/remember - save a long-term memory\n"
        "/memory - show long-term memories\n"
        "/forget - forget one memory\n"
        "/files - show uploaded files\n"
        "/deletefile - delete one uploaded file\n"
        "/clearfiles - delete uploaded files\n"
        "/clearimage - forget the latest image\n"
        "/clear - clear conversation memory"
    )


async def clear_memory(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    telegram_user_id = (
        update.effective_user.id
    )

    await delete_memory(
        telegram_user_id
    )

    await update.message.reply_text(
        "Conversation memory cleared. 🧹"
    )


async def files_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    telegram_user_id = (
        update.effective_user.id
    )

    documents = await load_documents(
        telegram_user_id
    )

    if not documents:
        await update.message.reply_text(
            "You haven't uploaded any files yet."
        )
        return

    lines = [
        "📁 Your uploaded files:",
        "",
    ]

    for (
        document_id,
        filename,
        file_type,
        created_at,
    ) in documents:
        lines.append(
            f"• {document_id}. {filename}"
        )

    from utils.telegram_text import send_long_message
    lines.extend(['', 'Remove one file: /deletefile ID'])
    await send_long_message(update, "\n".join(lines))


async def delete_file_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    try:
        document_id = int(args[0])
        valid = (len(args) in {1, 2} and 0 < document_id < 2**63
                 and (len(args) == 1 or args[1] == 'confirm'))
    except (IndexError, ValueError):
        valid = False
    if not valid:
        await update.message.reply_text('Usage: /deletefile ID\nUse /files to find the ID.')
        return
    user_id = update.effective_user.id
    if len(args) == 1:
        row = await get_document(user_id, document_id)
        if row is None:
            await update.message.reply_text('File not found.')
            return
        from utils.telegram_text import send_long_message
        await send_long_message(update,
            f'Delete file {row[0]}: {row[1]}?\n'
            'This permanently removes this file and its searchable content.\n'
            f'Confirm with /deletefile {document_id} confirm')
        return
    filename = await delete_document(user_id, document_id)
    if filename is None:
        await update.message.reply_text('File not found.')
        return
    from utils.telegram_text import send_long_message
    await send_long_message(update, f'Deleted file {document_id}: {filename}')


async def clear_files(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    telegram_user_id = (
        update.effective_user.id
    )

    await delete_documents(
        telegram_user_id
    )

    await update.message.reply_text(
        "Uploaded documents deleted. 🗑️"
    )


async def clear_image_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    telegram_user_id = (
        update.effective_user.id
    )

    await delete_latest_image(
        telegram_user_id
    )

    await update.message.reply_text(
        "Latest image cleared. 🧹"
    )


async def tasks_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    from database.personal import fetch
    from utils.telegram_text import send_long_message

    rows = await fetch(
        """SELECT
            id,
            title,
            status,
            due_date,
            priority,
            project,
            recurrence
        FROM tasks
        WHERE telegram_user_id = %s
        ORDER BY
            (status = 'open') DESC,
            priority DESC,
            id DESC
        LIMIT 100""",
        (update.effective_user.id,),
    )

    lines = []

    for task_number, task in enumerate(
        rows,
        start=1,
    ):
        icon = (
            "✅"
            if task["status"] == "done"
            else "⬜"
        )

        lines.append(
            f"{icon} {task_number}. "
            f"{task['title']}\n"
            f"Priority {task['priority']}/5 · "
            f"{task['project'] or 'No project'} · "
            f"Due: {task['due_date'] or 'Not set'} · "
            f"{task['recurrence']}"
        )

    await send_long_message(
        update,
        "\n\n".join(lines)
        or "You don't have any tasks yet.",
    )


async def add_task_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    telegram_user_id = (
        update.effective_user.id
    )

    if not context.args:
        await update.message.reply_text(
            "Usage:\n"
            "/addtask Finish calculus homework"
        )
        return

    title = " ".join(
        context.args
    ).strip()

    task_id = await create_task(
        telegram_user_id,
        title,
    )

    await update.message.reply_text(
        f"✅ Task added.\n\n"
        f"{title}\n"
        "Use /tasks to see its task number."
    )


async def done_task_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    telegram_user_id = (
        update.effective_user.id
    )

    if not context.args:
        await update.message.reply_text(
            "Usage:\n"
            "/donetask 3"
        )
        return

    try:
        task_id = int(
            context.args[0]
        )

    except ValueError:
        await update.message.reply_text(
            "Task number must be a number."
        )
        return

    database_task_id = await resolve_task_number(
        telegram_user_id,
        task_id,
    )

    if database_task_id is None:
        await update.message.reply_text(
            "I couldn't find that task number."
        )
        return

    title = await complete_task(
        telegram_user_id,
        database_task_id,
    )

    if not title:
        await update.message.reply_text(
            "I couldn't find that open task."
        )
        return

    await update.message.reply_text(
        f"✅ Completed:\n{title}"
    )


async def delete_task_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    telegram_user_id = (
        update.effective_user.id
    )

    if not context.args:
        await update.message.reply_text(
            "Usage:\n"
            "/deletetask 3"
        )
        return

    try:
        task_id = int(
            context.args[0]
        )

    except ValueError:
        await update.message.reply_text(
            "Task number must be a number."
        )
        return

    database_task_id = await resolve_task_number(
        telegram_user_id,
        task_id,
    )

    if database_task_id is None:
        await update.message.reply_text(
            "I couldn't find that task number."
        )
        return

    title = await delete_task(
        telegram_user_id,
        database_task_id,
    )

    if not title:
        await update.message.reply_text(
            "I couldn't find that task."
        )
        return

    await update.message.reply_text(
        f"🗑️ Deleted:\n{title}"
    )


# ==================================================
# LONG-TERM MEMORY COMMANDS
# ==================================================

async def remember_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    telegram_user_id = (
        update.effective_user.id
    )

    if not context.args:
        await update.message.reply_text(
            "Usage:\n"
            "/remember I prefer concise answers"
        )
        return

    content = " ".join(
        context.args
    ).strip()

    memory_id = (
        await remember_user_memory(
            telegram_user_id,
            content,
            memory_type="fact",
            importance=3,
            source="explicit_user",
        )
    )

    await update.message.reply_text(
        "🧠 Remembered.\n\n"
        f"{memory_id}. {content}"
    )


async def memory_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    telegram_user_id = (
        update.effective_user.id
    )

    memories = (
        await list_long_term_memories(
            telegram_user_id,
            limit=50,
        )
    )

    if not memories:
        await update.message.reply_text(
            "I don't have any long-term "
            "memories saved for you yet."
        )
        return

    lines = [
        "🧠 Long-term memory",
        "",
    ]

    for display_number, memory in enumerate(
        memories,
        start=1,
    ):
        (
            memory_id,
            content,
            memory_type,
            importance,
            source,
            created_at,
            updated_at,
        ) = memory

        lines.append(
            (
                f"{display_number}. {content}\n"
                f"   Type: {memory_type} | "
                f"Importance: {importance}/5"
            )
        )

    await update.message.reply_text(
        "\n\n".join(lines)
    )


async def forget_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    telegram_user_id = (
        update.effective_user.id
    )

    if not context.args:
        await update.message.reply_text(
            "Usage:\n"
            "/forget 3"
        )
        return

    try:
        display_number = int(
            context.args[0]
        )

    except ValueError:
        await update.message.reply_text(
            "Memory number must be a number."
        )
        return

    memories = (
        await list_long_term_memories(
            telegram_user_id,
            limit=50,
        )
    )

    if (
        display_number < 1
        or display_number > len(memories)
    ):
        await update.message.reply_text(
            "I couldn't find that memory."
        )
        return

    real_memory_id = (
        memories[
            display_number - 1
        ][0]
    )

    forgotten = (
        await forget_long_term_memory(
            telegram_user_id,
            real_memory_id,
        )
    )

    if not forgotten:
        await update.message.reply_text(
            "I couldn't find that memory."
        )
        return

    await update.message.reply_text(
        "🧹 Forgotten:\n"
        f"{forgotten}"
    )
