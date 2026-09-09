from telegram import Update
from telegram.ext import ContextTypes

from database.memory import (
    delete_memory,
)

from database.documents import (
    load_documents,
    delete_documents,
)

from database.images import (
    delete_latest_image,
)

from database.tasks import (
    get_tasks,
    create_task,
    complete_task,
    delete_task,
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
            f"• {filename}"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


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
    telegram_user_id = (
        update.effective_user.id
    )

    tasks = await get_tasks(
        telegram_user_id
    )

    if not tasks:
        await update.message.reply_text(
            "You don't have any tasks yet."
        )
        return

    lines = [
        "✅ Your Tasks",
        "",
    ]

    for (
        task_id,
        title,
        status,
        due_date,
    ) in tasks:
        icon = (
            "✅"
            if status == "done"
            else "⬜"
        )

        line = (
            f"{icon} "
            f"{task_id}. "
            f"{title}"
        )

        if due_date:
            line += (
                f" — due {due_date}"
            )

        lines.append(
            line
        )

    await update.message.reply_text(
        "\n".join(lines)
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
        f"{task_id}. {title}"
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
            "Task ID must be a number."
        )
        return

    title = await complete_task(
        telegram_user_id,
        task_id,
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
            "Task ID must be a number."
        )
        return

    title = await delete_task(
        telegram_user_id,
        task_id,
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

    for (
        memory_id,
        content,
        memory_type,
        importance,
        source,
        created_at,
        updated_at,
    ) in memories:

        lines.append(
            (
                f"{memory_id}. {content}\n"
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
        memory_id = int(
            context.args[0]
        )

    except ValueError:
        await update.message.reply_text(
            "Memory ID must be a number."
        )
        return

    forgotten = (
        await forget_long_term_memory(
            telegram_user_id,
            memory_id,
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
