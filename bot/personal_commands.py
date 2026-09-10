import asyncio
import logging
from collections import OrderedDict, deque
from time import monotonic

from telegram import BotCommand, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationHandlerStop

from database.personal import fetch, get_timezone, set_timezone
from tools.personal_tools import execute_personal_tool
from services.vision import analyze_image_with_vision
from utils.telegram_text import send_long_message

logger = logging.getLogger(__name__)
_recent = OrderedDict()


async def guard(update, context):
    """Personal context is private-chat only. Bound both request rate and state."""
    if not update.effective_user or not update.effective_message:
        raise ApplicationHandlerStop
    if update.effective_chat.type != 'private':
        raise ApplicationHandlerStop
    from config import ALLOWED_USER_IDS
    uid = update.effective_user.id
    if ALLOWED_USER_IDS and uid not in ALLOWED_USER_IDS:
        raise ApplicationHandlerStop
    now = monotonic()
    window = _recent.pop(uid, deque())
    while window and now - window[0] > 60:
        window.popleft()
    _recent[uid] = window
    if len(_recent) > 10000:
        _recent.popitem(last=False)
    if len(window) >= 20:
        raise ApplicationHandlerStop
    window.append(now)


async def help_command(update, context):
    keyboard = InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=f'panel:{key}') for key, label in row] for row in [[('tasks', 'Tasks'), ('reminders', 'Reminders')], [('memory', 'Memory'), ('notes', 'Notes'), ('images', 'Images')]]])
    await update.message.reply_text(
        'Your personal assistant\n\n'
        '/tasks — tasks\n/reminders — upcoming reminders\n'
        '/timezone Asia/Yerevan — set your timezone\n'
        '/memory — personal memories\n/notes [search] — saved notes\n'
        '/images — recent images\n/image ID question — ask about an older image\n'
        '/clearimages confirm — delete image history and latest image\n\n'
        'You can also ask naturally: “Remind me tomorrow at 5 PM”, '
        '“Make task 2 high priority”, “Save a project note”, or “Search the web for …”. '
        'Web search needs no API key and may be temporarily rate-limited. Recurring reminders support daily and weekly schedules.',
        reply_markup=keyboard,
    )


async def timezone_command(update, context):
    uid = update.effective_user.id
    if context.args:
        try:
            await set_timezone(uid, context.args[0])
        except (KeyError, ValueError):
            await update.message.reply_text('Use an IANA timezone such as Asia/Yerevan or Europe/Paris.')
            return
    zone = await get_timezone(uid)
    await update.message.reply_text(f'Timezone: {zone}' if zone else 'Set your timezone with /timezone Asia/Yerevan (or your own IANA timezone).')


async def reminders_command(update, context):
    rows = await fetch('''SELECT id, content, due_at, timezone, recurrence, status FROM reminders
        WHERE telegram_user_id = %s AND status IN ('pending', 'failed') ORDER BY due_at LIMIT 50''', (update.effective_user.id,))
    from zoneinfo import ZoneInfo
    lines = [f"{r['id']}. {r['content']}\n{r['due_at'].astimezone(ZoneInfo(r['timezone'])).isoformat()} · {r['recurrence']} · {r['status']}" for r in rows]
    await send_long_message(update, '\n\n'.join(lines) or 'No upcoming reminders.')


async def notes_command(update, context):
    import json
    result = json.loads(await execute_personal_tool(update.effective_user.id, 'search_notes', {'query': ' '.join(context.args)}))
    await send_long_message(update, '\n\n'.join(f"{r['id']}. {r['title']} [{r['project']}]\n{r['content']}" for r in result['results']) or 'No saved notes.')


async def images_command(update, context):
    rows = await fetch('SELECT id, filename, created_at FROM image_history WHERE telegram_user_id = %s ORDER BY id DESC LIMIT 10', (update.effective_user.id,))
    await send_long_message(update, '\n'.join(f"{r['id']}. {r['filename']} ({r['created_at']:%Y-%m-%d %H:%M UTC})" for r in rows) or 'No image history yet. Upload a photo to begin.')


async def image_command(update, context):
    if len(context.args) < 2 or not context.args[0].isdigit():
        await update.message.reply_text('Usage: /image ID your question\nUse /images to find the ID.')
        return
    rows = await fetch('SELECT image_data, ocr_text FROM image_history WHERE telegram_user_id = %s AND id = %s', (update.effective_user.id, int(context.args[0])))
    if not rows:
        await update.message.reply_text('Image not found in your history.')
        return
    answer = await analyze_image_with_vision(bytes(rows[0]['image_data']), ' '.join(context.args[1:]), rows[0]['ocr_text'] or '')
    await send_long_message(update, answer)


async def clear_images_command(update, context):
    if context.args != ['confirm']:
        await update.message.reply_text('This deletes all stored images. Use /clearimages confirm to continue.')
        return
    import psycopg
    from config import DATABASE_URL
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        for table in ('image_history', 'latest_images'):
            from psycopg import sql
            await conn.execute(sql.SQL('DELETE FROM {} WHERE telegram_user_id = %s').format(sql.Identifier(table)), (update.effective_user.id,))
    await update.message.reply_text('Image history cleared.')


async def on_error(update, context):
    # Avoid dumping Update objects, credentials or private message bodies.
    logger.error('Telegram handler failed error=%s', type(context.error).__name__)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text('That operation failed. Please try again shortly.')
        except Exception:
            pass


async def panel_callback(update, context):
    from types import SimpleNamespace
    from bot.commands import tasks_command, memory_command
    await update.callback_query.answer()
    handlers = {'tasks': tasks_command, 'memory': memory_command, 'reminders': reminders_command,
                'notes': notes_command, 'images': images_command}
    context.args = []
    proxy = SimpleNamespace(effective_user=update.effective_user, message=update.effective_message)
    await handlers[update.callback_query.data.split(':', 1)[1]](proxy, context)


async def setup_personal(application):
    from telegram.ext import CommandHandler, TypeHandler, CallbackQueryHandler
    application.add_handler(TypeHandler(Update, guard), group=-1)
    commands = {'help': help_command, 'timezone': timezone_command, 'reminders': reminders_command,
                'notes': notes_command, 'images': images_command, 'image': image_command,
                'clearimages': clear_images_command}
    for name, handler in commands.items():
        application.add_handler(CommandHandler(name, handler))
    application.add_handler(CallbackQueryHandler(panel_callback, pattern=r"^panel:(tasks|reminders|memory|notes|images)$"))
    application.add_error_handler(on_error)
    try:
        await application.bot.set_my_commands([BotCommand(name, description) for name, description in [
            ('help', 'Commands and examples'), ('tasks', 'Your tasks'), ('reminders', 'Upcoming reminders'),
            ('timezone', 'View or set timezone'), ('memory', 'Personal memories'), ('notes', 'Saved knowledge'),
            ('images', 'Recent images'), ('image', 'Ask about an older image'), ('clearimages', 'Delete stored images')]])
    except Exception as error:
        logger.warning('Could not register command menu: %s', type(error).__name__)
