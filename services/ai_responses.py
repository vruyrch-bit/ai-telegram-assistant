"""Keep provider reasoning out of replies and preserve completed tool results."""
import re
from urllib.parse import urlsplit


def final_answer(content):
    text = content or ''
    text = re.sub(r'<think\b[^>]*>.*?</think\s*>', '', text, flags=re.I | re.S)
    # A truncated reasoning block must never become the user-facing answer.
    text = re.split(r'<think\b[^>]*>', text, maxsplit=1, flags=re.I)[0]
    if re.search(r'</think\s*>', text, re.I):
        text = re.split(r'</think\s*>', text, flags=re.I)[-1]
    return text.strip()


_ACTION_NOTICES = {
    'create_task': 'Task saved. Check /tasks.',
    'edit_task': 'Task updated. Check /tasks.',
    'complete_task': 'Task completed. Check /tasks.',
    'delete_task': 'Task deleted. Check /tasks.',
    'create_reminder': 'Reminder saved. Check /reminders.',
    'edit_reminder': 'Reminder updated. Check /reminders.',
    'cancel_reminder': 'Reminder cancelled. Check /reminders.',
    'remember_memory': 'Memory saved. Check /memory.',
    'replace_memory': 'Memory updated. Check /memory.',
    'save_note': 'Note saved. Check /notes.',
    'delete_note': 'Note deleted. Check /notes.',
    'set_timezone': 'Timezone updated. Check /timezone.',
    'set_document_collection': 'Document collection updated.',
}


def interrupted_answer(message, tool_results):
    lines = [message]
    for name, result in tool_results:
        if not result.get('success'):
            continue
        if name in _ACTION_NOTICES:
            lines.append('• ' + _ACTION_NOTICES[name])
        elif name == 'web_search':
            sources = []
            for row in result.get('results', [])[:5]:
                url = row.get('url', '')
                try:
                    parsed = urlsplit(url)
                except (ValueError, TypeError):
                    continue
                if parsed.scheme in {'https', 'http'} and parsed.hostname:
                    sources.append(f"• {row.get('title', 'Source')}\n{url}")
            if sources:
                lines.append("Web search finished, but I couldn't compose the answer. Sources:")
                lines.extend(sources)
    if len(lines) > 1:
        lines.append('The results above are already complete; no need to repeat those actions.')
    return '\n\n'.join(lines)
