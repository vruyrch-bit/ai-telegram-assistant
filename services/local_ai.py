"""Ollama adapter using local HTTP only; no cloud fallback or API key."""
import json
import uuid
import httpx


class Object(dict):
    def __getattr__(self, key):
        return self.get(key)


def obj(value):
    if isinstance(value, dict):
        return Object({key: obj(item) for key, item in value.items()})
    if isinstance(value, list):
        return [obj(item) for item in value]
    return value


class LocalClient:
    def __init__(self):
        self.chat = self
        self.completions = self

    async def create(self, *, model, messages, tools=None, max_completion_tokens=1500, **kwargs):
        from config import OLLAMA_URL
        converted = []
        for original in messages:
            message = dict(original) if isinstance(original, dict) else original.model_dump(exclude_none=True)
            content = message.get('content') or ''
            target = {'role': message['role'], 'content': content}
            if isinstance(content, list):
                target['content'] = '\n'.join(part['text'] for part in content if part.get('type') == 'text')
                target['images'] = [part['image_url']['url'].split(',', 1)[1]
                                    for part in content if part.get('type') == 'image_url']
            if message.get('tool_calls'):
                target['tool_calls'] = []
                for call in message['tool_calls']:
                    function = dict(call['function'])
                    if isinstance(function.get('arguments'), str):
                        function['arguments'] = json.loads(function['arguments'])
                    target['tool_calls'].append({'function': function})
            if message.get('name'):
                target['tool_name'] = message['name']
            converted.append(target)
        payload = {'model': model, 'messages': converted, 'stream': False, 'think': False,
                   'keep_alive': '5m', 'options': {'num_ctx': 16384, 'num_predict': max_completion_tokens}}
        if tools:
            payload['tools'] = tools
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(f'{OLLAMA_URL}/api/chat', json=payload)
            response.raise_for_status()
        message = response.json()['message']
        calls = []
        for call in message.get('tool_calls', []):
            calls.append({'id': uuid.uuid4().hex, 'type': 'function', 'function': {
                'name': call['function']['name'], 'arguments': json.dumps(call['function']['arguments'])}})
        return obj({'choices': [{'message': {'role': 'assistant', 'content': message.get('content', ''), 'tool_calls': calls}}]})
