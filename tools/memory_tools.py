import json
import re

from services.memory import (
    remember_user_memory,
)


# ==================================================
# MEMORY TOOL DEFINITION
# ==================================================

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remember_memory",
            "description": (
                "Save a durable, non-sensitive fact about the "
                "user that will probably remain useful in future "
                "conversations. Good examples include preferences, "
                "long-term goals, ongoing projects, recurring "
                "workflows, and stable non-sensitive facts. "
                "Do not save temporary details, guesses, secrets, "
                "credentials, contact information, health data, "
                "financial information, precise location, religion, "
                "political beliefs, sexuality, or legal/criminal "
                "information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": (
                            "A concise standalone memory."
                        ),
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": [
                            "preference",
                            "goal",
                            "project",
                            "workflow",
                            "stable_fact",
                        ],
                    },
                    "importance": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    },
                },
                "required": [
                    "content",
                    "memory_type",
                    "importance",
                ],
                "additionalProperties": False,
            },
        },
    }
]


MEMORY_TOOL_NAMES = {
    "remember_memory",
}


# ==================================================
# SERVER-SIDE SAFETY
# ==================================================

SAFE_MEMORY_TYPES = {
    "preference",
    "goal",
    "project",
    "workflow",
    "stable_fact",
}


SENSITIVE_PATTERNS = (
    r"\bpassword\b",
    r"\bpasscode\b",
    r"\bapi[\s_-]*key\b",
    r"\baccess[\s_-]*token\b",
    r"\bauth[\s_-]*token\b",
    r"\bbearer\s+[a-z0-9._-]+",
    r"\bprivate[\s_-]*key\b",
    r"\bsecret[\s_-]*key\b",
    r"\bcredit[\s_-]*card\b",
    r"\bdebit[\s_-]*card\b",
    r"\bmedical\s+(condition|diagnosis|history)\b",
    r"\bdiagnosed\s+with\b",
    r"\bpolitical\s+(belief|party|ideology)\b",
    r"\breligious\s+(belief|affiliation)\b",
)


EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)


LONG_NUMBER_PATTERN = re.compile(
    r"\b(?:\d[ -]*?){13,19}\b"
)


def memory_is_safe(
    content: str,
):
    content = content.strip()

    if not content:
        return False

    if len(content) > 500:
        return False

    for pattern in SENSITIVE_PATTERNS:
        if re.search(
            pattern,
            content,
            flags=re.IGNORECASE,
        ):
            return False

    if EMAIL_PATTERN.search(
        content
    ):
        return False

    if LONG_NUMBER_PATTERN.search(
        content
    ):
        return False

    return True


# ==================================================
# MEMORY TOOL EXECUTION
# ==================================================

async def execute_memory_tool(
    telegram_user_id: int,
    tool_name: str,
    arguments: dict,
):
    if tool_name != "remember_memory":
        return json.dumps(
            {
                "success": False,
                "error": (
                    f"Unknown memory tool: "
                    f"{tool_name}"
                ),
            }
        )

    content = str(
        arguments.get(
            "content",
            "",
        )
    ).strip()

    memory_type = str(
        arguments.get(
            "memory_type",
            "stable_fact",
        )
    ).strip()

    try:
        importance = int(
            arguments.get(
                "importance",
                3,
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        importance = 3

    importance = max(
        1,
        min(
            importance,
            5,
        ),
    )

    if (
        memory_type
        not in SAFE_MEMORY_TYPES
    ):
        return json.dumps(
            {
                "success": False,
                "error": (
                    "Unsupported memory type."
                ),
            }
        )

    if not memory_is_safe(
        content
    ):
        return json.dumps(
            {
                "success": False,
                "error": (
                    "Memory was rejected by "
                    "the safety filter."
                ),
            }
        )

    memory_id = (
        await remember_user_memory(
            telegram_user_id,
            content,
            memory_type=memory_type,
            importance=importance,
            source="automatic",
        )
    )

    return json.dumps(
        {
            "success": True,
            "memory_id": memory_id,
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
        },
        ensure_ascii=False,
    )
