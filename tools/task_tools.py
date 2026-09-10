import json
import logging
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from database.tasks import (
    create_task,
    get_tasks,
    complete_task,
    delete_task,
)


logger = logging.getLogger(__name__)


class CreateTaskArguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=500)
    due_date: str | None = Field(default=None, max_length=100, description=(
        'Resolve relative deadlines from the current date and saved user timezone. '
        'Use ISO 8601 with UTC offset when a time is given, YYYY-MM-DD for date-only. '
        'Recurring tasks currently require date-only deadlines. Do not invent a deadline.'))
    priority: int = Field(default=3, ge=1, le=5, description='1 low, 3 medium, 5 high.')
    project: str = Field(default='', max_length=100)
    notes: str = Field(default='', max_length=5000)
    recurrence: Literal['none', 'daily', 'weekly'] = 'none'

    @model_validator(mode='after')
    def validate_recurrence(self):
        if self.recurrence != 'none':
            try:
                date.fromisoformat(self.due_date or '')
            except ValueError:
                raise ValueError('Recurring tasks require a YYYY-MM-DD deadline.')
        return self


# ==================================================
# TASK TOOL DEFINITIONS
# ==================================================

TASK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": (
                "Create a new task in the user's persistent "
                "task list. Use only when the user clearly "
                "asks to add, create, save, or remember a task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The task title.",
                    },
                    "due_date": {
                        "type": "string",
                        "description": (
                            "Optional due-date wording provided "
                            "by the user. Do not invent a date."
                        ),
                    },
                },
                "required": ["title"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": (
                "Retrieve the user's task list. Use when the "
                "user asks about their tasks or when a task ID "
                "is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": (
                "Mark one task as completed. Use a task ID. "
                "If only the task name is known, call "
                "list_tasks first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": (
                            "Database ID of the task to complete."
                        ),
                    },
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_task",
            "description": (
                "Delete one task. Use a task ID. If only the "
                "task name is known, call list_tasks first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": (
                            "Database ID of the task to delete."
                        ),
                    },
                },
                "required": ["task_id"],
                "additionalProperties": False,
            },
        },
    },
]

# Keep the advertised create schema and runtime validation in sync.
TASK_TOOLS[0]['function']['parameters'] = CreateTaskArguments.model_json_schema()


# ==================================================
# TASK TOOL EXECUTION
# ==================================================

async def execute_task_tool(
    telegram_user_id: int,
    tool_name: str,
    arguments: dict,
):
    logger.info(
        "Executing AI tool user_id=%s tool=%s",
        telegram_user_id,
        tool_name,
    )

    if tool_name == "create_task":
        try:
            parsed = CreateTaskArguments.model_validate(arguments)
        except ValidationError:
            return json.dumps({'success': False, 'error': (
                'Invalid task fields. Use a nonblank title, priority 1–5, and '
                'a YYYY-MM-DD deadline for recurring tasks.')})
        title = parsed.title
        due_date = parsed.due_date or None
        extra = parsed.model_dump(exclude={'title', 'due_date'}, exclude_unset=True)

        task_id = await create_task(
            telegram_user_id,
            title,
            due_date,
            **extra,
        )

        return json.dumps(
            {
                "success": True,
                "task_id": task_id,
                "title": title,
                "due_date": due_date,
                "priority": parsed.priority,
                "project": parsed.project,
                "notes": parsed.notes,
                "recurrence": parsed.recurrence,
            },
            ensure_ascii=False,
        )

    if tool_name == "list_tasks":
        rows = await get_tasks(
            telegram_user_id
        )

        tasks = []

        for (
            task_id,
            title,
            status,
            due_date,
        ) in rows:
            tasks.append(
                {
                    "task_id": task_id,
                    "title": title,
                    "status": status,
                    "due_date": due_date,
                }
            )

        return json.dumps(
            {
                "success": True,
                "tasks": tasks,
            },
            ensure_ascii=False,
        )

    if tool_name == "complete_task":
        try:
            task_id = int(
                arguments.get(
                    "task_id"
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "A valid numeric task ID "
                        "is required."
                    ),
                }
            )

        title = await complete_task(
            telegram_user_id,
            task_id,
        )

        if not title:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "That open task was not found."
                    ),
                }
            )

        return json.dumps(
            {
                "success": True,
                "task_id": task_id,
                "title": title,
                "status": "done",
            },
            ensure_ascii=False,
        )

    if tool_name == "delete_task":
        try:
            task_id = int(
                arguments.get(
                    "task_id"
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "A valid numeric task ID "
                        "is required."
                    ),
                }
            )

        title = await delete_task(
            telegram_user_id,
            task_id,
        )

        if not title:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "That task was not found."
                    ),
                }
            )

        return json.dumps(
            {
                "success": True,
                "task_id": task_id,
                "title": title,
                "deleted": True,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "success": False,
            "error": (
                f"Unknown tool: {tool_name}"
            ),
        }
    )
