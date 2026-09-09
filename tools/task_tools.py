import json
import logging

from database.tasks import (
    create_task,
    get_tasks,
    complete_task,
    delete_task,
)


logger = logging.getLogger(__name__)


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
        title = str(
            arguments.get(
                "title",
                "",
            )
        ).strip()

        due_date = arguments.get(
            "due_date"
        )

        if not title:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "Task title cannot be empty."
                    ),
                }
            )

        if due_date is not None:
            due_date = str(
                due_date
            ).strip()

            if not due_date:
                due_date = None

        task_id = await create_task(
            telegram_user_id,
            title,
            due_date,
        )

        return json.dumps(
            {
                "success": True,
                "task_id": task_id,
                "title": title,
                "due_date": due_date,
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
