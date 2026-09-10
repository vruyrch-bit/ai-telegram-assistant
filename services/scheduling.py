"""Timezone-aware calendar recurrence without an in-memory schedule."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


def parse_due(value, zone, now=None):
    tz = ZoneInfo(zone)
    due = datetime.fromisoformat(value)
    if due.tzinfo is None:
        raise ValueError('Use an ISO datetime with an explicit UTC offset.')
    # Require the offset to match the user's actual zone on this date.
    if due.utcoffset() != due.astimezone(tz).utcoffset():
        raise ValueError('The date offset does not match your saved timezone.')
    now = now or datetime.now(timezone.utc)
    if due <= now:
        raise ValueError('The reminder time must be in the future.')
    return due.astimezone(timezone.utc)


def next_occurrence(due, zone, recurrence, now=None):
    if recurrence == 'none':
        return None
    if recurrence not in {'daily', 'weekly'}:
        raise ValueError('Unsupported recurrence.')
    now = now or datetime.now(timezone.utc)
    tz = ZoneInfo(zone)
    local = due.astimezone(tz)
    step = 1 if recurrence == 'daily' else 7
    elapsed = max(0, (now.astimezone(tz).date() - local.date()).days)
    candidate = local + timedelta(days=max(step, (elapsed // step) * step))
    # Round-trip advances nonexistent spring-forward times to the next valid time.
    candidate = candidate.astimezone(timezone.utc).astimezone(tz)
    while candidate.astimezone(timezone.utc) <= now:
        candidate += timedelta(days=step)
        candidate = candidate.astimezone(timezone.utc).astimezone(tz)
    return candidate.astimezone(timezone.utc)
