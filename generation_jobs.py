"""Bounded in-process job event replay; one server process owns the GPU."""

import threading
import time

JOB_TTL_SECONDS = 900
MAX_RETAINED_JOBS = 16
MAX_ACTIVE_JOBS = 4


class GenerationJob:
    def __init__(self):
        self.condition = threading.Condition()
        self.events = []
        self.finished_at = None

    def put(self, event):
        if event is None:
            return
        with self.condition:
            self.events.append(event)
            if event["type"] in ("done", "error"):
                self.finished_at = time.monotonic()
            self.condition.notify_all()

    def stream(self, cursor=0):
        while True:
            with self.condition:
                if cursor >= len(self.events) and self.finished_at is None:
                    self.condition.wait(timeout=10)
                pending = list(enumerate(self.events[cursor:], start=cursor + 1))
                finished = self.finished_at is not None
            if not pending and not finished:
                yield None, None  # SSE comment heartbeat, even during model preparation.
            for event_id, event in pending:
                cursor = event_id
                yield event_id, event
            if finished:
                return


def prune_jobs(jobs):
    """Call while holding the registry lock. Never evict active jobs."""
    now = time.monotonic()
    completed = sorted(((key, job.finished_at) for key, job in jobs.items()
                        if job.finished_at is not None), key=lambda item: item[1])
    for key, finished_at in completed:
        if now - finished_at > JOB_TTL_SECONDS or len(jobs) >= MAX_RETAINED_JOBS:
            jobs.pop(key, None)
