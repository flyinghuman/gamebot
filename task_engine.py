"""Task engine providing generic workflow execution for the bot."""

from __future__ import annotations

import copy
import random
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


class TaskError(RuntimeError):
    """Raised when a task cannot be executed due to malformed configuration."""


@dataclass
class TaskResult:
    """Represents the outcome of a task execution."""

    executed: bool
    success: Optional[bool]
    detail: str
    context: "TaskContext" = field(repr=False)


@dataclass
class TaskDefinition:
    """In-memory representation of a task definition."""

    id: str
    name: str
    description: str
    enabled: bool
    trigger: Dict[str, Any]
    steps: List[Dict[str, Any]]
    raw: Dict[str, Any] = field(repr=False)
    protected_system_task: bool = False
    stats: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskDefinition":
        task_id = data.get("id")
        if not task_id:
            raise TaskError("Task definition is missing required field 'id'.")
        name = data.get("name", task_id)
        description = data.get("description", "")
        enabled = bool(data.get("enabled", True))
        trigger = copy.deepcopy(data.get("trigger", {}))
        steps = copy.deepcopy(data.get("steps", []))
        stats_meta = copy.deepcopy(data.get("stats", {}))
        protected_flag = bool(
            data.get("protected-system-task")
            or data.get("protected_system_task")
        )
        if not isinstance(steps, list):
            raise TaskError(
                f"Task '{task_id}' has invalid 'steps' section (must be list)."
            )
        return cls(
            id=task_id,
            name=name,
            description=description,
            enabled=enabled,
            trigger=trigger,
            steps=steps,
            raw=copy.deepcopy(data),
            protected_system_task=protected_flag,
            stats=stats_meta if isinstance(stats_meta, dict) else {},
        )


class TaskContext:
    """Holds execution state and shared data for a running task."""

    def __init__(self, task: TaskDefinition, env: Dict[str, Any]):
        self.task = task
        self.env = env
        self.data: Dict[str, Any] = {}
        self.executed: bool = False
        self.success: Optional[bool] = None
        self.detail: str = ""
        self._stop_requested: bool = False

    def request_stop(self):
        self._stop_requested = True

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def set_detail(self, text: str, append: bool = False):
        if append and self.detail:
            self.detail = f"{self.detail} {text}".strip()
        else:
            self.detail = text

    def format_text(self, template: str) -> str:
        class SafeMap(dict):
            def __missing__(self, key):
                return ""  # return empty string for unknown keys

        safe = SafeMap(**{k: v for k, v in self.data.items() if isinstance(k, str)})
        return template.format_map(safe)


class TaskActionExecutor:
    """Executes workflow actions defined in task steps."""

    def __init__(self, env: Dict[str, Any]):
        self.env = env

    def execute_steps(self, steps: Iterable[Dict[str, Any]], context: TaskContext):
        for step in steps:
            if context.stop_requested:
                break
            self.execute_step(step, context)

    def execute_step(self, step: Dict[str, Any], context: TaskContext):
        step_type = step.get("type")
        if not step_type:
            raise TaskError(f"Task '{context.task.id}' contains a step without 'type'.")
        handler_name = f"_handle_{step_type}"
        handler = getattr(self, handler_name, None)
        if not handler:
            raise TaskError(
                f"Unsupported action type '{step_type}' in task '{context.task.id}'."
            )
        handler(step, context)

    # --- Helpers ---
    def _resolve_threshold(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str):
            key = value.lower()
            if key == "default":
                return float(self.env.get("threshold_default", 0.85))
            if key == "loose":
                return float(self.env.get("threshold_loose", 0.8))
        try:
            return float(value)
        except (TypeError, ValueError):
            raise TaskError(f"Invalid threshold value: {value!r}")

    def _sleep(self, duration: Dict[str, Any]):
        if not duration:
            return
        min_v = duration.get("min")
        max_v = duration.get("max")
        seconds = duration.get("seconds")
        if seconds is not None:
            wait = float(seconds)
        else:
            min_v = float(min_v if min_v is not None else 0.0)
            max_v = float(max_v if max_v is not None else min_v)
            wait = random.uniform(min_v, max_v)
        time.sleep(wait)

    def _handle_sleep(self, step: Dict[str, Any], context: TaskContext):
        self._sleep(step)

    def _handle_set_flag(self, step: Dict[str, Any], context: TaskContext):
        name = step.get("name")
        if not name:
            raise TaskError("set_flag action requires a 'name'.")
        value = step.get("value", True)
        context.data[name] = value

    def _handle_set_detail(self, step: Dict[str, Any], context: TaskContext):
        text = step.get("text", "")
        append = bool(step.get("append", False))
        formatted = context.format_text(text)
        context.set_detail(formatted, append=append)

    def _handle_set_success(self, step: Dict[str, Any], context: TaskContext):
        if "value" in step:
            context.success = step.get("value")
            return
        flag = step.get("from_flag")
        default = step.get("default")
        if flag is None:
            raise TaskError(
                "set_success action requires either 'value' or 'from_flag'."
            )
        context.success = context.data.get(flag, default)

    def _handle_set_executed(self, step: Dict[str, Any], context: TaskContext):
        context.executed = bool(step.get("value", True))

    def _handle_log_message(self, step: Dict[str, Any], context: TaskContext):
        logger = self.env.get("logger")
        if logger:
            level = step.get("level", "info").lower()
            msg = context.format_text(step.get("message", ""))
            log_func = getattr(logger, level, logger.info)
            try:
                log_func(msg)
            except Exception:
                logger.info(msg)

    def _handle_press_back(self, step: Dict[str, Any], context: TaskContext):
        back_fn: Callable = self.env.get("key_back")
        if not back_fn:
            raise TaskError("press_back action requires key_back in the environment.")
        back_fn()
        sleep_after = step.get("sleep_after")
        if sleep_after not in (None, ""):
            self._sleep({"seconds": sleep_after})

    def _handle_swipe(self, step: Dict[str, Any], context: TaskContext):
        swipe_fn: Callable = self.env.get("swipe")
        if not swipe_fn:
            raise TaskError(
                "Swipe function is not available in the execution environment."
            )
        start = step.get("start")
        end = step.get("end")
        if not (
            isinstance(start, (list, tuple))
            and isinstance(end, (list, tuple))
            and len(start) == 2
            and len(end) == 2
        ):
            raise TaskError(
                "swipe action requires 'start' and 'end' coordinates with two values each."
            )
        duration_ms = int(step.get("duration_ms", step.get("duration", 400)))
        swipe_fn(
            int(start[0]), int(start[1]), int(end[0]), int(end[1]), dur_ms=duration_ms
        )
        if step.get("sleep_after"):
            self._sleep(step.get("sleep_after"))

    def _evaluate_condition(
        self, condition: Optional[Dict[str, Any]], context: TaskContext
    ) -> bool:
        if not condition:
            return True
        cond_type = condition.get("type", "always_true")
        if cond_type == "always_true":
            return True
        if cond_type == "always_false":
            return False
        if cond_type == "template":
            template = condition.get("template")
            if not template:
                raise TaskError("template condition requires 'template'.")
            use_roi = condition.get("use_roi", True)
            threshold = condition.get("threshold")
            threshold_value = (
                self._resolve_threshold(threshold) if threshold is not None else None
            )
            exists_fn: Callable = self.env.get("template_exists")
            if not exists_fn:
                raise TaskError("template_exists function not provided in environment.")
            result = bool(exists_fn(template, threshold_value, use_roi))
            if condition.get("negate"):
                result = not result
            return result
        if cond_type == "flag":
            flag = condition.get("flag")
            if flag is None:
                raise TaskError("flag condition requires 'flag'.")
            value = condition.get("value", True)
            result = context.data.get(flag) == value
            if condition.get("negate"):
                result = not result
            return result
        if cond_type == "success":
            result = context.success is True
            if condition.get("negate"):
                result = not result
            return result
        raise TaskError(f"Unsupported condition type '{cond_type}'.")

    def _handle_close_dialogs(self, step: Dict[str, Any], context: TaskContext):
        close_fn: Callable = self.env.get("close_any")
        if not close_fn:
            raise TaskError(
                "close_dialogs action requires a close_any function in the environment."
            )
        attempts = int(step.get("max_attempts", 3))
        close_fn(max_attempts=attempts)

    def _handle_if(self, step: Dict[str, Any], context: TaskContext):
        condition = step.get("condition")
        if condition is None:
            raise TaskError("if action requires 'condition'.")
        branches = (
            step.get("then_steps")
            if self._evaluate_condition(condition, context)
            else step.get("else_steps")
        )
        if branches:
            self.execute_steps(branches, context)

    def _handle_if_flag(self, step: Dict[str, Any], context: TaskContext):
        flag_name = step.get("flag")
        if flag_name is None:
            raise TaskError("if_flag action requires 'flag'.")
        equals = step.get("equals", True)
        flag_value = context.data.get(flag_name)
        branch = step.get("steps") if flag_value == equals else step.get("else_steps")
        if branch:
            self.execute_steps(branch, context)

    def _handle_loop(self, step: Dict[str, Any], context: TaskContext):
        steps = step.get("steps")
        if not isinstance(steps, list):
            raise TaskError("loop action requires a list of 'steps'.")
        max_iterations = int(step.get("max_iterations", 1))
        break_on_fail = bool(step.get("break_on_fail", True))
        delay = step.get("sleep_between")
        condition = step.get("condition")
        for _ in range(max_iterations):
            if condition is not None and not self._evaluate_condition(
                condition, context
            ):
                break
            if context.stop_requested:
                break
            before_success = context.success
            before_executed = context.executed
            before_detail = context.detail
            self.execute_steps(steps, context)
            if context.stop_requested:
                break
            loop_failed = context.success is False
            if break_on_fail and loop_failed:
                break
            if delay:
                self._sleep(delay)
            context.success = before_success
            context.executed = before_executed
            context.detail = before_detail

    def _handle_tap_template(self, step: Dict[str, Any], context: TaskContext):
        self._handle_tap_like(step, context, wait_mode=False)

    def _handle_wait_tap_template(self, step: Dict[str, Any], context: TaskContext):
        self._handle_tap_like(step, context, wait_mode=True)

    def _handle_tap_like(
        self, step: Dict[str, Any], context: TaskContext, wait_mode: bool
    ):
        fn_key = "wait_and_tap" if wait_mode else "find_and_tap"
        tap_fn: Callable = self.env.get(fn_key)
        if not tap_fn:
            raise TaskError(f"{fn_key} function is not provided in the environment.")

        template = step.get("template")
        if not template:
            raise TaskError("tap_template action requires 'template'.")

        threshold_value = self._resolve_threshold(step.get("threshold"))
        use_roi = step.get("use_roi", True)
        duration_ms = step.get("duration_ms")
        duration_ms = None if duration_ms in (None, "") else float(duration_ms)
        sleep_after = step.get("sleep_after")
        required = bool(step.get("required", True))
        stop_on_fail = bool(step.get("stop_on_fail", required))
        fail_detail = step.get("fail_detail")
        success_detail = step.get("success_detail")
        save_result_as = step.get("save_result_as")
        set_flag_on_success = step.get("set_flag_on_success")
        set_flag_on_failure = step.get("set_flag_on_failure")
        repeat_until_fail = bool(step.get("repeat_until_fail", False))
        max_repeats = step.get("max_repeats")
        max_repeats = int(max_repeats) if max_repeats not in (None, "") else None
        sleep_between = step.get("sleep_between_repeats")
        timeout = float(step.get("timeout", step.get("wait_timeout", 3.0)))
        sleep_interval = float(step.get("sleep_interval", 0.5))
        scroll_spec = step.get("scroll_if_not_found")

        attempt_count = 0
        success_any = False
        last_success = False

        while True:
            attempt_count += 1
            if wait_mode:
                last_success = bool(
                    tap_fn(
                        template,
                        timeout=timeout,
                        thresh=threshold_value,
                        sleep_interval=sleep_interval,
                        duration=duration_ms,
                    )
                )
            else:
                result = tap_fn(
                    template,
                    thresh=threshold_value,
                    use_roi=use_roi,
                    duration=duration_ms,
                )
                last_success = (
                    bool(result[0]) if isinstance(result, tuple) else bool(result)
                )

            if save_result_as:
                context.data[save_result_as] = last_success
            if last_success:
                success_any = True
                context.executed = True
                if set_flag_on_success:
                    context.data[set_flag_on_success] = True
                if success_detail:
                    context.set_detail(context.format_text(success_detail))
            else:
                if set_flag_on_failure:
                    context.data[set_flag_on_failure] = True
                if scroll_spec and attempt_count == 1:
                    start = scroll_spec.get("start")
                    end = scroll_spec.get("end")
                    duration_scroll = scroll_spec.get(
                        "duration_ms", scroll_spec.get("duration", 400)
                    )
                    swipe_fn: Callable = self.env.get("swipe")
                    if swipe_fn and start and end:
                        swipe_fn(
                            int(start[0]),
                            int(start[1]),
                            int(end[0]),
                            int(end[1]),
                            dur_ms=int(duration_scroll),
                        )
                        if scroll_spec.get("sleep_after"):
                            self._sleep(scroll_spec.get("sleep_after"))
                        continue  # retry after scroll

            if not repeat_until_fail:
                break
            if not last_success:
                break
            if max_repeats is not None and attempt_count >= max_repeats:
                break
            if sleep_between:
                self._sleep(sleep_between)

        final_success = success_any if repeat_until_fail else last_success

        context.executed = True

        if not final_success:
            if fail_detail:
                context.set_detail(context.format_text(fail_detail))
            if required:
                context.success = False
            if stop_on_fail:
                context.request_stop()
        else:
            if context.success is None and step.get("mark_success", True):
                context.success = True

        if final_success and step.get("success_steps"):
            self.execute_steps(step.get("success_steps") or [], context)
        if not final_success and step.get("failure_steps"):
            self.execute_steps(step.get("failure_steps") or [], context)

        if sleep_after:
            self._sleep(sleep_after)

    def _handle_set_success_from_flag(self, step: Dict[str, Any], context: TaskContext):
        flag = step.get("flag")
        if not flag:
            raise TaskError("set_success_from_flag requires 'flag'.")
        true_value = step.get("success_on_true", True)
        false_value = step.get("success_on_false")
        context.success = true_value if context.data.get(flag) else false_value

    def _handle_set_detail_from_flag(self, step: Dict[str, Any], context: TaskContext):
        flag = step.get("flag")
        if not flag:
            raise TaskError("set_detail_from_flag requires 'flag'.")
        if context.data.get(flag):
            detail = step.get("detail_on_true", "")
        else:
            detail = step.get("detail_on_false", "")
        if detail:
            context.set_detail(context.format_text(detail))

    def _handle_stop_task(self, step: Dict[str, Any], context: TaskContext):
        if "success" in step:
            context.success = step.get("success")
        if "detail" in step:
            context.set_detail(context.format_text(step.get("detail")))
        context.request_stop()

    def _handle_call_task(self, step: Dict[str, Any], context: TaskContext):
        task_id = step.get("task_id")
        if not task_id:
            raise TaskError("call_task action requires 'task_id'.")
        runner: Callable = self.env.get("run_task_by_id")
        if not runner:
            raise TaskError("call_task requires 'run_task_by_id' in the environment.")
        propagate = bool(step.get("propagate_success", True))
        stop_on_fail = bool(step.get("stop_on_fail", True))
        detail_override = step.get("detail")

        result = runner(task_id)
        context.executed = True
        if result is None:
            if stop_on_fail:
                context.success = False
                context.request_stop()
            return

        if propagate:
            context.success = result.success
        if detail_override:
            context.set_detail(detail_override)
        elif result.detail:
            context.set_detail(result.detail)
        if result.success is False and stop_on_fail:
            context.request_stop()


class TaskScheduler:
    """Keeps track of next-run timestamps for enabled tasks."""

    def __init__(self, tasks: List[TaskDefinition]):
        self.tasks = tasks
        self.state: Dict[str, Dict[str, Any]] = {
            task.id: {"next_run": 0.0} for task in tasks if task.enabled
        }

    def schedule_all(self, now: float):
        for task in self.tasks:
            if not task.enabled:
                continue
            state = self.state.setdefault(task.id, {})
            trigger = task.trigger or {}
            trigger_type = str(trigger.get("type", "interval")).lower()
            if trigger_type == "sheduled":
                trigger_type = "scheduled"
            if trigger_type == "interval":
                min_seconds = float(trigger.get("min_seconds", 60.0))
                max_seconds = float(trigger.get("max_seconds", min_seconds))
                initial = trigger.get("run_at_start", False)
                if state.get("next_run", 0.0) <= 0.0:
                    state["next_run"] = (
                        now
                        if initial
                        else now + random.uniform(min_seconds, max_seconds)
                    )
            elif trigger_type == "scheduled":
                times = self._parse_scheduled_times(trigger)
                state["scheduled_times"] = times
                if not times:
                    state["next_run"] = now
                    continue
                initial = trigger.get("run_at_start", False)
                if initial and state.get("next_run", 0.0) <= 0.0:
                    state["next_run"] = now
                else:
                    state["next_run"] = self._next_scheduled_timestamp(now, times)
            else:
                state["next_run"] = now + float(trigger.get("cooldown", 1.0))

    def due_tasks(self, now: float) -> List[TaskDefinition]:
        ready: List[TaskDefinition] = []
        for task in self.tasks:
            if not task.enabled:
                continue
            next_run = self.state.get(task.id, {}).get("next_run", 0.0)
            if now >= next_run:
                ready.append(task)
        return ready

    def mark_executed(self, task: TaskDefinition, now: float):
        trigger = task.trigger or {}
        trigger_type = str(trigger.get("type", "interval")).lower()
        if trigger_type == "sheduled":
            trigger_type = "scheduled"
        state = self.state.setdefault(task.id, {})
        if trigger_type == "interval":
            min_seconds = float(trigger.get("min_seconds", 60.0))
            max_seconds = float(trigger.get("max_seconds", min_seconds))
            state["next_run"] = now + random.uniform(min_seconds, max_seconds)
        elif trigger_type == "scheduled":
            times = state.get("scheduled_times") or self._parse_scheduled_times(trigger)
            state["scheduled_times"] = times
            if not times:
                state["next_run"] = now + float(trigger.get("cooldown", 60.0))
            else:
                state["next_run"] = self._next_scheduled_timestamp(now, times)
        else:
            state["next_run"] = now + float(trigger.get("cooldown", 1.0))

    def _parse_scheduled_times(self, trigger: Dict[str, Any]) -> List[int]:
        raw = trigger.get("times") or trigger.get("time")
        if not raw:
            return []
        if isinstance(raw, str):
            values = [raw]
        else:
            values = list(raw)
        parsed: List[int] = []
        for item in values:
            if not item:
                continue
            try:
                parsed.append(self._time_to_seconds(str(item)))
            except ValueError:
                continue
        return sorted(set(parsed))

    @staticmethod
    def _time_to_seconds(value: str) -> int:
        value = value.strip()
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                dt = datetime.strptime(value, fmt)
                return dt.hour * 3600 + dt.minute * 60 + dt.second
            except ValueError:
                continue
        raise ValueError(value)

    @staticmethod
    def _next_scheduled_timestamp(reference: float, times: List[int]) -> float:
        if not times:
            return reference + 60.0
        now_dt = datetime.fromtimestamp(reference)
        seconds_today = now_dt.hour * 3600 + now_dt.minute * 60 + now_dt.second
        for target in times:
            if target > seconds_today:
                midnight = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                return midnight.timestamp() + target
        midnight_next = (now_dt + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return midnight_next.timestamp() + times[0]


class TaskManager:
    """Coordinates task loading, scheduling and execution."""

    def __init__(self, tasks: List[TaskDefinition], action_env: Dict[str, Any]):
        self.tasks = tasks
        self.scheduler = TaskScheduler(tasks)
        self.action_env = action_env
        self.executor = TaskActionExecutor(action_env)
        self.action_env["run_task_by_id"] = self.run_task_by_id

    def prepare(self, now: Optional[float] = None):
        now = now or time.time()
        self.scheduler.schedule_all(now)

    def run_due_tasks(self, now: Optional[float] = None) -> List[TaskResult]:
        now = now or time.time()
        results: List[TaskResult] = []
        for task in self.scheduler.due_tasks(now):
            result = self.run_task(task)
            results.append(result)
            self.scheduler.mark_executed(task, now)
        return results

    def run_task_by_id(self, task_id: str) -> Optional[TaskResult]:
        for task in self.tasks:
            if task.id == task_id:
                return self.run_task(task)
        logger = self.action_env.get("logger")
        if logger:
            logger.warning(f"Task '{task_id}' not found when requested via call_task.")
        return None

    def has_task(self, task_id: str) -> bool:
        return any(task.id == task_id for task in self.tasks)

    def run_task(self, task: TaskDefinition) -> TaskResult:
        context = TaskContext(task, self.action_env)
        try:
            self.executor.execute_steps(task.steps, context)
        except Exception as exc:
            logger = self.action_env.get("logger")
            if logger:
                logger.error(f"Task '{task.name}' failed: {exc}")
            context.success = False
            if not context.detail:
                context.detail = str(exc)
        if context.detail == "" and task.description:
            context.detail = task.description
        return TaskResult(context.executed, context.success, context.detail, context)


def load_tasks_from_dict(data: Dict[str, Any]) -> List[TaskDefinition]:
    tasks_data = data.get("tasks")
    if not isinstance(tasks_data, list):
        raise TaskError("tasks.yaml must contain a top-level 'tasks' list.")
    return [TaskDefinition.from_dict(item) for item in tasks_data]
