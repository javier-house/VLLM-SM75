# SPDX-License-Identifier: Apache-2.0
# vllm-sm75 overlay tests: auto-sleep config + controller state machine.
#
# Runs two ways:
#   * in the SM75 image under pytest (vllm importable, overlay installed);
#   * locally with plain ``python3 tests/v1/engine/test_auto_sleep.py``
#     (no vllm/torch): the module under test is loaded from its file path.

from __future__ import annotations

import importlib.util
import os
import queue
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _load_auto_sleep() -> Any:
    try:
        from vllm.v1.engine import auto_sleep
    except ImportError:
        # Local run: the repo mirrors only overlay files (no package inits),
        # so load the module straight from its file path.
        path = (
            Path(__file__).resolve().parents[3]
            / "vllm"
            / "v1"
            / "engine"
            / "auto_sleep.py"
        )
        spec = importlib.util.spec_from_file_location("auto_sleep_under_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        # Register before exec: @dataclass resolves string annotations via
        # sys.modules[cls.__module__].
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return auto_sleep


auto_sleep = _load_auto_sleep()


@contextmanager
def _env(pairs: dict[str, str | None]):
    saved = {key: os.environ.get(key) for key in pairs}
    for key, value in pairs.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _FakeScheduler:
    def __init__(self) -> None:
        self.has_requests_flag = False

    def has_requests(self) -> bool:
        return self.has_requests_flag


class _FakeExecutor:
    def __init__(self, fake: "_FakeEngine") -> None:
        self._fake = fake

    def collective_rpc(self, method: str, kwargs: dict | None = None):
        self._fake.rpc_calls.append((method, kwargs))
        return None


class _FakeEngine:
    """Duck-typed stand-in for EngineCore (attributes used by the controller)."""

    def __init__(self) -> None:
        self.input_queue: queue.Queue = queue.Queue()
        self.scheduler = _FakeScheduler()
        self.model_executor = _FakeExecutor(self)
        self._idle_state_callbacks: list = []
        self.sleeping = False
        self.running = True
        self.sleep_calls: list[int] = []
        self.wake_up_calls = 0
        self.rpc_calls: list[tuple[str, dict | None]] = []
        self.wake_up_error: Exception | None = None

    def is_sleeping(self) -> bool:
        return self.sleeping

    def is_running(self) -> bool:
        return self.running

    def sleep(self, level: int = 1, mode: str = "abort") -> None:
        self.sleep_calls.append(level)
        self.sleeping = True

    def wake_up(self, tags: list[str] | None = None) -> None:
        if self.wake_up_error is not None:
            raise self.wake_up_error
        self.sleeping = False
        self.wake_up_calls += 1


def _make_controller(
    fake: _FakeEngine,
    auto_sleep_mod: Any,
    timeout_seconds: float = 0.05,
    offload_target: str = "cpu",
    reload_path: str = "",
) -> Any:
    config = auto_sleep_mod.AutoSleepConfig(
        timeout_seconds=timeout_seconds,
        offload_target=offload_target,
        reload_path=reload_path,
    )
    return auto_sleep_mod.AutoSleepController(fake, b"\x05", config=config)


def _drain_wakeup(fake: _FakeEngine, controller: Any, timeout: float = 3.0) -> None:
    """Wait for the timer to enqueue the WAKEUP sentinel, then dispatch it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            request_type, _ = fake.input_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.005)
            continue
        if request_type == controller._wakeup_request_type:
            controller.on_wakeup_poke()
            return
    raise AssertionError("WAKEUP sentinel was not enqueued by the timer")


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_config_from_env_disabled_without_envs():
    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: None,
            auto_sleep.OFFLOAD_TARGET_ENV: None,
            auto_sleep.RELOAD_PATH_ENV: None,
        }
    ):
        assert auto_sleep.AutoSleepConfig.from_env() is None


def test_config_from_env_parses_minutes_and_maps_target():
    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: "0.2",  # 0.2 minutes = 12 seconds
            auto_sleep.OFFLOAD_TARGET_ENV: "cpu",
            auto_sleep.RELOAD_PATH_ENV: None,
        }
    ):
        config = auto_sleep.AutoSleepConfig.from_env()
    assert config is not None
    assert config.timeout_seconds == 12.0
    assert config.offload_target == "cpu"
    assert config.sleep_level == 1


def test_config_from_env_reload_requires_path():
    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: "1",
            auto_sleep.OFFLOAD_TARGET_ENV: "reload",
            auto_sleep.RELOAD_PATH_ENV: "/ckpt",
        }
    ):
        config = auto_sleep.AutoSleepConfig.from_env()
    assert config is not None
    assert config.sleep_level == 2
    assert config.reload_path == "/ckpt"

    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: "1",
            auto_sleep.OFFLOAD_TARGET_ENV: "reload",
            auto_sleep.RELOAD_PATH_ENV: "",
        }
    ):
        try:
            auto_sleep.AutoSleepConfig.from_env()
        except ValueError as exc:
            assert "RELOAD_PATH" in str(exc)
        else:
            raise AssertionError("expected ValueError for reload without path")


def test_config_from_env_invalid_values():
    with _env({auto_sleep.IDLE_TIMEOUT_ENV: "not-a-number"}):
        assert auto_sleep.AutoSleepConfig.from_env() is None
    with _env({auto_sleep.IDLE_TIMEOUT_ENV: "-1"}):
        assert auto_sleep.AutoSleepConfig.from_env() is None
    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: "1",
            auto_sleep.OFFLOAD_TARGET_ENV: "gpu",
            auto_sleep.RELOAD_PATH_ENV: None,
        }
    ):
        try:
            auto_sleep.AutoSleepConfig.from_env()
        except ValueError as exc:
            assert "OFFLOAD_TARGET" in str(exc)
        else:
            raise AssertionError("expected ValueError for unknown target")


# ---------------------------------------------------------------------------
# Controller state machine
# ---------------------------------------------------------------------------


def test_disabled_controller_hooks_are_noops():
    fake = _FakeEngine()
    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: None,
            auto_sleep.OFFLOAD_TARGET_ENV: None,
            auto_sleep.RELOAD_PATH_ENV: None,
        }
    ):
        controller = auto_sleep.AutoSleepController(fake, b"\x05")
        assert not controller.enabled
    fake._idle_state_callbacks = [controller.on_idle]
    controller.on_idle(fake)  # must not re-register or touch the engine
    assert fake._idle_state_callbacks == [controller.on_idle]
    controller.on_request_arrival()
    controller.on_wakeup_poke()
    assert fake.sleep_calls == []
    assert fake.wake_up_calls == 0


def test_idle_arms_timer_and_wakeup_poke_sleeps():
    fake = _FakeEngine()
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.05)
    assert controller.enabled

    controller.on_idle(fake)
    assert fake._idle_state_callbacks == [controller.on_idle], "callback must re-register"
    assert controller._timer is not None, "timer must be armed while idle"

    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == [1], "cpu target must sleep at level 1"
    assert controller.state is auto_sleep.AutoSleepState.SLEEPING
    assert fake.sleeping


def test_reload_target_sleeps_at_level_2():
    fake = _FakeEngine()
    controller = _make_controller(
        fake, auto_sleep, timeout_seconds=0.05, offload_target="reload",
        reload_path="/ckpt",
    )
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == [2]


def test_request_arrival_cancels_armed_timer():
    fake = _FakeEngine()
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.05)
    controller.on_idle(fake)
    assert controller._timer is not None

    controller.on_request_arrival()
    assert controller._timer is None

    time.sleep(0.1)  # past the timeout: the cancelled timer must not fire
    assert fake.input_queue.empty()
    assert fake.sleep_calls == []


def test_wakeup_poke_ignores_pending_requests():
    fake = _FakeEngine()
    fake.scheduler.has_requests_flag = True
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.001)
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == []
    assert controller.state is auto_sleep.AutoSleepState.ACTIVE


def test_wakeup_poke_ignores_fresh_activity():
    fake = _FakeEngine()
    controller = _make_controller(fake, auto_sleep, timeout_seconds=60.0)
    controller.on_request_arrival()  # fresh last_activity
    # Poke directly (as the timer would) without waiting.
    controller.on_wakeup_poke()
    assert fake.sleep_calls == []


def test_request_wakes_sleeping_engine_and_reloads():
    fake = _FakeEngine()
    controller = _make_controller(
        fake, auto_sleep, timeout_seconds=0.05, offload_target="reload",
        reload_path="/ckpt",
    )
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == [2]

    controller.on_request_arrival()
    assert fake.sleeping is False, "wake_up must resume the engine"
    assert fake.wake_up_calls == 1
    assert fake.rpc_calls == [
        ("reload_weights", {"weights_path": "/ckpt", "is_checkpoint_format": True})
    ], "reload target must re-run the checkpoint load pipeline"
    assert controller.state is auto_sleep.AutoSleepState.ACTIVE


def test_cpu_target_wake_does_not_reload():
    fake = _FakeEngine()
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.05)
    controller.on_idle(fake)
    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == [1]

    controller.on_request_arrival()
    assert fake.sleeping is False
    assert fake.rpc_calls == [], "cpu target must not reload weights"


def test_wake_up_failure_propagates_and_stays_active():
    fake = _FakeEngine()
    fake.wake_up_error = RuntimeError("boom")
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.05)
    fake.sleeping = True  # engine went to sleep via the dev endpoint

    try:
        controller.on_request_arrival()
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("wake_up failure must propagate to the caller")
    assert controller.state is auto_sleep.AutoSleepState.ACTIVE


def test_no_double_sleep_while_sleeping():
    fake = _FakeEngine()
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.05)
    fake.sleeping = True  # manual sleep via dev endpoint, controller unaware

    controller.on_idle(fake)
    assert controller._timer is None, "no timer while engine is already sleeping"

    controller.on_wakeup_poke()
    assert fake.sleep_calls == [], "must not sleep an already-sleeping engine"


def test_no_sleep_during_shutdown():
    fake = _FakeEngine()
    controller = _make_controller(fake, auto_sleep, timeout_seconds=0.001)
    controller.on_idle(fake)
    fake.running = False
    _drain_wakeup(fake, controller)
    assert fake.sleep_calls == []


# ---------------------------------------------------------------------------
# CLI arg validation (needs the full vllm package; image-only)
# ---------------------------------------------------------------------------


def test_cli_arg_validation_and_env_propagation():
    try:
        from vllm.engine import arg_utils
    except ImportError:
        print("  skipped: vllm not importable in this environment")
        return

    with _env(
        {
            auto_sleep.IDLE_TIMEOUT_ENV: None,
            auto_sleep.OFFLOAD_TARGET_ENV: None,
            auto_sleep.RELOAD_PATH_ENV: None,
        }
    ):
        # timeout without --enable-sleep-mode must be rejected
        try:
            arg_utils.EngineArgs(
                model="/m", auto_sleep_idle_timeout=1.0, enable_sleep_mode=False
            )
        except ValueError as exc:
            assert "enable-sleep-mode" in str(exc)
        else:
            raise AssertionError("expected ValueError without --enable-sleep-mode")

        # explicit values propagate to the envs for the engine-core subprocess
        arg_utils.EngineArgs(
            model="/m",
            auto_sleep_idle_timeout=2.5,
            auto_sleep_offload_target="reload",
            enable_sleep_mode=True,
        )
        assert os.environ[auto_sleep.IDLE_TIMEOUT_ENV] == "2.5"
        assert os.environ[auto_sleep.OFFLOAD_TARGET_ENV] == "reload"
        assert os.environ[auto_sleep.RELOAD_PATH_ENV] == "/m"

        # explicit reload path wins over the model path
        arg_utils.EngineArgs(
            model="/m",
            auto_sleep_idle_timeout=2.5,
            auto_sleep_offload_target="reload",
            auto_sleep_reload_path="/other",
            enable_sleep_mode=True,
        )
        assert os.environ[auto_sleep.RELOAD_PATH_ENV] == "/other"

        # disabled by default: no env side effects
        os.environ.pop(auto_sleep.IDLE_TIMEOUT_ENV, None)
        arg_utils.EngineArgs(model="/m", enable_sleep_mode=True)
        assert auto_sleep.IDLE_TIMEOUT_ENV not in os.environ


if __name__ == "__main__":
    failures = 0
    tests = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
