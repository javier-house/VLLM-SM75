# SPDX-License-Identifier: Apache-2.0
# vllm-sm75 overlay: idle auto-sleep controller for the V1 EngineCore.
#
# Upstream vLLM ships --enable-sleep-mode, but sleeping/waking must be driven
# by external POST /sleep + POST /wake_up calls.  This controller adds the
# missing control plane:
#
#   * the engine sleeps by itself after a configurable idle timeout;
#   * the first arriving request wakes it transparently (no API call).
#
# Design notes (upstream v0.28.0, see tmp/PLAN-auto-sleep-mode.md):
#
#   * When fully idle the EngineCoreProc main loop blocks in
#     input_queue.get(block=True) and step() returns early, so a per-step
#     timeout check is impossible.  The controller reuses upstream facilities:
#       - the one-shot _idle_state_callbacks to observe "engine went idle";
#       - a daemon threading.Timer armed once per idle period;
#       - input_queue.put_nowait((WAKEUP, None)) (the same sentinel the
#         shutdown signal handler uses) to hand the decision back to the
#         main loop thread, where sleep()/wake_up() may only run.
#   * offload_target "cpu"    -> sleep(level=1): weights copied to pinned CPU
#                                memory, wake is fast (~1-2 s);
#     offload_target "reload" -> sleep(level=2): weights discarded, wake is
#     wake_up() + collective_rpc("reload_weights") which re-runs the startup
#     load pipeline (quantization repack included), wake is slow (~20-60 s).
#
# This module deliberately has no vllm imports so it can be unit-tested
# without a GPU environment; the EngineCore is accessed duck-typed.

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

IDLE_TIMEOUT_ENV = "VLLM_AUTO_SLEEP_IDLE_TIMEOUT"
OFFLOAD_TARGET_ENV = "VLLM_AUTO_SLEEP_OFFLOAD_TARGET"
RELOAD_PATH_ENV = "VLLM_AUTO_SLEEP_RELOAD_PATH"

_OFFLOAD_TARGETS = ("cpu", "reload")


@dataclass(frozen=True)
class AutoSleepConfig:
    """Resolved auto-sleep configuration (timeout in seconds)."""

    timeout_seconds: float
    offload_target: str = "cpu"
    reload_path: str = ""

    @property
    def sleep_level(self) -> int:
        return 1 if self.offload_target == "cpu" else 2

    @classmethod
    def from_env(cls) -> "AutoSleepConfig | None":
        """Build config from VLLM_AUTO_SLEEP_* envs; None when disabled.

        The envs are populated by EngineArgs.__post_init__ from the
        --auto-sleep-* CLI flags before the engine-core process is spawned.
        """
        raw = os.environ.get(IDLE_TIMEOUT_ENV, "0")
        try:
            timeout_minutes = float(raw)
        except ValueError:
            logger.warning(
                "auto-sleep: invalid %s=%r, auto-sleep disabled",
                IDLE_TIMEOUT_ENV,
                raw,
            )
            return None
        if timeout_minutes <= 0:
            return None
        target = os.environ.get(OFFLOAD_TARGET_ENV, "cpu")
        if target not in _OFFLOAD_TARGETS:
            raise ValueError(
                f"auto-sleep: {OFFLOAD_TARGET_ENV} must be one of "
                f"{_OFFLOAD_TARGETS}, got {target!r}"
            )
        reload_path = os.environ.get(RELOAD_PATH_ENV, "")
        if target == "reload" and not reload_path:
            raise ValueError(
                f"auto-sleep: offload target 'reload' requires "
                f"{RELOAD_PATH_ENV} set to a readable checkpoint path"
            )
        return cls(
            timeout_seconds=timeout_minutes * 60.0,
            offload_target=target,
            reload_path=reload_path,
        )


class AutoSleepState(str, Enum):
    ACTIVE = "active"
    SLEEPING = "sleeping"
    WAKING = "waking"


class AutoSleepController:
    """Idle auto-sleep state machine driven by EngineCore hooks.

    All public methods run on the EngineCore main loop thread except the
    timer callback (_poke), which only enqueues the WAKEUP sentinel.
    The engine_core object is accessed duck-typed; required attributes:
    input_queue, scheduler.has_requests(), is_sleeping(), sleep(level),
    wake_up(), model_executor.collective_rpc(), _idle_state_callbacks, and
    optionally is_running() (absent on the in-process EngineCore base class).
    """

    def __init__(
        self,
        engine_core: Any,
        wakeup_request_type: Any,
        config: AutoSleepConfig | None = None,
    ) -> None:
        if config is None:
            config = AutoSleepConfig.from_env()
        self._engine = engine_core
        self._wakeup_request_type = wakeup_request_type
        self._config = config
        self._state = AutoSleepState.ACTIVE
        self._timer: threading.Timer | None = None
        self._last_activity = time.monotonic()
        if config is not None:
            logger.info(
                "auto-sleep enabled: idle_timeout=%.1fs target=%s "
                "reload_path=%s",
                config.timeout_seconds,
                config.offload_target,
                config.reload_path or "<unset>",
            )

    @property
    def enabled(self) -> bool:
        return self._config is not None

    @property
    def state(self) -> AutoSleepState:
        return self._state

    # ------------------------------------------------------------------
    # Hooks installed in EngineCore (main loop thread)
    # ------------------------------------------------------------------

    def on_idle(self, engine_core: Any) -> None:
        """Idle-state callback: re-register and arm the sleep timer.

        Upstream pops one-shot callbacks on every idle-loop iteration, so
        re-appending ourselves keeps the controller subscribed for future
        idle periods.
        """
        if not self.enabled:
            return
        engine_core._idle_state_callbacks.append(self.on_idle)
        if self._state is not AutoSleepState.ACTIVE:
            return
        if self._timer is not None:
            return
        if self._engine.is_sleeping() or not self._engine_running():
            return
        delay = max(
            self._config.timeout_seconds - (time.monotonic() - self._last_activity),
            0.0,
        )
        self._timer = threading.Timer(delay, self._poke)
        self._timer.daemon = True
        self._timer.start()
        logger.debug("auto-sleep: armed sleep timer for %.1fs", delay)

    def on_request_arrival(self) -> None:
        """Hook for EngineCore.add_request: refresh idle clock / wake up."""
        if not self.enabled:
            return
        self._last_activity = time.monotonic()
        self._cancel_timer()
        if self._engine.is_sleeping():
            self._wake()

    def on_wakeup_poke(self) -> None:
        """Hook for the EngineCore WAKEUP input branch (main thread).

        The timer only pokes; every decision is re-validated here because
        requests may have arrived in the meantime.
        """
        if not self.enabled or self._state is not AutoSleepState.ACTIVE:
            return
        if self._engine.is_sleeping() or not self._engine_running():
            return
        if self._engine.scheduler.has_requests():
            return
        if time.monotonic() - self._last_activity < self._config.timeout_seconds:
            return
        self._sleep()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _engine_running(self) -> bool:
        is_running = getattr(self._engine, "is_running", None)
        return True if is_running is None else is_running()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _poke(self) -> None:
        # Timer thread: only nudge the main loop; every decision happens on
        # the main thread in on_wakeup_poke.
        self._timer = None
        self._engine.input_queue.put_nowait((self._wakeup_request_type, None))

    def _set_state(self, state: AutoSleepState) -> None:
        if self._state is not state:
            logger.info(
                "auto-sleep: state %s -> %s", self._state.value, state.value
            )
            self._state = state

    def _sleep(self) -> None:
        self._set_state(AutoSleepState.SLEEPING)
        level = self._config.sleep_level
        try:
            self._engine.sleep(level)
        except Exception:
            logger.exception(
                "auto-sleep: sleep(level=%d) failed; engine stays active",
                level,
            )
            self._set_state(AutoSleepState.ACTIVE)
            return
        logger.info(
            "auto-sleep: engine is sleeping (level=%d, target=%s)",
            level,
            self._config.offload_target,
        )

    def _wake(self) -> None:
        self._set_state(AutoSleepState.WAKING)
        start = time.monotonic()
        try:
            self._engine.wake_up()
            if self._config.offload_target == "reload":
                # Re-run the startup load pipeline (incl. quantization
                # repack) on every worker; weights are written back in place.
                self._engine.model_executor.collective_rpc(
                    "reload_weights",
                    kwargs={
                        "weights_path": self._config.reload_path,
                        "is_checkpoint_format": True,
                    },
                )
        except Exception:
            logger.exception("auto-sleep: wake_up failed")
            self._set_state(AutoSleepState.ACTIVE)
            raise
        elapsed = time.monotonic() - start
        self._set_state(AutoSleepState.ACTIVE)
        logger.info(
            "auto-sleep: engine woke in %.1fs (target=%s); the first "
            "request latency includes this time",
            elapsed,
            self._config.offload_target,
        )
