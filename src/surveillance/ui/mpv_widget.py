# Copyright (c) 2026, Renaud Allard <renaud@allard.it>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""MPV video player widget using GTK4 GLArea + mpv OpenGL render context."""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import logging
import math
import os
from collections.abc import Callable
from typing import Any

import gi

gi.require_version("Gtk", "4.0")

from gi.repository import GLib, Gtk  # type: ignore[import-untyped]

log = logging.getLogger(__name__)


def _load_gl_get_proc() -> ctypes.CDLL | None:
    """Load a native GL library that can resolve proc addresses."""
    candidates: list[str] = []
    for name in ("EGL", "GL", "GLX"):
        path = ctypes.util.find_library(name)
        if path:
            candidates.append(path)
    candidates.extend(("libEGL.so.1", "libGL.so.1", "libGLX.so.0"))
    for lib_name in candidates:
        try:
            lib = ctypes.CDLL(lib_name)
            # Verify it has a proc address function
            for fn_name in ("eglGetProcAddress", "glXGetProcAddressARB", "glXGetProcAddress"):
                fn = getattr(lib, fn_name, None)
                if fn is not None:
                    fn.argtypes = [ctypes.c_char_p]
                    fn.restype = ctypes.c_void_p
                    return lib
        except OSError:
            continue
    return None


_gl_lib = _load_gl_get_proc()

# video-zoom is mpv's own log2 scale (0 = 100%, 1 = 200%, ...). Zooming out
# below 100% would just letterbox the video within the widget for no
# benefit, so the range only goes up from there.
_ZOOM_MIN = 0.0
_ZOOM_MAX = 3.0  # 2**3 = 800%
# Clamp so scrolling can't pan the video entirely out of view.
_PAN_MAX = 0.8

# Adaptive playback-speed control keeps mpv's demuxer cache near its
# configured target instead of drifting unbounded in either direction:
# growing when decode falls behind, or draining during a network stall.
# Meaningful only for a profile actually running a cache: RTSP/default
# and muxed_audio always, low_latency only once a fast History rewind
# gives it one.
_CACHE_CONTROL_INTERVAL_MS = 500
_CACHE_CONTROL_SPEED_UP = 1.2
_CACHE_CONTROL_SPEED_DOWN = 0.8
_CACHE_CONTROL_SPEED_UP_ENTER = 1.5  # x target - reaches _SPEED_UP here
_CACHE_CONTROL_SPEED_DOWN_ENTER = 0.5  # x target - reaches _SPEED_DOWN here
# Periodic visibility into cache depth
_CACHE_LOG_INTERVAL_TICKS = round(2000 / _CACHE_CONTROL_INTERVAL_MS)  # ~2s

# Baseline cache target (seconds) for each playback profile, before
# _cache_target_seconds's own speed-aware adjustment. low_latency's
# cache stays off (see _apply_playback_options) whenever this ends up
# 0 (Live, or a History speed below _CACHE_HIGH_SPEED_ENTER), and
# turns on automatically once it isn't, for the extra buffer a fast
# History rewind needs.
_CACHE_SECONDS_DEFAULT = 0.5
_CACHE_SECONDS_MUXED_AUDIO = 0.5
_CACHE_SECONDS_LOW_LATENCY = 0.0

# Extra cache _high_speed_cache_seconds adds on top of a profile's
# baseline at a high History speed: 0 below _CACHE_HIGH_SPEED_ENTER,
# growing exponentially to _CACHE_HIGH_SPEED_MAX_SECONDS at a 100x
# factor (the fastest option Timeline's speed dropdown offers).
_CACHE_HIGH_SPEED_ENTER = 2.0  # x history speed - extra cache starts here
_CACHE_HIGH_SPEED_MAX_SECONDS = 8.0  # ~seconds added at a 100x factor
_CACHE_HIGH_SPEED_GROWTH = math.log(_CACHE_HIGH_SPEED_MAX_SECONDS) / (
    100.0 - _CACHE_HIGH_SPEED_ENTER
)

# Demuxer byte cap for every profile actually running a cache (see
# _apply_playback_options): mpv reads ahead by whichever of this and
# the seconds-based cache-secs/demuxer-readahead-secs above is larger,
# so too small a cap here can bottleneck a cache that's otherwise
# sized generously (e.g. low_latency's at a high History speed).
_DEMUXER_MAX_BYTES_MIB = 32.0

# Byte cap for a profile running with no cache at all -- today only a
# silent camera's Live stream. mpv's own docs put demuxer-max-bytes as
# the limit on "excessive readahead in case of broken files or desynced
# playback", so it is not the dead letter an unused cache would suggest:
# it is exactly the ceiling the demuxer buffers up to while playback is
# stalled, before backpressure reaches the pipe. Keeping it tight here
# bounds that per slot, which matters most on a full 4x4 grid, and costs
# nothing when nothing is stalled -- a stream reading ahead zero seconds
# never approaches it.
_DEMUXER_MAX_BYTES_UNCACHED = "512KiB"


# Runtime setters for the constants above, used by the Settings page's
# player-settings registry (surveillance.settings_registry) to override
# them without a restart. Each constant is read fresh wherever it's
# used in this module, so reassigning it here takes effect on the very
# next stream tick. Clamped to a sane floor: a hand-edited config file
# could otherwise hand these a negative (or, for the high-speed one,
# zero) value that mpv or math.log would reject.
def set_cache_seconds_default(value: float) -> None:
    global _CACHE_SECONDS_DEFAULT
    _CACHE_SECONDS_DEFAULT = max(value, 0.0)


def set_cache_seconds_muxed_audio(value: float) -> None:
    global _CACHE_SECONDS_MUXED_AUDIO
    _CACHE_SECONDS_MUXED_AUDIO = max(value, 0.0)


def set_demuxer_max_bytes_mib(value: float) -> None:
    global _DEMUXER_MAX_BYTES_MIB
    _DEMUXER_MAX_BYTES_MIB = max(value, 0.1)


def set_cache_seconds_low_latency(value: float) -> None:
    global _CACHE_SECONDS_LOW_LATENCY
    _CACHE_SECONDS_LOW_LATENCY = max(value, 0.0)


def set_high_speed_cache_max_seconds(value: float) -> None:
    """Update _CACHE_HIGH_SPEED_MAX_SECONDS and the growth rate derived
    from it together, since the growth rate is only ever computed once
    (not read fresh like the plain constants above) and would otherwise
    go stale."""
    global _CACHE_HIGH_SPEED_MAX_SECONDS, _CACHE_HIGH_SPEED_GROWTH
    value = max(value, 0.1)  # math.log below is undefined at or below 0
    _CACHE_HIGH_SPEED_MAX_SECONDS = value
    _CACHE_HIGH_SPEED_GROWTH = math.log(value) / (100.0 - _CACHE_HIGH_SPEED_ENTER)


# Whether _tick_cache_control draws its cache/target/speed readout as
# mpv OSD text: off by default, toggled from the Settings page (see
# set_osd_enabled). Checked fresh every tick, so toggling this takes
# effect within one tick (_CACHE_CONTROL_INTERVAL_MS) on an
# already-playing stream. The readout lands top-left, which is mpv's
# own default corner for osd-msg1 and the one sitting over video least
# often, so nothing has to ask for it.
_OSD_ENABLED = False


def set_osd_enabled(value: bool) -> None:
    global _OSD_ENABLED
    _OSD_ENABLED = value


def _high_speed_cache_seconds(high_playback_speed_factor: float) -> float:
    """Extra cache (seconds) to add on top of a profile's baseline once
    DSM is delivering frames at *high_playback_speed_factor* or faster:
    0 below _CACHE_HIGH_SPEED_ENTER, exactly 1.0 at
    _CACHE_HIGH_SPEED_ENTER, growing exponentially to
    _CACHE_HIGH_SPEED_MAX_SECONDS at a 100x factor: a low-speed/Live
    stream keeps the profile's own small baseline untouched, while a
    high-speed History stream gets real margin to swallow the increased
    traffic.
    """
    if high_playback_speed_factor < _CACHE_HIGH_SPEED_ENTER:
        return 0.0
    return math.exp(
        _CACHE_HIGH_SPEED_GROWTH * (high_playback_speed_factor - _CACHE_HIGH_SPEED_ENTER)
    )


def _cache_target_seconds(default_seconds: float, high_playback_speed_factor: float) -> float:
    """Cache target (seconds) to actually use, given a profile's own
    baseline *default_seconds* (one of the _CACHE_SECONDS_* constants)
    plus whatever _high_speed_cache_seconds adds for
    *high_playback_speed_factor* (DSM's own History-speed multiplier).
    """
    return default_seconds + _high_speed_cache_seconds(high_playback_speed_factor)


def _cache_control_speed(cache_seconds: float, target_seconds: float) -> float:
    """Playback-speed correction for a cache sitting at *cache_seconds*
    against *target_seconds*: 1.0x exactly on target, ramping linearly
    toward _CACHE_CONTROL_SPEED_UP as the ratio reaches
    _CACHE_CONTROL_SPEED_UP_ENTER, or toward _CACHE_CONTROL_SPEED_DOWN
    at _CACHE_CONTROL_SPEED_DOWN_ENTER.

    Bounded by those two either way, and deliberately unaware of DSM's
    History-speed multiplier. A high History speed is delivery rate,
    not a clock this correction sits on top of: DSM scales the frame
    rate and the msec header together and the client just relays what
    arrives (see WebSocketBridge._history_play_params' own "speed"
    bullet), so mpv's own speed is never already running that many
    times faster. Scaling this by the multiplier would make it a
    fast-forward rather than a correction -- 21x at 100x History --
    which empties the cache inside one tick and swings straight to the
    floor.
    """
    ratio = cache_seconds / target_seconds
    if ratio >= 1.0:
        fraction = min(1.0, (ratio - 1.0) / (_CACHE_CONTROL_SPEED_UP_ENTER - 1.0))
        return 1.0 + fraction * (_CACHE_CONTROL_SPEED_UP - 1.0)
    fraction = min(1.0, (1.0 - ratio) / (1.0 - _CACHE_CONTROL_SPEED_DOWN_ENTER))
    return 1.0 - fraction * (1.0 - _CACHE_CONTROL_SPEED_DOWN)


def _get_gl_proc_address(_ctx: ctypes.c_void_p, name: bytes) -> int:
    """Get OpenGL procedure address via native GL library.

    Signature matches mpv's MpvGlGetProcAddressFn:
      CFUNCTYPE(c_void_p, c_void_p, c_char_p)
    """
    if _gl_lib is None:
        return 0
    for fn_name in ("eglGetProcAddress", "glXGetProcAddressARB", "glXGetProcAddress"):
        fn = getattr(_gl_lib, fn_name, None)
        if fn is not None:
            addr = fn(name)
            if addr:
                return addr  # type: ignore[no-any-return]
    return 0


class MpvGLArea(Gtk.GLArea):
    """GTK4 GLArea widget that renders mpv video via OpenGL.

    Each instance has its own mpv player and render context.
    Works on both X11 and Wayland without wid embedding.
    """

    def __init__(self, tls_verify: bool = True) -> None:
        super().__init__()
        self._mpv: Any = None
        self._ctx: Any = None
        self._url: str = ""
        self._initialized = False
        self._render_pending = False
        self._tls_verify = tls_verify
        self._low_latency = False
        self._muxed_audio = False
        self._start_offset: float = 0
        self._zoom: float = 0.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._muted = False
        self._volume = 100
        self._cache_control_enabled = False
        self._cache_control_source: int | None = None
        self._cache_log_tick = 0
        # DSM's own History-speed multiplier (Live View's timeline speed
        # dropdown — see Timeline._SPEED_OPTIONS), already baked into how
        # fast DSM delivers frames. Used for additional throttling of
        # self._mpv.speed and so _tick_cache_control can report the true
        # playback multiplier.
        self._history_speed: float = 1.0

        self.set_auto_render(False)
        self.set_hexpand(True)
        self.set_vexpand(True)

        self.connect("realize", self._on_realize)
        self.connect("unrealize", self._on_unrealize)
        self.connect("render", self._on_render)

    def _on_realize(self, widget: Gtk.GLArea) -> None:
        """Initialize mpv and OpenGL render context when widget is realized."""
        self.make_current()

        if self.get_error():
            log.error("GLArea has error: %s", self.get_error())
            return

        try:
            import mpv

            # Audio output driver, left to mpv's own autoprobe unless
            # SURVEILLANCE_AO names one. The AppImage sets it to pulse where
            # it had to keep its own libpipewire, since mpv's PipeWire output
            # would then pair that copy with the host's modules and take the
            # process down with it (see TROUBLESHOOTING.md).
            ao_option: dict[str, str] = {}
            ao = os.environ.get("SURVEILLANCE_AO", "").strip()
            if ao:
                ao_option["ao"] = ao
                log.info("Audio output driver set to %s", ao)

            # SURVEILLANCE_HWDEC picks the hardware decoder (mpv --hwdec values:
            # auto, nvdec, nvdec-copy, vaapi, no ...). SURVEILLANCE_MPV_OPTS passes
            # extra mpv options as "name=value,name=value" (e.g.
            # "hwdec-extra-frames=32" when NVDEC reports "No decoder surfaces
            # left" on streams with deep reorder buffers). Both optional.
            hwdec = os.environ.get("SURVEILLANCE_HWDEC", "").strip() or "auto"
            extra_opts: dict[str, str] = {}
            for item in os.environ.get("SURVEILLANCE_MPV_OPTS", "").split(","):
                if "=" in item:
                    name, value = item.split("=", 1)
                    extra_opts[name.strip().replace("-", "_")] = value.strip()
            if extra_opts:
                log.info("Extra mpv options from SURVEILLANCE_MPV_OPTS: %s", extra_opts)

            self._mpv = mpv.MPV(
                vo="libmpv",
                hwdec=hwdec,
                keep_open="yes",
                idle="yes",
                input_default_bindings=False,
                input_vo_keyboard=False,
                log_handler=self._mpv_log,
                # A minimum level, not a filter: at "fatal" libmpv delivers
                # nothing else, so _mpv_log's error and warn branches could
                # never run and mpv was silent about every problem short of
                # a fatal one. It writes nothing to stderr either, terminal
                # being off, so this handler is the only way its diagnostics
                # reach a log at all.
                loglevel="debug" if log.isEnabledFor(logging.DEBUG) else "warn",
                demuxer_lavf_o="rtsp_transport=tcp",
                tls_verify=self._tls_verify,
                mute=self._muted,
                volume=self._volume,
                **ao_option,
                **extra_opts,
            )

            # Wrap with mpv's own CFUNCTYPE so ctypes type identity matches
            self._proc_addr_fn = mpv.MpvGlGetProcAddressFn(_get_gl_proc_address)

            # Set up OpenGL render context
            self._ctx = mpv.MpvRenderContext(
                self._mpv,
                "opengl",
                opengl_init_params={
                    "get_proc_address": self._proc_addr_fn,
                },
            )

            self._ctx.update_cb = self._mpv_update_cb
            self._initialized = True

            # If URL was set before realization, apply options and start playing
            if self._url:
                self._apply_playback_options()
                self._restart_cache_control()
                self._mpv["start"] = str(self._start_offset) if self._start_offset else "0"
                self._mpv.play(self._url)

        except Exception:
            log.exception("Failed to initialize mpv")
            self._initialized = False

    def _on_unrealize(self, widget: Gtk.GLArea) -> None:
        """Clean up mpv when widget is unrealized."""
        self.stop()
        if self._ctx:
            with contextlib.suppress(Exception):
                self._ctx.free()
            self._ctx = None
        if self._mpv:
            with contextlib.suppress(Exception):
                self._mpv.terminate()
            self._mpv = None
        self._initialized = False

    def _on_render(self, area: Gtk.GLArea, ctx: Any) -> bool:
        """Render callback - called by GTK when the area needs to be redrawn."""
        self._render_pending = False

        if not self._initialized or not self._ctx:
            return True

        try:
            if not self._url:
                # No active stream: paint black instead of asking mpv's render
                # context to redraw, since libmpv keeps repainting the last
                # decoded frame it has buffered until a new one arrives.
                from OpenGL.GL import GL_COLOR_BUFFER_BIT, glClear, glClearColor

                glClearColor(0.0, 0.0, 0.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT)
                return True

            width = self.get_width()
            height = self.get_height()
            scale = self.get_scale_factor()

            from OpenGL.GL import GL_FRAMEBUFFER_BINDING, glGetIntegerv

            fbo = int(glGetIntegerv(GL_FRAMEBUFFER_BINDING))

            self._ctx.render(
                flip_y=True,
                opengl_fbo={
                    "w": width * scale,
                    "h": height * scale,
                    "fbo": fbo,
                },
            )
            self._ctx.report_swap()
        except Exception as e:
            log.debug("Render error: %s", e)

        return True

    def _mpv_update_cb(self) -> None:
        """Called by mpv from its thread when a new frame is available.

        Coalesces multiple updates into a single render to avoid flooding
        the GTK main loop when many streams are active (e.g. 3x3 grid).
        Skips scheduling when stopped (no URL) to prevent idle render loops.
        """
        if self._url and not self._render_pending:
            self._render_pending = True
            GLib.idle_add(self._do_queue_render)

    def _do_queue_render(self) -> bool:
        """Queue a render on the main thread. Returns False to remove idle source."""
        if self._initialized:
            self.queue_render()
        return False

    def _mpv_log(self, loglevel: str, component: str, message: str) -> None:
        """Handle mpv log messages.

        mpv routes libavcodec and libavformat through the same client log
        under components named ffmpeg/*, and maps AV_LOG_ERROR to its own
        "error". Those are routine on a live stream: a decoder started
        mid-GOP reports a missing PPS and a failed slice header for every
        frame until the next keyframe, which is what a WebSocket bridge
        reconnecting onto the same pipe hands it. Reporting that as an
        application error would call a healthy stream broken, several
        hundred lines at a time and once per slot, so the decoder's own
        complaints stay at debug and only mpv's own components speak up.
        """
        text = message.strip()
        if component.startswith("ffmpeg"):
            log.debug("mpv [%s] %s: %s", component, loglevel, text)
        elif loglevel in ("error", "fatal"):
            log.error("mpv [%s]: %s", component, text)
        elif loglevel == "warn":
            log.warning("mpv [%s]: %s", component, text)
        else:
            # Only reachable on a debug run, which is the only time libmpv
            # is asked for these. Dropping them instead would mean paying
            # to carry every message across the C boundary for nothing.
            log.debug("mpv [%s] %s: %s", component, loglevel, text)

    def _apply_playback_options(self) -> None:
        """Apply buffering and timing options for the current playback profile."""
        if not self._mpv:
            return
        if self._muxed_audio:
            # A live-piped Matroska stream from our own ffmpeg mux (see
            # ws_bridge.py's audio muxing): unlike the other two
            # profiles' analyzeduration=0, this needs an explicit small
            # cap: 0 on a live (never-ending) pipe left libavformat
            # waiting well past any reasonable duration for "enough"
            # data to feel confident, a real, reproducible multi-second
            # stall. A well-formed MKV header (ffmpeg has already
            # resolved codec/timing info by the time mpv sees it) needs
            # nowhere near that.
            target_seconds = _cache_target_seconds(_CACHE_SECONDS_MUXED_AUDIO, self._history_speed)
            self._mpv["demuxer-lavf-analyzeduration"] = 0.3
        elif self._low_latency:
            # A silent camera's raw H.264/H.265 WebSocket pipe (video
            # only, no container at all to probe). demuxer-lavf-probesize,
            # in the shared block below, drops to the smallest value
            # libmpv accepts whenever this stream also stays untimed
            # (Live, or History below _CACHE_HIGH_SPEED_ENTER), shaving
            # startup latency off Live, where it matters most.
            target_seconds = _cache_target_seconds(_CACHE_SECONDS_LOW_LATENCY, self._history_speed)
            self._mpv["demuxer-lavf-analyzeduration"] = 0
        else:
            # RTSP/local file: 0 is mpv's own default for
            # analyzeduration, not a deliberate RTSP-specific tuning;
            # restated explicitly since every profile states its own
            # value for every option here (see
            # test_every_profile_writes_the_same_options).
            target_seconds = _cache_target_seconds(_CACHE_SECONDS_DEFAULT, self._history_speed)
            self._mpv["demuxer-lavf-analyzeduration"] = 0

        # Shared by all three profiles. cache-secs is a deliberate cap
        # everywhere, not mpv's own (much larger) default, and the
        # cache is on for as long as one is asked for at all: always
        # for muxed_audio/default's own nonzero baseline, and for
        # low_latency once a fast History rewind's extra cache
        # (_high_speed_cache_seconds) gives it one.
        #
        # The timing options key off the profile, not off the cache.
        # Only low_latency reads a headerless H.264/H.265 pipe, and
        # only it can be played untimed off a fixed frame rate with
        # libmpv's minimum probesize; muxed_audio has a real MKV
        # header and RTSP a real container, both with per-frame timing
        # worth trusting however deep their buffer happens to be. A
        # cache size of 0, which the Settings page accepts for either,
        # must not turn them into raw-NAL streams. The untimed pairing
        # still tracks the cache within low_latency itself: once a fast
        # rewind buffers, there is real timing to keep.
        cache_enabled = target_seconds > 0
        timed = cache_enabled or not self._low_latency
        self._mpv["cache"] = "yes" if cache_enabled else "no"
        self._mpv["demuxer-max-bytes"] = (
            f"{_DEMUXER_MAX_BYTES_MIB:g}MiB" if cache_enabled else _DEMUXER_MAX_BYTES_UNCACHED
        )
        self._mpv["demuxer-readahead-secs"] = target_seconds
        self._mpv["cache-secs"] = target_seconds
        self._mpv["correct-pts"] = timed
        self._mpv["untimed"] = not timed
        self._mpv["container-fps-override"] = 0 if timed else 25
        # 32 is libmpv's own minimum, for the fastest possible Live
        # startup on a raw pipe; 32768 is reliable for format/stream
        # detection everywhere else, and well below the probesize of
        # 5000000 the default profile used to inherit.
        self._mpv["demuxer-lavf-probesize"] = 32768 if timed else 32

        self._cache_control_enabled = self._mpv["cache-secs"] > 0

    def _restart_cache_control(self) -> None:
        """(Re)start the cache-speed ticker for the profile just applied
        by _apply_playback_options, stopping any previous one first.
        Called on every play()/profile change so a widget reused across
        streams doesn't keep ticking against a target or speed state left
        over from what it played before."""
        self._stop_cache_control()
        if self._cache_control_enabled:
            self._cache_control_source = GLib.timeout_add(
                _CACHE_CONTROL_INTERVAL_MS, self._tick_cache_control
            )

    def _stop_cache_control(self) -> None:
        """Cancel the ticker and reset playback to normal speed."""
        if self._cache_control_source is not None:
            GLib.source_remove(self._cache_control_source)
            self._cache_control_source = None
        if self._mpv is not None:
            with contextlib.suppress(Exception):
                self._mpv.speed = 1.0
                self._mpv["osd-msg1"] = ""

    def _tick_cache_control(self) -> bool:
        """Nudge playback speed to keep the demuxer cache near its
        target depth rather than drifting: growing without bound when
        decode falls behind, or draining to nothing during a network
        stall, both of which would otherwise mean the app silently
        lagging further behind real time. See _cache_control_speed for
        the ramp itself.
        """
        if not self._mpv or not self._url:
            self._cache_control_source = None
            return False
        with contextlib.suppress(Exception):
            if self._mpv.pause:
                return True
        # Attribute access, not mpv[name]: mpv[name] reads options/*, not
        # this runtime property, so it would silently read back None
        # instead of the live cache duration.
        cache_seconds: float | None = None
        with contextlib.suppress(Exception):
            cache_seconds = getattr(self._mpv, "demuxer_cache_duration", None)
        if cache_seconds is None:
            return True
        # Dict access here, not attribute: the inverse of the gotcha
        # above. cache-secs is an option _apply_playback_options wrote,
        # not a runtime property, so mpv[name] reads back exactly what
        # was set instead of options/* silently returning nothing.
        target_seconds: float | None = None
        with contextlib.suppress(Exception):
            target_seconds = self._mpv["cache-secs"]
        if not target_seconds:
            return True

        mpv_playback_speed = _cache_control_speed(cache_seconds, target_seconds)
        with contextlib.suppress(Exception):
            self._mpv.speed = mpv_playback_speed
        # The number shown/logged is the true playback multiplier against
        # real time: DSM already delivers frames self._history_speed
        # times faster, and mpv_playback_speed is only cache control's own
        # correction on top of that, so self._mpv.speed alone would
        # understate it.
        effective_speed = mpv_playback_speed * self._history_speed
        with contextlib.suppress(Exception):
            # Cleared rather than left as-is when disabled, so toggling
            # the setting off clears an already-playing stream's OSD
            # within one tick instead of leaving stale text on screen.
            self._mpv["osd-msg1"] = (
                (
                    f"cache {cache_seconds:.2f}secs\n"
                    f"target {target_seconds:.2f}secs\n"
                    f"speed {effective_speed:.2f}X"
                )
                if _OSD_ENABLED
                else ""
            )

        self._cache_log_tick += 1
        due = self._cache_log_tick % _CACHE_LOG_INTERVAL_TICKS == 0
        if due:
            log.debug(
                "Cache control: url=%s cache=%.2fs target=%.2fs speed=%.2f",
                self._url,
                cache_seconds,
                target_seconds,
                effective_speed,
            )
        return True

    def play(
        self,
        url: str,
        *,
        low_latency: bool = False,
        muxed_audio: bool = False,
        start_offset: float = 0,
        history_speed: float = 1.0,
    ) -> None:
        """Start playing a stream URL.

        When *low_latency* is True, disable caching and read-ahead so the
        stream plays in near real-time (used for WebSocket pipe bridges
        without a usable audio track, piping raw video only).

        When *muxed_audio* is True, use the buffering profile tuned for a
        live-piped Matroska stream instead (used for WebSocket pipe
        bridges that muxed real audio in via ffmpeg — see ws_bridge.py).
        Takes precedence over *low_latency* if both are set.

        *start_offset* seeks to that position (seconds) as the file loads,
        for playing a moment within a much longer recording file.

        *history_speed* is DSM's own History-speed multiplier already in
        effect for this stream (1.0 for Live, which has no speed concept);
        see set_history_speed for updating it on an already-playing
        History stream.
        """
        self._url = url
        self._low_latency = low_latency
        self._muxed_audio = muxed_audio
        self._start_offset = start_offset
        self._history_speed = history_speed
        if self._initialized and self._mpv:
            try:
                self._apply_playback_options()
                self._restart_cache_control()
                self._mpv["start"] = str(start_offset) if start_offset else "0"
                self._mpv.play(url)
            except Exception:
                log.exception("Failed to play %s", url)
                # _restart_cache_control() above already started the
                # ticker; without this it would keep polling and nudging
                # speed against a URL that never actually started playing.
                self._stop_cache_control()

    def stop(self) -> None:
        """Stop playback."""
        self._url = ""
        self._stop_cache_control()
        if self._mpv:
            with contextlib.suppress(Exception):
                self._mpv.command("stop")
        if self._initialized:
            self.queue_render()

    def pause(self) -> None:
        """Toggle pause."""
        if self._mpv:
            with contextlib.suppress(Exception):
                self._mpv.pause = not self._mpv.pause

    def set_paused(self, paused: bool) -> None:
        """Explicitly pause or resume, unlike pause()'s toggle -- for a
        caller tracking its own paused state (Live View's timeline
        Pause/Play button) where a toggle could drift out of sync with
        it if this and some other trigger ever raced."""
        if self._mpv:
            with contextlib.suppress(Exception):
                self._mpv.pause = paused

    def set_history_speed(self, value: float) -> None:
        """Update DSM's own History-speed multiplier for an already-
        playing stream (Live View's timeline speed dropdown, changed
        mid-session); see play()'s *history_speed* for the value set
        when a stream starts. Re-applies playback options, then
        restarts cache control against the new target.

        The cache options land on the running stream; the demuxer ones
        (demuxer-lavf-probesize/analyzeduration) are read when the
        demuxer is created, so if crossing _CACHE_HIGH_SPEED_ENTER
        changes them, the stream keeps the values it started with until
        the next play().
        """
        self._history_speed = value
        self._apply_playback_options()
        self._restart_cache_control()

    def _letterbox_fraction(self, width: int, height: int) -> tuple[float, float]:
        """Fraction of the widget's width/height the fitted (unzoomed)
        video actually occupies — 1.0 on an axis mpv doesn't letterbox to
        preserve the camera's aspect ratio, less than 1.0 on one it does.
        Falls back to (1.0, 1.0) if mpv hasn't reported a video size yet.
        """
        dwidth = getattr(self._mpv, "dwidth", None)
        dheight = getattr(self._mpv, "dheight", None)
        if not dwidth or not dheight or width <= 0 or height <= 0:
            return 1.0, 1.0
        fit_scale = min(width / dwidth, height / dheight)
        return (dwidth * fit_scale) / width, (dheight * fit_scale) / height

    def zoom_at(self, delta: float, cursor_x: float, cursor_y: float) -> None:
        """Zoom in/out by *delta* (mpv's own log2 video-zoom units — e.g.
        0.15 per scroll tick), keeping whatever's under the cursor
        (cursor_x, cursor_y — widget-relative pixel coordinates) fixed on
        screen — the standard "zoom to point" behavior of image viewers,
        maps, and graphics editors.

        mpv's video-pan-x/y are fractions of the *scaled* video size, not
        the widget, so a screen position maps to a video-space point via
        `screen = scale * (point + pan)`. Solving for the pan that keeps a
        screen position mapped to the same point across a scale change
        gives `pan' = pan + cursor_offset * (1/scale' - 1/scale)` — exact
        at every step either direction, and self-limiting at high zoom
        since 1/scale shrinks. cursor_offset is relative to the actual
        displayed video rect (see _letterbox_fraction), not the raw
        widget.
        """
        if not self._mpv or not self._initialized:
            return

        width = self.get_width()
        height = self.get_height()
        if width <= 0 or height <= 0:
            return

        new_zoom = max(_ZOOM_MIN, min(_ZOOM_MAX, self._zoom + delta))

        if new_zoom <= _ZOOM_MIN:
            # Back at (or already at) 1:1 — pan is meaningless with no
            # extra zoomed content to shift around, and leaving it nonzero
            # would show the video pushed off into a corner instead of
            # filling the slot. Reset it here unconditionally, even if the
            # zoom level itself isn't changing — e.g. scrolling out
            # further while already at the floor after panning around at
            # 1:1, which would otherwise never reach this branch at all.
            if self._zoom == new_zoom and self._pan_x == 0.0 and self._pan_y == 0.0:
                return  # genuinely nothing to do
            self._zoom = new_zoom
            self._pan_x = 0.0
            self._pan_y = 0.0
        else:
            if new_zoom == self._zoom:
                return
            scale_old = 2**self._zoom
            scale_new = 2**new_zoom
            inv_delta = 1 / scale_new - 1 / scale_old
            letterbox_x, letterbox_y = self._letterbox_fraction(width, height)
            # Cursor position relative to widget center, normalized to
            # [-0.5, 0.5], then rescaled from widget-relative to
            # actual-video-rect-relative.
            nx = (cursor_x / width - 0.5) / letterbox_x
            ny = (cursor_y / height - 0.5) / letterbox_y
            self._pan_x = max(-_PAN_MAX, min(_PAN_MAX, self._pan_x + nx * inv_delta))
            self._pan_y = max(-_PAN_MAX, min(_PAN_MAX, self._pan_y + ny * inv_delta))
            self._zoom = new_zoom

        with contextlib.suppress(Exception):
            self._mpv.video_zoom = self._zoom
            self._mpv.video_pan_x = self._pan_x
            self._mpv.video_pan_y = self._pan_y

    def pan_by(self, dx_fraction: float, dy_fraction: float) -> None:
        """Pan by an incremental amount, as fractions of the widget's own
        size (same units zoom_at() takes a cursor offset in). Divided by
        the current scale and the letterbox fraction to convert to mpv's
        own pan units, so a drag tracks the cursor at a constant
        screen-space rate regardless of zoom level or the camera's aspect
        ratio. Allowed even at 1:1 zoom (scrolling back out already
        re-centers via reset-on-floor in zoom_at(), so panning off-center
        at 1:1 is always one scroll-out away from being undone)."""
        if not self._mpv or not self._initialized:
            return
        scale = 2**self._zoom
        letterbox_x, letterbox_y = self._letterbox_fraction(self.get_width(), self.get_height())
        self._pan_x = max(
            -_PAN_MAX, min(_PAN_MAX, self._pan_x + dx_fraction / (scale * letterbox_x))
        )
        self._pan_y = max(
            -_PAN_MAX, min(_PAN_MAX, self._pan_y + dy_fraction / (scale * letterbox_y))
        )
        with contextlib.suppress(Exception):
            self._mpv.video_pan_x = self._pan_x
            self._mpv.video_pan_y = self._pan_y

    def reset_zoom(self) -> None:
        """Reset zoom/pan to the default 1:1 view."""
        self._zoom = 0.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        if self._mpv:
            with contextlib.suppress(Exception):
                self._mpv.video_zoom = 0.0
                self._mpv.video_pan_x = 0.0
                self._mpv.video_pan_y = 0.0

    def set_volume(self, volume: int) -> None:
        """Set volume (0-100). Independent of mute — the level set here is
        what's restored on unmute, silent or not in the meantime. Stored
        whether or not mpv has been realized yet, same as set_mute()."""
        self._volume = volume
        if self._mpv:
            with contextlib.suppress(Exception):
                self._mpv.volume = volume

    @property
    def muted(self) -> bool:
        return self._muted

    def set_mute(self, muted: bool) -> None:
        """Mute/unmute without touching the volume level (mpv's own `mute`
        property, not zeroing volume) — stored regardless of whether mpv
        has been realized yet, same as _low_latency/_start_offset, and
        applied immediately if it has."""
        self._muted = muted
        if self._mpv:
            with contextlib.suppress(Exception):
                self._mpv.mute = muted

    def seek(self, seconds: float) -> None:
        """Seek relative to current position."""
        if self._mpv:
            with contextlib.suppress(Exception):
                self._mpv.seek(seconds)

    def seek_absolute(self, seconds: float) -> None:
        """Seek to absolute position."""
        if self._mpv:
            with contextlib.suppress(Exception):
                self._mpv.seek(seconds, reference="absolute")

    @property
    def duration(self) -> float | None:
        """Get duration of current media."""
        if self._mpv:
            try:
                result: float | None = self._mpv.duration
            except Exception:
                return None
            else:
                return result
        return None

    @property
    def time_pos(self) -> float | None:
        """Get current playback position."""
        if self._mpv:
            try:
                result: float | None = self._mpv.time_pos
            except Exception:
                return None
            else:
                return result
        return None

    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        if self._mpv:
            try:
                return not self._mpv.pause and self._mpv.time_pos is not None
            except Exception:
                return False
        return False


# video-zoom units (mpv's own log2 scale) applied per scroll wheel tick.
_ZOOM_STEP = 0.15
# A drag shorter than this (pixels, either axis) is still treated as a
# plain click rather than a pan.
_DRAG_CLICK_THRESHOLD = 4


def attach_zoom_pan_controls(player: MpvGLArea, on_click: Callable[[], None] | None = None) -> None:
    """Wire up scroll-to-zoom (centered on the cursor) and click-and-drag
    panning on an MpvGLArea. Shared by Live View slots and the recording
    player dialog so both behave identically.

    If `on_click` is given, a drag shorter than _DRAG_CLICK_THRESHOLD is
    treated as a plain click and calls it with no arguments — used by Live
    View slots, which select the slot on click. Pass None (the default)
    where the video area has no click behavior to preserve, so every drag
    is treated as a pan from the first pixel.
    """
    # EventControllerScroll's "scroll" signal has no position, so a motion
    # controller tracks the last-known pointer position for it to use.
    pointer = {"x": 0.0, "y": 0.0}

    def on_motion(controller: Gtk.EventControllerMotion, x: float, y: float) -> None:
        pointer["x"] = x
        pointer["y"] = y

    motion = Gtk.EventControllerMotion()
    motion.connect("motion", on_motion)
    player.add_controller(motion)

    def on_scroll(controller: Gtk.EventControllerScroll, dx: float, dy: float) -> bool:
        # Scroll up (dy negative — "away from the user") zooms in, matching
        # the near-universal convention (maps, image viewers, etc.).
        player.zoom_at(-dy * _ZOOM_STEP, pointer["x"], pointer["y"])
        return True  # handled — don't let it bubble past the widget

    scroll = Gtk.EventControllerScroll(flags=Gtk.EventControllerScrollFlags.VERTICAL)
    scroll.connect("scroll", on_scroll)
    player.add_controller(scroll)

    # drag-update/drag-end report the offset cumulative from drag-begin,
    # not incrementally — pan_by() wants an increment, so track how much
    # of that cumulative offset has already been applied.
    drag_last = {"x": 0.0, "y": 0.0}

    def on_drag_begin(gesture: Gtk.GestureDrag, start_x: float, start_y: float) -> None:
        drag_last["x"] = 0.0
        drag_last["y"] = 0.0

    def on_drag_update(gesture: Gtk.GestureDrag, offset_x: float, offset_y: float) -> None:
        width = player.get_width()
        height = player.get_height()
        if width <= 0 or height <= 0:
            return
        dx = (offset_x - drag_last["x"]) / width
        dy = (offset_y - drag_last["y"]) / height
        drag_last["x"] = offset_x
        drag_last["y"] = offset_y
        # Content should follow the cursor (grab-and-drag feel): dragging
        # right/down reveals what was previously off to the left/top.
        player.pan_by(dx, dy)

    def on_drag_end(gesture: Gtk.GestureDrag, offset_x: float, offset_y: float) -> None:
        moved = abs(offset_x) >= _DRAG_CLICK_THRESHOLD or abs(offset_y) >= _DRAG_CLICK_THRESHOLD
        if not moved and on_click is not None:
            on_click()

    drag = Gtk.GestureDrag(button=1)
    drag.connect("drag-begin", on_drag_begin)
    drag.connect("drag-update", on_drag_update)
    drag.connect("drag-end", on_drag_end)
    player.add_controller(drag)
