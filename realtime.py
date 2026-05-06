#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实时读取 ToF 数据，仅调用 tof_mtf 包的 run_all_checks 接口：
- 清晰度(MTF)
- 倾斜(Tilt)

界面只显示两张图：
- MTF output.bmp
- Tilt output.bmp
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np

from tof_mtf import run_all_checks


TOF_H = 30
TOF_W = 40
TOF_C = 64
TOF_RAW_HEADER_BYTES = 5120

TARGET_FPS = 10.0
ADB_PULL_TIMEOUT_S = 0.9

TMP_DIR = Path("./tmp")
TMP_PULL_RAW_PATH = TMP_DIR / "tof_pull.raw"
TMP_CHECK_RAW_PATH = TMP_DIR / "realtime_check.raw"
REC_DIR = TMP_DIR
REC_FPS = 20.0


@dataclass(frozen=True)
class ToFFrame:
    ts: float
    raw_bytes: bytes


class ToFRealtimeServer:
    def __init__(
        self,
        *,
        queue_maxlen: int = 3,
        min_peak_count: float = 100.0,
        target_fps: float = TARGET_FPS,
        raw_expected_bytes: int | None = None,
        read_retry: int = 3,
    ) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._q: Deque[ToFFrame] = deque(maxlen=int(max(queue_maxlen, 1)))
        self._target_dt = 1.0 / float(max(target_fps, 1.0))
        self._read_retry = int(max(read_retry, 0))
        self._min_peak_count = float(max(min_peak_count, 0.0))
        if raw_expected_bytes is None:
            self._raw_expected_bytes = int(TOF_RAW_HEADER_BYTES + TOF_H * TOF_W * TOF_C * 2)
        else:
            self._raw_expected_bytes = int(raw_expected_bytes)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ToFRealtimeServerInline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.8)
        self._thread = None

    def get_latest(self) -> Optional[ToFFrame]:
        with self._lock:
            return self._q[-1] if self._q else None

    @staticmethod
    def _adb_trigger_generate_raw() -> bool:
        cmd = "if [ -e /tmp/sv_tof ]; then rm /tmp/sv_tof && rm /tmp/tof.raw; fi && touch /tmp/sv_tof"
        try:
            r = subprocess.run(
                ["adb", "shell", cmd],
                timeout=0.6,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return int(r.returncode) == 0
        except Exception:
            return False

    @staticmethod
    def _adb_pull_raw_bytes(*, expected_bytes: int, retry: int) -> bytes | None:
        expected = int(expected_bytes)
        retr = int(max(retry, 0))
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        for k in range(retr + 1):
            try:
                if TMP_PULL_RAW_PATH.exists():
                    TMP_PULL_RAW_PATH.unlink(missing_ok=True)
                r = subprocess.run(
                    ["adb", "pull", "/tmp/tof.raw", str(TMP_PULL_RAW_PATH)],
                    timeout=float(ADB_PULL_TIMEOUT_S),
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if int(r.returncode) != 0:
                    if k < retr:
                        time.sleep(0.01)
                    continue
                if (not TMP_PULL_RAW_PATH.exists()) or int(TMP_PULL_RAW_PATH.stat().st_size) < expected:
                    if k < retr:
                        time.sleep(0.01)
                    continue
                out = TMP_PULL_RAW_PATH.read_bytes()
                if len(out) >= expected:
                    return bytes(out[:expected])
            except Exception:
                pass
            if k < retr:
                time.sleep(0.01)
        return None

    def _run(self) -> None:
        fail_sleep = 0.15
        while not self._stop.is_set():
            if not self._adb_trigger_generate_raw():
                time.sleep(fail_sleep)
                continue
            time.sleep(0.03)
            t0 = time.perf_counter()
            raw_bytes = self._adb_pull_raw_bytes(expected_bytes=self._raw_expected_bytes, retry=self._read_retry)
            if not raw_bytes:
                time.sleep(fail_sleep)
                continue
            if self._min_peak_count > 0.0:
                raw_u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
                hists = tof_histograms_from_u16(raw_u16)
                if hists.shape != (TOF_H, TOF_W, TOF_C):
                    time.sleep(fail_sleep)
                    continue
                peak = float(np.max(hists[:, :, :62]))
                if peak < self._min_peak_count:
                    time.sleep(fail_sleep)
                    continue
            frame = ToFFrame(ts=time.time(), raw_bytes=raw_bytes)
            with self._lock:
                self._q.append(frame)
            dt = time.perf_counter() - t0
            sleep = self._target_dt - dt
            if sleep > 0:
                time.sleep(min(sleep, 0.2))


def tof_histograms_from_u16(raw_u16: np.ndarray) -> np.ndarray:
    header_words = int(TOF_RAW_HEADER_BYTES // 2)
    if raw_u16.size <= header_words:
        return np.zeros((TOF_H, TOF_W, TOF_C), dtype=np.uint16)
    data = raw_u16[header_words:]
    expected = TOF_H * TOF_W * TOF_C
    if data.size < expected:
        return np.zeros((TOF_H, TOF_W, TOF_C), dtype=np.uint16)
    return data[:expected].reshape((TOF_H, TOF_W, TOF_C)).astype(np.uint16, copy=False)


def _placeholder_view(h: int = 344, w: int = 800) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _overlay_rec(view: np.ndarray) -> np.ndarray:
    out = view.copy()
    cv2.circle(out, (out.shape[1] - 24, 22), 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        out,
        "REC",
        (out.shape[1] - 72, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main() -> int:
    window_name = "TOF_REALTIME_CHECK"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    tof_srv = ToFRealtimeServer(queue_maxlen=3, min_peak_count=100.0, target_fps=float(TARGET_FPS))
    tof_srv.start()

    last_ts = 0.0
    image_cache: np.ndarray | None = None
    latest_raw_bytes: bytes | None = None
    rec_writer: cv2.VideoWriter | None = None
    rec_path: Path | None = None
    view_cache: np.ndarray = _placeholder_view()

    try:
        while True:
            frame = tof_srv.get_latest()
            if frame is not None and float(frame.ts) > float(last_ts):
                try:
                    latest_raw_bytes = bytes(frame.raw_bytes)
                    TMP_CHECK_RAW_PATH.write_bytes(frame.raw_bytes)
                    _passed, image, _params = run_all_checks(str(TMP_CHECK_RAW_PATH))
                    if image is not None:
                        image_cache = image
                except Exception:
                    pass
                last_ts = float(frame.ts)

            base_view = image_cache if image_cache is not None else _placeholder_view()
            view_cache = _overlay_rec(base_view) if rec_writer is not None else base_view
            cv2.imshow(window_name, view_cache)
            if rec_writer is not None:
                rec_writer.write(view_cache)

            key = int(cv2.waitKey(1) & 0xFF)
            if key == 32:  # Space: toggle recording
                if rec_writer is None:
                    try:
                        REC_DIR.mkdir(parents=True, exist_ok=True)
                        rec_path = REC_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(rec_path),
                            fourcc,
                            max(float(REC_FPS), 1.0),
                            (view_cache.shape[1], view_cache.shape[0]),
                        )
                        if writer.isOpened():
                            rec_writer = writer
                            print(f"[存储成功] 视频开始录制: {rec_path}")
                        else:
                            writer.release()
                    except Exception:
                        pass
                else:
                    rec_writer.release()
                    rec_writer = None
                    if rec_path is not None:
                        print(f"[存储成功] 视频保存完成: {rec_path}")
                    rec_path = None
            if key == 48:  # '0': save current raw
                try:
                    if latest_raw_bytes is not None:
                        TMP_DIR.mkdir(parents=True, exist_ok=True)
                        raw_path = TMP_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.raw"
                        raw_path.write_bytes(latest_raw_bytes)
                        print(f"[存储成功] RAW已保存: {raw_path}")
                except Exception:
                    pass
            if key == 27:  # ESC
                break
    finally:
        if rec_writer is not None:
            rec_writer.release()
        tof_srv.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

