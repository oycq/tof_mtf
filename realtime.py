#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实时读取 ToF 数据，仅调用 tof_mtf 包的 run_all_checks 接口：
- 清晰度(MTF)
- 倾斜(Tilt)

单线程主循环：trigger -> pull -> check -> 显示。
所有外部子进程超时统一为 0.3 秒，任何一步超时/失败都丢这帧继续下一帧。
"""

from __future__ import annotations

import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from tof_mtf import run_all_checks


TOF_H = 30
TOF_W = 40
TOF_C = 64
TOF_RAW_HEADER_BYTES = 5120
RAW_EXPECTED_BYTES = TOF_RAW_HEADER_BYTES + TOF_H * TOF_W * TOF_C * 2

STEP_TIMEOUT_S = 0.3

TMP_DIR = Path("./tmp")
TMP_PULL_RAW_PATH = TMP_DIR / "tof_pull.raw"
TMP_CHECK_RAW_PATH = TMP_DIR / "realtime_check.raw"
REC_DIR = TMP_DIR
REC_FPS = 20.0


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


def _check_adb_connected() -> bool:
    try:
        r = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            timeout=3.0,
            check=False,
            text=True,
        )
    except Exception:
        return False
    if r.returncode != 0:
        return False
    for ln in (r.stdout or "").splitlines()[1:]:
        if "\tdevice" in ln:
            return True
    return False


def _adb_trigger() -> tuple[bool, str]:
    cmd = "if [ -e /tmp/sv_tof ]; then rm /tmp/sv_tof && rm /tmp/tof.raw; fi && touch /tmp/sv_tof"
    try:
        r = subprocess.run(
            ["adb", "shell", cmd],
            timeout=STEP_TIMEOUT_S,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if int(r.returncode) == 0:
            return True, ""
        err = (r.stderr or b"").decode("utf-8", errors="ignore").strip()
        return False, f"adb shell rc={r.returncode}: {err[:200]}"
    except subprocess.TimeoutExpired:
        return False, "adb shell timeout"
    except Exception as e:
        return False, f"adb shell exc: {e!r}"


def _adb_pull_raw() -> tuple[bytes | None, str]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if TMP_PULL_RAW_PATH.exists():
            TMP_PULL_RAW_PATH.unlink(missing_ok=True)
        r = subprocess.run(
            ["adb", "pull", "/tmp/tof.raw", str(TMP_PULL_RAW_PATH)],
            timeout=STEP_TIMEOUT_S,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if int(r.returncode) != 0:
            err = (r.stderr or b"").decode("utf-8", errors="ignore").strip() or f"rc={r.returncode}"
            return None, err
        if not TMP_PULL_RAW_PATH.exists():
            return None, "pulled file missing"
        size = int(TMP_PULL_RAW_PATH.stat().st_size)
        if size < RAW_EXPECTED_BYTES:
            return None, f"pulled too small: {size} < {RAW_EXPECTED_BYTES}"
        out = TMP_PULL_RAW_PATH.read_bytes()
        return bytes(out[:RAW_EXPECTED_BYTES]), ""
    except subprocess.TimeoutExpired:
        return None, "adb pull timeout"
    except Exception as e:
        return None, f"adb pull exc: {e!r}"


def _throttle_log(state: dict, msg: str) -> None:
    now = time.time()
    if now - state.get("last_log_ts", 0.0) > 2.0:
        print(f"[realtime] {msg}", flush=True)
        state["last_log_ts"] = now


def main() -> int:
    if not _check_adb_connected():
        print("[realtime] adb 无法连接（请检查设备是否插好、是否授权）", flush=True)
        return 1

    window_name = "TOF_REALTIME_CHECK"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    image_cache: np.ndarray | None = None
    latest_raw_bytes: bytes | None = None
    rec_writer: cv2.VideoWriter | None = None
    rec_path: Path | None = None
    view_cache: np.ndarray = _placeholder_view()
    log_state: dict = {"last_log_ts": 0.0}

    try:
        while True:
            ok, err = _adb_trigger()
            if ok:
                raw_bytes, perr = _adb_pull_raw()
                if raw_bytes is not None:
                    latest_raw_bytes = raw_bytes
                    try:
                        TMP_CHECK_RAW_PATH.write_bytes(raw_bytes)
                        _passed, image, _params = run_all_checks(str(TMP_CHECK_RAW_PATH))
                        if image is not None:
                            image_cache = image
                    except Exception:
                        traceback.print_exc()
                else:
                    _throttle_log(log_state, f"pull 失败: {perr}")
            else:
                _throttle_log(log_state, f"trigger 失败: {err}")

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
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
