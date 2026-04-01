import os
import glob
import numpy as np
import cv2
import re
import subprocess


# 你的 raw 数据固定是 31*40*64 的 int16
RAW_ROWS = 31
RAW_COLS = 40
RAW_BINS = 64
TOTAL_COUNT = RAW_ROWS * RAW_COLS * RAW_BINS
K = 500.0
INPUT_FILE = "input.raw"
OUTPUT_FILE = "tof.pgm"


def convert():
    data = np.fromfile(INPUT_FILE, dtype=np.uint16)
    if data.size != TOTAL_COUNT:
        print("跳过(长度不对):", INPUT_FILE, "实际:", data.size, "期望:", TOTAL_COUNT)
        return

    data = data.reshape(RAW_ROWS, RAW_COLS, RAW_BINS).astype(np.float32)

    hist = data[1:, :, :62]
    sat = data[1:, :, 62] * 1024 + data[1:, :, 63]
    data = hist * 50000 / sat[:, :, None]


    img = data.mean(axis=2) / K * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)  # 形状: (30, 40)

    cv2.imwrite(OUTPUT_FILE, img)
    print("已输出:", OUTPUT_FILE)

def check_mtf():
    try:
        result = subprocess.run(
            ["./mtf.exe"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        return 0, 0, f"failed to run mtf.exe: {e}"

    output = (result.stdout or "") + "\n" + (result.stderr or "")

    # 提取清晰度值，优先匹配 "value = 0.7739" 这种格式
    clarity_value = None
    value_match = re.search(r"value\s*=\s*([0-9]*\.?[0-9]+)", output, re.IGNORECASE)
    if value_match:
        try:
            clarity_value = float(value_match.group(1))
        except ValueError:
            clarity_value = None

    if clarity_value is None:
        return 0, 0, "clarity value not found in mtf output"

    # GOOD 直接通过，原因为空字符串
    if "clarity is GOOD!" in output:
        return 1, clarity_value, "GOOD"

    # 特定报错优先返回
    if "The light panel is too bright" in output:
        return 0, clarity_value, "The light panel is too bright"

    # 解析条件行，如:
    # field 0.60  n =  2 >  1 value = 0.7739 > 0.50|
    cond_match = re.search(
        r"(n\s*=\s*([0-9]+)\s*>\s*([0-9]+)\s*value\s*=\s*([0-9]*\.?[0-9]+)\s*>\s*([0-9]*\.?[0-9]+)\|?)",
        output,
        re.IGNORECASE,
    )
    if not cond_match:
        return 0, clarity_value, "condition line not found: n = ... value = ..."

    raw_line = cond_match.group(1).strip()
    n_actual = int(cond_match.group(2))
    n_threshold = int(cond_match.group(3))
    value_actual = float(cond_match.group(4))
    value_threshold = float(cond_match.group(5))

    failed_reasons = []
    if n_actual <= n_threshold:
        failed_reasons.append(
            f"required at least {n_threshold + 1} MTF boxes, actual {n_actual}"
        )
    if value_actual <= value_threshold:
        failed_reasons.append(
            f"mtf value{value_actual:.4f} below threshold: {value_threshold:.2f}"
        )

    if failed_reasons:
        return 0, clarity_value, failed_reasons
    return 0, clarity_value, f"Unknown error, contract to developer"


if __name__ == "__main__":   # raw 文件目录
    convert()
    result, clarity_value, reason = check_mtf()
    print(f"result: {result}, clarity_value: {clarity_value}, reason: {reason}")
