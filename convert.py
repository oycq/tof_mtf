import os
import glob
import numpy as np
import cv2
import re
import subprocess


# 原始数据固定为 31*40*64 个 uint16 点位
# 其中前 62 个 bin 是直方图能量，后 2 个 bin 用于拼接饱和参考值 sat
RAW_ROWS = 31
RAW_COLS = 40
RAW_BINS = 64
TOTAL_COUNT = RAW_ROWS * RAW_COLS * RAW_BINS
# K 用于把深度/强度映射到 0~255 灰度范围
K = 2500.0
INPUT_FILE = "input.raw"
OUTPUT_FILE = "tof.pgm"


def convert():
    # 读取原始二进制数据，按 uint16 解释
    data = np.fromfile(INPUT_FILE, dtype=np.uint16)
    # 长度检查：防止输入文件尺寸异常导致后续 reshape 崩溃
    if data.size != TOTAL_COUNT:
        print("跳过(长度不对):", INPUT_FILE, "实际:", data.size, "期望:", TOTAL_COUNT)
        return

    data = data.reshape(RAW_ROWS, RAW_COLS, RAW_BINS).astype(np.float32)

    # 丢弃第 0 行（通常是无效行/校准行），只保留有效成像区域
    hist = data[1:, :, :62]
    # sat 由 bin62 和 bin63 拼成 20bit 量级参考值（高位*1024 + 低位）
    sat = data[1:, :, 62] * 1024 + data[1:, :, 63]

    data = hist * 50000 / sat[:, :, None]

    # 对 62 个 bin 取均值，映射为 8bit 灰度图
    img = data.mean(axis=2) / K * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    cv2.imwrite(OUTPUT_FILE, img)
    print("已输出:", OUTPUT_FILE)

def check_mtf():
    # 调用外部 mtf 工具并抓取 stdout/stderr，统一做文本解析
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

    # 提取清晰度值，匹配 "value = 0.7739" 这种格式
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
    # 含义：
    # - n_actual 必须 > n_threshold
    # - value_actual 必须 > value_threshold
    cond_match = re.search(
        r"(n\s*=\s*([0-9]+)\s*>\s*([0-9]+)\s*value\s*=\s*([0-9]*\.?[0-9]+)\s*>\s*([0-9]*\.?[0-9]+)\|?)",
        output,
        re.IGNORECASE,
    )
    if not cond_match:
        return 0, clarity_value, "condition line not found: n = ... value = ..."

    n_actual = int(cond_match.group(2))
    n_threshold = int(cond_match.group(3))
    value_actual = float(cond_match.group(4))
    value_threshold = float(cond_match.group(5))

    # 收集所有失败原因，便于排查（一次返回多个条件失败信息）
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
    # 兜底：出现未覆盖分支时给开发者提示
    return 0, clarity_value, f"Unknown error, contract to developer"


if __name__ == "__main__":   # 入口：先产出 pgm，再执行 MTF 判定
    convert()
    result, clarity_value, reason = check_mtf()
    print(f"result: {result}, clarity_value: {clarity_value}, reason: {reason}")
