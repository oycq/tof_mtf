import os
import glob
import numpy as np
import cv2


# 你的 raw 数据固定是 31*40*64 的 int16
RAW_ROWS = 31
RAW_COLS = 40
RAW_BINS = 64
TOTAL_COUNT = RAW_ROWS * RAW_COLS * RAW_BINS
K = 2500.0
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


if __name__ == "__main__":   # raw 文件目录
    convert()

