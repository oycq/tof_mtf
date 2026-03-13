import os
import glob
import numpy as np
import cv2


# 你的 raw 数据固定是 31*40*64 的 int16
RAW_ROWS = 31
RAW_COLS = 40
RAW_BINS = 64
TOTAL_COUNT = RAW_ROWS * RAW_COLS * RAW_BINS
K = 2
OUTPUT_FORMAT = "pgm"  # 可选: "pgm" / "bmp" / "png"


def convert_one(raw_file, out_dir, output_format):
    data = np.fromfile(raw_file, dtype=np.int16)
    if data.size != TOTAL_COUNT:
        print("跳过(长度不对):", raw_file, "实际:", data.size, "期望:", TOTAL_COUNT)
        return

    data = data.reshape(RAW_ROWS, RAW_COLS, RAW_BINS)

    # 第一行不要，只要后 30 行；64bin 中只要前 62 个
    data = data[1:, :, :62]  # 形状: (30, 40, 62)

    # 亮度: 62bin 均值 /1024 *255 *K，转 8bit
    img = data.mean(axis=2) / 1024.0 * 255.0 * K
    img = np.clip(img, 0, 255).astype(np.uint8)  # 形状: (30, 40)

    name = os.path.splitext(os.path.basename(raw_file))[0]
    out_file = os.path.join(out_dir, name + "." + output_format)
    cv2.imwrite(out_file, img)
    print("已输出:", out_file)


if __name__ == "__main__":
    input_dir = "raw"        # raw 文件目录
    output_format = OUTPUT_FORMAT.lower()
    if output_format not in ["pgm", "bmp", "png"]:
        raise ValueError("OUTPUT_FORMAT 只能是 pgm / bmp / png")
    output_dir = output_format  # 根据格式自动输出到同名目录
    os.makedirs(output_dir, exist_ok=True)

    raw_list = sorted(glob.glob(os.path.join(input_dir, "*.raw")))
    if len(raw_list) == 0:
        print(f"{input_dir} 目录没找到 .raw 文件")
    else:
        for raw_file in raw_list:
            convert_one(raw_file, output_dir, output_format)

