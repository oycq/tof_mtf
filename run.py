"""命令行入口：调用 tof_mtf 包，显示拼接结果图。"""

import sys

import cv2

from tof_mtf import run_all_checks


def main():
    tof_raw_path = sys.argv[1]

    passed, image, params = run_all_checks(tof_raw_path)

    cv2.imshow("tof_mtf", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
