"""命令行入口：调用 tof_mtf 包，显示拼接结果图。"""

import sys

import cv2

from tof_mtf import run_all_checks


def main():
    args = sys.argv[1:]
    tof_raw_path = args[0] if len(args) >= 1 else "tof.raw"

    passed, image, params = run_all_checks(tof_raw_path)
    mtf_value, roll, pitch, yaw, tx, ty, tz, bright_mean, dirt_mean = params

    print(f"pass : {passed}")
    print(f"mtf  : {mtf_value:.4f}")
    print(
        f"tilt : roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}, "
        f"tx={tx:.2f}, ty={ty:.2f}, tz={tz:.2f}"
    )
    print(f"img  : bright_top20={bright_mean:.1f}, dark_bottom20={dirt_mean:.1f}")

    if image is not None:
        cv2.imshow("tof_mtf", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
