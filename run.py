"""命令行入口：调用 tof_mtf 包，显示 MTF / Tilt 两张结果图。"""

import sys

import cv2

from tof_mtf import run_all_checks


def main():
    args = sys.argv[1:]
    tof_raw_path = args[0] if len(args) >= 1 else "tof.raw"

    result = run_all_checks(tof_raw_path)

    mtf = result["mtf"]
    tilt = result["tilt"]
    roll_deg, pitch_deg, yaw_deg, tx_mm, ty_mm, tz_mm = tilt["values"]

    print("=== MTF ===")
    print(f"pass: {mtf['pass']}")
    print(f"value: {mtf['value']}")
    print(f"reason: {mtf['reason']}")
    print("=== TILT ===")
    print(f"pass: {tilt['pass']}")
    print(f"reason: {tilt['reason']}")
    print(
        f"roll={roll_deg:.2f}, pitch={pitch_deg:.2f}, yaw={yaw_deg:.2f}, "
        f"tx={tx_mm:.2f}, ty={ty_mm:.2f}, tz={tz_mm:.2f}"
    )

    if mtf["image"] is not None:
        cv2.imshow("MTF output.bmp", mtf["image"])
    if tilt["image"] is not None:
        cv2.imshow("Tilt output", tilt["image"])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
