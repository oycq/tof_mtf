import py_compile
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RELEASE_DIR = ROOT / "release"
MODULE_DIR = RELEASE_DIR / "tof_mtf"


def _clean_package_dir():
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    MODULE_DIR.mkdir(parents=True, exist_ok=True)


def _copy_required_files():
    run_py = ROOT / "run.py"
    tof_raw = ROOT / "tof.raw"
    config_ini = ROOT / "config.ini"
    mtf_exe = ROOT / "mtf.exe"

    required = [run_py, tof_raw, config_ini, mtf_exe]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少打包文件: " + ", ".join(missing))

    # 打包为 run.pyc，不携带 run.py 源码
    py_compile.compile(str(run_py), cfile=str(MODULE_DIR / "run.pyc"), doraise=True)

    shutil.copy2(tof_raw, RELEASE_DIR / "tof.raw")
    shutil.copy2(config_ini, MODULE_DIR / "config.ini")
    shutil.copy2(mtf_exe, MODULE_DIR / "mtf.exe")


def _write_init():
    init_text = '''"""
tof_mtf 对外只暴露一个主入口函数: run_all_checks

用法:
    from tof_mtf import run_all_checks
    result = run_all_checks(input_path, output_dir="output")

输入:
- input_path (str): tof.raw 路径
- output_dir (str): 输出目录，默认 "output"

输出:
- 返回 dict，结构如下:
  {
      "mtf": {
          "pass": bool,          # MTF 是否通过
          "value": float|None,   # MTF 值
          "reason": str|list,    # 失败原因或 "GOOD"
          "image": numpy.ndarray # output.bmp 图像
      },
      "tilt": {
          "pass": bool,          # 倾斜检查是否通过
          "values": list[float], # [roll, pitch, yaw, tx, ty, tz]
          "reason": str,         # 倾斜失败原因或 "GOOD"
          "image": numpy.ndarray # tilt.bmp 图像
      }
  }
"""

from .run import run_all_checks

__all__ = ["run_all_checks"]
'''
    (MODULE_DIR / "__init__.py").write_text(init_text, encoding="utf-8")


def _write_demo():
    demo_text = """import cv2
from tof_mtf import run_all_checks

# 直接用相对路径
raw_path = "tof.raw"
output_dir = "output"

res = run_all_checks(raw_path, output_dir=output_dir)

# 打印所有关键结果
print("=== MTF ===")
print("pass :", res["mtf"]["pass"])
print("value:", res["mtf"]["value"])
print("reason:", res["mtf"]["reason"])

print("=== TILT ===")
print("pass :", res["tilt"]["pass"])
print("reason:", res["tilt"]["reason"])
print("values:", res["tilt"]["values"])  # [roll, pitch, yaw, tx, ty, tz]

# 显示图像
cv2.imshow("MTF output.bmp", res["mtf"]["image"])
cv2.imshow("Tilt output.bmp", res["tilt"]["image"])
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
    (RELEASE_DIR / "demo.py").write_text(demo_text, encoding="utf-8")


def main():
    _clean_package_dir()
    _copy_required_files()
    _write_init()
    _write_demo()
    print(f"打包完成: {RELEASE_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"打包失败: {e}")
        sys.exit(1)

