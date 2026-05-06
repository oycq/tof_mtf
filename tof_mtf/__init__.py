"""
tof_mtf
=======

一个开箱即用的 ToF MTF + 倾斜检测包。
该包内自带 ``mtf.exe`` 和 ``config.ini``，可在任意目录被 ``import``。

用法::

    from tof_mtf import run_all_checks

    # tof.raw 路径相对于"调用时 Python 的当前工作目录"
    # 所有中间产物会落到 tof_mtf/tmp/ 内，不污染调用方目录
    result = run_all_checks("tof.raw")

返回结构详见 :func:`run_all_checks`。
"""

from ._runner import run_all_checks

__all__ = ["run_all_checks"]
