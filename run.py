import os
import re
import shutil
import subprocess
import sys

import cv2
import numpy as np


RAW_ROWS = 30
RAW_COLS = 40
RAW_BINS = 64
K = 2500.0

BOTTOM_MM = 300.0
H_MM = 240.0
TOP_MM = BOTTOM_MM - H_MM * np.tan(np.deg2rad(10.0)) * 2
HFOV_DEG = 52.0

ANGLE_ABS_DEG_MAX = 10.0
XY_ABS_MM_MAX = 20.0
Z_MM_MIN = 400.0
Z_MM_MAX = 500.0


def _convert_raw_to_pgm(raw_path, pgm_path):
    raw_rows_with_header = RAW_ROWS + 1
    total_count = RAW_ROWS * RAW_COLS * RAW_BINS
    total_count_with_header = raw_rows_with_header * RAW_COLS * RAW_BINS

    data = np.fromfile(raw_path, dtype=np.uint16)
    if data.size not in (total_count, total_count_with_header):
        raise ValueError(
            "raw长度不对: "
            f"actual={data.size}, "
            f"expected={total_count} or {total_count_with_header}, "
            f"path={raw_path}"
        )

    rows = RAW_ROWS if data.size == total_count else raw_rows_with_header
    data = data.reshape(rows, RAW_COLS, RAW_BINS).astype(np.float32)
    # 兼容有头/无头：统一只取最后 30x40 区域。
    data = data[-RAW_ROWS:, :, :]

    hist = data[:, :, :62]
    sat = data[:, :, 62] * 1024 + data[:, :, 63]
    depth_like = hist * 50000 / sat[:, :, None]

    img = depth_like.mean(axis=2) / K * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(pgm_path, img)
    if not ok:
        raise RuntimeError(f"写入pgm失败: {pgm_path}")


def _check_mtf_with_exe(mtf_exe_path, work_dir):
    try:
        old_cwd = os.getcwd()
        os.chdir(work_dir)
        result = subprocess.run(
            [os.path.basename(mtf_exe_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        return 0, 0.0, f"failed to run mtf.exe: {e}"
    finally:
        # 不论成功失败，都回到调用前目录（run.py 所在目录）。
        os.chdir(old_cwd)

    output = (result.stdout or "") + "\n" + (result.stderr or "")

    clarity_value = None
    value_match = re.search(r"value\s*=\s*([0-9]*\.?[0-9]+)", output, re.IGNORECASE)
    if value_match:
        try:
            clarity_value = float(value_match.group(1))
        except ValueError:
            clarity_value = None

    if clarity_value is None:
        return 0, 0.0, "clarity value not found in mtf output"

    if "clarity is GOOD!" in output:
        return 1, clarity_value, "GOOD"

    if "The light panel is too bright" in output:
        return 0, clarity_value, "The light panel is too bright"

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

    failed_reasons = []
    if n_actual <= n_threshold:
        failed_reasons.append(
            f"required at least {n_threshold + 1} MTF boxes, actual {n_actual}"
        )
    if value_actual <= value_threshold:
        failed_reasons.append(
            f"mtf value {value_actual:.4f} below threshold: {value_threshold:.2f}"
        )

    if failed_reasons:
        return 0, clarity_value, failed_reasons
    return 0, clarity_value, "Unknown error, contact developer"


def _prepare_mtf_runtime(script_dir, output_dir):
    # 只复制 exe/config 到 output 目录，不修改 config.ini 内容。
    src_exe = os.path.join(script_dir, "mtf.exe")
    src_cfg = os.path.join(script_dir, "config.ini")
    dst_exe = os.path.join(output_dir, "mtf.exe")
    dst_cfg = os.path.join(output_dir, "config.ini")
    shutil.copyfile(src_exe, dst_exe)
    shutil.copyfile(src_cfg, dst_cfg)


class TiltChecker:
    """Tilt 检查器：封装角点提取、PnP、阈值判断和可视化输出。"""

    def __init__(self):
        self.angle_abs_deg_max = ANGLE_ABS_DEG_MAX
        self.xy_abs_mm_max = XY_ABS_MM_MAX
        self.z_mm_min = Z_MM_MIN
        self.z_mm_max = Z_MM_MAX
        self.nan_values = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    def run_from_pgm(self, pgm_path, tilt_output_path):
        try:
            img = cv2.imread(pgm_path, cv2.IMREAD_UNCHANGED)
            gray_f32 = self._to_gray_float(img)
            gray_up, scale_x, scale_y = self._upscale_to_300x400(gray_f32)
            vis = self._build_display_image(gray_up)

            results = self._detect_strongest_corners_per_region(gray_f32)
            self._draw_results(vis, results, scale_x, scale_y)
            pose = self._solve_pose(results, gray_f32.shape[1], gray_f32.shape[0])
            self._draw_pose_origin_and_axes(
                vis,
                pose,
                width=gray_f32.shape[1],
                height=gray_f32.shape[0],
                scale_x=scale_x,
                scale_y=scale_y,
            )
            passed, tilt_values, tilt_reason = self._check_tilt_and_extract_values(
                gray_f32, results, pose
            )
            display = self._compose_display_with_header(vis, pose)

            ok = cv2.imwrite(tilt_output_path, display)
            if not ok:
                raise RuntimeError(f"写入倾斜结果图失败: {tilt_output_path}")
            return passed, tilt_values, display, tilt_reason
        except Exception as e:
            # 兜底图，避免异常时上层流程中断。
            fallback = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(
                fallback,
                "tilt failed",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(tilt_output_path, fallback)
            return False, self.nan_values, fallback, f"tilt exception: {e}"

    def _to_gray_float(self, image):
        if image is None:
            raise ValueError("读取图像失败，请检查输入路径")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.astype(np.float32)

    def _build_display_image(self, gray):
        # 对齐 MTF 的亮度显示方式：使用固定 0~255 映射，不做逐帧 min-max 拉伸。
        disp = np.clip(gray, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

    def _upscale_to_300x400(self, gray_f32):
        h, w = gray_f32.shape
        if (h, w) == (30, 40):
            up = cv2.resize(gray_f32, (400, 300), interpolation=cv2.INTER_NEAREST)
            return up, 10.0, 10.0
        up = cv2.resize(gray_f32, None, fx=10.0, fy=10.0, interpolation=cv2.INTER_NEAREST)
        return up, 10.0, 10.0

    def _detect_strongest_corners_per_region(self, gray_f32):
        h, w = gray_f32.shape
        regions = [
            ("top-left", (0, 0, w // 2, h // 2)),
            ("top-right", (w // 2, 0, w, h // 2)),
            ("bottom-right", (w // 2, h // 2, w, h)),
            ("bottom-left", (0, h // 2, w // 2, h)),
        ]
        initial_points = []
        found_regions = []

        for name, (x0, y0, x1, y1) in regions:
            roi = gray_f32[y0:y1, x0:x1]
            corners = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=1,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=3,
                useHarrisDetector=False,
            )
            if corners is None:
                found_regions.append((name, (x0, y0, x1, y1), None))
                continue
            pt = corners[0, 0]
            pt_global = np.array([pt[0] + x0, pt[1] + y0], dtype=np.float32)
            initial_points.append(pt_global)
            found_regions.append((name, (x0, y0, x1, y1), pt_global.copy()))

        if initial_points:
            pts = np.array(initial_points, dtype=np.float32).reshape(-1, 1, 2)
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                40,
                1e-3,
            )
            cv2.cornerSubPix(gray_f32, pts, (3, 3), (-1, -1), criteria)
            refined = [p[0].copy() for p in pts]
        else:
            refined = []

        refined_idx = 0
        results = []
        for name, rect, init_pt in found_regions:
            if init_pt is None:
                results.append((name, rect, None, None))
            else:
                results.append((name, rect, init_pt, refined[refined_idx]))
                refined_idx += 1
        return results

    def _draw_results(self, vis, results, scale_x, scale_y):
        for _name, _rect, _init_pt, refined_pt in results:
            if refined_pt is None:
                continue
            x_up = float(refined_pt[0]) * scale_x
            y_up = float(refined_pt[1]) * scale_y
            xi = int(round(x_up))
            yi = int(round(y_up))
            if 0 <= xi < vis.shape[1] and 0 <= yi < vis.shape[0]:
                r = 3
                xs = max(0, xi - r)
                ys = max(0, yi - r)
                xe = min(vis.shape[1], xi + r + 1)
                ye = min(vis.shape[0], yi + r + 1)
                vis[ys:ye, xs:xe] = (0, 0, 255)

    def _draw_pose_origin_and_axes(self, vis, pose, width, height, scale_x, scale_y):
        if pose is None:
            return
        rvec = pose.get("rvec")
        tvec = pose.get("tvec")
        if rvec is None or tvec is None:
            return

        camera_matrix = self._build_camera_matrix(width, height)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        axis_len_mm = 60.0
        # O 为梯形中心原点，X/Y 为平面内轴，Z 为中心旋转轴（法向）。
        obj_pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_len_mm, 0.0, 0.0],
                [0.0, axis_len_mm, 0.0],
                [0.0, 0.0, axis_len_mm],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 3)
        img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
        if img_pts is None or img_pts.shape[0] < 4:
            return

        pts = img_pts.reshape(-1, 2).astype(np.float32)

        def _to_up(pt):
            return (
                int(round(float(pt[0]) * float(scale_x))),
                int(round(float(pt[1]) * float(scale_y))),
            )

        o = _to_up(pts[0])
        x_end = _to_up(pts[1])
        y_end = _to_up(pts[2])
        z_end = _to_up(pts[3])

        def _label_pos(origin, end, along=14.0, ortho=0.0):
            vec = np.array([float(end[0] - origin[0]), float(end[1] - origin[1])], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm < 1e-6:
                return int(end[0] + 8), int(end[1] - 8)
            unit = vec / norm
            perp = np.array([-unit[1], unit[0]], dtype=np.float32)
            p = np.array([float(end[0]), float(end[1])], dtype=np.float32) + unit * float(along) + perp * float(ortho)
            px = int(np.clip(round(float(p[0])), 0, vis.shape[1] - 1))
            py = int(np.clip(round(float(p[1])), 0, vis.shape[0] - 1))
            return px, py

        cv2.circle(vis, o, 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(vis, o, x_end, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(vis, o, y_end, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(vis, o, z_end, (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.2)
        x_text = _label_pos(o, x_end, along=14.0, ortho=-6.0)
        y_text = _label_pos(o, y_end, along=14.0, ortho=6.0)
        z_text = _label_pos(o, z_end, along=18.0, ortho=0.0)
        cv2.putText(vis, "X", x_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "Y", y_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, "Z", z_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 0, 0), 1, cv2.LINE_AA)

    def _get_image_points(self, results):
        point_map = {}
        for name, _rect, _init_pt, refined_pt in results:
            if refined_pt is not None:
                point_map[name] = refined_pt

        ordered_names = ["top-left", "top-right", "bottom-right", "bottom-left"]
        if not all(name in point_map for name in ordered_names):
            return None
        img_pts = np.array([point_map[name] for name in ordered_names], dtype=np.float32)
        return img_pts.reshape(-1, 1, 2)

    def _build_trapezoid_object_points(self):
        half_bottom = BOTTOM_MM * 0.5
        half_top = TOP_MM * 0.5
        half_h = H_MM * 0.5
        return np.array(
            [
                [-half_top, -half_h, 0.0],
                [half_top, -half_h, 0.0],
                [half_bottom, half_h, 0.0],
                [-half_bottom, half_h, 0.0],
            ],
            dtype=np.float32,
        )

    def _build_camera_matrix(self, width, height):
        hfov_rad = np.deg2rad(HFOV_DEG)
        fx = (width * 0.5) / np.tan(hfov_rad * 0.5)
        fy = fx
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
        return np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _rotation_matrix_to_euler_zyx_deg(self, rot_mat):
        sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            roll_x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
            pitch_y = np.arctan2(-rot_mat[2, 0], sy)
            yaw_z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        else:
            roll_x = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
            pitch_y = np.arctan2(-rot_mat[2, 0], sy)
            yaw_z = 0.0
        return (
            float(np.rad2deg(roll_x)),
            float(np.rad2deg(pitch_y)),
            float(np.rad2deg(yaw_z)),
        )

    def _solve_pose(self, results, width, height):
        img_pts = self._get_image_points(results)
        if img_pts is None:
            return None
        obj_pts = self._build_trapezoid_object_points()
        camera_matrix = self._build_camera_matrix(width, height)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not success:
            return None
        rot_mat, _ = cv2.Rodrigues(rvec)
        roll_deg, pitch_deg, yaw_deg = self._rotation_matrix_to_euler_zyx_deg(rot_mat)
        return {
            "roll_deg": float(roll_deg),
            "pitch_deg": float(pitch_deg),
            "yaw_deg": float(yaw_deg),
            "tx_mm": float(tvec[0, 0]),
            "ty_mm": float(tvec[1, 0]),
            "tz_mm": float(tvec[2, 0]),
            "rvec": rvec.astype(np.float64, copy=True),
            "tvec": tvec.astype(np.float64, copy=True),
        }

    def _check_tilt_and_extract_values(self, gray_f32, results, pose):
        detected_points = sum(1 for _n, _r, _i, p in results if p is not None)
        if detected_points < 4:
            return False, self.nan_values, f"tilt corner extraction failed: need 4 points, got {detected_points}"
        if pose is None:
            return False, self.nan_values, "tilt pnp failed"

        roll_deg = float(pose["roll_deg"])
        pitch_deg = float(pose["pitch_deg"])
        yaw_deg = float(pose["yaw_deg"])
        tx_mm = float(pose["tx_mm"])
        ty_mm = float(pose["ty_mm"])
        tz_mm = float(pose["tz_mm"])
        values = [roll_deg, pitch_deg, yaw_deg, tx_mm, ty_mm, tz_mm]

        passed = (
            abs(roll_deg) <= self.angle_abs_deg_max
            and abs(pitch_deg) <= self.angle_abs_deg_max
            and abs(yaw_deg) <= self.angle_abs_deg_max
            and abs(tx_mm) <= self.xy_abs_mm_max
            and abs(ty_mm) <= self.xy_abs_mm_max
            and self.z_mm_min <= tz_mm <= self.z_mm_max
        )
        if passed:
            return True, values, "GOOD"

        fail_reasons = []
        if abs(roll_deg) > self.angle_abs_deg_max:
            fail_reasons.append(
                f"roll out of range: {roll_deg:.2f} (limit +/-{self.angle_abs_deg_max:.2f})"
            )
        if abs(pitch_deg) > self.angle_abs_deg_max:
            fail_reasons.append(
                f"pitch out of range: {pitch_deg:.2f} (limit +/-{self.angle_abs_deg_max:.2f})"
            )
        if abs(yaw_deg) > self.angle_abs_deg_max:
            fail_reasons.append(
                f"yaw out of range: {yaw_deg:.2f} (limit +/-{self.angle_abs_deg_max:.2f})"
            )
        if abs(tx_mm) > self.xy_abs_mm_max:
            fail_reasons.append(
                f"tx out of range: {tx_mm:.2f} (limit +/-{self.xy_abs_mm_max:.2f})"
            )
        if abs(ty_mm) > self.xy_abs_mm_max:
            fail_reasons.append(
                f"ty out of range: {ty_mm:.2f} (limit +/-{self.xy_abs_mm_max:.2f})"
            )
        if not (self.z_mm_min <= tz_mm <= self.z_mm_max):
            fail_reasons.append(
                f"tz out of range: {tz_mm:.2f} (limit [{self.z_mm_min:.2f}, {self.z_mm_max:.2f}])"
            )
        return False, values, "; ".join(fail_reasons) if fail_reasons else "tilt check failed"

    def _compose_display_with_header(self, vis, pose):
        # header 上展示关键姿态值 + 坐标方向示意，便于定位异常轴。
        threshold_lines = [
            f"angle_limit: [{-self.angle_abs_deg_max:.1f}, {self.angle_abs_deg_max:.1f}]",
            f"xy_limit   : [{-self.xy_abs_mm_max:.1f}, {self.xy_abs_mm_max:.1f}]",
            f"z_range    : [{int(self.z_mm_min)}, {int(self.z_mm_max)}]",
        ]
        axis_lines = [
            "roll: around +X",
            "pitch: around +Y",
            "yaw: around +Z",
        ]
        if pose is None:
            value_lines = [
                "roll_deg :      nan",
                "pitch_deg:      nan",
                "yaw_deg  :      nan",
                "tx_mm    :      nan",
                "ty_mm    :      nan",
                "tz_mm    :      nan",
            ]
            value_colors = [(255, 255, 255)] * len(value_lines)
        else:
            roll = pose["roll_deg"]
            pitch = pose["pitch_deg"]
            yaw = pose["yaw_deg"]
            tx = pose["tx_mm"]
            ty = pose["ty_mm"]
            tz = pose["tz_mm"]
            value_lines = [
                f"{'roll_deg':<8}: {roll:>8.2f}",
                f"{'pitch_deg':<8}: {pitch:>8.2f}",
                f"{'yaw_deg':<8}: {yaw:>8.2f}",
                f"{'tx_mm':<8}: {tx:>8.2f}",
                f"{'ty_mm':<8}: {ty:>8.2f}",
                f"{'tz_mm':<8}: {tz:>8.2f}",
            ]
            def _color(ok):
                return (255, 255, 255) if ok else (0, 0, 255)
            value_colors = [
                _color(abs(roll) <= self.angle_abs_deg_max),
                _color(abs(pitch) <= self.angle_abs_deg_max),
                _color(abs(yaw) <= self.angle_abs_deg_max),
                _color(abs(tx) <= self.xy_abs_mm_max),
                _color(abs(ty) <= self.xy_abs_mm_max),
                _color(self.z_mm_min <= tz <= self.z_mm_max),
            ]

        lines = value_lines + threshold_lines + axis_lines
        line_colors = value_colors + [(255, 255, 255)] * (len(threshold_lines) + len(axis_lines))
        line_h = 20
        top_pad = 8
        bottom_pad = 8
        header_h = max(top_pad + len(lines) * line_h + bottom_pad, 260)
        header = np.zeros((header_h, vis.shape[1], 3), dtype=np.uint8)

        y = top_pad + 14
        for idx, line in enumerate(lines):
            cv2.putText(
                header,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                line_colors[idx],
                1,
                cv2.LINE_AA,
            )
            y += line_h

        return np.vstack([header, vis])


def _run_tilt_from_pgm(pgm_path, tilt_output_path):
    checker = TiltChecker()
    return checker.run_from_pgm(pgm_path, tilt_output_path)


def run_all_checks(tof_raw_path, output_dir="output"):
    """
    输入 tof.raw 路径，返回：
    1) mtf 是否 pass、mtf 值、报错原因、output.bmp 的 numpy 图像
    2) 倾斜检查是否 pass、6 个值、输出图像 numpy
    """
    original_cwd = os.getcwd()
    launch_cwd = original_cwd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_tof_raw_path = (
        tof_raw_path if os.path.isabs(tof_raw_path) else os.path.abspath(os.path.join(launch_cwd, tof_raw_path))
    )
    abs_output_dir = (
        output_dir if os.path.isabs(output_dir) else os.path.abspath(os.path.join(launch_cwd, output_dir))
    )

    if not os.path.isfile(abs_tof_raw_path):
        raise FileNotFoundError(f"找不到输入raw: {abs_tof_raw_path}")

    try:
        # 第一件事先切到当前脚本目录，确保 mtf.exe/config.ini 等相对路径稳定。
        os.chdir(script_dir)

        os.makedirs(abs_output_dir, exist_ok=True)

        output_raw_path = os.path.join(abs_output_dir, "tof.raw")
        output_pgm_path = os.path.join(abs_output_dir, "tof.pgm")
        output_mtf_bmp_path = os.path.join(abs_output_dir, "output.bmp")
        output_tilt_bmp_path = os.path.join(abs_output_dir, "tilt.bmp")

        # mtf.exe 需要实体 raw/pgm 文件，统一放到 output 目录。
        shutil.copyfile(abs_tof_raw_path, output_raw_path)
        _convert_raw_to_pgm(output_raw_path, output_pgm_path)

        # 把 mtf.exe 和 config.ini 放到 output 目录，在 output 目录内执行。
        _prepare_mtf_runtime(script_dir, abs_output_dir)
        mtf_exe_path = os.path.join(abs_output_dir, "mtf.exe")
        mtf_pass, mtf_value, mtf_reason = _check_mtf_with_exe(
            mtf_exe_path, abs_output_dir
        )
        mtf_img = cv2.imread(output_mtf_bmp_path, cv2.IMREAD_UNCHANGED)
        if mtf_img is None:
            output_mtf_bmp_path_alt = os.path.join(abs_output_dir, "output", "output.bmp")
            mtf_img = cv2.imread(output_mtf_bmp_path_alt, cv2.IMREAD_UNCHANGED)

        tilt_pass, tilt_values, tilt_img, tilt_reason = _run_tilt_from_pgm(
            output_pgm_path, output_tilt_bmp_path
        )

        return {
            "mtf": {
                "pass": bool(mtf_pass),
                "value": float(mtf_value) if mtf_value is not None else None,
                "reason": mtf_reason,
                "image": mtf_img,
            },
            "tilt": {
                "pass": bool(tilt_pass),
                "values": tilt_values,
                "reason": tilt_reason,
                "image": tilt_img,
            },
        }
    finally:
        # 最后切回原始目录（push 回去）。
        os.chdir(original_cwd)


if __name__ == "__main__":
    default_raw = "input.raw" if os.path.isfile("input.raw") else "tof.raw"
    args = sys.argv[1:]
    tof_raw_path = args[0] if len(args) >= 1 else default_raw
    output_dir = args[1] if len(args) >= 2 else "output"

    result = run_all_checks(tof_raw_path, output_dir=output_dir)

    mtf_pass = result["mtf"]["pass"]
    mtf_value = result["mtf"]["value"]
    mtf_reason = result["mtf"]["reason"]
    mtf_img = result["mtf"]["image"]

    tilt_pass = result["tilt"]["pass"]
    tilt_reason = result["tilt"]["reason"]
    roll_deg, pitch_deg, yaw_deg, tx_mm, ty_mm, tz_mm = result["tilt"]["values"]
    tilt_img = result["tilt"]["image"]

    print("=== MTF ===")
    print(f"pass: {mtf_pass}")
    print(f"value: {mtf_value}")
    print(f"reason: {mtf_reason}")
    print("=== TILT ===")
    print(f"pass: {tilt_pass}")
    print(f"reason: {tilt_reason}")
    print(
        f"roll={roll_deg:.2f}, pitch={pitch_deg:.2f}, yaw={yaw_deg:.2f}, "
        f"tx={tx_mm:.2f}, ty={ty_mm:.2f}, tz={tz_mm:.2f}"
    )
    if mtf_img is not None:
        cv2.imshow("MTF output.bmp", mtf_img)
    if tilt_img is not None:
        cv2.imshow("Tilt output", tilt_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

