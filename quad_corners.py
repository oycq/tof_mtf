import argparse
import cv2
import numpy as np


BOTTOM_MM = 300.0
H_MM = 240.0
TOP_MM = BOTTOM_MM - H_MM * np.tan(np.deg2rad(10.0))
HFOV_DEG = 52.0


def to_gray_float(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Failed to read image. Check input path.")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32)


def build_display_image(gray: np.ndarray) -> np.ndarray:
    disp = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    disp = disp.astype(np.uint8)
    return cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)


def upscale_to_300x400(gray_f32: np.ndarray):
    h, w = gray_f32.shape
    if (h, w) == (30, 40):
        up = cv2.resize(gray_f32, (400, 300), interpolation=cv2.INTER_NEAREST)
        return up, 10.0, 10.0
    # If input is not 30x40, still upscale by 10x for display.
    up = cv2.resize(gray_f32, None, fx=10.0, fy=10.0, interpolation=cv2.INTER_NEAREST)
    return up, 10.0, 10.0


def detect_strongest_corners_per_region(gray_f32: np.ndarray):
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
            maxCorners=1,      # one strongest corner per region
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


def draw_results(vis: np.ndarray, results, scale_x: float, scale_y: float):
    for name, (x0, y0, x1, y1), init_pt, refined_pt in results:
        if init_pt is not None:
            # Refine on original image, then map by scale for display.
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


def get_image_points(results):
    point_map = {}
    for name, _rect, _init_pt, refined_pt in results:
        if refined_pt is not None:
            point_map[name] = refined_pt

    ordered_names = ["top-left", "top-right", "bottom-right", "bottom-left"]
    if not all(name in point_map for name in ordered_names):
        return None

    img_pts = np.array([point_map[name] for name in ordered_names], dtype=np.float32)
    return img_pts.reshape(-1, 1, 2)


def build_trapezoid_object_points():
    half_bottom = BOTTOM_MM * 0.5
    half_top = TOP_MM * 0.5
    half_h = H_MM * 0.5
    # Keep board center at origin. Z=0 means board lies on one plane.
    obj_pts = np.array(
        [
            [-half_top, -half_h, 0.0],       # top-left
            [half_top, -half_h, 0.0],        # top-right
            [half_bottom, half_h, 0.0],      # bottom-right
            [-half_bottom, half_h, 0.0],     # bottom-left
        ],
        dtype=np.float32,
    )
    return obj_pts


def build_camera_matrix(width: int, height: int):
    hfov_rad = np.deg2rad(HFOV_DEG)
    fx = (width * 0.5) / np.tan(hfov_rad * 0.5)
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    k = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return k


def rotation_matrix_to_euler_zyx_deg(rot_mat: np.ndarray):
    # ZYX convention: yaw(Z), pitch(Y), roll(X)
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


def solve_pose(results, width: int, height: int):
    img_pts = get_image_points(results)
    if img_pts is None:
        return None

    obj_pts = build_trapezoid_object_points()
    camera_matrix = build_camera_matrix(width, height)
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
    roll_deg, pitch_deg, yaw_deg = rotation_matrix_to_euler_zyx_deg(rot_mat)

    tx_mm = float(tvec[0, 0])
    ty_mm = float(tvec[1, 0])
    tz_mm = float(tvec[2, 0])
    distance_mm = float(np.linalg.norm(tvec))
    return {
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "yaw_deg": yaw_deg,
        "tx_mm": tx_mm,
        "ty_mm": ty_mm,
        "tz_mm": tz_mm,
        "distance_mm": distance_mm,
        "camera_matrix": camera_matrix,
    }


def compose_display_with_header(vis: np.ndarray, pose):
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
    else:
        value_lines = [
            f"{'roll_deg':<8}: {pose['roll_deg']:>8.2f}",
            f"{'pitch_deg':<8}: {pose['pitch_deg']:>8.2f}",
            f"{'yaw_deg':<8}: {pose['yaw_deg']:>8.2f}",
            f"{'tx_mm':<8}: {pose['tx_mm']:>8.2f}",
            f"{'ty_mm':<8}: {pose['ty_mm']:>8.2f}",
            f"{'tz_mm':<8}: {pose['tz_mm']:>8.2f}",
        ]

    lines = value_lines + axis_lines
    line_h = 20
    top_pad = 8
    bottom_pad = 8
    header_h = max(top_pad + len(lines) * line_h + bottom_pad, 210)
    header = np.zeros((header_h, vis.shape[1], 3), dtype=np.uint8)

    y = top_pad + 14
    for line in lines:
        cv2.putText(
            header,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_h

    # Coordinate/rotation legend in the header (pose of trapezoid w.r.t camera).
    ox = vis.shape[1] - 95
    oy = header_h // 2
    arrow_len = 45
    cv2.arrowedLine(
        header,
        (ox, oy),
        (ox + arrow_len, oy),
        (255, 255, 255),
        1,
        cv2.LINE_AA,
        tipLength=0.2,
    )
    cv2.arrowedLine(
        header,
        (ox, oy),
        (ox, oy + arrow_len),
        (255, 255, 255),
        1,
        cv2.LINE_AA,
        tipLength=0.2,
    )
    cv2.putText(
        header,
        "+X",
        (ox + arrow_len + 6, oy + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "+Y",
        (ox - 4, oy + arrow_len + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    # +Z with an oblique arrow style.
    cv2.arrowedLine(
        header,
        (ox, oy),
        (ox + 30, oy - 30),
        (255, 255, 255),
        1,
        cv2.LINE_AA,
        tipLength=0.2,
    )
    cv2.putText(
        header,
        "+Z",
        (ox + 34, oy - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return np.vstack([header, vis])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tof.pgm", help="Input PGM path")
    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    gray_f32 = to_gray_float(img)
    gray_up, scale_x, scale_y = upscale_to_300x400(gray_f32)
    vis = build_display_image(gray_up)

    results = detect_strongest_corners_per_region(gray_f32)
    draw_results(vis, results, scale_x, scale_y)
    pose = solve_pose(results, gray_f32.shape[1], gray_f32.shape[0])

    if pose is None:
        print("PnP failed: need 4 valid corners.")
    else:
        print(
            "Rotation(deg) "
            f"roll={pose['roll_deg']:.2f}, "
            f"pitch={pose['pitch_deg']:.2f}, "
            f"yaw={pose['yaw_deg']:.2f}"
        )
        print(
            "Translation(mm) "
            f"x={pose['tx_mm']:.2f}, "
            f"y={pose['ty_mm']:.2f}, "
            f"z={pose['tz_mm']:.2f}, "
            f"|t|={pose['distance_mm']:.2f}"
        )
    display = compose_display_with_header(vis, pose)
    cv2.imshow("TOF Corners (Upscaled + GFTT + SubPix 3x3)", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
