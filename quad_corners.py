import argparse
import cv2
import numpy as np


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

    cv2.imshow("TOF Corners (Upscaled + GFTT + SubPix 3x3)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
