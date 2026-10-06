import os
import glob
import threading
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

sys.setrecursionlimit(1000000)

MAX_KMEANS_ITERATIONS = 100


def custom_rgb_to_gray(rgb_img):
    r = rgb_img[:, :, 0].astype(np.float32)
    g = rgb_img[:, :, 1].astype(np.float32)
    b = rgb_img[:, :, 2].astype(np.float32)
    gray = 0.30 * r + 0.59 * g + 0.11 * b
    return np.clip(gray, 0, 255).astype(np.uint8)

def custom_filter(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect').astype(np.float32)
    output = np.zeros((h, w), dtype=np.float32)

    for i in range(kh):
        for j in range(kw):
            output += padded[i:i + h, j:j + w] * kernel[i, j]

    return np.clip(output, 0, 255).astype(np.uint8)

###

def custom_fill(img, labels, x, y, L):
    if labels[y, x] == 0 and img[y, x] == 1:
        labels[y, x] = L
        h, w = img.shape
        if x > 0:
             custom_fill(img, labels, x - 1, y, L)
        if x < w - 1:
            custom_fill(img, labels, x + 1, y, L)
        if y > 0:
            custom_fill(img, labels, x, y - 1, L)
        if y < h - 1:
             custom_fill(img, labels, x, y + 1, L)

def custom_labeling(binary_img):
    img_bin = (binary_img > 0).astype(np.uint8)
    h, w = img_bin.shape
    labels = np.zeros((h, w), dtype=np.int32)

    L = 1
    for y in range(h):
        for x in range(w):
            if labels[y, x] == 0 and img_bin[y, x] == 1:
                custom_fill(img_bin, labels, x, y, L)
                L += 1
    return labels

def custom_kmeans(data, k):
    np.random.seed(676767)
    n_samples, n_features = data.shape

    idx = np.random.choice(n_samples, k, replace=False)
    centroids = data[idx].copy()
    labels = np.zeros(n_samples, dtype=np.int32)

    distances = None
    for _ in range(MAX_KMEANS_ITERATIONS):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for j in range(k):
            points = data[labels == j]
            if len(points) > 0:
                centroids[j] = np.mean(points, axis=0)
            else:
                centroids[j] = data[np.random.choice(n_samples)]

    compactness = float(np.sum(np.min(distances, axis=1) ** 2))
    return compactness, labels.reshape(-1, 1), centroids

def feature_area(mask):
    return int(np.count_nonzero(mask))

def feature_perimeter(mask):
    p = np.pad(mask, 1, constant_values=False)
    neighbor_is_bg = (
        ~p[:-2, 1:-1] |   # верх
        ~p[2:,  1:-1] |   # низ
        ~p[1:-1, :-2] |   # лево
        ~p[1:-1, 2:]      # право
    )
    return int(np.count_nonzero(mask & neighbor_is_bg))

def feature_centroid(mask):
    ys, xs = np.where(mask)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    return cx, cy

def cluster_features(features, k, method='custom'):
    n = len(features)
    if n < 2:
        return np.zeros(n, dtype=np.int32)

    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1.0
    normalized = (features - mean) / std

    actual_k = min(k, n)
    if actual_k < 2:
        return np.zeros(n, dtype=np.int32)

    if method == 'custom':
        _, labels, _ = custom_kmeans(normalized, actual_k)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, _ = cv2.kmeans(normalized, actual_k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels.flatten()

def render_clusters(rgb, object_masks, labels):
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    img = np.zeros_like(rgb)

    for i, mask in enumerate(object_masks):
        label = int(labels[i]) if i < len(labels) else 0
        img[mask] = colors[label % 3]

    return img

def custom_hsv_threshold(rgb_img):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # mask = (hue >= 35) & (hue <= 135) & (sat > 45) & (val > 40) # кубики
    mask = (hue >= 35) & (hue <= 91) & (sat > 40) & (val > 40) # цифры

    return mask.astype(np.uint8) * 255

###

def extract_features_and_cluster(cleaned):

    binary_mask = custom_hsv_threshold(cleaned)

    processed_mask = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    _, labels_cc, stats, _ = cv2.connectedComponentsWithStats(processed_mask, connectivity=8)
    min_area = (cleaned.shape[0] * cleaned.shape[1]) * 0.002
    keep = np.where(stats[:, cv2.CC_STAT_AREA] >= min_area)[0]
    keep = keep[keep != 0]
    processed_mask = np.isin(labels_cc, keep).astype(np.uint8) * 255

    ###

    labels_img = custom_labeling(processed_mask)
    unique_labels = np.unique(labels_img)

    features_list = []
    object_masks = []

    for L in unique_labels:
        if L == 0:
            continue
        obj_mask = (labels_img == L)

        obj_mask = binary_fill_holes(obj_mask) # ? заливка

        area = feature_area(obj_mask)
        perimeter = feature_perimeter(obj_mask)
        centroid_x, centroid_y = feature_centroid(obj_mask)
        compactness = (perimeter ** 2) / area if area > 0 else 0

        features_list.append([area, perimeter, centroid_x, centroid_y, compactness])

        object_masks.append(obj_mask)

    if len(object_masks) == 0:
        empty = np.zeros_like(cleaned)
        return binary_mask, processed_mask, empty, empty, "No objects found"

    ###

    features = np.array(features_list, dtype=np.float32)

    labels_custom = cluster_features(features, 3, method='custom')
    labels_lib = cluster_features(features, 3, method='lib')

    clustered_custom = render_clusters(cleaned, object_masks, labels_custom)
    clustered_lib = render_clusters(cleaned, object_masks, labels_lib)

    features_str = "Object Features:\n"
    features_str += (f"{'No':<4} {'Area':<9} {'Perim':<9} {'C_x':<8} {'C_y':<8}"
                     f"{'Comp':<8} {'C_cust':<8} {'C_lib':<6}\n")
    features_str += "-" * 64 + "\n"

    for i in range(len(object_masks)):
        f = features[i]
        lc = int(labels_custom[i]) if i < len(labels_custom) else 0
        ll = int(labels_lib[i]) if i < len(labels_lib) else 0
        features_str += (f"{i+1:<4} {f[0]:<9.0f} {f[1]:<9.0f} {f[2]:<8.1f} {f[3]:<8.1f}"
                         f"{f[4]:<8.2f} {lc:<8} {ll:<6}\n")
    return binary_mask, processed_mask, clustered_custom, clustered_lib, features_str

###

def process_pipeline(img_path):
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image at path: {img_path}")

    if bgr.shape[0] > bgr.shape[1]:
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)

    max_dim = 1400
    h, w = bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    gaussian_kernel_3x3 = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16.0

    gray_custom = custom_rgb_to_gray(rgb)
    blurred_custom = custom_filter(gray_custom, gaussian_kernel_3x3)

    gray_lib = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred_lib = cv2.filter2D(gray_lib, -1, gaussian_kernel_3x3, borderType=cv2.BORDER_REFLECT)

    gray_mae = float(np.mean(np.abs(gray_custom.astype(np.float32) - gray_lib.astype(np.float32))))
    blur_mae = float(np.mean(np.abs(blurred_custom.astype(np.float32) - blurred_lib.astype(np.float32))))

    diff_map = np.abs(gray_custom.astype(np.float32) - blurred_custom.astype(np.float32))
    diff_denoised = np.maximum(diff_map - 2.0, 0.0)
    diff_normalized = np.clip((diff_denoised / 8.0) * 255.0, 0, 255).astype(np.uint8)

    grad_x = cv2.Sobel(blurred_custom, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_custom, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.magnitude(grad_x, grad_y)
    edges = np.clip(edges, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    foreground_seed = (hue >= 35) & (hue <= 135) & (sat > 45) & (val > 40)
    raw_mask = foreground_seed.astype(np.uint8) * 255

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(opened)
    min_area = (rgb.shape[0] * rgb.shape[1]) * 0.004

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    cleaned = cv2.bitwise_and(rgb, rgb, mask=final_mask)

    binary_mask, processed_mask, clustered_custom, clustered_lib, features_str = extract_features_and_cluster(cleaned)

    return {
        "rgb": rgb,
        "gray": gray_custom,
        "blurred": blurred_custom,
        "diff_map": diff_normalized,
        "edges": edges,
        "cleaned": cleaned,
        "gray_mae": gray_mae,
        "blur_mae": blur_mae,
        "binary_mask" : binary_mask,
        "processed_mask" : processed_mask,
        "clustered_custom": clustered_custom,
        "clustered_lib": clustered_lib,
        "features_str": features_str,
    }

###

class DatasetViewer:
    def __init__(self, image_paths):
        self.image_paths = sorted(image_paths)
        self.current_idx = 0
        self.cache = {}
        self.stop_worker = False

        # 2x4 = 8 слотов
        self.fig, self.axes = plt.subplots(2, 4, figsize=(20, 12))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        first_path = self.image_paths[0]
        self.cache[first_path] = process_pipeline(first_path)
        self.update_view()

        self.worker_thread = threading.Thread(target=self._preload_remaining, daemon=True)
        self.worker_thread.start()

    def _preload_remaining(self):
        for path in self.image_paths[1:]:
            if self.stop_worker:
                break
            if path not in self.cache:
                self.cache[path] = process_pipeline(path)
                print(f"[Preloaded] {os.path.basename(path)}")

    def update_view(self):
        current_path = self.image_paths[self.current_idx]
        file_name = os.path.basename(current_path)

        if current_path not in self.cache:
            self.cache[current_path] = process_pipeline(current_path)

        res = self.cache[current_path]
        print(f"[{self.current_idx + 1}/{len(self.image_paths)}] {file_name}")

        stages = [
            ("Original RGB",                res["rgb"],              None),
            ("Cleaned Output",              res["cleaned"],          None),
            ("Binary Mask",                 res["binary_mask"],      "gray"),
            ("Processed Mask",              res["processed_mask"],      "gray"),
            ("Clustering — Custom k-means", res["clustered_custom"], None),
            ("Clustering — cv2.kmeans",     res["clustered_lib"],    None),
        ]

        self.fig.suptitle(
            f"File: {file_name} ({self.current_idx + 1}/{len(self.image_paths)}) (<- / ->)",
            fontsize=14, y=0.97
        )

        flat_axes = self.axes.flatten()

        for ax, (title, img, cmap) in zip(flat_axes, stages):
            ax.clear()
            ax.set_title(title, fontsize=12, pad=8)
            if cmap is None:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap)
            ax.axis('off')

        # 7-й слот — таблица признаков
        ax_text = flat_axes[6]
        ax_text.clear()
        ax_text.text(0.02, 0.5, " " * 18 + "Calculated Features\n\n" + res["features_str"], fontsize=10, va='center', ha='left', family='monospace')
        ax_text.axis('off')

        self.fig.subplots_adjust(top=0.92, bottom=0.04, left=0.03, right=0.97, hspace=0.20, wspace=0.08)
        self.fig.canvas.draw_idle()

        ax_none = flat_axes[7]
        ax_none.clear()
        ax_none.axis('off')


    def on_key(self, event):
        if event.key in ('right', 'd', ' '):
            self.current_idx = (self.current_idx + 1) % len(self.image_paths)
            self.update_view()
        elif event.key in ('left', 'a'):
            self.current_idx = (self.current_idx - 1) % len(self.image_paths)
            self.update_view()
        elif event.key in ('q', 'escape'):
            self.stop_worker = True
            plt.close(self.fig)

def main():
    dataset_dir = "../dataset"
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    sample_files = []
    for ext in extensions:
        sample_files.extend(glob.glob(os.path.join(dataset_dir, ext)))

    if not sample_files:
        print(f"No image files found in {dataset_dir}.")
        return

    viewer = DatasetViewer(sample_files)
    plt.show()


if __name__ == "__main__":
    main()