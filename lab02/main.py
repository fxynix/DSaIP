import os
import glob
import threading
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    h, w = img.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return
    if labels[y, x] == 0 and img[y, x] == 1:
        labels[y, x] = L
        custom_fill(img, labels, x - 1, y, L)
        custom_fill(img, labels, x + 1, y, L)
        custom_fill(img, labels, x, y - 1, L)
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

def custom_find_contours(binary_img):
    labels_img = custom_labeling(binary_img)
    unique_labels = np.unique(labels_img)

    contours = []
    for L in unique_labels:
        if L == 0:
            continue
        obj_mask = (labels_img == L).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            contours.append(max(cnts, key=cv2.contourArea))

    return contours, None

def custom_hsv_threshold(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    foreground_seed = (hue >= 35) & (hue <= 135) & (sat > 45) & (val > 40)
    raw_mask = foreground_seed.astype(np.uint8) * 255
    return raw_mask

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

def compute_features_from_contours(contours):
    valid_contours = []
    features = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0

        if len(cnt) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            elongation = max(MA, ma) / min(MA, ma) if min(MA, ma) > 0 else 0
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            elongation = max(w, h) / min(w, h) if min(w, h) > 0 else 0

        features.append([area, perimeter, compactness, elongation])
        valid_contours.append((cnt, cx, cy))

    features = np.array(features, dtype=np.float32)
    return features, valid_contours

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

def render_clusters(rgb, valid_contours, labels):
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Красный, Зелёный, Синий
    img = np.zeros_like(rgb)

    for i, (cnt, cx, cy) in enumerate(valid_contours):
        label = int(labels[i]) if i < len(labels) else 0
        color = colors[label % 3]

        cv2.drawContours(img, [cnt], -1, color, thickness=cv2.FILLED)
        cv2.drawContours(img, [cnt], -1, (255, 255, 255), 2)
        cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1)
        cv2.putText(img, f"C{label}", (cx - 10, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img

def extract_features_and_cluster(rgb, mask):
    (contours, _) = custom_find_contours(mask)
    # (contours, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features, valid_contours = compute_features_from_contours(contours)

    if len(valid_contours) == 0:
        empty = np.zeros_like(rgb)
        return empty, empty, "No objects found"

    labels_custom = cluster_features(features, 3, method='custom')
    labels_lib = cluster_features(features, 3, method='lib')


    clustered_custom = render_clusters(rgb, valid_contours, labels_custom)
    clustered_lib = render_clusters(rgb, valid_contours, labels_lib)

    features_str = "Object Features:\n"
    features_str += (f"{'ID':<4} {'Area':<9} {'Perim':<9} {'Comp':<8} "
                     f"{'Elong':<8} {'C_cust':<8} {'C_lib':<6}\n")
    features_str += "-" * 66 + "\n"

    for i in range(len(valid_contours)):
        f = features[i]
        lc = int(labels_custom[i]) if i < len(labels_custom) else 0
        ll = int(labels_lib[i]) if i < len(labels_lib) else 0
        features_str += (f"{i+1:<4} {f[0]:<9.1f} {f[1]:<9.1f} "
                         f"{f[2]:<8.2f} {f[3]:<8.2f} {lc:<8} {ll:<6}\n")

    return clustered_custom, clustered_lib, features_str

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

    hsv_binary = custom_hsv_threshold(bgr)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(hsv_binary, cv2.MORPH_CLOSE, close_kernel)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(opened)
    min_area = (rgb.shape[0] * rgb.shape[1]) * 0.004

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    cleaned = cv2.bitwise_and(rgb, rgb, mask=final_mask)

    clustered_custom, clustered_lib, features_str = extract_features_and_cluster(rgb, final_mask)

    return {
        "rgb": rgb,
        "gray": gray_custom,
        "hsv_binary": hsv_binary,
        "final_mask": final_mask,
        "cleaned": cleaned,
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


        self.fig, self.axes = plt.subplots(2, 4, figsize=(22, 11))
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
            ("Original RGB", res["rgb"], None),
            ("Grayscale", res["gray"], "gray"),
            ("HSV Binary (raw)", res["hsv_binary"], "gray"),
            ("Final Mask", res["final_mask"], "gray"),
            ("Cleaned Output", res["cleaned"], None),
            ("Clustering — Custom k-means", res["clustered_custom"], None),
            ("Clustering — cv2.kmeans", res["clustered_lib"], None),
        ]

        self.fig.suptitle(
            f"File: {file_name} ({self.current_idx + 1}/{len(self.image_paths)}) (<- / ->)",
            fontsize=14, y=0.98
        )

        flat_axes = self.axes.flatten()

        for ax, (title, img, cmap) in zip(flat_axes, stages):
            ax.clear()
            ax.set_title(title, fontsize=11, pad=8)
            if cmap is None:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap)
            ax.axis('off')

        # 8-й слот — таблица признаков
        ax_text = flat_axes[7]
        ax_text.clear()
        text_block = "Calculated Features\n\n" + res["features_str"]
        ax_text.text(0.02, 0.5, text_block, fontsize=10, va='center', ha='left', family='monospace')
        ax_text.axis('off')

        self.fig.subplots_adjust(top=0.93, bottom=0.04, left=0.03,
                                 right=0.97, hspace=0.20, wspace=0.08)
        self.fig.canvas.draw_idle()

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