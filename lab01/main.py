import os
import glob
import threading
import numpy as np
import cv2
import matplotlib.pyplot as plt


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

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)


    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(opened)
    min_area = (rgb.shape[0] * rgb.shape[1]) * 0.0004

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    cleaned = cv2.bitwise_and(rgb, rgb, mask=final_mask)

    return {
        "rgb": rgb,
        "gray": gray_custom,
        "blurred": blurred_custom,
        "diff_map": diff_normalized,
        "edges": edges,
        "cleaned": cleaned,
        "gray_mae": gray_mae,
        "blur_mae": blur_mae
    }


class DatasetViewer:
    def __init__(self, image_paths):
        self.image_paths = sorted(image_paths)
        self.current_idx = 0
        self.cache = {}
        self.stop_worker = False

        self.fig, self.axes = plt.subplots(2, 3, figsize=(16, 9))
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
        print(f"[{self.current_idx + 1}/{len(self.image_paths)}] {file_name} | "
              f"Gray MAE: {res['gray_mae']:.4f} | Blur MAE: {res['blur_mae']:.4f}")

        stages = [
            ("Original RGB", res["rgb"], None),
            ("Grayscale", res["gray"], "gray"),
            ("Low-Pass Filtered (H3)", res["blurred"], "gray"),
            ("LPF Difference Heatmap", res["diff_map"], "magma"),
            ("Sobel Gradient Edges", res["edges"], "gray"),
            ("Cleaned Output", res["cleaned"], None)
        ]

        self.fig.suptitle(f"File: {file_name} ({self.current_idx + 1}/{len(self.image_paths)}) (<- / ->)", fontsize=13, y=0.97)

        for ax, (title, img, cmap) in zip(self.axes.flatten(), stages):
            ax.clear()
            ax.set_title(title, fontsize=11, pad=8)
            if cmap is None:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap)
            ax.axis('off')

        self.fig.subplots_adjust(top=0.92, bottom=0.04, left=0.03, right=0.97, hspace=0.25, wspace=0.08)
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