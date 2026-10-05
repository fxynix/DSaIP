import os
import glob
import threading
import numpy as np
import cv2
import matplotlib.pyplot as plt


def custom_filter2d(img, kernel):
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

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    high_pass_kernel = np.array([
        [ 1, -2,  1],
        [-2,  5, -2],
        [ 1, -2,  1]
    ], dtype=np.float32)

    sharpened = custom_filter2d(gray, high_pass_kernel)

    return {
        "gray": gray,
        "sharp": sharpened
    }

class DatasetViewer:
    def __init__(self, image_paths):
        self.image_paths = sorted(image_paths)
        self.current_idx = 0
        self.cache = {}
        self.stop_worker = False

        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7))
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
        print(f"[{self.current_idx + 1}/{len(self.image_paths)}] Displaying: {file_name}")

        stages = [
            ("Original Grayscale", res["gray"]),
            ("High-Pass Filtered (Sharpened H3)", res["sharp"])
        ]

        self.fig.suptitle(f"File: {file_name} ({self.current_idx + 1}/{len(self.image_paths)}) (<- / ->)", fontsize=13)

        for ax, (title, img) in zip(self.axes, stages):
            ax.clear()
            ax.set_title(title, fontsize=12, pad=10)
            ax.imshow(img, cmap="gray")
            ax.axis('off')

        self.fig.subplots_adjust(top=0.90, bottom=0.05, left=0.03, right=0.97, wspace=0.05)
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