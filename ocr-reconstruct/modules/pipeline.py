"""High-level iterative pipeline orchestrating enhancement, reconstruction and OCR."""
import os
import cv2
import numpy as np
from .enhance import to_gray, sharpen, denoise, adaptive_threshold, upscale_and_smooth
from .reconstruct import depixelate_naive, inpaint_bbox
from .ocr import image_to_text


class IterativeOCR:
    def __init__(self, image_path, iterations=3, save_iterations=False, output_dir="./iterations"):
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        self.image_path = image_path
        self.iterations = iterations
        self.save_iterations = save_iterations
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self):
        img = cv2.imread(self.image_path)
        gray = to_gray(img)
        current = gray.copy()
        best_text = ""
        best_conf = 0
        meta = {"iterations": []}

        for i in range(self.iterations):
            # Enhance
            enhanced = sharpen(current)
            den = denoise(enhanced)
            th = adaptive_threshold(den)

            # OCR
            text = image_to_text(th)

            # Save
            if self.save_iterations:
                cv2.imwrite(os.path.join(self.output_dir, f"iter_{i+1}.png"), th)

            meta["iterations"].append({"iteration": i+1, "text": text})

            # Heuristic: if text is short, attempt depixelation or inpaint
            if len(text) < 10:
                # Depixelate naively
                dep = depixelate_naive(current)
                dep_th = adaptive_threshold(dep)
                maybe_text = image_to_text(dep_th)
                meta["iterations"].append({"iteration": f"depixelate_{i+1}", "text": maybe_text})
                if len(maybe_text) > len(text):
                    text = maybe_text
                    current = dep
                else:
                    # Try inpaint over bright spots (simple mask)
                    mask = (th == 255).astype('uint8')*255
                    inpainted = inpaint_bbox(current, mask)
                    maybe_text2 = image_to_text(adaptive_threshold(inpainted))
                    meta["iterations"].append({"iteration": f"inpaint_{i+1}", "text": maybe_text2})
                    if len(maybe_text2) > len(text):
                        text = maybe_text2
                        current = inpainted

            # Update best
            if len(text) > len(best_text):
                best_text = text

            # Stop early if we've got some non-trivial text
            if len(best_text) > 20:
                break

        return best_text.strip(), {"iterations": meta["iterations"]}
