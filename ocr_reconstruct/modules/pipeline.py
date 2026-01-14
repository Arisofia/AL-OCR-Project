"""
High-level iterative pipeline orchestrating enhancement, reconstruction and OCR.
"""

import os
import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, List

from .enhance import ImageEnhancer
from .reconstruct import PixelReconstructor
from .ocr import image_to_text

logger = logging.getLogger("ocr-reconstruct.pipeline")


class IterativeOCR:
    """
    Main pipeline for iterative OCR processing with enhancement and reconstruction.
    """

    def __init__(
        self,
        iterations: int = 3,
        save_iterations: bool = False,
        output_dir: str = "./iterations",
    ):
        """
        Initializes the pipeline with settings.
        """
        self.iterations = iterations
        self.save_iterations = save_iterations
        self.output_dir = output_dir
        self.enhancer = ImageEnhancer()
        self.reconstructor = PixelReconstructor()

        if self.save_iterations:
            os.makedirs(self.output_dir, exist_ok=True)

    def _save_debug_image(self, img: np.ndarray, iteration: int, suffix: str = ""):
        """Saves image for debugging if requested."""
        if self.save_iterations:
            filename = f"iter_{iteration + 1}{suffix}.png"
            path = os.path.join(self.output_dir, filename)
            cv2.imwrite(path, img)

    def _apply_feedback_strategies(
        self,
        current: np.ndarray,
        thresholded: np.ndarray,
        iteration: int,
    ) -> Tuple[str, np.ndarray, List[Dict[str, Any]]]:
        """
        Applies heuristic strategies like depixelation and inpainting
        """
        strategies_meta = []
        best_local_text = ""
        best_local_img = current

        # Strategy 1: Depixelation
        depixelated = self.reconstructor.depixelate_naive(current)
        dep_th = self.enhancer.apply_threshold(depixelated)
        text_dep = image_to_text(dep_th)

        strategies_meta.append(
            {"strategy": "depixelate", "iteration": iteration + 1, "text": text_dep}
        )

        if len(text_dep) > len(best_local_text):
            best_local_text = text_dep
            best_local_img = depixelated

        # Strategy 2: Inpainting
        mask = (thresholded == 255).astype("uint8") * 255
        inpainted = cv2.inpaint(current, mask, 3, cv2.INPAINT_TELEA)
        inp_th = self.enhancer.apply_threshold(inpainted)
        text_inp = image_to_text(inp_th)

        strategies_meta.append(
            {"strategy": "inpaint", "iteration": iteration + 1, "text": text_inp}
        )

        if len(text_inp) > len(best_local_text):
            best_local_text = text_inp
            best_local_img = inpainted

        return best_local_text, best_local_img, strategies_meta

    def process_image(self, img: np.ndarray) -> Tuple[str, np.ndarray, Dict[str, Any]]:
        """
        Internal core logic for processing an image (numpy array).
        Returns best text found, final processed image, and metadata.
        """
        current = self.enhancer.to_gray(img)
        best_overall_text = ""
        meta = {"iterations": []}

        for i in range(self.iterations):
            # 1. Standard Enhancement & OCR
            enhanced = self.enhancer.sharpen(current)
            denoised = self.enhancer.denoise(enhanced)
            thresholded = self.enhancer.apply_threshold(denoised)

            text = image_to_text(thresholded)
            self._save_debug_image(thresholded, i)

            current_meta = {"iteration": i + 1, "type": "standard", "text": text}
            meta["iterations"].append(current_meta)

            # 2. Feedback Loop / Heuristics
            if len(text) < 10:
                fb_text, fb_img, fb_meta = self._apply_feedback_strategies(
                    current,
                    thresholded,
                    i,
                )
                meta["iterations"].extend(fb_meta)

                if len(fb_text) > len(text):
                    text = fb_text
                    current = fb_img

            # 3. Update global best
            if len(text) > len(best_overall_text):
                best_overall_text = text

            # 4. Early exit if we found a strong result
            if len(best_overall_text) > 20:
                break

        return best_overall_text.strip(), current, meta

    def process_file(self, image_path: str) -> Tuple[str, Dict[str, Any]]:
        """Processes an image from a file path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        text, _, meta = self.process_image(img)
        return text, meta

    def process_bytes(
        self,
        image_bytes: bytes,
    ) -> Tuple[
        str,
        Optional[bytes],
        Dict[str, Any],
    ]:
        """Processes an image provided as bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return (
                "",
                None,
                {"iterations": [], "error": "Invalid image bytes"},
            )

        text, final_img, meta = self.process_image(img)

        # Encode processed image back to bytes
        success, buf = cv2.imencode(".png", final_img)
        img_bytes = buf.tobytes() if success else None

        return text, img_bytes, meta


def process_bytes(
    image_bytes: bytes, iterations: int = 3, save_iterations: bool = False
) -> Tuple[str, Optional[bytes], Dict[str, Any]]:
    """Stand-alone function for processing image bytes."""
    pipeline = IterativeOCR(iterations=iterations, save_iterations=save_iterations)
    return pipeline.process_bytes(image_bytes)
