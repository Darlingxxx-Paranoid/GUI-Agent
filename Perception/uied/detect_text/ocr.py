import os
import cv2
import numpy as np
from PIL import Image
import logging

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)


class OCRModelWrapper:
    def __init__(self, model):
        self.model = model

    def ocr(self, img):
        result = self.model.ocr(np.array(img))

        if not result:
            return [[]]

        page = result[0]
        logger.debug(
            "PaddleOCR page keys: %s",
            list(page.keys()) if hasattr(page, "keys") else None
        )

        rec_texts = page.get("rec_texts", []) if hasattr(page, "get") else []
        rec_scores = page.get("rec_scores", []) if hasattr(page, "get") else []
        rec_polys = page.get("rec_polys", []) if hasattr(page, "get") else []

        logger.debug(
            "PaddleOCR counts: rec_texts=%d, rec_scores=%d, rec_polys=%d",
            len(rec_texts), len(rec_scores), len(rec_polys)
        )

        converted = []
        count = min(len(rec_texts), len(rec_scores), len(rec_polys))

        for i in range(count):
            try:
                pts = rec_polys[i]
                text = rec_texts[i]
                score = rec_scores[i]

                if isinstance(pts, np.ndarray):
                    pts = pts.tolist()

                converted.append([pts, (text, float(score))])
            except Exception:
                continue

        logger.debug("PaddleOCR converted count: %d", len(converted))
        return [converted]


class OCRDetector:
    def __init__(self):
        self.threshold = 0.8
        self.model = PaddleOCR(lang='ch')
        self.wrapper = OCRModelWrapper(self.model)
        logger.info("OCRDetector initialized successfully")

    def get_model(self, ocr_model: str = None):
        return self.wrapper

    def apply_model(self, img):
        try:
            result = self.wrapper.ocr(np.array(img))
        except Exception:
            logger.exception("OCR execution failed")
            return []

        boxes_filtered = []
        if not result or not result[0]:
            return boxes_filtered

        for box in result[0]:
            try:
                pts = box[0]
                text = box[1][0]
                score = box[1][1]

                if score > self.threshold:
                    boxes_filtered.append([
                        int(pts[0][0]),
                        int(pts[0][1]),
                        int(pts[2][0]),
                        int(pts[2][1]),
                        text
                    ])
            except Exception:
                continue

        return boxes_filtered

    def detect(self, img):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return self.apply_model(img)