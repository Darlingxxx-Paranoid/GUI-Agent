from Perception.uied.detect_text.ocr import OCRDetector
from Perception.uied.detect_text.Text import Text
from utils.utils import remove_punctuation
import numpy as np
import cv2
import json
import time
import os
from os.path import join as p_join
import logging

logger = logging.getLogger(__name__)
_OCR_DETECTOR = None


def _get_ocr_detector() -> OCRDetector:
    global _OCR_DETECTOR
    if _OCR_DETECTOR is None:
        _OCR_DETECTOR = OCRDetector()
    return _OCR_DETECTOR


def save_detection_json(file_path, texts, img_shape):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    output = {'img_shape': img_shape, 'texts': []}
    for text in texts:
        c = {'id': text.id, 'content': text.content}
        loc = text.location
        c['column_min'], c['row_min'], c['column_max'], c['row_max'] = \
            loc['left'], loc['top'], loc['right'], loc['bottom']
        c['width'] = text.width
        c['height'] = text.height
        output['texts'].append(c)

    with open(file_path, 'w', encoding='utf-8') as f_out:
        json.dump(output, f_out, indent=4, ensure_ascii=False)


def visualize_texts(org_img, texts, shown_resize_height=None, show=False, write_path=None):
    img = org_img.copy()
    for text in texts:
        text.visualize_element(img, line=2)

    img_resize = img
    if shown_resize_height is not None:
        img_resize = cv2.resize(
            img,
            (int(shown_resize_height * (img.shape[1] / img.shape[0])), shown_resize_height)
        )

    if show:
        cv2.imshow('texts', img_resize)
        cv2.waitKey(0)
        cv2.destroyWindow('texts')
    if write_path is not None:
        cv2.imwrite(write_path, img)


def text_sentences_recognition(texts):
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                if text_a.is_on_same_line(
                    text_b, 'h',
                    bias_justify=0.2 * min(text_a.height, text_b.height),
                    bias_gap=2 * max(text_a.word_width, text_b.word_width)
                ):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()

    for i, text in enumerate(texts):
        text.id = i
    return texts


def merge_intersected_texts(texts):
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                if text_a.is_intersected(text_b, bias=2):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()
    return texts


def text_cvt_orc_format(ocr_result):
    texts = []
    if ocr_result is not None:
        for i, result in enumerate(ocr_result):
            error = False
            x_coordinates = []
            y_coordinates = []
            text_location = result['boundingPoly']['vertices']
            content = result['description']
            for loc in text_location:
                if 'x' not in loc or 'y' not in loc:
                    error = True
                    break
                x_coordinates.append(loc['x'])
                y_coordinates.append(loc['y'])
            if error:
                continue
            location = {
                'left': min(x_coordinates),
                'top': min(y_coordinates),
                'right': max(x_coordinates),
                'bottom': max(y_coordinates),
            }
            texts.append(Text(i, content, location))
    return texts


def text_cvt_orc_format_paddle(paddle_result):
    texts = []
    if paddle_result is not None:
        for i, line in enumerate(paddle_result):
            try:
                points = np.array(line[0])
                location = {
                    'left': int(min(points[:, 0])),
                    'top': int(min(points[:, 1])),
                    'right': int(max(points[:, 0])),
                    'bottom': int(max(points[:, 1])),
                }
                content = line[1][0]
                texts.append(Text(i, content, location))
            except Exception:
                continue
    return texts


def text_filter_noise(texts, img_shape=None):
    valid_texts = []
    img_h = int(img_shape[0]) if img_shape is not None and len(img_shape) > 0 else 0
    for text in texts:
        content = str(text.content or "").strip()
        if not content:
            continue

        # Drop symbolic one-char OCR artifacts (status icons / separators), while
        # keeping meaningful one-char alnum labels.
        if len(content) <= 1 and not content.isalnum():
            continue

        if remove_punctuation(content, more_punc=["+"]) == "Q":
            continue

        # Suppress tiny status-bar text fragments near the very top.
        if img_h > 0:
            top_noise_h = max(24, int(img_h * 0.03))
            top_noise_bottom = int(img_h * 0.08)
            if text.height <= top_noise_h and text.location["bottom"] <= top_noise_bottom:
                continue

        valid_texts.append(text)
    return valid_texts


def text_detection(input_file, output_file, show=False):
    start = time.perf_counter()
    name = input_file.split('/')[-1][:-4]
    ocr_root = p_join(output_file, 'ocr')
    os.makedirs(ocr_root, exist_ok=True)

    img = cv2.imread(input_file)
    if img is None:
        logger.error("Failed to read image: %s", input_file)
        raise FileNotFoundError(f"Cannot read image: {input_file}")

    detector = _get_ocr_detector()
    ocr_model = detector.get_model('ch_ppocr_mobile_v2.0_xx')

    raw = ocr_model.ocr(np.array(img))
    result = raw[0] if raw else []

    texts = text_cvt_orc_format_paddle(result)
    logger.debug("After convert: %d", len(texts))

    texts = merge_intersected_texts(texts)
    logger.debug("After merge: %d", len(texts))

    texts = text_filter_noise(texts, img_shape=img.shape)
    logger.debug("After filter: %d", len(texts))

    texts = text_sentences_recognition(texts)
    logger.debug("After sentence merge: %d", len(texts))

    for i, text in enumerate(texts[:5]):
        logger.debug("Final text[%d]: %r", i, text.content)

    visualize_texts(
        img,
        texts,
        shown_resize_height=800,
        show=show,
        write_path=p_join(ocr_root, name + '.png')
    )
    save_detection_json(p_join(ocr_root, name + '.json'), texts, img.shape)

    logger.info(
        "Text detection completed in %.3fs, text_count=%d",
        time.perf_counter() - start,
        len(texts)
    )
