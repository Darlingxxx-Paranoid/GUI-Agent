from Perception.uied.detect_text.ocr import OCRDetector
from Perception.uied.detect_text.Text import Text
from utils.utils import remove_punctuation
import numpy as np
import cv2
import json
import time
import os
import re
from os.path import join as p_join
import logging

logger = logging.getLogger(__name__)
_OCR_DETECTOR = None
_KEY_TOKEN_RE = re.compile(r"^[0-9+\-*/xX×÷=%().,:]+$")


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


def _is_symbolic_key_token(value):
    token = str(value or "").strip()
    if not token or len(token) > 2:
        return False
    return bool(_KEY_TOKEN_RE.fullmatch(token))


def _should_skip_same_line_merge(text_a, text_b, img_shape=None):
    token_a = str(getattr(text_a, "content", "") or "").strip()
    token_b = str(getattr(text_b, "content", "") or "").strip()
    if not (_is_symbolic_key_token(token_a) and _is_symbolic_key_token(token_b)):
        return False

    a = text_a.location
    b = text_b.location
    span = max(a["right"], b["right"]) - min(a["left"], b["left"])
    gap = min(abs(a["right"] - b["left"]), abs(b["right"] - a["left"]))

    h_ratio = max(text_a.height, text_b.height) / max(1, min(text_a.height, text_b.height))
    if h_ratio > 1.8:
        return False

    if img_shape is not None and len(img_shape) > 1:
        img_w = int(img_shape[1])
        if img_w > 0 and (text_a.width / img_w > 0.35 or text_b.width / img_w > 0.35):
            return False
        if img_w > 0 and span / img_w >= 0.62:
            return True
        if img_w > 0 and gap > max(40, int(0.20 * img_w)):
            return False
    elif gap > 80:
        return False
    return True


def text_sentences_recognition(texts, img_shape=None):
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                if _should_skip_same_line_merge(text_a, text_b, img_shape=img_shape):
                    continue
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
                confidence = float(line[1][1]) if len(line) > 1 and len(line[1]) > 1 else 1.0
                text_obj = Text(i, content, location)
                # Keep OCR confidence for downstream noise suppression.
                text_obj.confidence = confidence
                texts.append(text_obj)
            except Exception:
                continue
    return texts


def text_filter_noise(texts, img_shape=None):
    valid_texts = []
    img_h = int(img_shape[0]) if img_shape is not None and len(img_shape) > 0 else 0
    img_w = int(img_shape[1]) if img_shape is not None and len(img_shape) > 1 else 0
    img_area = img_h * img_w if img_h > 0 and img_w > 0 else 0
    for text in texts:
        content = str(text.content or "").strip()
        if not content:
            continue

        confidence = float(getattr(text, "confidence", 1.0))
        text_area = int(max(0, text.width) * max(0, text.height))

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

        # Suppress oversized low-confidence regions (common false positives in
        # popup/dropdown overlays), which can incorrectly absorb nearby lines
        # during intersection merge.
        if img_area > 0:
            area_ratio = text_area / img_area
            if area_ratio > 0.08 and confidence < 0.45:
                continue
            if area_ratio > 0.20 and len(content) <= 3:
                continue

        valid_texts.append(text)
    return valid_texts


def split_compact_symbol_sequences(texts, img_shape=None):
    """
    Split compact OCR rows like "4 5 6" into per-token text boxes.

    This is intentionally generic: it only triggers for short symbolic tokens
    in a wide horizontal OCR region and leaves natural-language phrases intact.
    """
    split_texts = []
    next_id = 0
    img_w = int(img_shape[1]) if img_shape is not None and len(img_shape) > 1 else 0

    for text in texts:
        content = str(getattr(text, "content", "") or "").strip()
        if not content or (" " not in content and "\t" not in content):
            text.id = next_id
            next_id += 1
            split_texts.append(text)
            continue

        tokens = [token for token in re.split(r"\s+", content) if token]
        if len(tokens) < 2 or len(tokens) > 6:
            text.id = next_id
            next_id += 1
            split_texts.append(text)
            continue

        if not all(_is_symbolic_key_token(token) for token in tokens):
            text.id = next_id
            next_id += 1
            split_texts.append(text)
            continue

        if text.width < max(90, int(0.16 * img_w) if img_w > 0 else 90):
            text.id = next_id
            next_id += 1
            split_texts.append(text)
            continue

        if text.height <= 0 or (text.width / max(1, text.height)) < 2.2:
            text.id = next_id
            next_id += 1
            split_texts.append(text)
            continue

        left = int(text.location["left"])
        right = int(text.location["right"])
        top = int(text.location["top"])
        bottom = int(text.location["bottom"])
        if right <= left or bottom <= top:
            text.id = next_id
            next_id += 1
            split_texts.append(text)
            continue

        total_units = sum(max(1, len(token)) for token in tokens)
        start = left
        used_units = 0
        created = []
        for index, token in enumerate(tokens):
            token_units = max(1, len(token))
            end_units = used_units + token_units
            if index == len(tokens) - 1:
                end = right
            else:
                end = int(round(left + (end_units / float(total_units)) * (right - left)))
            end = max(start + 1, min(end, right))
            piece = Text(
                id=-1,
                content=str(token),
                location={
                    "left": int(start),
                    "top": int(top),
                    "right": int(end),
                    "bottom": int(bottom),
                },
            )
            piece.confidence = float(getattr(text, "confidence", 1.0))
            created.append(piece)
            start = end
            used_units = end_units

        if len(created) >= 2:
            for piece in created:
                piece.id = next_id
                next_id += 1
                split_texts.append(piece)
            continue

        text.id = next_id
        next_id += 1
        split_texts.append(text)

    return split_texts


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

    # Filter obvious OCR noise before any merge step. If left untouched, a single
    # oversized false-positive text region can absorb multiple nearby candidates.
    texts = text_filter_noise(texts, img_shape=img.shape)
    logger.debug("After pre-merge filter: %d", len(texts))

    texts = merge_intersected_texts(texts)
    logger.debug("After merge: %d", len(texts))

    texts = text_filter_noise(texts, img_shape=img.shape)
    logger.debug("After filter: %d", len(texts))

    texts = split_compact_symbol_sequences(texts, img_shape=img.shape)
    logger.debug("After compact symbol split: %d", len(texts))

    texts = text_sentences_recognition(texts, img_shape=img.shape)
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
