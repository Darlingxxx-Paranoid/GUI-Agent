"""
通用工具函数
包含被 uied/detect_text/text_detection.py 引用的 remove_punctuation 等
"""
import re
import string


def remove_punctuation(text: str, more_punc: list = None) -> str:
    """
    移除文本中的标点符号
    :param text: 输入文本
    :param more_punc: 额外需要移除的标点列表
    :return: 清理后的文本
    """
    punc = string.punctuation
    if more_punc:
        for p in more_punc:
            punc += p
    return text.translate(str.maketrans("", "", punc)).strip()


def calc_iou(box_a: tuple, box_b: tuple) -> float:
    """
    计算两个矩形框的 IoU
    :param box_a: (x1, y1, x2, y2)
    :param box_b: (x1, y1, x2, y2)
    :return: IoU 值
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def cosine_similarity(text_a: str, text_b: str) -> float:
    """
    简单的基于字符级n-gram的余弦相似度
    用于经验检索的文本匹配
    """
    def char_ngrams(text: str, n: int = 2) -> dict:
        grams = {}
        for i in range(len(text) - n + 1):
            g = text[i:i + n]
            grams[g] = grams.get(g, 0) + 1
        return grams

    grams_a = char_ngrams(text_a)
    grams_b = char_ngrams(text_b)

    all_keys = set(grams_a.keys()) | set(grams_b.keys())
    if not all_keys:
        return 0.0

    dot_product = sum(grams_a.get(k, 0) * grams_b.get(k, 0) for k in all_keys)
    mag_a = sum(v ** 2 for v in grams_a.values()) ** 0.5
    mag_b = sum(v ** 2 for v in grams_b.values()) ** 0.5

    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product / (mag_a * mag_b)
