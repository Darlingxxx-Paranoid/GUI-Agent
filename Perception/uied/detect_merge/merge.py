import json
import cv2
import numpy as np
from os.path import join as p_join
import os
import time
import shutil

from Perception.uied.detect_merge.Element import Element


def show_elements(org_img, eles, show=False, win_name="element", wait_key=0, shown_resize=None, line=2):
    img = org_img.copy()
    for ele in eles:
        color = (0, 255, 0)
        ele.visualize_element(img, color, line, mark_bbox=True, mark_id=False)
        # ele.visualize_element(img, color, line, mark_bbox=False, mark_id=True)
    img_resize = img
    if shown_resize is not None:
        img_resize = cv2.resize(img, shown_resize)
    if show:
        cv2.imshow(win_name, img_resize)
        cv2.waitKey(wait_key)
        if wait_key == 0:
            cv2.destroyWindow(win_name)
    return img_resize


def save_elements(output_file, elements, img_shape):
    components = {"compos": [], "img_shape": img_shape}
    for i, ele in enumerate(elements):
        c = ele.wrap_info()
        # c["id"] = i
        components["compos"].append(c)
    json.dump(components, open(output_file, "w"), indent=4)
    return components


def reassign_ids(elements):
    def custom_key(e):
        key2, key1, _, _ = e.put_bbox()
        return key1, key2

    elements = sorted(elements, key=custom_key)
    for i, element in enumerate(elements):
        element.id = i


def refine_texts(texts, img_shape):
    refined_texts = []
    for text in texts:
        # remove potential noise
        if len(text.text_content) > 1 and text.height / img_shape[0] < 0.075:
            refined_texts.append(text)
    return refined_texts


def merge_text_line_to_paragraph(elements, max_line_gap=10, max_col_gap=10, img_shape=None):
    texts = []
    non_texts = []
    for ele in elements:
        if ele.category == "Text":
            texts.append(ele)
        elif ele.area < 100:  # possible text fragment ignored in ocr
            texts.append(ele)
        else:
            non_texts.append(ele)

    img_h = int(img_shape[0]) if img_shape and len(img_shape) >= 1 else 0
    img_w = int(img_shape[1]) if img_shape and len(img_shape) >= 2 else 0
    img_area = img_h * img_w if img_h > 0 and img_w > 0 else 0

    def blocks_non_text(nx1, ny1, nx2, ny2):
        n_area = max(0, nx2 - nx1) * max(0, ny2 - ny1)
        if n_area <= 0:
            return True
        for e in non_texts:
            if e.category == "Block":
                continue
            ex1, ey1, ex2, ey2 = e.put_bbox()
            ix1 = max(nx1, ex1)
            iy1 = max(ny1, ey1)
            ix2 = min(nx2, ex2)
            iy2 = min(ny2, ey2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter <= 0:
                continue
            if inter / n_area > 0.12 and inter / max(1, e.area) > 0.25:
                return True
        return False

    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                ax1, ay1, ax2, ay2 = text_a.put_bbox()
                bx1, by1, bx2, by2 = text_b.put_bbox()

                overlap_w = max(0, min(ax2, bx2) - max(ax1, bx1))
                min_w = max(1, min(ax2 - ax1, bx2 - bx1))
                overlap_ratio = overlap_w / min_w

                a_h = max(1, ay2 - ay1)
                b_h = max(1, by2 - by1)
                h_ratio = max(a_h, b_h) / max(1, min(a_h, b_h))

                vertical_gap = max(0, max(ay1, by1) - min(ay2, by2))
                if overlap_ratio < 0.65:
                    continue
                if h_ratio > 2.0:
                    continue
                if vertical_gap > min(max_line_gap, int(0.35 * min(a_h, b_h))):
                    continue

                if abs(ax1 - bx1) > max(max_col_gap, int(0.18 * min_w)):
                    continue
                if abs(ax2 - bx2) > max(max_col_gap, int(0.18 * min_w)):
                    continue

                nx1 = min(ax1, bx1)
                ny1 = min(ay1, by1)
                nx2 = max(ax2, bx2)
                ny2 = max(ay2, by2)
                n_area = max(0, nx2 - nx1) * max(0, ny2 - ny1)

                if img_area > 0:
                    if n_area / img_area > 0.10:
                        continue
                if img_h > 0:
                    if (ny2 - ny1) / img_h > 0.35:
                        continue
                if blocks_non_text(nx1, ny1, nx2, ny2):
                    continue

                text_b.element_merge(text_a)
                if text_b.category != "Text":
                    text_b.category = "Text"
                merged = True
                changed = True
                break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()
    return non_texts + texts


def refine_elements(compos, texts, intersection_bias=(2, 2), containment_ratio=0.8):
    """
    1. remove compos contained in text
    2. remove compos containing text area that"s too large
    3. store text in a compo if it"s contained by the compo as the compo"s text child element
    """
    elements = []
    contained_texts = []
    for compo in compos:
        # if compo.area > 10000:
        #     continue
        is_valid = True
        text_area = 0
        for text in texts:
            inter, iou, ioa, iob = compo.calc_intersection_area(text, bias=intersection_bias)
            if inter > 0:
                # the non-text is contained in the text compo
                if ioa >= containment_ratio:
                    is_valid = False
                    break
                text_area += inter
                # the text is contained in the non-text compo
                if iob >= containment_ratio and compo.category != "Block":
                    contained_texts.append(text)
        if is_valid and text_area / compo.area < containment_ratio:
            # for t in contained_texts:
            #     t.parent_id = compo.id
            # compo.children += contained_texts
            elements.append(compo)

    # elements += texts
    for text in texts:
        if text not in contained_texts:
            elements.append(text)
    return elements


def remove_fragments(elements, img_path):
    fragments = []
    screen = cv2.imread(img_path)
    if screen is None:
        return elements

    max_x = 0
    max_y = 0
    for element in elements:
        x1, y1, x2, y2 = element.put_bbox()
        if x2 > max_x:
            max_x = x2
        if y2 > max_y:
            max_y = y2

    target_w = max(1, int(max_x))
    target_h = max(1, int(max_y))
    if target_w != screen.shape[1] or target_h != screen.shape[0]:
        screen = cv2.resize(screen, (target_w, target_h))

    # elements contained in text blocks
    for i in range(len(elements) - 1):
        for j in range(i + 1, len(elements)):
            relation = elements[i].element_relation(elements[j], bias=(2, 2))
            if relation == -1 and elements[j].category == "Text":
                fragments.append(elements[i])
            if relation == 1 and elements[i].category == "Text":
                fragments.append(elements[j])

    for element in elements:
        if element.height > element.width * 10 or element.width > element.height * 30:  # over-thin elements
            fragments.append(element)
        # elif element.area > screen.shape[1] * screen.shape[0] // 9:  # over-large elements
        #     fragments.append(element)
        elif element.category == "Block":  # remove block-elements
            fragments.append(element)
        # elif element.area < screen.shape[1] * screen.shape[0] // 14400:  # over-small elements
        #     fragments.append(element)
        else:  # blank elements
            x1, y1, x2, y2 = element.put_bbox()
            ele = screen[y1:y2, x1:x2]
            if ele.size == 0:
                fragments.append(element)
                continue
            ele_hsv = cv2.cvtColor(ele, cv2.COLOR_BGR2HSV)
            hue_var = np.var(ele_hsv[:, :, 0])
            sat_var = np.var(ele_hsv[:, :, 1])
            val_var = np.var(ele_hsv[:, :, 2])
            total_var = hue_var + sat_var + val_var
            if total_var < 10:
                fragments.append(element)
            # ele_gray = cv2.cvtColor(ele, cv2.COLOR_BGR2GRAY)
            # ele_gray = cv2.GaussianBlur(ele_gray, (3, 3), 1)
            # ele_gray = np.array(ele_gray)
            # std = np.std(ele_gray)
            # if std < 1:
            #     fragments.append(element)

    new_elements = [ele for ele in elements if ele not in fragments]
    return new_elements


def merge_related_elements(elements):
    new_elements = elements.copy()
    changed = True
    while changed:
        changed = False
        temp_set = []
        for i in range(len(new_elements)):
            merged = False
            ele_a = new_elements[i]
            for j in range(i + 1, len(new_elements)):
                ele_b = new_elements[j]
                if (ele_a.category == "Compo" and ele_b.category == "Text" and ele_a.row_min < ele_b.row_min) \
                        or (ele_b.category == "Compo" and ele_a.category == "Text" and ele_a.row_min > ele_b.row_min):
                    inter_area, _, _, _ = ele_a.calc_intersection_area(ele_b, bias=(2, 10))
                    if inter_area > 0:
                        merged = True
                        changed = True
                        ele_b.element_merge(ele_a)
                        ele_b.category = "Combined"
                        break
                elif ele_a.category == "Compo" and ele_b.category == "Compo":
                    inter_area, _, _, _ = ele_a.calc_intersection_area(ele_b, bias=(4, 4))
                    if inter_area > 0:
                        merged = True
                        changed = True
                        ele_b.element_merge(ele_a)
                        ele_b.category = "Combined"
                        break
            if not merged:
                temp_set.append(ele_a)
        new_elements = temp_set.copy()
    return new_elements


def merge_list_rows(elements, img_shape):
    img_h = int(img_shape[0]) if img_shape and len(img_shape) >= 1 else 0
    img_w = int(img_shape[1]) if img_shape and len(img_shape) >= 2 else 0
    if img_h <= 0 or img_w <= 0:
        return elements

    if len(elements) < 35:
        return elements

    def center_y(e):
        return (e.row_min + e.row_max) / 2

    candidates = [
        e
        for e in elements
        if e.category != "Block"
        and e.height / img_h < 0.30
        and e.width / img_w < 0.98
    ]
    if len(candidates) < 35:
        return elements

    candidates = sorted(candidates, key=lambda e: (center_y(e), e.col_min))
    band_gap = max(12, int(0.06 * img_h))
    max_band_height = int(0.24 * img_h)

    bands = []
    cur = []
    cur_y_min = None
    cur_y_max = None
    cur_center = None

    for e in candidates:
        cy = center_y(e)
        if not cur:
            cur = [e]
            cur_y_min = e.row_min
            cur_y_max = e.row_max
            cur_center = cy
            continue

        next_y_min = min(cur_y_min, e.row_min)
        next_y_max = max(cur_y_max, e.row_max)
        if abs(cy - cur_center) <= band_gap and (next_y_max - next_y_min) <= max_band_height:
            cur.append(e)
            cur_y_min = next_y_min
            cur_y_max = next_y_max
            cur_center = (cur_center * (len(cur) - 1) + cy) / len(cur)
        else:
            bands.append(cur)
            cur = [e]
            cur_y_min = e.row_min
            cur_y_max = e.row_max
            cur_center = cy

    if cur:
        bands.append(cur)

    new_elements = elements.copy()
    for band in bands:
        if len(band) < 4:
            continue

        bx1 = min(e.col_min for e in band)
        by1 = min(e.row_min for e in band)
        bx2 = max(e.col_max for e in band)
        by2 = max(e.row_max for e in band)

        b_w = bx2 - bx1
        b_h = by2 - by1
        if b_w / img_w < 0.62:
            continue
        if b_h / img_h > 0.24:
            continue

        band_texts = sum(1 for e in band if e.category in ["Text", "Combined"])
        if not (len(band) >= 6 or (len(band) >= 4 and band_texts >= 2)):
            continue

        merged = Element(-1, (bx1, by1, bx2, by2), "Combined")
        for e in band:
            merged.element_merge(e)

        band_ids = {e.id for e in band}
        new_elements = [e for e in new_elements if e.id not in band_ids]
        new_elements.append(merged)

    return new_elements


def check_containment(elements):
    for i in range(len(elements) - 1):
        for j in range(i + 1, len(elements)):
            relation = elements[i].element_relation(elements[j], bias=(2, 2))
            if relation == -1:
                elements[j].children.append(elements[i])
                elements[i].parent_id = elements[j].id
            if relation == 1:
                elements[i].children.append(elements[j])
                elements[j].parent_id = elements[i].id


def remove_top_bar(elements, img_height):
    new_elements = []
    max_height = img_height * 0.04
    for ele in elements:
        if ele.row_min < 10 and ele.height < max_height:
            continue
        new_elements.append(ele)
    return new_elements


def remove_bottom_bar(elements, img_height):
    new_elements = []
    for ele in elements:
        # parameters for 800-height GUI
        if ele.row_min > 750 and 20 <= ele.height <= 30 and 20 <= ele.width <= 30:
            continue
        new_elements.append(ele)
    return new_elements


def compos_clip_and_fill(clip_root, org, compos):
    def most_pix_around(pad=6, offset=2):
        """
        determine the filled background color according to the most surrounding pixel
        """
        up = row_min - pad if row_min - pad >= 0 else 0
        left = col_min - pad if col_min - pad >= 0 else 0
        bottom = row_max + pad if row_max + pad < org.shape[0] - 1 else org.shape[0] - 1
        right = col_max + pad if col_max + pad < org.shape[1] - 1 else org.shape[1] - 1
        most = []
        for i in range(3):
            val = np.concatenate(
                (
                    org[up:row_min - offset, left:right, i].flatten(),
                    org[row_max + offset:bottom, left:right, i].flatten(),
                    org[up:bottom, left:col_min - offset, i].flatten(),
                    org[up:bottom, col_max + offset:right, i].flatten()
                )
            )
            most.append(int(np.argmax(np.bincount(val))))
        return most

    if os.path.exists(clip_root):
        shutil.rmtree(clip_root)
    os.mkdir(clip_root)

    bkg = org.copy()
    cls_dirs = []
    for compo in compos:
        cls = compo["class"]
        if cls == "Background":
            compo["path"] = p_join(clip_root, "bkg.png")
            continue
        c_root = p_join(clip_root, cls)
        c_path = p_join(c_root, str(compo["id"]) + ".jpg")
        compo["path"] = c_path
        if cls not in cls_dirs:
            os.mkdir(c_root)
            cls_dirs.append(cls)

        position = compo["position"]
        col_min, row_min, col_max, row_max = position["column_min"], position["row_min"], position["column_max"], \
            position["row_max"]
        cv2.imwrite(c_path, org[row_min:row_max, col_min:col_max])
        # Fill up the background area
        cv2.rectangle(bkg, (col_min, row_min), (col_max, row_max), most_pix_around(), -1)
    cv2.imwrite(p_join(clip_root, "bkg.png"), bkg)


def merge(img_path, compo_path, text_path, merge_root=None, is_paragraph=False, is_remove_bar=True, show=False,
          wait_key=0):

    if merge_root is not None:
        os.makedirs(merge_root, exist_ok=True)
        
    compo_json = json.load(open(compo_path, "r"))
    text_json = json.load(open(text_path, "r"))

    # load text and non-text compo
    ele_id = 0
    compos = []
    for compo in compo_json["compos"]:
        element = Element(
            ele_id,
            (compo["column_min"], compo["row_min"], compo["column_max"], compo["row_max"]),
            compo["class"],
        )
        compos.append(element)
        ele_id += 1
    texts = []
    for text in text_json["texts"]:
        element = Element(
            ele_id,
            (text["column_min"], text["row_min"], text["column_max"], text["row_max"]),
            "Text",
            text_content=text["content"],
        )
        texts.append(element)
        ele_id += 1
    if compo_json["img_shape"] != text_json["img_shape"]:
        resize_ratio = compo_json["img_shape"][0] / text_json["img_shape"][0]
        for text in texts:
            text.resize(resize_ratio)
    else:
        resize_ratio = 1

    # check the original detected elements
    img = cv2.imread(img_path)
    img_resize = cv2.resize(img, (compo_json["img_shape"][1], compo_json["img_shape"][0]))
    show_elements(img_resize, texts + compos, show=show, win_name="all elements before merging", wait_key=wait_key)

    # refine elements
    # texts = refine_texts(texts, compo_json["img_shape"])
    elements = refine_elements(compos, texts)
    if is_remove_bar:
        elements = remove_top_bar(elements, img_height=compo_json["img_shape"][0])
        elements = remove_bottom_bar(elements, img_height=compo_json["img_shape"][0])
    if is_paragraph:
        elements = merge_text_line_to_paragraph(elements, max_line_gap=7, img_shape=compo_json.get("img_shape"))

    show_elements(img_resize, elements, show=show, win_name="1", wait_key=wait_key)

    elements = remove_fragments(elements, img_path)
    show_elements(img_resize, elements, show=show, win_name="2", wait_key=wait_key)
    elements = merge_related_elements(elements)
    show_elements(img_resize, elements, show=show, win_name="3", wait_key=wait_key)
    elements = merge_list_rows(elements, img_shape=compo_json.get("img_shape"))
    show_elements(img_resize, elements, show=show, win_name="4", wait_key=wait_key)
    reassign_ids(elements)
    check_containment(elements)
    board = show_elements(img_resize, elements, show=show, win_name="elements after merging", wait_key=wait_key)

    # save all merged elements, clips and blank background
    name = img_path.replace("\\", "/").split("/")[-1][:-4]
    components = save_elements(p_join(merge_root, name + ".json"), elements, img_resize.shape)
    output_path = p_join(merge_root, name + ".jpg")
    cv2.imwrite(output_path, board)
    # print("[Merge Completed] Input: %s Output: %s" % (img_path, p_join(merge_root, name + ".jpg")))
    return output_path, components, resize_ratio
