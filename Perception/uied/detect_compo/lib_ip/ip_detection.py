import cv2
import numpy as np

import Perception.uied.detect_compo.lib_ip.ip_draw as draw
import Perception.uied.detect_compo.lib_ip.ip_preprocessing as pre
from Perception.uied.detect_compo.lib_ip.Component import Component
import Perception.uied.detect_compo.lib_ip.Component as Compo
from Perception.uied.CONFIG_UIED import Config
C = Config()


def merge_intersected_corner(compos, org, is_merge_contained_ele, max_gap=(0, 0), max_ele_height=25):
    """
    :param is_merge_contained_ele: if true, merge compos nested in others
    :param max_gap: (horizontal_distance, vertical_distance) to be merge into one line/column
    :param max_ele_height: if higher than it, recognize the compo as text
    :return:
    """
    changed = False
    new_compos = []
    Compo.compos_update(compos, org.shape)
    for i in range(len(compos)):
        merged = False
        cur_compo = compos[i]
        for j in range(len(new_compos)):
            relation = cur_compo.compo_relation(new_compos[j], max_gap)
            # print(relation)
            # draw.draw_bounding_box(org, [cur_compo, new_compos[j]], name='b-merge', show=True)
            # merge compo[i] to compo[j] if
            # 1. compo[j] contains compo[i]
            # 2. compo[j] intersects with compo[i] with certain iou
            # 3. is_merge_contained_ele and compo[j] is contained in compo[i]
            if relation == 1 or \
                    relation == 2 or \
                    (is_merge_contained_ele and relation == -1):
                # (relation == 2 and new_compos[j].height < max_ele_height and cur_compo.height < max_ele_height) or\

                new_compos[j].compo_merge(cur_compo)
                cur_compo = new_compos[j]
                # draw.draw_bounding_box(org, [new_compos[j]], name='a-merge', show=True)
                merged = True
                changed = True
                # break
        if not merged:
            new_compos.append(compos[i])

    if not changed:
        return compos
    else:
        return merge_intersected_corner(new_compos, org, is_merge_contained_ele, max_gap, max_ele_height)


def merge_intersected_compos(compos):
    changed = True
    while changed:
        changed = False
        temp_set = []
        for compo_a in compos:
            merged = False
            for compo_b in temp_set:
                if compo_a.compo_relation(compo_b) == 2:
                    compo_b.compo_merge(compo_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(compo_a)
        compos = temp_set.copy()
    return compos


def rm_contained_compos_not_in_block(compos):
    """
    remove all components contained by others that are not Block
    """
    marked = np.full(len(compos), False)
    for i in range(len(compos) - 1):
        for j in range(i + 1, len(compos)):
            relation = compos[i].compo_relation(compos[j])
            if relation == -1 and compos[j].category != 'Block':
                marked[i] = True
            if relation == 1 and compos[i].category != 'Block':
                marked[j] = True
    new_compos = []
    for i in range(len(marked)):
        if not marked[i]:
            new_compos.append(compos[i])
    return new_compos


def merge_text(compos, org_shape, max_word_gad=4, max_word_height=20):
    def is_text_line(compo_a, compo_b):
        (col_min_a, row_min_a, col_max_a, row_max_a) = compo_a.put_bbox()
        (col_min_b, row_min_b, col_max_b, row_max_b) = compo_b.put_bbox()

        col_min_s = max(col_min_a, col_min_b)
        col_max_s = min(col_max_a, col_max_b)
        row_min_s = max(row_min_a, row_min_b)
        row_max_s = min(row_max_a, row_max_b)

        # on the same line
        # if abs(row_min_a - row_min_b) < max_word_gad and abs(row_max_a - row_max_b) < max_word_gad:
        if row_min_s < row_max_s:
            # close distance
            if col_min_s < col_max_s or \
                    (0 < col_min_b - col_max_a < max_word_gad) or (0 < col_min_a - col_max_b < max_word_gad):
                return True
        return False

    changed = False
    new_compos = []
    row, col = org_shape[:2]
    for i in range(len(compos)):
        merged = False
        height = compos[i].height
        # ignore non-text
        # if height / row > max_word_height_ratio\
        #         or compos[i].category != 'Text':
        if height > max_word_height:
            new_compos.append(compos[i])
            continue
        for j in range(len(new_compos)):
            # if compos[j].category != 'Text':
            #     continue
            if is_text_line(compos[i], new_compos[j]):
                new_compos[j].compo_merge(compos[i])
                merged = True
                changed = True
                break
        if not merged:
            new_compos.append(compos[i])

    if not changed:
        return compos
    else:
        return merge_text(new_compos, org_shape)


def rm_top_or_bottom_corners(components, org_shape, top_bottom_height=C.THRESHOLD_TOP_BOTTOM_BAR):
    new_compos = []
    height, width = org_shape[:2]
    for compo in components:
        (column_min, row_min, column_max, row_max) = compo.put_bbox()
        # remove big ones
        # if (row_max - row_min) / height > 0.65 and (column_max - column_min) / width > 0.8:
        #     continue
        if not (row_max < height * top_bottom_height[0] or row_min > height * top_bottom_height[1]):
            new_compos.append(compo)
    return new_compos


def rm_line_v_h(binary, show=False, max_line_thickness=C.THRESHOLD_LINE_THICKNESS):
    def check_continuous_line(line, edge):
        continuous_length = 0
        line_start = -1
        for j, p in enumerate(line):
            if p > 0:
                if line_start == -1:
                    line_start = j
                continuous_length += 1
            elif continuous_length > 0:
                if continuous_length / edge > 0.6:
                    return [line_start, j]
                continuous_length = 0
                line_start = -1

        if continuous_length / edge > 0.6:
            return [line_start, len(line)]
        else:
            return None

    def extract_line_area(line, start_idx, flag='v'):
        for e, l in enumerate(line):
            if flag == 'v':
                map_line[start_idx + e, l[0]:l[1]] = binary[start_idx + e, l[0]:l[1]]

    map_line = np.zeros(binary.shape[:2], dtype=np.uint8)
    cv2.imshow('binary', binary)

    width = binary.shape[1]
    start_row = -1
    line_area = []
    for i, row in enumerate(binary):
        line_v = check_continuous_line(row, width)
        if line_v is not None:
            # new line
            if start_row == -1:
                start_row = i
                line_area = []
            line_area.append(line_v)
        else:
            # checking line
            if start_row != -1:
                if i - start_row < max_line_thickness:
                    # binary[start_row: i] = 0
                    # map_line[start_row: i] = binary[start_row: i]
                    print(line_area, start_row, i)
                    extract_line_area(line_area, start_row)
                start_row = -1

    height = binary.shape[0]
    start_col = -1
    for i in range(width):
        col = binary[:, i]
        line_h = check_continuous_line(col, height)
        if line_h is not None:
            # new line
            if start_col == -1:
                start_col = i
        else:
            # checking line
            if start_col != -1:
                if i - start_col < max_line_thickness:
                    # binary[:, start_col: i] = 0
                    map_line[:, start_col: i] = binary[:, start_col: i]
                start_col = -1

    binary -= map_line

    if show:
        cv2.imshow('no-line', binary)
        cv2.imshow('lines', map_line)
        cv2.waitKey()


def rm_line(binary,
            max_line_thickness=C.THRESHOLD_LINE_THICKNESS,
            min_line_length_ratio=C.THRESHOLD_LINE_MIN_LENGTH,
            show=False, wait_key=0):
    def is_valid_line(line):
        line_length = 0
        line_gap = 0
        for j in line:
            if j > 0:
                if line_gap > 5:
                    return False
                line_length += 1
                line_gap = 0
            elif line_length > 0:
                line_gap += 1
        if line_length / width > 0.95:
            return True
        return False

    height, width = binary.shape[:2]
    board = np.zeros(binary.shape[:2], dtype=np.uint8)

    start_row, end_row = -1, -1
    check_line = False
    check_gap = False
    for i, row in enumerate(binary):
        # line_ratio = (sum(row) / 255) / width
        # if line_ratio > 0.9:
        if is_valid_line(row):
            # new start: if it is checking a new line, mark this row as start
            if not check_line:
                start_row = i
                check_line = True
        else:
            # end the line
            if check_line:
                # thin enough to be a line, then start checking gap
                if i - start_row < max_line_thickness:
                    end_row = i
                    check_gap = True
                else:
                    start_row, end_row = -1, -1
                check_line = False
        # check gap
        if check_gap and i - end_row > max_line_thickness:
            binary[start_row: end_row] = 0
            start_row, end_row = -1, -1
            check_line = False
            check_gap = False

    if (check_line and (height - start_row) < max_line_thickness) or check_gap:
        binary[start_row: end_row] = 0

    if show:
        cv2.imshow('no-line binary', binary)
        if wait_key is not None:
            cv2.waitKey(wait_key)
        if wait_key == 0:
            cv2.destroyWindow('no-line binary')


def rm_noise_compos(compos):
    compos_new = []
    for compo in compos:
        if compo.category == 'Noise':
            continue
        compos_new.append(compo)
    return compos_new


def rm_noise_in_large_img(compos, org,
                      max_compo_scale=C.THRESHOLD_COMPO_MAX_SCALE):
    row, column = org.shape[:2]
    remain = np.full(len(compos), True)
    new_compos = []
    for compo in compos:
        if compo.category == 'Image':
            for i in compo.contain:
                remain[i] = False
    for i in range(len(remain)):
        if remain[i]:
            new_compos.append(compos[i])
    return new_compos


def detect_compos_in_img(compos, binary, org, max_compo_scale=C.THRESHOLD_COMPO_MAX_SCALE, show=False):
    compos_new = []
    row, column = binary.shape[:2]
    for compo in compos:
        if compo.category == 'Image':
            compo.compo_update_bbox_area()
            # org_clip = compo.compo_clipping(org)
            # bin_clip = pre.binarization(org_clip, show=show)
            bin_clip = compo.compo_clipping(binary)
            bin_clip = pre.reverse_binary(bin_clip, show=show)

            compos_rec, compos_nonrec = component_detection(bin_clip, test=False, step_h=10, step_v=10, rec_detect=True)
            for compo_rec in compos_rec:
                compo_rec.compo_relative_position(compo.bbox.col_min, compo.bbox.row_min)
                if compo_rec.bbox_area / compo.bbox_area < 0.8 and compo_rec.bbox.height > 20 and compo_rec.bbox.width > 20:
                    compos_new.append(compo_rec)
                    # draw.draw_bounding_box(org, [compo_rec], show=True)

            # compos_inner = component_detection(bin_clip, rec_detect=False)
            # for compo_inner in compos_inner:
            #     compo_inner.compo_relative_position(compo.bbox.col_min, compo.bbox.row_min)
            #     draw.draw_bounding_box(org, [compo_inner], show=True)
            #     if compo_inner.bbox_area / compo.bbox_area < 0.8:
            #         compos_new.append(compo_inner)
    compos += compos_new


def compo_filter(compos, min_area, img_shape):
    max_height = img_shape[0] * 0.8
    compos_new = []
    for compo in compos:
        if compo.area < min_area:
            continue
        if compo.height > max_height:
            continue
        ratio_h = compo.width / compo.height
        ratio_w = compo.height / compo.width
        if ratio_h > 50 or ratio_w > 40 or \
                (min(compo.height, compo.width) < 8 and max(ratio_h, ratio_w) > 10):
            continue
        compos_new.append(compo)

    # Suppress text-like tiny component clusters in content area (common OCR-like
    # character fragments), while preserving small icons in top/bottom toolbars.
    if len(compos_new) < 6:
        return compos_new

    img_h, img_w = img_shape[:2]
    tiny_max_w = max(40, int(img_w * 0.11))
    tiny_max_h = max(32, int(img_h * 0.05))
    tiny_max_area = max(1200, int(img_h * img_w * 0.004))
    mid_top = int(img_h * 0.12)
    mid_bottom = int(img_h * 0.90)
    row_eps = max(8, int(img_h * 0.015))
    col_eps = max(80, int(img_w * 0.22))

    tiny_indices = []
    centers = []
    for idx, compo in enumerate(compos_new):
        cx = (compo.bbox.col_min + compo.bbox.col_max) // 2
        cy = (compo.bbox.row_min + compo.bbox.row_max) // 2
        centers.append((cx, cy))
        if compo.width <= tiny_max_w and compo.height <= tiny_max_h and compo.area <= tiny_max_area:
            tiny_indices.append(idx)

    remove_indices = set()
    for i in tiny_indices:
        cx, cy = centers[i]
        if cy <= mid_top or cy >= mid_bottom:
            continue
        neighbors = 0
        for j in tiny_indices:
            if i == j:
                continue
            cx2, cy2 = centers[j]
            if abs(cy - cy2) <= row_eps and abs(cx - cx2) <= col_eps:
                neighbors += 1
                if neighbors >= 2:
                    remove_indices.add(i)
                    break

    if not remove_indices:
        return compos_new
    return [compo for idx, compo in enumerate(compos_new) if idx not in remove_indices]


def is_block(clip, thread=0.15):
    """
    Block is a rectangle border enclosing a group of compos (consider it as a wireframe)
    Check if a compo is block by checking if the inner side of its border is blank
    """
    side = 4  # scan 4 lines inner forward each border
    # top border - scan top down
    blank_count = 0
    for i in range(1, 5):
        if np.count_nonzero(clip[side + i]) > thread * clip.shape[1]:
            blank_count += 1
    if blank_count > 2: return False
    # left border - scan left to right
    blank_count = 0
    for i in range(1, 5):
        if np.count_nonzero(clip[:, side + i]) > thread * clip.shape[0]:
            blank_count += 1
    if blank_count > 2: return False

    side = -4
    # bottom border - scan bottom up
    blank_count = 0
    for i in range(-1, -5, -1):
        if np.count_nonzero(clip[side + i]) > thread * clip.shape[1]:
            blank_count += 1
    if blank_count > 2: return False
    # right border - scan right to left
    blank_count = 0
    for i in range(-1, -5, -1):
        if np.count_nonzero(clip[:, side + i]) > thread * clip.shape[0]:
            blank_count += 1
    if blank_count > 2: return False
    return True


def compo_block_recognition(binary, compos, block_side_length=0.15):
    height, width = binary.shape
    for compo in compos:
        if compo.height / height > block_side_length and compo.width / width > block_side_length:
            clip = compo.compo_clipping(binary)
            if is_block(clip):
                compo.category = 'Block'


def _bbox_metrics(compo):
    x1, y1, x2, y2 = compo.put_bbox()
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    return x1, y1, x2, y2, w, h, cx


def _overlap_ratio_1d(a1, a2, b1, b2):
    inter = max(0, min(a2, b2) - max(a1, b1))
    denom = max(1, min(a2 - a1, b2 - b1))
    return inter / float(denom)


def _vertical_gap(compo_a, compo_b):
    _, ay1, _, ay2, _, _, _ = _bbox_metrics(compo_a)
    _, by1, _, by2, _, _, _ = _bbox_metrics(compo_b)
    if ay2 < by1:
        return by1 - ay2
    if by2 < ay1:
        return ay1 - by2
    return 0


def _is_top_thin_fragment(compo, img_w, top_limit, thin_max_h, thin_min_w, thin_max_w):
    x1, _, x2, y2, w, h, _ = _bbox_metrics(compo)
    if y2 > top_limit:
        return False
    if h > thin_max_h:
        return False
    if w < thin_min_w or w > thin_max_w:
        return False
    if (w / float(max(1, h))) < 4.0:
        return False
    # Avoid merging tiny status-bar speckles at screen edges.
    if x1 <= 0 or x2 >= img_w:
        return False
    return True


def _fragments_related(compo_a, compo_b, max_vertical_gap, min_x_overlap_ratio, max_center_shift):
    ax1, _, ax2, _, aw, _, acx = _bbox_metrics(compo_a)
    bx1, _, bx2, _, bw, _, bcx = _bbox_metrics(compo_b)
    x_overlap = _overlap_ratio_1d(ax1, ax2, bx1, bx2)
    if x_overlap < min_x_overlap_ratio and abs(acx - bcx) > max_center_shift:
        return False
    if abs(aw - bw) > max(8, int(0.6 * max(aw, bw))):
        return False
    if _vertical_gap(compo_a, compo_b) > max_vertical_gap:
        return False
    return True


def merge_top_thin_icon_fragments(compos, img_shape):
    """
    Merge thin horizontal fragments in top toolbar into one icon candidate.

    This targets cases like hamburger/back icons where each stroke is detected as
    a separate 2-3px component and later becomes hard to anchor.
    """
    if len(compos) < 2:
        return compos

    img_h, img_w = img_shape[:2]
    top_limit = int(img_h * 0.18)
    thin_max_h = 3
    thin_min_w = max(8, int(img_w * 0.01))
    thin_max_w = max(thin_min_w + 1, int(img_w * 0.12))
    max_vertical_gap = max(5, int(img_h * 0.01))
    min_x_overlap_ratio = 0.60
    max_center_shift = max(6, int(img_w * 0.02))
    max_group_h = max(8, int(img_h * 0.08))
    max_group_w = max(16, int(img_w * 0.20))

    candidate_indices = []
    for idx, compo in enumerate(compos):
        if _is_top_thin_fragment(
                compo=compo,
                img_w=img_w,
                top_limit=top_limit,
                thin_max_h=thin_max_h,
                thin_min_w=thin_min_w,
                thin_max_w=thin_max_w):
            candidate_indices.append(idx)

    if len(candidate_indices) < 2:
        return compos

    neighbors = {idx: set() for idx in candidate_indices}
    for i, idx_a in enumerate(candidate_indices):
        compo_a = compos[idx_a]
        for idx_b in candidate_indices[i + 1:]:
            compo_b = compos[idx_b]
            if _fragments_related(
                    compo_a=compo_a,
                    compo_b=compo_b,
                    max_vertical_gap=max_vertical_gap,
                    min_x_overlap_ratio=min_x_overlap_ratio,
                    max_center_shift=max_center_shift):
                neighbors[idx_a].add(idx_b)
                neighbors[idx_b].add(idx_a)

    visited = set()
    remove_indices = set()
    for start in candidate_indices:
        if start in visited:
            continue
        stack = [start]
        component_indices = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component_indices.append(current)
            for nxt in neighbors[current]:
                if nxt not in visited:
                    stack.append(nxt)

        if len(component_indices) < 2:
            continue

        component_indices.sort(key=lambda idx: (compos[idx].bbox.row_min, compos[idx].bbox.col_min))
        bx1 = min(compos[idx].bbox.col_min for idx in component_indices)
        by1 = min(compos[idx].bbox.row_min for idx in component_indices)
        bx2 = max(compos[idx].bbox.col_max for idx in component_indices)
        by2 = max(compos[idx].bbox.row_max for idx in component_indices)
        group_h = by2 - by1
        group_w = bx2 - bx1
        if group_h > max_group_h or group_w > max_group_w:
            continue

        base_idx = component_indices[0]
        base = compos[base_idx]
        for idx in component_indices[1:]:
            base.compo_merge(compos[idx])
            remove_indices.add(idx)

    if not remove_indices:
        return compos
    return [compo for idx, compo in enumerate(compos) if idx not in remove_indices]


# take the binary image as input
# calculate the connected regions -> get the bounding boundaries of them -> check if those regions are rectangles
# return all boundaries and boundaries of rectangles
def component_detection(binary, min_obj_area,
                        line_thickness=C.THRESHOLD_LINE_THICKNESS,
                        min_rec_evenness=C.THRESHOLD_REC_MIN_EVENNESS,
                        max_dent_ratio=C.THRESHOLD_REC_MAX_DENT_RATIO,
                        step_h = 5, step_v = 2,
                        rec_detect=False, show=False, test=False):
    """
    :param binary: Binary image from pre-processing
    :param min_obj_area: If not pass then ignore the small object
    :param min_obj_perimeter: If not pass then ignore the small object
    :param line_thickness: If not pass then ignore the slim object
    :param min_rec_evenness: If not pass then this object cannot be rectangular
    :param max_dent_ratio: If not pass then this object cannot be rectangular
    :return: boundary: [top, bottom, left, right]
                        -> up, bottom: list of (column_index, min/max row border)
                        -> left, right: list of (row_index, min/max column border) detect range of each row
    """
    mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), dtype=np.uint8)
    compos_all = []
    compos_rec = []
    compos_nonrec = []
    row, column = binary.shape[0], binary.shape[1]
    for i in range(0, row, step_h):
        for j in range(i % 2, column, step_v):
            if binary[i, j] == 255 and mask[i, j] == 0:
                # get connected area
                # region = util.boundary_bfs_connected_area(binary, i, j, mask)

                mask_copy = mask.copy()
                ff = cv2.floodFill(binary, mask, (j, i), None, 0, 0, cv2.FLOODFILL_MASK_ONLY)
                if ff[0] < min_obj_area: continue
                mask_copy = mask - mask_copy
                region = np.reshape(cv2.findNonZero(mask_copy[1:-1, 1:-1]), (-1, 2))
                region = [(p[1], p[0]) for p in region]

                # filter out some compos
                component = Component(region, binary.shape)
                # calculate the boundary of the connected area
                # ignore small area
                if component.width < 3 or component.height < 3:
                    continue
                # check if it is line by checking the length of edges
                # if component.compo_is_line(line_thickness):
                #     continue

                if test:
                    print('Area:%d' % (len(region)))
                    draw.draw_boundary([component], binary.shape, show=True)

                compos_all.append(component)

                if rec_detect:
                    # rectangle check
                    if component.compo_is_rectangle(min_rec_evenness, max_dent_ratio):
                        component.rect_ = True
                        compos_rec.append(component)
                    else:
                        component.rect_ = False
                        compos_nonrec.append(component)

                if show:
                    print('Area:%d' % (len(region)))
                    draw.draw_boundary(compos_all, binary.shape, show=True)

    # draw.draw_boundary(compos_all, binary.shape, show=True)
    if rec_detect:
        return compos_rec, compos_nonrec
    else:
        return compos_all


def nested_components_detection(grey, org, grad_thresh,
                   show=False, write_path=None,
                   step_h=10, step_v=10,
                   line_thickness=C.THRESHOLD_LINE_THICKNESS,
                   min_rec_evenness=C.THRESHOLD_REC_MIN_EVENNESS,
                   max_dent_ratio=C.THRESHOLD_REC_MAX_DENT_RATIO):
    """
    :param grey: grey-scale of original image
    :return: corners: list of [(top_left, bottom_right)]
                        -> top_left: (column_min, row_min)
                        -> bottom_right: (column_max, row_max)
    """
    compos = []
    mask = np.zeros((grey.shape[0]+2, grey.shape[1]+2), dtype=np.uint8)
    broad = np.zeros((grey.shape[0], grey.shape[1], 3), dtype=np.uint8)
    broad_all = broad.copy()

    row, column = grey.shape[0], grey.shape[1]
    for x in range(0, row, step_h):
        for y in range(0, column, step_v):
            if mask[x, y] == 0:
                # region = flood_fill_bfs(grey, x, y, mask)

                # flood fill algorithm to get background (layout block)
                mask_copy = mask.copy()
                ff = cv2.floodFill(grey, mask, (y, x), None, grad_thresh, grad_thresh, cv2.FLOODFILL_MASK_ONLY)
                # ignore small regions
                if ff[0] < 500: continue
                mask_copy = mask - mask_copy
                region = np.reshape(cv2.findNonZero(mask_copy[1:-1, 1:-1]), (-1, 2))
                region = [(p[1], p[0]) for p in region]

                compo = Component(region, grey.shape)
                # draw.draw_region(region, broad_all)
                # if block.height < 40 and block.width < 40:
                #     continue
                if compo.height < 30:
                    continue

                # print(block.area / (row * column))
                if compo.area / (row * column) > 0.9:
                    continue
                elif compo.area / (row * column) > 0.7:
                    compo.redundant = True

                # get the boundary of this region
                # ignore lines
                if compo.compo_is_line(line_thickness):
                    continue
                # ignore non-rectangle as blocks must be rectangular
                if not compo.compo_is_rectangle(min_rec_evenness, max_dent_ratio):
                    continue
                # if block.height/row < min_block_height_ratio:
                #     continue
                compos.append(compo)
                # draw.draw_region(region, broad)
    if show:
        cv2.imshow('flood-fill all', broad_all)
        cv2.imshow('block', broad)
        cv2.waitKey()
    if write_path is not None:
        cv2.imwrite(write_path, broad)
    return compos
