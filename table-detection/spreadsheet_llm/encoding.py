"""Spreadsheet encoding — V3, faithful to supplementary material code.

Module 1 implements the full C# heuristic pipeline from:
  heuristics_structural_anchor_extraction/code/

Pipeline (TableSenseDetect path ≤ 30 000 cells, RegionGrowthDetect otherwise):

  1.  CellFeatures extraction            (CellFeatures.cs)
      — added alpha_ratio, number_ratio, sp_char_ratio, text_length
  2.  Value maps + prefix sums           (SheetMap.CalculateBasicValueMaps)
  3.  computeValueDiff                   (SheetMap.computeValueDiff)
  4.  ProposeBoundaryLines               (SheetMap.ProposeBoundaryLines)
  5.  CohensionDetection                 (SheetMap: merged + color + border)
      — dealWithBorderCohensionRegions now implemented from SheetMap.cs
  6.  GenerateBlockRegions               (SheetMap.GenerateBlockRegions)
  7.  RegionGrowthDetector × 16 calls    (RegionGrowthDetector.FindConnectedRanges)
  8.  GenerateRawCandidateBoxes per block (TableDetectionHybrid)
  9.  BlockCandidatesRefineAndFilter      (TableDetectionHybrid)
      — UpHeaderTrim → OverlapCohensionFilter → OverlapBorderCohesionFilter
        → LittleBoxesFilter → OverlapUpHeaderFilter → SurroundingBoundariesTrim
        → OverlapUpHeaderFilter → SplittedEmptyLinesFilter
  10. CandidatesRefineAndFilter           (TableDetectionHybrid, partial)
      — BorderCohensionsAddition → LittleBoxesFilter → RetrieveDistantUpHeader
        → VerticalRelationalMerge → SuppressionSoftFilter → HeaderPriorityFilter
        → PairAlikeContainsFilter → PairContainsFilter → NestingCombinationFilter
        → ForcedBorderFilter → AdjoinHeaderFilter → LittleBoxesFilter
        → AddRegionGrowth → AddCompactRegionGrowth → MergeFilter
        → RetrieveLeftHeader → LeftHeaderTrim → BottomTrim
        → RetrieveUpHeader(1) → RetrieveUpHeader(2) → UpTrimSimple
        → LittleBoxesFilter
      Skipped (incomplete source): OverlapHeaderFilter, PairContainsFilterHard,
        CombineContainsFilterHard, CombineContainsFillArea/LineFilterSoft,
        ContainsLittleFilter
  11. RegionGrowthDetect (fallback) full filter chain
  12. DELTA=4 expansion
  13. delete_space
  14. coordinate_rearrangement

Module 2 — unchanged (inverted-index translation).
Module 3 — deactivated for fine-tuning per paper Table 2 (M1+M2 best).
"""
from __future__ import annotations

import re
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from .utils import cell_address, parse_address
from .spreadsheet_io import SpreadsheetData


# ============================================================
#  Constants
# ============================================================

DELTA = 4
MAX_SIMPLE_CELLS = 30_000

_THRESH_PAIRS: List[Tuple[int, int]] = [
    (h, v)
    for h in range(1, 7)
    for v in range(1, (3 if h < 3 else 2))
]

Box = Tuple[int, int, int, int]   # (r0, c0, r1, c1) 0-indexed


# ============================================================
#  Section A: 2-D prefix sum
# ============================================================

class _PS2D:
    """2-D prefix sum for fast rectangular sum queries (0-indexed)."""

    def __init__(self, matrix: List[List[int]]):
        nr = len(matrix)
        nc = len(matrix[0]) if nr else 0
        ps = [[0] * (nc + 1) for _ in range(nr + 1)]
        for r in range(nr):
            for c in range(nc):
                ps[r + 1][c + 1] = (matrix[r][c]
                                     + ps[r][c + 1]
                                     + ps[r + 1][c]
                                     - ps[r][c])
        self._ps = ps
        self._nr = nr
        self._nc = nc

    def query(self, r0: int, c0: int, r1: int, c1: int) -> int:
        r0 = max(r0, 0); c0 = max(c0, 0)
        r1 = min(r1, self._nr - 1); c1 = min(c1, self._nc - 1)
        if r0 > r1 or c0 > c1:
            return 0
        ps = self._ps
        return (ps[r1 + 1][c1 + 1]
                - ps[r0][c1 + 1]
                - ps[r1 + 1][c0]
                + ps[r0][c0])


# ============================================================
#  Section B: CellFeatures extraction
# ============================================================

def _parse_merged_cells(data: SpreadsheetData) -> List[Tuple[int, int, int, int]]:
    result = []
    for rng_str in (data.merged_cells or []):
        parts = rng_str.replace(' ', '').split(':')
        if len(parts) == 2:
            a = parse_address(parts[0])
            b = parse_address(parts[1])
            if a and b:
                r0, c0 = min(a[0], b[0]), min(a[1], b[1])
                r1, c1 = max(a[0], b[0]), max(a[1], b[1])
                result.append((r0, c0, r1, c1))
    return result


_SP_CHARS = set('*/：\\-+(')


def _extract_features(data: SpreadsheetData):
    """Return all cell features.

    Returns:
      (content_strs, has_content,
       has_top, has_bottom, has_left, has_right,
       has_color, merged_regions,
       alpha_ratio, number_ratio, sp_char_ratio, text_length)

    alpha_ratio, number_ratio, sp_char_ratio, text_length follow
    CellFeatures.cs: computed after merged-cell value propagation.
    sp_char_ratio = fraction of special chars among chars AFTER the first.
    """
    nr, nc = data.n_rows, data.n_cols

    content_strs: List[List[str]] = [
        ['' if v is None else str(v).strip() for v in row]
        for row in data.values
    ]

    merged_regions = _parse_merged_cells(data)

    for r0, c0, r1, c1 in merged_regions:
        top_val = content_strs[r0][c0]
        for r in range(r0, min(r1 + 1, nr)):
            for c in range(c0, min(c1 + 1, nc)):
                content_strs[r][c] = top_val

    has_content = [[bool(content_strs[r][c]) for c in range(nc)] for r in range(nr)]

    has_top    = [[False] * nc for _ in range(nr)]
    has_bottom = [[False] * nc for _ in range(nr)]
    has_left   = [[False] * nc for _ in range(nr)]
    has_right  = [[False] * nc for _ in range(nr)]
    has_color  = [[False] * nc for _ in range(nr)]

    for r in range(nr):
        for c in range(nc):
            attrs = (data.format_attrs[r][c]
                     if data.format_attrs
                        and r < len(data.format_attrs)
                        and c < len(data.format_attrs[r])
                     else [])
            has_top[r][c]    = 'TopBorder'    in attrs
            has_bottom[r][c] = 'BottomBorder' in attrs
            has_left[r][c]   = 'LeftBorder'   in attrs
            has_right[r][c]  = 'RightBorder'  in attrs
            has_color[r][c]  = 'FillColor'    in attrs

    # Per-cell ratios (CellFeatures.cs)
    alpha_ratio  = [[0.0] * nc for _ in range(nr)]
    number_ratio = [[0.0] * nc for _ in range(nr)]
    sp_char_ratio = [[0.0] * nc for _ in range(nr)]
    text_length  = [[0]   * nc for _ in range(nr)]

    for r in range(nr):
        for c in range(nc):
            s = content_strs[r][c]
            n = len(s)
            text_length[r][c] = n
            if n == 0:
                continue
            alpha_ratio[r][c]  = sum(ch.isalpha() for ch in s) / n
            number_ratio[r][c] = sum(ch.isdigit() for ch in s) / n
            if n > 1:
                tail = s[1:]
                sp_char_ratio[r][c] = sum(ch in _SP_CHARS for ch in tail) / len(tail)

    return (content_strs, has_content,
            has_top, has_bottom, has_left, has_right,
            has_color, merged_regions,
            alpha_ratio, number_ratio, sp_char_ratio, text_length)


# ============================================================
#  Section C: Value maps
# ============================================================

def _build_value_maps(has_content, has_top, has_bottom, has_left, has_right,
                      has_color, nr: int, nc: int):
    vm_content = [[0] * nc for _ in range(nr)]
    vm_ce      = [[0] * nc for _ in range(nr)]
    vm_border  = [[0] * nc for _ in range(nr)]
    vm_bcol    = [[0] * nc for _ in range(nr)]
    vm_brow    = [[0] * nc for _ in range(nr)]
    vm_color   = [[0] * nc for _ in range(nr)]
    vm_all     = [[0] * nc for _ in range(nr)]

    for r in range(nr):
        for c in range(nc):
            cnt = 2 if has_content[r][c] else 0
            vm_content[r][c] = cnt
            vm_ce[r][c]      = cnt
            n_b = (has_top[r][c] + has_bottom[r][c]
                   + has_left[r][c] + has_right[r][c])
            brd = (n_b - 1 if n_b >= 3 else n_b) * 2
            vm_border[r][c] = brd
            vm_bcol[r][c]   = 2 if (has_left[r][c] or has_right[r][c]) else 0
            vm_brow[r][c]   = 2 if (has_top[r][c]  or has_bottom[r][c]) else 0
            col = 2 if has_color[r][c] else 0
            vm_color[r][c] = col
            vm_all[r][c]   = min(cnt + brd + col, 16)

    return (_PS2D(vm_content), _PS2D(vm_ce),
            _PS2D(vm_border),  _PS2D(vm_bcol), _PS2D(vm_brow),
            _PS2D(vm_color),   _PS2D(vm_all),
            vm_content, vm_ce, vm_border, vm_color)


# ============================================================
#  Section D: computeValueDiff + ProposeBoundaryLines
# ============================================================

def _verify_opposite(x: int, y: int) -> bool:
    return (x == 0) != (y == 0)


def _compute_value_diff(vm_content_raw, vm_ce_raw, vm_color_raw,
                        ps_all: _PS2D, ps_ce: _PS2D,
                        has_top, nr: int, nc: int):
    row_db = [[False] * nc for _ in range(nr - 1)]
    col_db = [[False] * nr for _ in range(nc - 1)]

    for r in range(nr - 1):
        sum_r1 = ps_all.query(r + 1, 0, r + 1, nc - 1)
        sum_r  = ps_all.query(r,     0, r,     nc - 1)
        ce_r1  = ps_ce.query(r + 1, 0, r + 1, nc - 1)
        ce_r   = ps_ce.query(r,     0, r,     nc - 1)

        for c in range(nc - 1):
            if sum_r1 == sum_r:
                row_db[r][c] = False
            elif (_verify_opposite(vm_content_raw[r][c], vm_content_raw[r + 1][c])
                  and _verify_opposite(ce_r, ce_r1)):
                row_db[r][c] = True
            elif abs(ce_r1 - ce_r) > 5:
                row_db[r][c] = True
            elif _verify_opposite(vm_color_raw[r][c], vm_color_raw[r + 1][c]):
                row_db[r][c] = True
            elif (has_top[r + 1][c]
                  and (not (r + 2 < nr and has_top[r + 2][c])
                       or  not has_top[r][c])):
                row_db[r][c] = True

    for c in range(nc - 1):
        sum_c1 = ps_all.query(0, c + 1, nr - 1, c + 1)
        sum_c  = ps_all.query(0, c,     nr - 1, c)
        ce_c1  = ps_ce.query(0, c + 1, nr - 1, c + 1)
        ce_c   = ps_ce.query(0, c,     nr - 1, c)

        for r in range(nr - 1):
            if sum_c1 == sum_c:
                col_db[c][r] = False
            elif (_verify_opposite(vm_content_raw[r][c], vm_content_raw[r][c + 1])
                  and _verify_opposite(ce_c, ce_c1)):
                col_db[c][r] = True
            elif abs(ce_c1 - ce_c) > 5:
                col_db[c][r] = True
            elif _verify_opposite(vm_color_raw[r][c], vm_color_raw[r][c + 1]):
                col_db[c][r] = True
            elif (has_top[r][c + 1]
                  and (not (c + 2 < nc and has_top[r][c + 2])
                       or  not has_top[r][c])):
                col_db[c][r] = True

    return row_db, col_db


def _propose_boundary_lines(row_db, col_db, nr: int, nc: int,
                             thresh: float = 1.0
                             ) -> Tuple[List[int], List[int]]:
    row_bounds = [r for r in range(nr - 1) if sum(row_db[r]) >= thresh]
    col_bounds = [c for c in range(nc - 1) if sum(col_db[c]) >= thresh]
    return row_bounds, col_bounds


# ============================================================
#  Section E: _S class  (SheetMap equivalent)
# ============================================================

def _r1(a: List[float], b: List[float]) -> float:
    """Mean absolute difference (Utils.r1)."""
    if not a:
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


class _S:
    """Sheet state.  All box coordinates 0-indexed throughout."""

    def __init__(self, content_strs, has_content,
                 has_top, has_bottom, has_left, has_right,
                 has_color, merged_regions,
                 alpha_ratio, number_ratio, sp_char_ratio, text_length,
                 ps_content, ps_ce, ps_border, ps_bcol, ps_brow,
                 ps_color, ps_all,
                 vm_content_raw, vm_ce_raw, vm_border_raw, vm_color_raw,
                 nr: int, nc: int):
        self.content_strs  = content_strs
        self.has_top       = has_top
        self.has_bottom    = has_bottom
        self.has_left      = has_left
        self.has_right     = has_right
        self.merged_regions = merged_regions
        self.merge_boxes    = merged_regions  # alias used by MergeFilter
        self.alpha_ratio   = alpha_ratio
        self.number_ratio  = number_ratio
        self.sp_char_ratio = sp_char_ratio
        self.text_length   = text_length
        # prefix-sum maps
        self.ps_content = ps_content   # sumContent
        self.ps_ce      = ps_ce        # sumContentExist
        self.ps_border  = ps_border    # sumBorder
        self.ps_bcol    = ps_bcol      # sumBorderCol
        self.ps_brow    = ps_brow      # sumBorderRow
        self.ps_color   = ps_color     # sumColor
        self.ps_all     = ps_all       # sumAll
        # raw 2-D lists
        self.vm_content = vm_content_raw
        self.vm_ce      = vm_ce_raw
        self.vm_border  = vm_border_raw
        self.vm_color   = vm_color_raw
        self.nr = nr
        self.nc = nc
        # cohesion regions (populated during CohesionDetection)
        self.cohesion_regions:             List[Box] = []
        self.cohesion_border_regions:      List[Box] = []
        self.small_cohesion_border_regions: List[Box] = []
        # boundary lines (populated after ProposeBoundaryLines)
        self.row_boundary_lines: List[int] = []
        self.col_boundary_lines: List[int] = []

    # ── SheetMap helper methods ─────────────────────────────────────────────

    def ced(self, box: Box) -> float:
        """ContentExistValueDensity."""
        r0, c0, r1, c1 = box
        if r1 < r0 or c1 < c0:
            return 0.0
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        return self.ps_ce.query(r0, c0, r1, c1) / area

    def tdc(self, box: Box) -> int:
        """TextDistinctCount: distinct non-empty content strings in box."""
        r0, c0, r1, c1 = box
        s: Set[str] = set()
        for r in range(max(r0, 0), min(r1 + 1, self.nr)):
            for c in range(max(c0, 0), min(c1 + 1, self.nc)):
                v = self.content_strs[r][c]
                if v:
                    s.add(v)
        return len(s)

    def exists_merged(self, box: Box) -> bool:
        r0, c0, r1, c1 = box
        for mr0, mc0, mr1, mc1 in self.merged_regions:
            if mr0 <= r1 and mr1 >= r0 and mc0 <= c1 and mc1 >= c0:
                return True
        return False

    def row_ced_split(self, box: Box, split: float = 4) -> float:
        """RowContentExistValueDensitySplit."""
        r0, c0, r1, c1 = box
        if r1 < r0 or c1 < c0:
            return 0.0
        stride = (c1 - c0 + 1) / split
        cnt = 0.0
        for i in range(int(split)):
            l = int(c0 + i * stride)
            rr = int(c0 + stride * (i + 1)) - 1
            if self.ps_ce.query(r0, l, r1, rr) > 0:
                cnt += 1
        return cnt / split

    def col_ced_split(self, box: Box, split: float = 4) -> float:
        """ColContentExistValueDensitySplit."""
        r0, c0, r1, c1 = box
        if r1 < r0 or c1 < c0:
            return 0.0
        stride = (r1 - r0 + 1) / split
        cnt = 0.0
        for i in range(int(split)):
            t = int(r0 + i * stride)
            b = int(r0 + stride * (i + 1)) - 1
            if self.ps_ce.query(t, c0, b, c1) > 0:
                cnt += 1
        return cnt / split

    def compute_similar_row(self, box1: Box, box2: Box) -> float:
        """ComputeSimilarRow: mean-L1 distance of alpha/number ratio vectors."""
        r00, c00, r01, c01 = box1
        r10, _, r11, _     = box2
        left  = c00
        right = c01
        similars: List[float] = []
        for i in range(r00, r01 + 1):
            for j in range(r10, r11 + 1):
                if i == j:
                    continue
                l, r = left, right
                # trim leading/trailing cols empty in both rows
                while l <= r and (self.ps_ce.query(i, l, i, l) == 0
                                   and self.ps_ce.query(j, l, j, l) == 0):
                    l += 1
                while r >= l and (self.ps_ce.query(i, r, i, r) == 0
                                   and self.ps_ce.query(j, r, j, r) == 0):
                    r -= 1
                if l > r:
                    continue
                n = r - l + 1
                ce1 = self.ps_ce.query(i, l, i, r)
                ce2 = self.ps_ce.query(j, l, j, r)
                if ce1 <= 2 or ce1 / n <= 0.3 or ce2 <= 2 or ce2 / n <= 0.3:
                    continue
                a1 = [self.alpha_ratio[i][c]  for c in range(l, r + 1)]
                a2 = [self.alpha_ratio[j][c]  for c in range(l, r + 1)]
                n1 = [self.number_ratio[i][c] for c in range(l, r + 1)]
                n2 = [self.number_ratio[j][c] for c in range(l, r + 1)]
                similars.append(_r1(a1, a2))
                similars.append(_r1(n1, n2))
        if not similars:
            return 1.0
        return sum(similars) / len(similars)

    def cnt_none_border(self, box: Box) -> int:
        """CntNoneBorder: count how many of the 4 edges have no content/color."""
        r0, c0, r1, c1 = box
        edges = [
            (r0, c0, r0, c1), (r1, c0, r1, c1),
            (r0, c0, r1, c0), (r0, c1, r1, c1),
        ]
        cnt = 0
        for e in edges:
            if self.ps_content.query(*e) == 0 and self.ps_color.query(*e) == 0:
                cnt += 1
        return cnt

    def find_upheaders(self, detector, boxes: List[Box]) -> List[Box]:
        """FindoutUpheaders."""
        result: List[Box] = []
        for box in boxes:
            r0, c0, r1, c1 = box
            for k in range(5):
                cand = _up_row(box, start=k)
                cr0, cc0, cr1, cc1 = cand
                ce_sum = self.ps_ce.query(cr0, cc0, cr1, cc1)
                tdc = self.tdc(cand)
                ced = self.ced(cand)
                if ((ce_sum >= 6 and tdc > 1 and ced >= 1.0)
                        or (ce_sum >= 4 and tdc > 1 and ced >= 2.0)):
                    if detector.is_header_up(cand) and ced > 1.4:
                        result.append(cand)
                    break
        return _deduplicate_boxes(result)

    def find_leftheaders(self, detector, boxes: List[Box]) -> List[Box]:
        """FindoutLeftheaders."""
        result: List[Box] = []
        for box in boxes:
            for k in range(5):
                cand = _left_col(box, start=k)
                ce_sum = self.ps_ce.query(*cand)
                tdc = self.tdc(cand)
                ced = self.ced(cand)
                if ce_sum >= 6 and tdc > 1 and ced >= 1.0:
                    if detector.is_header_left(cand):
                        result.append(cand)
                    break
        return _deduplicate_boxes(result)


# ============================================================
#  Section F: Box utilities
# ============================================================

def _up_row(box: Box, start: int = 0, step: int = 1) -> Box:
    r0, c0, r1, c1 = box
    return (r0 + start, c0, r0 + start + step - 1, c1)

def _down_row(box: Box, start: int = 0, step: int = 1) -> Box:
    r0, c0, r1, c1 = box
    return (r1 - start - step + 1, c0, r1 - start, c1)

def _left_col(box: Box, start: int = 0, step: int = 1) -> Box:
    r0, c0, r1, c1 = box
    return (r0, c0 + start, r1, c0 + start + step - 1)

def _right_col(box: Box, start: int = 0, step: int = 1) -> Box:
    r0, c0, r1, c1 = box
    return (r0, c1 - start - step + 1, r1, c1 - start)

def _is_overlap(a: Box, b: Box) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]

def _contains_box(big: Box, small: Box, step: int = 0) -> bool:
    return (big[0] <= small[0] + step and big[2] >= small[2] - step
            and big[1] <= small[1] + step and big[3] >= small[3] - step)

def _is_suppression_box(b1: Box, b2: Box, step: int = 2) -> bool:
    return (abs(b1[0] - b2[0]) <= step and abs(b1[2] - b2[2]) <= step
            and abs(b1[1] - b2[1]) <= step and abs(b1[3] - b2[3]) <= step)

def _area_size(box: Box) -> float:
    return max(0, box[2] - box[0] + 1) * max(0, box[3] - box[1] + 1)

def _is_overlap_any(box: Box, targets,
                    except_forward: bool = False,
                    except_backward: bool = False,
                    except_suppression: bool = False) -> bool:
    for t in targets:
        if not _is_overlap(box, t):
            continue
        if except_forward  and _contains_box(box, t):
            continue
        if except_backward and _contains_box(t, box):
            continue
        if except_suppression and _is_suppression_box(box, t):
            continue
        return True
    return False

def _distinct_strs(content_strs, r0: int, r1: int, c0: int, c1: int) -> int:
    nr = len(content_strs)
    nc = len(content_strs[0]) if nr else 0
    s: Set[str] = set()
    for r in range(max(r0, 0), min(r1 + 1, nr)):
        for c in range(max(c0, 0), min(c1 + 1, nc)):
            v = content_strs[r][c]
            if v:
                s.add(v)
    return len(s)

def _unify_box(b1: Box, b2: Box) -> Box:
    return (min(b1[0], b2[0]), min(b1[1], b2[1]),
            max(b1[2], b2[2]), max(b1[3], b2[3]))

def _overlap_box(b1: Box, b2: Box) -> Box:
    return (max(b1[0], b2[0]), max(b1[1], b2[1]),
            min(b1[2], b2[2]), min(b1[3], b2[3]))

def _deduplicate_boxes(boxes) -> List[Box]:
    return list(dict.fromkeys(boxes))


# ============================================================
#  Section G: Border cohesion  (SheetMap.dealWithBorderCohensionRegions)
# ============================================================

def _deal_with_border_cohesion_regions(s: _S) -> None:
    """Translated from SheetMap.dealWithBorderCohensionRegions +
    locateBorderCohensions + VerifyBorderRegion (0-indexed throughout).
    Populates s.cohesion_border_regions and s.small_cohesion_border_regions.
    """
    nr, nc = s.nr, s.nc
    # bm[r][c][d]: 0=bottom 1=top 2=left 3=right
    bm = [[[False] * 4 for _ in range(nc)] for _ in range(nr)]

    for i in range(nr):
        for j in range(nc):
            bm[i][j][0] = s.has_bottom[i][j]
            bm[i][j][1] = s.has_top[i][j]
            bm[i][j][2] = s.has_left[i][j]
            bm[i][j][3] = s.has_right[i][j]
            if bm[i][j][0] and i - 1 >= 0:  bm[i-1][j][1] = True
            if bm[i][j][1] and i + 1 < nr:  bm[i+1][j][0] = True
            if bm[i][j][2] and j - 1 >= 0:  bm[i][j-1][3] = True
            if bm[i][j][3] and j + 1 < nc:  bm[i][j+1][2] = True

    # UL→DR
    for i in range(nr):
        for j in range(nc):
            if bm[i][j][1] and bm[i][j][2]:
                bm[i][j][0] = True; bm[i][j][3] = True
                if j + 1 < nc: bm[i][j+1][2] = True
                if i + 1 < nr: bm[i+1][j][1] = True
    # UR→DL
    for i in range(nr):
        for j in range(nc - 1, -1, -1):
            if bm[i][j][1] and bm[i][j][3]:
                bm[i][j][0] = True; bm[i][j][2] = True
                if j - 1 >= 0: bm[i][j-1][3] = True
                if i + 1 < nr: bm[i+1][j][1] = True
    # DL→UR
    for i in range(nr - 1, -1, -1):
        for j in range(nc):
            if bm[i][j][0] and bm[i][j][2]:
                bm[i][j][1] = True; bm[i][j][3] = True
                if j + 1 < nc: bm[i][j+1][2] = True
                if i - 1 >= 0: bm[i-1][j][0] = False
    # DR→UL
    for i in range(nr - 1, -1, -1):
        for j in range(nc - 1, -1, -1):
            if bm[i][j][0] and bm[i][j][3]:
                bm[i][j][1] = True; bm[i][j][2] = True
                if j - 1 >= 0: bm[i][j-1][3] = True
                if i - 1 >= 0: bm[i-1][j][0] = True

    def _verify(r0, c0, r1, c1) -> bool:
        # Check consistent bottom border on last row (start from c0+2)
        for col in range(c0 + 2, c1 + 1):
            b1 = s.has_bottom[r1][col]     if r1 < nr and col < nc     else False
            b2 = s.has_bottom[r1][col - 1] if r1 < nr and col - 1 >= 0 else False
            if b1 != b2:
                return False
        # Consistent right border on last col (start from r0+2)
        for row in range(r0 + 2, r1 + 1):
            rb1 = s.has_right[row][c1]     if row < nr and c1 < nc     else False
            rb2 = s.has_right[row-1][c1]   if row-1 >= 0 and c1 < nc  else False
            if rb1 != rb2:
                return False
        # Outside: no left borders below/above
        for col in range(c0, c1 + 1):
            if col > c0:
                if r1 + 1 < nr and col < nc and s.has_left[r1+1][col]:
                    return False
                if r0 - 1 >= 0 and col < nc and s.has_left[r0-1][col]:
                    return False
        # Outside: no top borders right/left of region
        for row in range(r0, r1 + 1):
            if row > r0:
                if c1 + 1 < nc and row < nr and s.has_top[row][c1+1]:
                    return False
                if c0 - 1 >= 0 and row < nr and s.has_top[row][c0-1]:
                    return False
        return True

    mark = [[True] * nc for _ in range(nr)]
    for i in range(nr):
        for j in range(nc):
            if not mark[i][j]:
                continue
            sr, sc = 0, 0
            while (i + sr < nr and j + sc < nc
                   and all(bm[i + sr][j + sc][d] for d in range(4))):
                sr += 1; sc += 1
            if sr > 0: sr -= 1
            if sc > 0: sc -= 1
            while (i + sr + 1 < nr
                   and all(bm[i + sr + 1][j + sc][d] for d in range(4))):
                sr += 1
            while (j + sc + 1 < nc
                   and all(bm[i + sr][j + sc + 1][d] for d in range(4))):
                sc += 1
            if sr == 0 and sc == 0:
                continue
            r0, c0, r1, c1 = i, j, i + sr, j + sc
            for p in range(sr + 1):
                for q in range(sc + 1):
                    mark[i + p][j + q] = False
            if s.cnt_none_border((r0, c0, r1, c1)) >= 2:
                continue
            if not _verify(r0, c0, r1, c1):
                continue
            if sr >= 3 and sc >= 2:
                s.cohesion_border_regions.append((r0, c0, r1, c1))
            elif sr >= 2 or sc >= 2:
                s.small_cohesion_border_regions.append((r0, c0, r1, c1))


# ============================================================
#  Section H: CohesionDetection  (merged + color + border)
# ============================================================

def _cohesion_detection(s: _S, merged_regions, has_color, nr: int, nc: int,
                        row_bounds: List[int], col_bounds: List[int]):
    """Returns (cohesion_regions, updated_row_bounds, updated_col_bounds)."""
    cohesion: List[Box] = []

    for r0, c0, r1, c1 in merged_regions:
        cohesion.append((r0, c0, r1, c1))
        if r0 > 0 and (r0 - 1) not in row_bounds:       row_bounds.append(r0 - 1)
        if r1 < nr - 1 and r1 not in row_bounds:          row_bounds.append(r1)
        if c0 > 0 and (c0 - 1) not in col_bounds:        col_bounds.append(c0 - 1)
        if c1 < nc - 1 and c1 not in col_bounds:          col_bounds.append(c1)

    color_mark = [[has_color[r][c] for c in range(nc)] for r in range(nr)]
    COLOR_H = 5; COLOR_W = 3
    for i in range(nr):
        for j in range(nc):
            if not color_mark[i][j]:
                continue
            sr, sc = 0, 0
            while i + sr < nr and j + sc < nc and color_mark[i + sr][j + sc]:
                sr += 1; sc += 1
            if sr > 0: sr -= 1
            if sc > 0: sc -= 1
            while i + sr + 1 < nr and color_mark[i + sr + 1][j + sc]:
                sr += 1
            while j + sc + 1 < nc and color_mark[i + sr][j + sc + 1]:
                sc += 1
            for p in range(sr + 1):
                for q in range(sc + 1):
                    color_mark[i + p][j + q] = False
            if sr >= COLOR_H or sc >= COLOR_W:
                cohesion.append((i, j, i + sr, j + sc))

    # Border cohesion
    _deal_with_border_cohesion_regions(s)
    cohesion.extend(s.cohesion_border_regions)

    s.cohesion_regions = cohesion
    row_bounds.sort(); col_bounds.sort()
    return cohesion, row_bounds, col_bounds


# ============================================================
#  Section I: GenerateBlockRegions
# ============================================================

def _trim_empty_edges(r0, c0, r1, c1, ps_content, ps_color, ps_border):
    while r0 < r1:
        if (ps_content.query(r0, c0, r0, c1) + ps_color.query(r0, c0, r0, c1)
                + ps_border.query(r0, c0, r0, c1)) != 0:
            break
        r0 += 1
    while r1 > r0:
        if (ps_content.query(r1, c0, r1, c1) + ps_color.query(r1, c0, r1, c1)
                + ps_border.query(r1, c0, r1, c1)) != 0:
            break
        r1 -= 1
    while c0 < c1:
        if (ps_content.query(r0, c0, r1, c0) + ps_color.query(r0, c0, r1, c0)
                + ps_border.query(r0, c0, r1, c0)) != 0:
            break
        c0 += 1
    while c1 > c0:
        if (ps_content.query(r0, c1, r1, c1) + ps_color.query(r0, c1, r1, c1)
                + ps_border.query(r0, c1, r1, c1)) != 0:
            break
        c1 -= 1
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _split_block_region(r0, c0, r1, c1, ps_content, ps_ce, ps_color):
    def _dens(ar0, ac0, ar1, ac1):
        area = (ar1 - ar0 + 1) * (ac1 - ac0 + 1)
        return ps_ce.query(ar0, ac0, ar1, ac1) / area if area > 0 else 0.0

    for i in range(r0 + 4, r1 - 3):
        e5r0, e5r1 = max(r0, i - 2), min(r1, i + 2)
        e3r0, e3r1 = max(r0, i - 1), min(r1, i + 1)
        c_e1  = ps_content.query(i, c0, i, c1) + ps_color.query(i, c0, i, c1)
        ce_e3 = ps_ce.query(e3r0, c0, e3r1, c1)
        ce_e5 = ps_ce.query(e5r0, c0, e5r1, c1)
        dens5 = _dens(e5r0, c0, e5r1, c1)
        if (c_e1 == 0 and (ce_e3 == 0 or (ce_e3 < 6 and ce_e5 < 10))
                or (dens5 < 0.1 and ce_e5 < 10)):
            k = i + 2
            while k < r1 and ps_ce.query(k, c0, k, c1) <= 2:
                k += 1
            k2 = i - 2
            while k2 > r0 and ps_ce.query(k2, c0, k2, c1) <= 2:
                k2 -= 1
            top_box = (r0, c0, k2, c1)
            bot_box = (k,  c0, r1, c1)
            if top_box[0] <= top_box[2] and bot_box[0] <= bot_box[2]:
                return top_box, bot_box

    for j in range(c0 + 4, c1 - 3):
        e5c0, e5c1 = max(c0, j - 2), min(c1, j + 2)
        e3c0, e3c1 = max(c0, j - 1), min(c1, j + 1)
        c_e1  = ps_content.query(r0, j, r1, j) + ps_color.query(r0, j, r1, j)
        ce_e3 = ps_ce.query(r0, e3c0, r1, e3c1)
        ce_e5 = ps_ce.query(r0, e5c0, r1, e5c1)
        dens5 = _dens(r0, e5c0, r1, e5c1)
        if (c_e1 == 0 and (ce_e3 == 0 or (ce_e3 < 6 and ce_e5 < 10))
                or (dens5 < 0.1 and ce_e5 < 10)):
            k = j + 2
            while k < c1 and ps_ce.query(r0, k, r1, k) <= 2:
                k += 1
            k2 = j - 2
            while k2 > c0 and ps_ce.query(r0, k2, r1, k2) <= 2:
                k2 -= 1
            left_box  = (r0, c0, r1, k2)
            right_box = (r0, k,  r1, c1)
            if left_box[1] <= left_box[3] and right_box[1] <= right_box[3]:
                return left_box, right_box

    return None


def _generate_block_regions(nr, nc, ps_content, ps_ce, ps_color, ps_border):
    pending = [(0, 0, nr - 1, nc - 1)]
    blocks: List[Box] = []
    while pending:
        box = pending.pop()
        r0, c0, r1, c1 = box
        if r0 > r1 or c0 > c1:
            continue
        trimmed = _trim_empty_edges(r0, c0, r1, c1, ps_content, ps_color, ps_border)
        if trimmed is None:
            continue
        if trimmed != box:
            pending.append(trimmed)
            continue
        split = _split_block_region(r0, c0, r1, c1, ps_content, ps_ce, ps_color)
        if split is not None:
            pending.append(split[0])
            pending.append(split[1])
        else:
            blocks.append(box)
    return blocks


# ============================================================
#  Section J: RegionGrowthDetector
# ============================================================

def _find_connected_ranges(content_strs, vm_border_raw, nr, nc,
                           thresh_hor=1, thresh_ver=1, direct=0):
    visited  = [[False] * nc for _ in range(nr)]
    cnt_hor  = [[thresh_hor] * nc for _ in range(nr)]
    cnt_ver  = [[thresh_ver] * nc for _ in range(nr)]
    rows = list(range(nr)) if direct == 0 else list(range(nr - 1, -1, -1))
    raw: List[Box] = []

    for r in rows:
        for c in range(nc):
            if visited[r][c] or not content_strs[r][c]:
                continue
            q: deque = deque([(r, c)])
            visited[r][c] = True
            min_r = max_r = r
            min_c = max_c = c
            while q:
                cr, cc = q.popleft()
                if cnt_hor[cr][cc] == 0 or cnt_ver[cr][cc] == 0:
                    continue
                min_r = min(min_r, cr); min_c = min(min_c, cc)
                max_r = max(max_r, cr); max_c = max(max_c, cc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr2, nc2 = cr + dr, cc + dc
                    if not (0 <= nr2 < nr and 0 <= nc2 < nc):
                        continue
                    if visited[nr2][nc2]:
                        continue
                    visited[nr2][nc2] = True
                    empty = not content_strs[nr2][nc2]
                    has_border = (vm_border_raw is not None
                                  and vm_border_raw[nr2][nc2] != 0)
                    if empty and not has_border:
                        if dc != 0: cnt_hor[nr2][nc2] = cnt_hor[cr][cc] - 1
                        if dr != 0: cnt_ver[nr2][nc2] = cnt_ver[cr][cc] - 1
                    elif has_border:
                        if dc != 0: cnt_hor[nr2][nc2] = thresh_hor
                        if dr != 0: cnt_ver[nr2][nc2] = thresh_ver
                    q.append((nr2, nc2))
            if max_r > min_r:
                raw.append((min_r, min_c, max_r, max_c))
                for ri in range(min_r, max_r + 1):
                    for rj in range(min_c, max_c + 1):
                        visited[ri][rj] = True

    def _trim(boxes):
        result = []
        for r0, c0, r1, c1 in boxes:
            for _ in range(3):
                while r0 <= r1 and all(not content_strs[r0][c] for c in range(c0, c1+1)):
                    r0 += 1
                while r1 >= r0 and all(not content_strs[r1][c] for c in range(c0, c1+1)):
                    r1 -= 1
                while c0 <= c1 and all(not content_strs[r][c0] for r in range(r0, r1+1)):
                    c0 += 1
                while c1 >= c0 and all(not content_strs[r][c1] for r in range(r0, r1+1)):
                    c1 -= 1
            if r0 <= r1 and c0 <= c1 and r0 < r1:
                result.append((r0, c0, r1, c1))
        return result

    return _trim(raw)



# ============================================================
#  Section K: _TableDetector  (TableDetectionHybrid equivalent)
# ============================================================

class _TableDetector:
    """Full C# TableDetectionHybrid pipeline, translated to Python."""

    def __init__(self, s: _S):
        self._s = s
        self._boxes: List[Box] = []
        self._region_growth_boxes: List[Box] = []
        self._header_up_cache:   Dict[Box, bool] = {}
        self._header_left_cache: Dict[Box, bool] = {}

    # ── K1: Header detection ──────────────────────────────────────────────

    def header_rate(self, box: Box, step: int = 6) -> float:
        """HeaderRate: fraction of header-like cells in box."""
        s = self._s
        r0, c0, r1, c1 = box
        cnt_all = 0; cnt_exist = 0; cnt_header = 0
        for i in range(max(r0, 0), min(r1 + 1, s.nr)):
            for j in range(max(c0, 0), min(c1 + 1, s.nc)):
                cnt_all += 1
                # headerControlledSurroundingRegion
                if r0 == r1:
                    rr = (i, i + step, j, j)
                elif c0 == c1:
                    rr = (i, i, j, j + step)
                else:
                    rr = (i, i, j, j)
                if s.ps_ce.query(*rr) != 0:
                    cnt_exist += 1
                    v = s.content_strs[i][j]
                    if v:
                        ar = s.alpha_ratio[i][j]
                        nr_ = s.number_ratio[i][j]
                        sp = s.sp_char_ratio[i][j]
                        tl = s.text_length[i][j]
                        if ((ar >= nr_ and ar != 0)
                                or (ar * tl > 2.5)
                                or sp > 0):
                            cnt_header += 1
        if cnt_all == 0:
            return 0.0
        denom = max(cnt_exist, cnt_all / 3.0)
        return cnt_header / denom

    def is_header_up_simple(self, box: Box) -> bool:
        r0, c0, r1, c1 = box
        if c1 == c0:
            return False
        s = self._s
        ce_sum = s.ps_ce.query(r0, c0, r1, c1)
        if ce_sum <= 4 and self.header_rate(box, step=0) <= 0.5:
            return False
        area = _area_size(box)
        tdc = s.tdc(box)
        if (area > 4 and tdc <= 2) or (area > 3 and tdc < 2):
            return False
        right_part = (r0, r1, min(c0, c1 - 5) + 3, c1)
        if (s.ced(box) > 0.6
                and self.header_rate(box) > 0.4
                and self.header_rate(right_part) > 0.3):
            return True
        return False

    def is_header_left_simple(self, box: Box) -> bool:
        r0, c0, r1, c1 = box
        if r1 == r0:
            return False
        s = self._s
        if r1 - r0 == 1 and self.header_rate(box) >= 0.5:
            return True
        area = _area_size(box)
        tdc = s.tdc(box)
        if (area > 4 and tdc <= 2) or (area > 3 and tdc < 2):
            return False
        ce_sum = s.ps_ce.query(r0, c0, r1, c1)
        if ce_sum <= 4 and self.header_rate(box) <= 0.5:
            return False
        up_part = (min(r0, r1 - 5) + 3, r1, c0, c1)
        if (s.ced(box) > 0.6
                and self.header_rate(box) > 0.4
                and self.header_rate(up_part) > 0.3):
            return True
        return False

    def is_header_up(self, box: Box) -> bool:
        if box in self._header_up_cache:
            return self._header_up_cache[box]
        s = self._s
        if self.is_header_up_simple(box):
            dn = _down_row(box, start=-1)
            similar = s.compute_similar_row(box, dn)
            if similar < 0.15 and self.is_header_up_simple(dn):
                self._header_up_cache[box] = False
                return False
            ce1 = s.ps_ce.query(*box)
            ce2 = s.ps_ce.query(*dn)
            if (ce1 == 2 and ce2 == 2
                    and self.header_rate(box, step=0) == self.header_rate(dn, step=0)):
                self._header_up_cache[box] = False
                return False
            self._header_up_cache[box] = True
            return True
        self._header_up_cache[box] = False
        return False

    def is_header_left(self, box: Box) -> bool:
        if box in self._header_left_cache:
            return self._header_left_cache[box]
        s = self._s
        if self.is_header_left_simple(box):
            self._header_left_cache[box] = True
            return True
        right = _right_col(box, start=-1)
        if (s.ced(right) >= 1.5 * s.ced(box)
                and s.ced(right) > 1.2
                and self.is_header_left_simple(right)):
            self._header_left_cache[box] = True
            return True
        self._header_left_cache[box] = False
        return False

    def is_header_up_with_data_area(self, upper_row: Box, box: Box) -> bool:
        r0, c0, r1, c1 = upper_row
        if not self.is_header_up(upper_row):
            return False
        cur = upper_row
        init_top = r0
        while (self.is_header_up_simple(cur)
               and cur[0] <= init_top + 2
               and cur[0] < box[2]):
            cur = _down_row(cur, start=-1)
        mark = True
        for k in range(1, 4):
            nxt = (cur[0] + k, cur[1], cur[0] + k, cur[3])
            if nxt[0] <= box[2] and self.is_header_up(nxt):
                mark = False; break
        return mark

    # ── K2: Basic filters ─────────────────────────────────────────────────

    def _verify_box_border_not_null(self, box: Box) -> bool:
        s = self._s
        r0, c0, r1, c1 = box
        for e in [(r0,c0,r0,c1),(r1,c0,r1,c1),(r0,c0,r1,c0),(r0,c1,r1,c1)]:
            if s.ps_content.query(*e) + s.ps_color.query(*e) == 0:
                return False
        return True

    def _verify_box_border_in_out_simple(self, box: Box) -> bool:
        s = self._s
        r0, c0, r1, c1 = box
        for e in [(r0-1,c0,r0-1,c1),(r1+1,c0,r1+1,c1),
                  (r0,c0-1,r1,c0-1),(r0,c1+1,r1,c1+1)]:
            if s.ps_ce.query(*e) >= 6:
                return False
        for e in [(r0,c0,r0,c1),(r1,c0,r1,c1),(r0,c0,r1,c0),(r0,c1,r1,c1)]:
            if s.ps_content.query(*e) + s.ps_color.query(*e) == 0:
                return False
        return True

    def _verify_box_border_out_sparse(self, box: Box) -> bool:
        s = self._s
        r0, c0, r1, c1 = box
        su = s.ps_ce.query(r0-1,c0,r0-1,c1)
        sd = s.ps_ce.query(r1+1,c0,r1+1,c1)
        sl = s.ps_ce.query(r0,c0-1,r1,c0-1)
        sr = s.ps_ce.query(r0,c1+1,r1,c1+1)
        if su >= 6 or sd >= 6 or sl >= 6 or sr >= 6: return False
        if r1 - r0 <= 2 and (sl >= 2 or sr >= 2):   return False
        if c1 - c0 <= 1 and (su >= 2 or sd >= 2):   return False
        if r1 - r0 <= 4 and (sl >= 4 or sr >= 4):   return False
        if c1 - c0 <= 4 and (su >= 4 or sd >= 4):   return False
        return True

    def _verify_box_split(self, box: Box) -> bool:
        """Full VerifyBoxSplit from DetectorFilters.cs."""
        s = self._s
        up, left, down, right = box[0], box[1], box[2], box[3]
        up_off  = 2 if (down - up)   > 12 else 0
        left_off = 2 if (right - left) > 12 else 0

        for i in range(up + 3 + up_off, down - 4):
            e3 = (i, left, i + 2, right)
            e1 = (i + 1, left, i + 1, right)
            if s.ps_content.query(*e1) < 3:
                if (s.ps_content.query(*e1) + s.ps_color.query(*e1) == 0
                        and s.ps_ce.query(*e3) == 0):
                    k = i + 3
                    while k < down and s.ps_content.query(k, left, k, right) <= 5:
                        k += 1
                    k2 = i - 1
                    while k2 > up and s.ps_content.query(k2, left, k2, right) <= 5:
                        k2 -= 1
                    eup  = (k2, left, k2, right)
                    edn  = (k,  left, k,  right)
                    if (s.ps_ce.query(*eup) > 5 and s.ps_ce.query(*edn) > 5):
                        return False
                elif (s.ps_color.query(*e1) + s.ps_brow.query(*e1) < 5
                      and not _is_overlap_any(e1, s.cohesion_regions,
                                               except_forward=True)):
                    ul = (up, left, i+1, left+2)
                    ur = (up, right-2, i+1, right)
                    dl = (i+1, left, down, left+2)
                    dr = (i+1, right-2, down, right)
                    def _qd(b):
                        a = _area_size(b)
                        if a == 0: return 0.0
                        return (s.ps_content.query(*b)+s.ps_color.query(*b)+s.ps_bcol.query(*b)) / a
                    dul = _qd(ul); dur = _qd(ur); ddl = _qd(dl); ddr = _qd(dr)
                    if dul == 0 and ddl > 0.4: return False
                    if dur == 0 and ddr > 0.4: return False
                    if ddl == 0 and dul > 0.4: return False
                    if ddr == 0 and dur > 0.4: return False

        for i in range(left + 3 + left_off, right - 4):
            e3 = (up, i, down, i + 2)
            e1 = (up, i+1, down, i+1)
            if s.ps_content.query(*e1) < 3:
                if (s.ps_content.query(*e1) + s.ps_color.query(*e1) == 0
                        and s.ps_ce.query(*e3) == 0):
                    k = i + 3
                    while k < right and s.ps_content.query(up, k, down, k) <= 5:
                        k += 1
                    k2 = i - 1
                    while k2 > left and s.ps_content.query(up, k2, down, k2) <= 5:
                        k2 -= 1
                    if k - k2 >= 3:
                        return False
                elif (s.ps_color.query(*e1) + s.ps_brow.query(*e1) < 5
                      and not _is_overlap_any(e1, s.cohesion_regions,
                                               except_forward=True)):
                    ul = (up, left, up+2, i+1)
                    ur = (up, i+1, up+2, right)
                    dl = (down-2, left, down, i+1)
                    dr = (down-2, i+1, down, right)
                    def _qdr(b):
                        a = _area_size(b)
                        if a == 0: return 0.0
                        return (s.ps_content.query(*b)+s.ps_color.query(*b)+s.ps_brow.query(*b)) / a
                    dul = _qdr(ul); dur = _qdr(ur); ddl = _qdr(dl); ddr = _qdr(dr)
                    if dul == 0 and dur / max(_area_size(ur), 1) > 0.4: return False
                    if dur == 0 and dul / max(_area_size(ul), 1) > 0.4: return False
                    if ddl == 0 and ddr / max(_area_size(dr), 1) > 0.4: return False
                    if ddr == 0 and ddl / max(_area_size(dl), 1) > 0.4: return False
        return True

    def _general_filter(self, box: Box) -> bool:
        r0, c0, r1, c1 = box
        if r1 - r0 < 1 or c1 - c0 < 1:
            return False
        if not self._verify_box_border_in_out_simple(box): return False
        if not self._verify_box_border_out_sparse(box):    return False
        if not self._verify_box_border_not_null(box):      return False
        if not self._verify_box_split(box):                return False
        return True

    # ── K3: LittleBoxesFilter ─────────────────────────────────────────────

    def _little_boxes_filter(self) -> None:
        s = self._s
        kept = []
        for box in self._boxes:
            r0, c0, r1, c1 = box
            h = r1 - r0; w = c1 - c0
            area = (h + 1) * (w + 1)
            dens = s.ced(box)
            if h < 1 or w < 1: continue
            if (h < 2 or w < 2) and area < 8  and dens < 1.4: continue
            if (h < 2 or w < 2) and area < 24 and dens < 1.2: continue
            if (h < 2 or w < 2) and dens < 1.1: continue
            if (h < 3 or w < 3) and dens < 0.7: continue
            if area < 7: continue
            if (h < 5 and w < 3) or (h < 3 and w < 5):
                if dens < 1.1: continue
            elif h < 5 and w < 5:
                if dens < 0.8: continue
            elif h < 8 and w < 8:
                if dens < 0.7: continue
            elif h < 14 and w < 14:
                if dens < 0.5: continue
            if h == 2:
                inner = s.ps_ce.query(r0+1, c0, r1-1, c1)
                if inner <= 5 and dens < 0.9: continue
            if w == 2:
                inner = s.ps_ce.query(r0, c0+1, r1, c1-1)
                if inner <= 5 and dens < 0.9: continue
            if 3 <= h <= 4:
                skip = False
                for idx in range(r0+1, r1):
                    if s.ps_ce.query(idx, c0, idx+1, c1) <= 3 and dens < 0.8:
                        skip = True; break
                if skip: continue
            if 2 < w <= 4:
                skip = False
                for idx in range(c0+2, c1):
                    if s.ps_ce.query(r0, idx, r1, idx+1) <= 3 and dens < 0.8:
                        skip = True; break
                if skip: continue
            if 4 < w <= 7:
                skip = False
                for idx in range(c0+2, c1-1):
                    if s.ps_ce.query(r0, idx, r1, idx+1) <= 3 and dens < 1.0:
                        skip = True; break
                if skip: continue
            if 4 < h <= 7:
                skip = False
                for idx in range(r0+1, r1-1):
                    if s.ps_ce.query(idx, c0, idx+1, c1) <= 3 and dens < 1.0:
                        skip = True; break
                if skip: continue
            kept.append(box)
        self._boxes = kept

    # ── K4: Overlap filters ───────────────────────────────────────────────

    def _overlap_cohesion_filter(self) -> None:
        s = self._s
        rm = set()
        for box in self._boxes:
            if _is_overlap_any(box, s.cohesion_regions,
                                except_forward=True, except_backward=True):
                rm.add(box)
        self._boxes = [b for b in self._boxes if b not in rm]

    def _overlap_border_cohesion_filter(self) -> None:
        s = self._s
        rm = set()
        for box in self._boxes:
            if _is_overlap_any(box, s.small_cohesion_border_regions,
                                except_forward=True, except_backward=True,
                                except_suppression=True):
                rm.add(box)
        self._boxes = [b for b in self._boxes if b not in rm]

    def _none_border_filter(self) -> None:
        s = self._s
        rm = set()
        for box in self._boxes:
            r0, c0, r1, c1 = box
            for e in [(r0,c0,r0,c1),(r1,c0,r1,c1),(r0,c0,r1,c0),(r0,c1,r1,c1)]:
                if s.ps_content.query(*e) + s.ps_color.query(*e) == 0:
                    rm.add(box); break
        self._boxes = [b for b in self._boxes if b not in rm]

    def _splitted_empty_lines_filter(self) -> None:
        rm = set()
        for box in self._boxes:
            if not self._verify_box_split(box):
                rm.add(box)
        self._boxes = [b for b in self._boxes if b not in rm]

    def _overlap_up_header_filter(self) -> None:
        s = self._s
        up_headers = s.find_upheaders(self, self._boxes)
        rm = set()
        for box in self._boxes:
            r0, c0, r1, c1 = box
            for hdr in up_headers:
                hr0, hc0, hr1, hc1 = hdr
                uside = (hr0-1, hc0, hr1-1, hc1)
                if (((uside[1] == c0 and uside[3] == c1)
                     or (abs(uside[1]-c0) <= 1 and abs(uside[3]-c1) <= 1 and c1-c0 > 5)
                     or (abs(uside[1]-c0) <= 2 and abs(uside[3]-c1) <= 2 and c1-c0 > 10)
                     or (abs(r1 - uside[0] - 1) < 2 and uside[3]-uside[1] > 3))
                        and _is_overlap(box, uside) and abs(uside[0]-r0) > 1):
                    rm.add(box); break
                if (abs(uside[0]+1 - r0) <= 1 and _is_overlap(box, hdr)
                        and c0 >= uside[1]+1 and c0 <= uside[3]-1):
                    dw = (hr0, hr1, c0-2, c0)
                    if s.ps_ce.query(*dw) >= 6 and self.header_rate(dw) == 1:
                        rm.add(box); break
                if (abs(uside[0]+1 - r0) <= 1 and _is_overlap(box, hdr)
                        and c1 >= uside[1]+1 and c1 <= uside[3]-1):
                    dw = (hr0, hr1, c1, c1+2)
                    if s.ps_ce.query(*dw) >= 6 and self.header_rate(dw) == 1:
                        rm.add(box); break
        self._boxes = [b for b in self._boxes if b not in rm]

    def _eliminate_overlaps(self) -> None:
        def area(b): return (b[2]-b[0]+1)*(b[3]-b[1]+1)
        boxes = sorted(self._boxes, key=area, reverse=True)
        rm: Set[Box] = set()
        for i in range(len(boxes)):
            if boxes[i] in rm: continue
            for j in range(i+1, len(boxes)):
                if boxes[j] in rm: continue
                if _is_overlap(boxes[i], boxes[j]):
                    rm.add(boxes[j])
        self._boxes = [b for b in boxes if b not in rm]

    # ── K5: Trim functions ────────────────────────────────────────────────

    def _find_up_boundary_not_sparse(self) -> None:
        s = self._s
        new_list = []
        for box in self._boxes:
            r0, c0, r1, c1 = box
            cand = (r0, c0, r0, c1)
            cand_dn = (r0+1, c0, r0+1, c1)
            while cand[0] < r1:
                cr, _, _, _ = cand
                if (s.ced(cand) < 0.4
                        or ((c1-c0+1) >= 5 and s.tdc(cand) <= 1)):
                    cand = (cr+1, c0, cr+1, c1)
                elif (not self.is_header_up(cand) and c1-c0 > 7
                      and 2*s.ps_ce.query(*cand) <= s.ps_ce.query(*cand_dn)
                      and (s.ps_ce.query(*cand) < 7 or s.ced(cand) < 0.6)):
                    cand = (cr+1, c0, cr+1, c1)
                else:
                    break
                cand_dn = (cand[0]+1, c0, cand[0]+1, c1)
            new_top = cand[0]
            nb = (new_top, c0, r1, c1) if new_top < r1 else box
            new_list.append(nb)
        self._boxes = _deduplicate_boxes(new_list)

    def _find_up_boundary_is_header(self) -> None:
        s = self._s
        new_list = []
        for box in self._boxes:
            r0, c0, r1, c1 = box
            cand = (r0, c0, r0, c1)
            while not self.is_header_up(cand) and cand[0] <= r0+3 and cand[0] < r1:
                cand = (cand[0]+1, c0, cand[0]+1, c1)
            if cand[0] != r0 and self.is_header_up_with_data_area(cand, box) and cand[0] < r1:
                new_list.append((cand[0], c0, r1, c1))
            else:
                new_list.append(box)
        self._boxes = _deduplicate_boxes(new_list)

    def _find_up_boundary_is_clear_header(self) -> None:
        s = self._s
        new_list = []
        for box in self._boxes:
            r0, c0, r1, c1 = box
            cand = (r0, c0, r0, c1)
            while s.ps_ce.query(*cand) > 3 and cand[0] < r0+6 and cand[0] < r1:
                cand = (cand[0]+1, c0, cand[0]+1, c1)
            if s.ps_ce.query(*cand) > 3:
                new_list.append(box); continue
            while s.ps_ce.query(*cand) < 3 and cand[0] < r0+2 and cand[0] < r1:
                cand = (cand[0]+1, c0, cand[0]+1, c1)
            if s.ps_ce.query(*cand) >= 3 and cand[0] != r0 and cand[0] < r1:
                if self.is_header_up_with_data_area(cand, box):
                    new_list.append((cand[0], c0, r1, c1)); continue
            new_list.append(box)
        self._boxes = _deduplicate_boxes(new_list)

    def _find_up_boundary_is_compact_header(self, lo: float, hi: float) -> None:
        s = self._s
        new_list = []
        for box in self._boxes:
            r0, c0, r1, c1 = box
            if c1 - c0 + 1 <= 4:
                new_list.append(box); continue
            cand = (r0, c0, r0, c1)
            while s.ced(cand) < 2*lo and cand[0] < r0+6 and cand[0] < r1:
                cand = (cand[0]+1, c0, cand[0]+1, c1)
            if cand[0] == r0 or s.ced(cand) < 2*lo:
                new_list.append(box); continue
            if self.is_header_up_with_data_area(cand, box) and cand[0] < r1:
                new_list.append((cand[0], c0, r1, c1))
            else:
                new_list.append(box)
        self._boxes = _deduplicate_boxes(new_list)

    def _up_header_trim(self) -> None:
        self._find_up_boundary_not_sparse()
        self._find_up_boundary_is_header()
        self._find_up_boundary_is_clear_header()
        self._find_up_boundary_is_compact_header(0.6, 0.8)
        self._find_up_boundary_is_compact_header(0.4, 0.7)
        self._find_up_boundary_is_compact_header(0.2, 0.5)

    def _sparse_boundaries_trim(self) -> None:
        s = self._s
        changed = True
        new_list = []
        for box in self._boxes:
            r0, c0, r1, c1 = list(box)
            changed_box = True
            while changed_box and c0 < c1 and r0 < r1:
                changed_box = False
                # left
                ln = (r0, c0, r1, c0)
                while s.ps_ce.query(*ln) < 3:
                    ln = (r0, ln[1]+1, r1, ln[1]+1)
                    if s.ps_ce.query(*ln) == 0: break
                while ln[1] < c1 and s.ps_ce.query(*ln) == 0:
                    c0 = ln[1]+1; ln = (r0, c0, r1, c0); changed_box = True
                # right
                ln = (r0, c1, r1, c1)
                while s.ps_ce.query(*ln) < 3:
                    ln = (r0, ln[1]-1, r1, ln[1]-1)
                    if s.ps_ce.query(*ln) == 0: break
                while ln[1] > c0 and s.ps_ce.query(*ln) == 0:
                    c1 = ln[1]-1; ln = (r0, c1, r1, c1); changed_box = True
                # up
                ln = (r0, c0, r0, c1)
                while (not self.is_header_up(ln)
                       and (s.ps_ce.query(*ln) < 3 or s.ced(ln) < 0.2
                            or (s.ps_ce.query(*ln) < 5 and c1-c0+1 > 7))):
                    ln = (ln[0]+1, c0, ln[0]+1, c1)
                    if s.ps_ce.query(*ln) == 0: break
                while ln[0] < r1 and s.ps_ce.query(*ln) == 0:
                    r0 = ln[0]+1; ln = (r0, c0, r0, c1); changed_box = True
                # down
                ln = (r1, c0, r1, c1)
                while s.ps_ce.query(*ln) < 3:
                    ln = (ln[0]-1, c0, ln[0]-1, c1)
                    if s.ps_ce.query(*ln) == 0: break
                while ln[0] > r0 and s.ps_ce.query(*ln) == 0:
                    r1 = ln[0]-1; ln = (r1, c0, r1, c1); changed_box = True
            nb = (r0, c0, r1, c1)
            if nb != box and c0 < c1 and r0 < r1:
                new_list.append(nb)
            else:
                new_list.append(box)
        self._boxes = _deduplicate_boxes(new_list)

    def _find_left_boundary_not_sparse(self) -> None:
        s = self._s
        new_list = []
        for box in self._boxes:
            r0, c0, r1, c1 = box
            mark = True
            while mark and c0 < c1:
                mark = False
                ln = (r0, c0, r1, c0)
                h = r1 - r0 + 1
                if (h >= 5 and s.ced(ln) < 0.7 and s.tdc(ln) <= 1):
                    mark = True; c0 += 1
                elif ((h > 3 and s.ps_ce.query(*ln) < 5)
                      or (h > 10 and (s.ps_ce.query(*ln) < 7 or s.ced(ln) < 0.3))):
                    mark = True; c0 += 1
            nb = (r0, c0, r1, c1)
            if nb != box and c0 < c1 and r0 < r1:
                new_list.append(nb)
            else:
                new_list.append(box)
        self._boxes = _deduplicate_boxes(new_list)

    def _bottom_boundary_trim(self) -> None:
        s = self._s
        new_list = []
        for box in self._boxes:
            r0, c0, r1, c1 = box
            bl = r1
            while bl > r0:
                w = c1 - c0 + 1
                blrow = (bl, c0, bl, c1)
                if (w > 6 and (
                        s.exists_merged((bl, min(c0+5, c1-1), bl, c1))
                        or (s.exists_merged(blrow)
                            and not s.exists_merged((max(r0, bl-4), bl-2, c0, c1)))
                        or (s.ps_ce.query(bl, c0, bl, c0) == 0
                            and s.ced(blrow) < 0.3)
                        or (s.ps_ce.query(bl, c0, bl, c0+2) == 0
                            and s.ced(blrow) < 0.6))):
                    bl -= 1; continue
                if w >= 2 and s.ps_ce.query(*blrow) < 3:
                    bl -= 1; continue
                break
            # skip trailing header-like rows
            bls = bl; cnt = 0
            while cnt < 5 and bl > r0:
                bls_row = (bls, c0, bls, c1)
                if (s.ps_ce.query(*bls_row) >= 2
                        and self.header_rate(bls_row, step=0) > 0.6):
                    cnt += 1; bls -= 1
                else:
                    break
            if cnt < 3 and s.ps_ce.query(bls, c0, bls, c1) == 0:
                while bls > r0 and s.ps_ce.query(bls, c0, bls, c1) == 0:
                    bls -= 1
                bl = bls
            nb = (r0, c0, bl, c1) if bl != r1 and bl > r0 else box
            new_list.append(nb)
        self._boxes = _deduplicate_boxes(new_list)

    def _up_boundary_compact_trim(self) -> None:
        s = self._s
        new_list = []
        for box in self._boxes:
            r0, c0, r1, c1 = box
            row1 = (r0, c0, r0, c1)
            row2 = (r0+1, c0, r0+1, c1)
            while (not (self.is_header_up(row1) and not self.is_header_up(row2))
                   and 2*s.ced(row1) <= s.ced(row2)
                   and row1[0] < r0+6 and row2[0] < r1):
                row1 = (row1[0]+1, c0, row1[0]+1, c1)
                row2 = (row2[0]+1, c0, row2[0]+1, c1)
            new_top = row1[0]
            if new_top == r0 or s.ced(row1) <= 0.6*s.ced(row2):
                new_list.append(box); continue
            if new_top < r1:
                new_list.append((new_top, c0, r1, c1))
            else:
                new_list.append(box)
        self._boxes = _deduplicate_boxes(new_list)

    def _surrounding_boundaries_trim(self) -> None:
        cnt = -1
        while len(self._boxes) != cnt:
            self._boxes = _deduplicate_boxes(self._boxes)
            cnt = len(self._boxes)
            self._find_left_boundary_not_sparse()
            self._find_up_boundary_not_sparse()
            self._sparse_boundaries_trim()
            self._bottom_boundary_trim()
            self._up_boundary_compact_trim()
            self._none_border_filter()

    def _left_header_trim(self) -> None:
        new_list = []
        for box in self._boxes:
            r0, c0, r1, c1 = box
            if _distinct_strs(self._s.content_strs, r0, r1, c0, c0) <= 1:
                new_list.append((r0, c0+1, r1, c1))
            else:
                new_list.append(box)
        self._boxes = _deduplicate_boxes(new_list)

    def _bottom_trim(self) -> None:
        s = self._s
        new_list = []
        for box in self._boxes:
            r0, c0, r1, c1 = box
            k = 0
            while k < r1 - r0:
                brow = (r1-k, c0, r1-k, c1)
                if (s.ced(brow) < 0.6
                        or (c1-c0+1 > 3 and _distinct_strs(
                                s.content_strs, r1-k, r1-k, c0, c1) <= 1)):
                    k += 1
                else:
                    break
            if k > 0:
                new_list.append((r0, c0, r1-k, c1))
            else:
                new_list.append(box)
        self._boxes = _deduplicate_boxes(new_list)

    def _up_trim_simple(self) -> None:
        new_list = []
        for box in self._boxes:
            r0, c0, r1, c1 = box
            k = 0
            while k < r1 - r0:
                if _distinct_strs(self._s.content_strs, r0+k, r0+k, c0, c1) <= 1:
                    k += 1
                else:
                    break
            if k > 0:
                new_list.append((r0+k, c0, r1, c1))
            else:
                new_list.append(box)
        self._boxes = _deduplicate_boxes(new_list)

    # ── K6: Retrieve functions ────────────────────────────────────────────

    def _retrieve_up_header(self, step: int) -> None:
        s = self._s
        new_list = []
        for box in self._boxes:
            r0, c0, r1, c1 = box
            mark_header = self.is_header_up((r0, c0, r0, c1))
            up = r0 - step
            while up > 0:
                bup = (up, c0, up, c1)
                if mark_header and not self.is_header_up(bup) and not s.exists_merged(bup):
                    break
                if _distinct_strs(s.content_strs, up, up, c0, c1) >= 2:
                    up -= 1; continue
                if (s.exists_merged(bup)
                        and _distinct_strs(s.content_strs, up, up, c0, c1) >= 2):
                    up -= 1; continue
                elif (s.ced(bup) >= 0.8 and s.ps_ce.query(*bup) >= 4
                      and _distinct_strs(s.content_strs, up, up, c0, c1) >= 2):
                    up -= 1; continue
                elif (c1-c0 >= 8
                      and (s.row_ced_split(bup, 4) >= 0.7 or s.row_ced_split(bup, 8) >= 0.7)
                      and not (s.ced(bup) >= 0.8 and not self.is_header_up(bup))):
                    up -= 1; continue
                else:
                    break
            if up < r0 - step and up >= r0 - 6:
                new_list.append((up+1, c0, r1, c1))
            else:
                new_list.append(box)
        self._boxes = _deduplicate_boxes(new_list)

    def _retrieve_left(self, step: int) -> None:
        s = self._s
        new_list = []
        for box in self._boxes:
            r0, c0, r1, c1 = box
            left = c0 - step
            while left > 0:
                bcol = (r0, left, r1, left)
                if s.exists_merged(bcol):
                    left -= 1; continue
                elif s.ced(bcol) >= 0.8 and s.ps_ce.query(*bcol) >= 4:
                    left -= 1; continue
                elif (r1-r0 >= 8
                      and (s.col_ced_split(bcol, 4) == 1 or s.col_ced_split(bcol, 8) >= 0.8)
                      and not (s.ced(bcol) >= 0.8 and not self.is_header_left(bcol))):
                    left -= 1; continue
                else:
                    break
            ok = (left < c0 - step and left >= c0 - 5
                  and not _is_overlap_any(
                      (r0, left+1, r1, c0-1), self._boxes))
            if ok:
                new_list.append((r0, left+1, r1, c1))
            else:
                new_list.append(box)
        self._boxes = _deduplicate_boxes(new_list)

    def _retrieve_left_header(self) -> None:
        s = self._s
        new_list = []
        for box in self._boxes:
            r0, c0, r1, c1 = box
            nb = box
            while nb[1] > 1:
                c0n = nb[1]
                bl  = _left_col(nb)
                bl1 = _left_col(nb, start=-1)
                bl2 = _left_col(nb, start=-2)
                bl3 = _left_col(nb, start=-3)
                if _is_overlap_any(bl2, self._boxes):
                    break
                if (not self.is_header_left(bl)
                        and s.ps_content.query(*bl1) > 3
                        and s.ps_content.query(*bl2) == 0):
                    nb = (nb[0], nb[1]-1, nb[2], nb[3])
                elif (not self.is_header_left(bl)
                      and s.ps_content.query(*bl1) + s.ps_content.query(*bl2) > 3
                      and s.ps_content.query(*bl3) == 0):
                    nb = (nb[0], nb[1]-2, nb[2], nb[3])
                elif (s.ps_content.query(*bl1) > 5
                      and s.col_ced_split(bl1, 5) > 0.2
                      and s.ps_brow.query(*bl2) == 0):
                    nb = (nb[0], nb[1]-1, nb[2], nb[3])
                elif s.ps_brow.query(*bl1) > 0:
                    nb = (nb[0], nb[1]-1, nb[2], nb[3])
                else:
                    break
            new_list.append(nb)
        self._boxes = _deduplicate_boxes(new_list)

    def _retrieve_distant_up_header(self) -> None:
        s = self._s
        rm: Set[Box] = set()
        add: Set[Box] = set()
        for box in self._boxes:
            if self.is_header_up_with_data_area(_up_row(box), box):
                continue
            compact = _up_row(box, start=-1)
            for hp in range(1, 5):
                compact = _up_row(box, start=-hp)
                ce = s.ps_ce.query(*compact)
                if ce >= 6 and s.tdc(compact) > 1 and s.ced(compact) >= 1.0:
                    break
            if not (self.is_header_up(compact) and self.header_rate(compact) > 0.8):
                continue
            cnt_hh = 0
            while (cnt_hh < 3
                   and self.is_header_up_simple(_up_row(compact, start=-1))):
                cnt_hh += 1
                compact = _up_row(compact, start=-1)
            checks = [_up_row(compact, start=-1), _up_row(compact, start=-2), _up_row(compact, start=-3)]
            if any(s.ps_ce.query(*ch) < 3 for ch in checks):
                rm.add(box)
                add.add((compact[0], box[1], box[2], box[3]))
        for b in rm: self._boxes.remove(b) if b in self._boxes else None
        self._boxes.extend(b for b in add if b not in self._boxes)
        self._boxes = _deduplicate_boxes(self._boxes)

    # ── K7: Merge/combine filters ─────────────────────────────────────────

    def _vertical_relational_merge(self) -> None:
        s = self._s
        rm: Set[Box] = set()
        forced: List[Box] = []
        for box in self._boxes:
            dr  = _down_row(box)
            for k in range(1, 6):
                dr2 = _down_row(box, start=-k)
                if (s.ps_ce.query(*_left_col(dr2)) != 0 and k > 1) or s.ps_ce.query(*dr2) >= 4:
                    break
            if (s.ced(dr2) >= 0.4 and s.ps_ce.query(*dr2) >= 4
                    and s.compute_similar_row(dr, dr2) < 0.1
                    and not self.is_header_up(dr2)
                    and not s.exists_merged(dr2)):
                forced.append((dr[0], dr2[0], box[1], box[3]))
            ur  = _up_row(box)
            for k in range(1, 6):
                ur2 = _up_row(box, start=-k)
                if (s.ps_ce.query(*_left_col(ur2)) != 0 and k > 1) or s.ps_ce.query(*ur2) >= 4:
                    break
            if (s.ced(ur2) >= 0.4 and s.ps_ce.query(*ur2) >= 4
                    and s.compute_similar_row(ur, ur2) < 0.1
                    and not self.is_header_up(ur)
                    and not s.exists_merged(ur)):
                forced.append((ur2[0], ur[2], box[1], box[3]))
        forced = _deduplicate_boxes(forced)
        for fb in forced:
            for box in list(self._boxes):
                if (fb[1] == box[1] and fb[3] == box[3]
                        and _is_overlap(box, fb)
                        and not _contains_box(box, fb)):
                    rm.add(box)
        self._boxes = [b for b in self._boxes if b not in rm]
        self._boxes = _deduplicate_boxes(self._boxes)

    def _border_cohesions_addition(self) -> None:
        s = self._s
        border_regions: List[Box] = list(s.cohesion_border_regions)
        for b in s.small_cohesion_border_regions:
            if b[2]-b[0] > 1 and b[3]-b[1] > 1:
                border_regions.append(b)
        clear = []
        for b in border_regions:
            bup = _up_row(b, start=-1)
            bdn = _down_row(b, start=-1)
            bl  = _left_col(b, start=-1)
            br  = _right_col(b, start=-1)
            if b[3]-b[1] > 3 and s.ced(bup) >= 1.0: continue
            if b[3]-b[1] > 3 and s.ced(bdn) >= 1.0: continue
            if b[2]-b[0] > 3 and s.ced(bl)  >= 1.0: continue
            if b[2]-b[0] > 3 and s.ced(br)  >= 1.0: continue
            clear.append(b)
        # suppression filter
        rm: Set[Box] = set()
        for b1 in self._boxes:
            for bc in clear:
                if _is_suppression_box(b1, bc) and b1 != bc:
                    rm.add(bc)
        clear = [b for b in clear if b not in rm]
        self._boxes.extend(clear)
        self._boxes = _deduplicate_boxes(self._boxes)

    def _forced_border_filter(self) -> None:
        s = self._s
        border_in = [b for b in s.small_cohesion_border_regions if b in self._boxes]
        rm: Set[Box] = set()
        for b in self._boxes:
            if _is_overlap_any(b, border_in, except_forward=True):
                rm.add(b)
        self._boxes = [b for b in self._boxes if b not in rm]

    def _merge_filter(self) -> None:
        # C# Utils.ContainsBox has list overloads:
        #   ContainsBox(box, list, step) = list.Any(mb => ContainsBox(box, mb, step) && box!=mb)
        #   ContainsBox(list, box, step) = list.Any(mb => ContainsBox(mb, box, step) && mb!=box)
        s = self._s
        rm: Set[Box] = set()
        for b in self._boxes:
            b_contains_any = any(_contains_box(b, mb, 2) and b != mb for mb in s.merge_boxes)
            any_contains_b = any(_contains_box(mb, b, 2) and mb != b for mb in s.merge_boxes)
            if ((_is_overlap_any(b, s.merge_boxes) and b_contains_any and any_contains_b)
                    or b in s.merge_boxes):
                rm.add(b)
        self._boxes = [b for b in self._boxes if b not in rm]

    # ── K8: Suppression + header filters ─────────────────────────────────

    def _check_sparsity_up_row(self, box: Box, depth: int) -> int:
        s = self._s
        area = _area_size(box)
        if self.is_header_up(box): return 1
        if s.ced(box) >= 0.6 and s.tdc(box) >= max(0.2*area, 3): return 1
        if depth == 1 and s.ps_ce.query(*box) <= 4 and area >= 6: return 2
        if depth == 1 and s.ps_ce.query(*box) <= 6 and area >= 10: return 2
        return 0

    def _check_sparsity_down_row(self, box: Box, depth: int) -> int:
        s = self._s
        area = _area_size(box)
        if s.ced(box) >= 0.6 and s.tdc(box) >= max(0.2*area, 3): return 1
        if depth == 1 and (s.ps_ce.query(*box) <= 4 or s.tdc(box) < 2) and area >= 6: return 2
        if depth == 1 and (s.ps_ce.query(*box) <= 6 or (s.ced(box) <= 0.6 and s.tdc(box) < 3)) and area >= 10: return 2
        return 0

    def _check_sparsity_col(self, box: Box, depth: int) -> int:
        s = self._s
        area = _area_size(box)
        if s.ced(box) >= 0.6 and s.tdc(box) >= max(area*0.2, 3): return 1
        if depth == 1 and s.ps_ce.query(*box) <= 4 and area >= 5: return 2
        return 0

    def _compare_suppression_header(self, b1: Box, b2: Box) -> int:
        if b1[0] != b2[0]:
            hu1 = self.is_header_up(_up_row(b1))
            hu2a = self.is_header_up(_up_row(b2))
            hu2b = self.is_header_up(_up_row(b2, start=1))
            if hu1 and not hu2a: return 1
            if not hu1 and (hu2a or hu2b): return 2
        if b1[1] != b2[1]:
            hl1 = self.is_header_left(_left_col(b1))
            hl2 = self.is_header_left(_left_col(b2))
            if hl1 and not hl2: return 1
            if not hl1 and hl2: return 2
        if b1[2] != b2[2]:
            if (self.is_header_up(_down_row(b1)) and not self.is_header_up(_down_row(b2))
                    and self._s.ced(_down_row(b2, start=-1)) < 0.4):
                return 2
        if b1[3] != b2[3]:
            if (self.is_header_left(_right_col(b1)) and not self.is_header_left(_right_col(b2))
                    and self._s.ced(_right_col(b2, start=-1)) < 0.4):
                return 2
        return 0

    def _compare_suppression_sparsity(self, b1: Box, b2: Box) -> int:
        if b1[0] != b2[0]:
            v = self._check_sparsity_up_row(_up_row(b1), 1)
            if v != 0: return v
        if b1[2] != b2[2]:
            v = self._check_sparsity_down_row(_down_row(b1), 1)
            if v != 0: return v
        if b1[1] != b2[1]:
            v = self._check_sparsity_col(_left_col(b1), 1)
            if v != 0: return v
        if b1[3] != b2[3]:
            v = self._check_sparsity_col(_right_col(b1), 1)
            if v != 0: return v
        for i in range(b2[0] - b1[0]):
            v = self._check_sparsity_up_row(_up_row(b1, start=i), i+1)
            if v != 0: return v
        for i in range(b1[2] - b2[2]):
            v = self._check_sparsity_down_row(_down_row(b1, start=i), i+1)
            if v != 0: return v
        for i in range(b2[1] - b1[1]):
            v = self._check_sparsity_col(_left_col(b1, i), i+1)
            if v != 0: return v
        for i in range(b1[3] - b2[3]):
            v = self._check_sparsity_col(_right_col(b1, i), i+1)
            if v != 0: return v
        return 0

    def _compare_suppression_merge(self, b1: Box, b2: Box) -> int:
        s = self._s
        if s.exists_merged(_right_col(b1, step=2)) and not s.exists_merged(_right_col(b2, step=2)): return 2
        if s.exists_merged(_down_row(b1, step=2)) and not s.exists_merged(_down_row(b2, step=2)): return 2
        return 0

    def _compare_suppression(self, b1: Box, b2: Box) -> int:
        if b1 == b2: return 0
        v = self._compare_suppression_header(b1, b2)
        if v != 0: return v
        v = self._compare_suppression_sparsity(b1, b2)
        if v != 0: return v
        v = self._compare_suppression_merge(b1, b2)
        return v

    def _suppression_soft_filter(self) -> None:
        rm: Set[Box] = set()
        for i in range(len(self._boxes)):
            b1 = self._boxes[i]
            if b1 in rm: continue
            for j in range(i+1, len(self._boxes)):
                b2 = self._boxes[j]
                if b2 in rm: continue
                if (not _is_suppression_box(b1, b2) or b1 == b2
                        or not _is_overlap(b1, b2)):
                    continue
                h1 = b1[2]-b1[0]+1; h2 = b2[2]-b2[0]+1
                w1 = b1[3]-b1[1]+1; w2 = b2[3]-b2[1]+1
                if h2 < 0.6*h1 or h1 < 0.6*h2 or w2 < 0.6*w1 or w1 < 0.6*w2:
                    continue
                ov = _overlap_box(b1, b2)
                v1 = self._compare_suppression(b1, ov)
                v2 = self._compare_suppression(b2, ov) if v1 == 0 else 0
                if v1 == 1 or v2 == 2:
                    rm.add(b2)
                elif v1 == 2 or v2 == 1:
                    rm.add(b1); break
        self._boxes = [b for b in self._boxes if b not in rm]

    def _header_priority_filter(self) -> None:
        rm: Set[Box] = set()
        boxes = self._boxes
        for i in range(len(boxes)):
            b1 = boxes[i]
            for j in range(len(boxes)):
                b2 = boxes[j]
                if b1 == b2 or not _is_overlap(b1,b2) or _contains_box(b2,b1,2): continue
                if abs(b1[0]-b2[0]) > 2 or abs(b1[2]-b2[2]) > 2: continue
                if abs(b1[1]-b2[1]) <= 1: continue
                if (_contains_box(b1,b2) and self.is_header_up(b2)
                        and not self.is_header_up((b1[0],b1[1],b1[0],b2[1]-1))):
                    rm.add(b1)
                elif (_contains_box(b1,b2) and b2[1]-b1[1] > 3
                      and self.is_header_up(b2)
                      and not self.is_header_up((b1[0],b1[1],b1[0],b2[1]-1))):
                    rm.add(b1)
                elif (self.is_header_left(_left_col(b1))
                      and not self.is_header_left(_left_col(b2))):
                    rm.add(b2)
                elif (self.is_header_left(_left_col(b2))
                      and not self.is_header_left(_left_col(b1))):
                    rm.add(b1); break
        for i in range(len(boxes)):
            b1 = boxes[i]
            for j in range(len(boxes)):
                b2 = boxes[j]
                if b2 == b1 or not _is_overlap(b1,b2): continue
                if abs(b1[1]-b2[1]) > 2 or abs(b1[3]-b2[3]) > 2: continue
                if abs(b1[0]-b2[0]) <= 1: continue
                if self.is_header_up(_up_row(b1)) and not self.is_header_up(_up_row(b2)):
                    rm.add(b2)
                elif self.is_header_up(_up_row(b2)) and not self.is_header_up(_up_row(b1)):
                    rm.add(b1); break
        self._boxes = [b for b in self._boxes if b not in rm]

    def _pair_alike_contains_filter(self) -> None:
        s = self._s
        rm: Set[Box] = set()
        # vertical
        for b1 in self._boxes:
            for b2 in self._boxes:
                if not _contains_box(b1, b2, 1) or b1[2] <= b2[2]: continue
                if (abs(b1[1]-b2[1]) > 2 or abs(b1[3]-b2[3]) > 2
                        or b1[3]-b1[1] >= 2*(b2[3]-b2[1])
                        or b1[3]-b1[1] <= 0.5*(b2[3]-b2[1])): continue
                if abs(b1[0]-b2[0]) > 2: continue
                cnt = 0
                remain = (b2[2]+1, b1[1], b1[2], b1[3])
                while remain[0] < remain[2] and s.ps_ce.query(*_up_row(remain)) < 3:
                    remain = (remain[0]+1, remain[1], remain[2], remain[3])
                for b3 in self._boxes:
                    if b3 in rm: continue
                    if _is_overlap(b2, b3) or b1 == b3 or b2 == b3: continue
                    if _contains_box(b3, remain, 2) or (_contains_box(remain, b3, 2) and self.is_header_up(_up_row(remain))):
                        cnt = 1; break
                    if ((abs(b3[1]-b1[1]) <= 2 or abs(b3[3]-b1[3]) <= 2) and _contains_box(b1, b3, 1)):
                        cnt = 1; break
                if cnt == 0:
                    b2b = (max(b2[0], b2[2]-12), b2[1], b2[2], b2[3])
                    if (not self.is_header_left(_left_col(remain))
                            and s.ced(b2b) > 2*s.ced(remain)
                            and (s.ced(remain) > 0.5 or s.ced(b2b) > 1.0)):
                        rm.add(b1)
                    elif (b2[2]-b2[0] >= 4 and s.ced(remain) < 0.5 and s.ced(b2b) > 1.0):
                        rm.add(b1)
                    elif (b2[2]-b2[0] >= 4 and s.ced(remain) < 0.2 and s.ced(b2b) > 0.7):
                        rm.add(b1)
                    elif (b1[2]-b2[2] <= 4 and b2[2]-b2[0] >= 5 and self.is_header_left(_left_col(b2)) and not self.is_header_left(_left_col(b1))):
                        rm.add(b1)
                    elif (self.is_header_up(_down_row(b1)) and s.ced(_down_row(b1, start=1)) < 0.4 and not self.is_header_up(_down_row(b1, start=3)) and not self.is_header_up(_down_row(b1, start=2))):
                        rm.add(b1)
                    else:
                        rm.add(b2)
        # horizontal
        for b1 in self._boxes:
            for b2 in self._boxes:
                if not _contains_box(b1, b2, 1): continue
                if (abs(b1[0]-b2[0]) > 2 or abs(b1[2]-b2[2]) > 2
                        or b1[2]-b1[0] >= 2*(b2[2]-b2[0])
                        or b1[2]-b1[0] <= 0.5*(b2[2]-b2[0])
                        or abs(b1[1]-b2[1]) > 2 or b1[3] <= b2[3]): continue
                cnt = 0
                remain = (b1[0], b2[3]+1, b1[2], b1[3])
                while remain[1] < remain[3] and s.ps_ce.query(*_left_col(remain)) < 3:
                    remain = (remain[0], remain[1]+1, remain[2], remain[3])
                for b3 in self._boxes:
                    if b3 in rm: continue
                    if _is_overlap(b2, b3) or b1 == b3 or b2 == b3: continue
                    if _contains_box(b3, remain, 2) or (_contains_box(remain, b3, 2) and self.is_header_left(_left_col(remain))):
                        cnt = 1; break
                    if ((abs(b3[0]-b1[0]) <= 2 or abs(b3[2]-b1[2]) <= 2) and _contains_box(b1, b3, 1)):
                        cnt = 1; break
                if cnt == 0:
                    b2r = (b2[0], max(b2[1], b2[3]-12), b2[2], b2[3])
                    if (not self.is_header_up(_up_row(remain)) and s.ced(b2r) > 2*s.ced(remain) and (s.ced(b2r) > 0.5 or s.ced(remain) > 0.25)):
                        rm.add(b1)
                    elif (b2[3]-b2[1] >= 4 and s.ced(remain) < 0.5 and s.ced(b2r) > 1.0):
                        rm.add(b1)
                    elif (b2[3]-b2[1] >= 4 and s.ced(remain) < 0.2 and s.ced(b2r) > 0.7):
                        rm.add(b1)
                    elif (b1[3]-b2[3] <= 4 and b2[3]-b2[1] >= 5 and self.is_header_up(_up_row(b2)) and not self.is_header_up(_up_row(b1))):
                        rm.add(b1)
                    elif (self.is_header_left(_right_col(b1)) and s.ced(_right_col(b1, start=1)) < 0.4 and not self.is_header_left(_right_col(b1, start=2)) and not self.is_header_left(_right_col(b1, start=3))):
                        rm.add(b1)
                    else:
                        rm.add(b2)
        self._boxes = [b for b in self._boxes if b not in rm]

    def _pair_contains_filter(self) -> None:
        s = self._s
        rm: Set[Box] = set()
        for b1 in self._boxes:
            has_partial = any(_is_overlap(b1, b2) and not _contains_box(b1, b2) and not _contains_box(b2, b1)
                              for b2 in self._boxes if b2 != b1)
            if has_partial: continue
            for b2 in self._boxes:
                if not _contains_box(b1, b2) or b1 == b2: continue
                # keep the one with better structure
                inner_dens = s.ced(b2)
                outer_extra = (b1[2]-b2[2])*(b1[3]-b1[1]+1) + (b2[0]-b1[0])*(b1[3]-b1[1]+1)
                if (inner_dens > 0.8 and s.ced(b1) < 0.5*inner_dens
                        and outer_extra > (b2[2]-b2[0]+1)*(b2[3]-b2[1]+1)):
                    rm.add(b1); break
                if b2[2]-b2[0] >= 4 and b1[2]-b2[2] <= 3 and self.is_header_up(_up_row(b2)):
                    rm.add(b1); break
        self._boxes = [b for b in self._boxes if b not in rm]

    def _nesting_combination_filter(self) -> None:
        s = self._s
        rm: List[Box] = []
        def _find_inter(ranges):
            res = []
            for b in ranges:
                in_ = any(not b == b2 and _contains_box(b2, b, 2) for b2 in ranges)
                out_ = any(not b == b2 and _contains_box(b, b2, 2) for b2 in ranges)
                if in_ and out_: res.append(b)
            return res
        for left in s.col_boundary_lines:
            for right in s.col_boundary_lines:
                if left >= right: continue
                cands = [b for b in self._boxes
                         if b[1] >= left-1 and b[1] <= left+3
                         and b[3] >= right-1 and b[3] <= right+3]
                rm.extend(_find_inter(cands))
        for up in s.row_boundary_lines:
            for dn in s.row_boundary_lines:
                if up >= dn: continue
                cands = [b for b in self._boxes
                         if b[0] >= up-1 and b[2] >= dn-1
                         and b[0] <= up+3 and b[2] <= dn+3]
                rm.extend(_find_inter(cands))
        rm_set = set(rm)
        self._boxes = [b for b in self._boxes if b not in rm_set]

    def _adjoin_header_filter(self) -> None:
        rm: Set[Box] = set()
        add: Set[Box] = set()
        for i, b1 in enumerate(self._boxes):
            for j, b2 in enumerate(self._boxes):
                if i >= j: continue
                if b1 == b2 or not _is_overlap(b1, b2): continue
                same_top_wide = (b1[0] == b2[0] and b2[2]-b2[0] > 4 and b1[2]-b1[0] > 4
                                 and self.is_header_up((b1[0], b1[0], min(b1[1],b2[1]), max(b1[3],b2[3]))))
                same_left_tall = (b1[1] == b2[1] and b2[3]-b2[1] > 4 and b1[3]-b1[1] > 4
                                  and self.is_header_left((min(b1[0],b2[0]), max(b1[2],b2[2]), b1[1], b1[1])))
                if not (same_top_wide or same_left_tall): continue
                merged = _unify_box(b1, b2)
                overlap_others = any(
                    not (b3 == b1 or b3 == b2) and _is_overlap(merged, b3)
                    for b3 in self._boxes)
                if not overlap_others:
                    if b1 != merged: rm.add(b1)
                    if b2 != merged: rm.add(b2)
                    add.add(merged)
        self._boxes = [b for b in self._boxes if b not in rm]
        self._boxes.extend(b for b in add if b not in self._boxes)
        self._boxes = _deduplicate_boxes(self._boxes)

    # ── K9: Region growth addition ────────────────────────────────────────

    def _pro_process_reduce_to_compact(self, ranges: List[Box]) -> List[Box]:
        s = self._s
        result = []
        for box in ranges:
            r0, c0, r1, c1 = box
            changed = True
            while changed and c0 <= c1 and r0 <= r1:
                changed = False
                while r0 < r1 and s.ced(_up_row((r0,c0,r1,c1))) < 0.8*2:
                    r0 += 1; changed = True
                while r1 > r0 and s.ced(_down_row((r0,c0,r1,c1))) < 0.8*2:
                    r1 -= 1; changed = True
                while c0 < c1 and s.ced(_left_col((r0,c0,r1,c1))) < 0.8*2:
                    c0 += 1; changed = True
                while c1 > c0 and s.ced(_right_col((r0,c0,r1,c1))) < 0.8*2:
                    c1 -= 1; changed = True
            if c0 < c1 and r0 < r1:
                result.append((r0, c0, r1, c1))
        return result

    def _add_region_growth(self) -> None:
        s = self._s
        rg = _find_connected_ranges(s.content_strs, None, s.nr, s.nc, 1, 1)
        add: Set[Box] = set()
        rm: Set[Box] = set()
        for box in rg:
            if s.ps_ce.query(*box) < 24: continue
            overlaps = [b for b in self._boxes if _is_overlap(b, box)]
            inside   = [b for b in self._boxes if _contains_box(box, b, 1) and not _is_suppression_box(box, b)]
            if not overlaps:
                add.add(box)
            elif (len(overlaps) == 1 and len(inside) == 1
                  and _area_size(box) - _area_size(inside[0]) > 20):
                extra_area = _area_size(box) - _area_size(inside[0])
                extra_ce   = s.ps_ce.query(*box) - s.ps_ce.query(*inside[0])
                if extra_ce / extra_area > 1.0:
                    add.add(box); rm.add(inside[0])
        for b in rm:
            if b in self._boxes: self._boxes.remove(b)
        self._boxes.extend(b for b in add if b not in self._boxes)
        self._boxes = _deduplicate_boxes(self._boxes)
        self._region_growth_boxes = rg

    def _add_compact_region_growth(self) -> None:
        s = self._s
        compact = self._pro_process_reduce_to_compact(self._region_growth_boxes)
        compact = self._pro_process_reduce_to_compact(compact)
        add: Set[Box] = set()
        for box in compact:
            if s.ps_ce.query(*box) < 7: continue
            exp = (box[0]-1, box[1]-1, box[2]+1, box[3]+1)
            if s.ps_ce.query(*exp) - s.ps_ce.query(*box) > 10: continue
            if not _is_overlap_any(box, self._boxes):
                add.add(box)
        self._boxes.extend(b for b in add if b not in self._boxes)

    # ── K10: GenerateRawCandidateBoxes ────────────────────────────────────

    def _generate_raw_candidate_boxes(self, block: Box) -> List[Box]:
        s = self._s
        br0, bc0, br1, bc1 = block

        # Filter boundary lines to block
        row_bl = []
        for row in s.row_boundary_lines:
            if row < br0 - 2 or row > br1 - 1: continue
            for idx in range(bc0 - 3, bc1):
                bup = (row+1, max(idx,0), row+1, min(idx+3, s.nc-1))
                bdn = (row+2, max(idx,0), row+2, min(idx+3, s.nc-1))
                cup = s.ps_ce.query(*bup); cdn = s.ps_ce.query(*bdn)
                if (cup > 0 and cdn == 0) or (cdn > 0 and cup == 0):
                    row_bl.append(row); break
        row_bl = list(set(row_bl))

        col_bl = []
        for col in s.col_boundary_lines:
            if col < bc0 - 2 or col > bc1 - 1: continue
            for idx in range(br0 - 3, br1):
                bl = (max(idx,0), col+1, min(idx+3, s.nr-1), col+1)
                br = (max(idx,0), col+2, min(idx+3, s.nr-1), col+2)
                cl = s.ps_ce.query(*bl); cr_ = s.ps_ce.query(*br)
                if (cl > 0 and cr_ == 0) or (cr_ > 0 and cl == 0):
                    col_bl.append(col); break
        col_bl = list(set(col_bl))

        mark_complex = False
        if len(row_bl) > 300:
            row_bl = row_bl[:300]; mark_complex = True
        if len(col_bl) > 150:
            col_bl = col_bl[:150]; mark_complex = True

        result: List[Box] = []
        for left in col_bl:
            if left < bc0 - 2: continue
            for right in col_bl:
                if left >= right or right > bc1 - 1: continue
                not_valid_down: Set[int] = set()
                for up in row_bl:
                    if up < br0 - 2: continue
                    out_up    = (up+1, up+1, left+2, right+1)
                    out_left  = (up+2, up+4, left+1, left+1)
                    out_right = (up+2, up+4, right+2, right+2)
                    in_up     = (up+2, up+2, left+2, right+1)
                    if s.ps_ce.query(*out_up)    >= 6: continue
                    if s.ps_ce.query(*out_left)  >= 6: continue
                    if s.ps_ce.query(*out_right) >= 6: continue
                    if s.ps_ce.query(*in_up)     == 0: continue
                    for down in row_bl:
                        if up >= down or down > br1 - 1: continue
                        if down in not_valid_down: continue
                        out_down = (down+2, down+2, left+2, right+1)
                        in_down  = (down+1, down+1, left+2, right+1)
                        if s.ps_ce.query(*out_down) >= 6:
                            not_valid_down.add(down); continue
                        if s.ps_ce.query(*in_down) == 0:
                            not_valid_down.add(down); continue
                        # candidate: 1-indexed in C# → 0-indexed here
                        cand = (up+1, left+1, down, right)
                        if (mark_complex
                                and s.ps_all.query(*cand) / max(_area_size(cand), 1) < 0.6):
                            continue
                        if self._general_filter(cand):
                            result.append(cand)
        return result

    # ── K11: Block + sheet-level pipelines ────────────────────────────────

    def _block_candidates_refine_and_filter(self) -> None:
        self._up_header_trim()
        self._overlap_cohesion_filter()
        self._overlap_border_cohesion_filter()
        self._little_boxes_filter()
        self._overlap_up_header_filter()
        self._surrounding_boundaries_trim()
        self._overlap_up_header_filter()
        self._splitted_empty_lines_filter()

    def _candidates_refine_and_filter(self) -> None:
        self._border_cohesions_addition()
        self._little_boxes_filter()
        self._retrieve_distant_up_header()
        self._vertical_relational_merge()
        self._suppression_soft_filter()
        self._header_priority_filter()
        self._pair_alike_contains_filter()
        self._pair_contains_filter()
        # CombineContainsFillAreaFilterSoft, CombineContainsFillLineFilterSoft,
        # ContainsLittleFilter: skipped (incomplete source)
        self._pair_alike_contains_filter()
        self._pair_contains_filter()
        self._nesting_combination_filter()
        # OverlapHeaderFilter: skipped (needs FindoutUpheaders/Leftheaders fully)
        self._boxes = _deduplicate_boxes(self._boxes)
        self._forced_border_filter()
        self._adjoin_header_filter()
        self._little_boxes_filter()
        # PairContainsFilterHard, CombineContainsFilterHard: skipped
        self._add_region_growth()
        self._add_compact_region_growth()
        self._merge_filter()
        self._retrieve_left_header()
        self._left_header_trim()
        self._bottom_trim()
        self._retrieve_up_header(1)
        self._retrieve_up_header(2)
        self._up_trim_simple()
        self._little_boxes_filter()

    def _region_growth_detect(self) -> None:
        s = self._s
        self._boxes = _find_connected_ranges(
            s.content_strs, s.vm_border, s.nr, s.nc, 1, 1, direct=1)
        self._little_boxes_filter()
        self._up_header_trim()
        self._surrounding_boundaries_trim()
        self._retrieve_up_header(1)
        self._retrieve_up_header(2)
        self._retrieve_left_header()
        self._retrieve_left(1)
        self._retrieve_left(2)

    def _table_sense_detect(self) -> None:
        s = self._s
        large_thresh = s.nr * s.nc > 10_000
        all_rg: List[Box] = []

        for th, tv in _THRESH_PAIRS:
            cands = _find_connected_ranges(s.content_strs, s.vm_border, s.nr, s.nc, th, tv, 0)
            for box in cands:
                if self._general_filter(box):
                    all_rg.append(box)
            if not large_thresh:
                cands2 = _find_connected_ranges(s.content_strs, None, s.nr, s.nc, th, tv, 0)
                for box in cands2:
                    if self._general_filter(box):
                        all_rg.append(box)

        all_rg = _deduplicate_boxes(all_rg)

        blocks = _generate_block_regions(s.nr, s.nc, s.ps_content, s.ps_ce,
                                          s.ps_color, s.ps_border)
        if not blocks:
            blocks = [(0, 0, s.nr - 1, s.nc - 1)]

        sheet_boxes: List[Box] = []
        if not large_thresh:
            for block in blocks:
                raw = self._generate_raw_candidate_boxes(block)
                for box in all_rg:
                    if _is_overlap(block, box):
                        raw.append(box)
                self._boxes = _deduplicate_boxes(raw)
                self._block_candidates_refine_and_filter()
                sheet_boxes.extend(self._boxes)
        else:
            sheet_boxes = all_rg

        self._boxes = _deduplicate_boxes(sheet_boxes)
        self._candidates_refine_and_filter()

    def detect(self) -> List[Box]:
        s = self._s
        if s.nr > 1000 or s.nr * s.nc > MAX_SIMPLE_CELLS:
            self._region_growth_detect()
        else:
            self._table_sense_detect()
        self._eliminate_overlaps()
        return self._boxes


# ============================================================
#  Section L: _detect_tables — top-level table detection driver
# ============================================================

def _detect_tables(data: SpreadsheetData) -> List[Box]:
    """Run the full C# detection pipeline on *data* and return detected boxes."""
    nr, nc = data.n_rows, data.n_cols
    if nr == 0 or nc == 0:
        return []

    (content_strs, has_content,
     has_top, has_bottom, has_left, has_right,
     has_color, merged_regions,
     alpha_ratio, number_ratio, sp_char_ratio, text_length
     ) = _extract_features(data)

    (ps_content, ps_ce, ps_border, ps_bcol, ps_brow,
     ps_color, ps_all,
     vm_content_raw, vm_ce_raw, vm_border_raw, vm_color_raw
     ) = _build_value_maps(has_content, has_top, has_bottom, has_left, has_right,
                           has_color, nr, nc)

    row_db, col_db = _compute_value_diff(
        vm_content_raw, vm_ce_raw, vm_color_raw,
        ps_all, ps_ce,
        has_top, nr, nc,
    )
    row_bounds, col_bounds = _propose_boundary_lines(row_db, col_db, nr, nc)

    s = _S(
        content_strs, has_content,
        has_top, has_bottom, has_left, has_right,
        has_color, merged_regions,
        alpha_ratio, number_ratio, sp_char_ratio, text_length,
        ps_content, ps_ce, ps_border, ps_bcol, ps_brow,
        ps_color, ps_all,
        vm_content_raw, vm_ce_raw, vm_border_raw, vm_color_raw,
        nr, nc,
    )
    s.row_boundary_lines = row_bounds
    s.col_boundary_lines = col_bounds

    _cohesion_detection(s, merged_regions, has_color, nr, nc, row_bounds, col_bounds)

    return _TableDetector(s).detect()


# ============================================================
#  Section M: compress_struture pipeline helpers
#             (ported from C# Compress_structure.py)
# ============================================================

def _col_num_to_letter(n: int) -> str:
    """1-indexed column number → Excel column letter (A, B, …, Z, AA, …)."""
    s = ''
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _is_row_empty(content_strs: List[List[str]], r: int) -> bool:
    if r < 0 or r >= len(content_strs):
        return True
    return all(v == '' for v in content_strs[r])


def _is_col_empty(content_strs: List[List[str]], c: int) -> bool:
    for row in content_strs:
        if c < len(row) and row[c] != '':
            return False
    return True


def _find_consecutive_ones_intervals(tag: List[int]) -> List[Tuple[int, int]]:
    """Return (start, end) inclusive intervals of runs of 1s with length ≥ 2."""
    intervals: List[Tuple[int, int]] = []
    start = None
    count = 0
    for i, v in enumerate(tag):
        if v == 1:
            if start is None:
                start = i
            count += 1
        if v != 1 or i == len(tag) - 1:
            if count >= 2:
                end = i if v != 1 else i + 1
                intervals.append((start, end - 1))
            start = None
            count = 0
    return intervals


def _apply_delete_space(
    content_strs: List[List[str]],
) -> Tuple[List[int], List[int]]:
    """Return kept_rows and kept_cols after removing runs of ≥2 empty rows/cols
    that are not at the edges (matching C# delete_space logic)."""
    nr = len(content_strs)
    nc = len(content_strs[0]) if nr else 0

    row_tag = [1 if _is_row_empty(content_strs, r) else 0 for r in range(nr)]
    row_intervals = _find_consecutive_ones_intervals(row_tag)
    for r0_int, r1_int in row_intervals:
        if r0_int == 0 or r1_int == nr - 1:
            continue
        row_tag[r0_int] = 2
        row_tag[r1_int] = 2

    col_tag = [1 if _is_col_empty(content_strs, c) else 0 for c in range(nc)]
    col_intervals = _find_consecutive_ones_intervals(col_tag)
    for c0_int, c1_int in col_intervals:
        if c0_int == 0 or c1_int == nc - 1:
            continue
        col_tag[c0_int] = 2
        col_tag[c1_int] = 2

    kept_rows = [r for r in range(nr) if row_tag[r] != 1]
    kept_cols = [c for c in range(nc) if col_tag[c] != 1]
    return kept_rows, kept_cols


# ============================================================
#  Section N: extract_anchors_original  (Module 1 — faithful)
# ============================================================

def extract_anchors_original(
    data: SpreadsheetData,
    k: int = DELTA,
    gt_ranges: Optional[List[str]] = None,
) -> Tuple[List[List[Any]], List[List[str]], Dict[int, int], Dict[int, int]]:
    """Module 1: Structural-anchor-based extraction, faithful to C# code.

    Steps:
      1. Run full C# TableDetectionHybrid pipeline to find table bounding boxes.
      2. Collect row/col boundary indices from table corners.
         If gt_ranges is provided (training mode), GT corners are added too,
         guaranteeing that ground-truth table boundaries survive extraction.
      3. Expand each anchor by ±k (DELTA=4).
      4. Apply delete_space (remove runs of ≥2 empty rows/cols in interior).
      5. Re-index coordinates to be contiguous (coordinate_rearrangement).

    Returns:
        compressed_values, compressed_nfs,
        row_map (compressed_idx -> original_idx),
        col_map (compressed_idx -> original_idx).
    """
    matrix = data.values
    nfs = data.number_formats
    nr, nc = data.n_rows, data.n_cols

    if nr == 0 or nc == 0:
        return [], [], {}, {}

    # Step 1: Detect tables
    boxes = _detect_tables(data)

    # Step 2+3: Collect anchor rows/cols from table boundaries, expand by k
    anchor_rows: Set[int] = set()
    anchor_cols: Set[int] = set()

    # In training mode, seed anchors from GT corners so they are never dropped
    if gt_ranges:
        for rng in gt_ranges:
            parts = rng.replace(' ', '').split(':')
            for addr in parts:
                parsed = parse_address(addr)
                if parsed is not None:
                    r_gt, c_gt = parsed
                    anchor_rows.add(max(0, min(r_gt, nr - 1)))
                    anchor_cols.add(max(0, min(c_gt, nc - 1)))

    for r0, c0, r1, c1 in boxes:
        anchor_rows.add(r0)
        anchor_rows.add(r1)
        anchor_cols.add(c0)
        anchor_cols.add(c1)

    # Fallback: if no tables detected, keep all rows/cols
    if not anchor_rows:
        kept_rows_set: Set[int] = set(range(nr))
        kept_cols_set: Set[int] = set(range(nc))
    else:
        kept_rows_set = set()
        for a in anchor_rows:
            for i in range(max(0, a - k), min(nr, a + k + 1)):
                kept_rows_set.add(i)
        kept_cols_set = set()
        for a in anchor_cols:
            for j in range(max(0, a - k), min(nc, a + k + 1)):
                kept_cols_set.add(j)

    kept_rows_list = sorted(kept_rows_set)
    kept_cols_list = sorted(kept_cols_set)

    # Step 4: delete_space on the retained sub-matrix
    sub_content = [
        ['' if matrix[r][c] is None else str(matrix[r][c]).strip()
         for c in kept_cols_list]
        for r in kept_rows_list
    ]
    local_kept_rows, local_kept_cols = _apply_delete_space(sub_content)

    # Map local indices back to original indices
    final_rows = [kept_rows_list[i] for i in local_kept_rows]
    final_cols = [kept_cols_list[j] for j in local_kept_cols]

    # Step 5: Build compressed matrices and coordinate maps
    compressed_values = [[matrix[r][c] for c in final_cols] for r in final_rows]
    compressed_nfs    = [[nfs[r][c]    for c in final_cols] for r in final_rows]

    row_map = {new: old for new, old in enumerate(final_rows)}
    col_map = {new: old for new, old in enumerate(final_cols)}

    return compressed_values, compressed_nfs, row_map, col_map


# ============================================================
#  VANILLA ENCODING
# ============================================================

def encode_vanilla(data: SpreadsheetData,
                   include_format: bool = False,
                   format_attrs: Optional[List[List[List[str]]]] = None) -> str:
    """Section 3.1 vanilla encoding: '|Address,Value|...\\n' row by row."""
    lines: List[str] = []
    for r in range(data.n_rows):
        cells = []
        for c in range(data.n_cols):
            val = data.values[r][c]
            val_str = '' if val is None else str(val)
            cells.append(f"{cell_address(r, c)},{val_str}")
        lines.append("|" + "|".join(cells) + "|")
    text = "\n".join(lines)

    if include_format and format_attrs is not None:
        fmt_lines = []
        for r in range(min(data.n_rows, len(format_attrs))):
            cells = []
            for c in range(min(data.n_cols, len(format_attrs[r]))):
                attrs = format_attrs[r][c] or []
                attr_str = ','.join(attrs)
                cells.append(f"{cell_address(r, c)},{attr_str}")
            fmt_lines.append("|" + "|".join(cells) + "|")
        text += "\n\nFormat Input:\n" + "\n".join(fmt_lines)
    return text


# ============================================================
#  MODULE 2: INVERTED-INDEX TRANSLATION
# ============================================================

def find_rectangles_with_value(
    positions: List[Tuple[int, int]],
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Greedy rectangle merger for same-value cells."""
    if not positions:
        return []
    pos_set = set(positions)
    used: Set[Tuple[int, int]] = set()
    rectangles: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    for r0, c0 in sorted(positions):
        if (r0, c0) in used:
            continue
        c1 = c0
        while (r0, c1 + 1) in pos_set and (r0, c1 + 1) not in used:
            c1 += 1
        r1 = r0
        while True:
            ok = True
            for c in range(c0, c1 + 1):
                if (r1 + 1, c) not in pos_set or (r1 + 1, c) in used:
                    ok = False
                    break
            if not ok:
                break
            r1 += 1
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                used.add((r, c))
        rectangles.append(((r0, c0), (r1, c1)))
    return rectangles


# ============================================================
#  MODULE 3: DATA-TYPE DETECTION (deactivated for fine-tuning)
# ============================================================

DATA_TYPE_LABELS = (
    "Year", "IntNum", "FloatNum", "Percentage", "ScientificNum",
    "DateData", "TimeData", "CurrencyData", "EmailData",
)

_RE_YEAR  = re.compile(r'^(?:19|20)\d{2}$')
_RE_INT   = re.compile(r'^-?\d{1,3}(?:,\d{3})+$|^-?\d+$')
_RE_FLOAT = re.compile(r'^-?\d{1,3}(?:,\d{3})*\.\d+$|^-?\d*\.\d+$')
_RE_PCT   = re.compile(r'^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%$|^-?\d+(?:\.\d+)?\s*%$')
_RE_SCI   = re.compile(r'^-?\d+(?:\.\d+)?[eE][+-]?\d+$')
_RE_DATE_NUMERIC = re.compile(r'^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$')
_RE_DATE_MMM = re.compile(
    r'^\d{1,2}[-/\s](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    r'(?:[-/\s]\d{2,4})?$', re.IGNORECASE)
_RE_TIME     = re.compile(r'^\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?$')
_RE_CURRENCY = re.compile(r'^[\$\€\£\¥]\s?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$')
_RE_EMAIL    = re.compile(r'^[\w.+-]+@[\w-]+\.[\w.-]+$')


def detect_data_type(value: Any, nfs: str = 'General') -> Optional[str]:
    import datetime
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, datetime.datetime):
        return "DateData"
    if isinstance(value, datetime.date):
        return "DateData"
    if isinstance(value, datetime.time):
        return "TimeData"
    if isinstance(value, int):
        return "Year" if 1900 <= value <= 2100 else "IntNum"
    if isinstance(value, float):
        return "FloatNum"

    s = str(value).strip()
    if s == '':
        return None

    if nfs and nfs not in ('General', '@'):
        nfs_l = nfs.lower()
        if '%' in nfs_l:
            return "Percentage"
        if 'e+' in nfs_l or 'e-' in nfs_l:
            return "ScientificNum"
        if any(tok in nfs_l for tok in ('yyyy', 'yy', 'mmm', 'dd', 'mm-', '-mm')) and 'h' not in nfs_l:
            return "DateData"
        if any(tok in nfs_l for tok in ('h:mm', 'hh:', 'ss', 'am/pm')):
            return "TimeData"
        if any(tok in nfs_l for tok in ('$', '€', '£', '¥')):
            return "CurrencyData"
        if '0' in nfs_l or '#' in nfs_l:
            try:
                f = float(s.replace(',', ''))
                return "IntNum" if f.is_integer() else "FloatNum"
            except ValueError:
                pass

    if _RE_YEAR.match(s):  return "Year"
    if _RE_PCT.match(s):   return "Percentage"
    if _RE_SCI.match(s):   return "ScientificNum"
    if _RE_FLOAT.match(s): return "FloatNum"
    if _RE_INT.match(s):   return "IntNum"
    if _RE_TIME.match(s):  return "TimeData"
    if _RE_DATE_NUMERIC.match(s) or _RE_DATE_MMM.match(s): return "DateData"
    if _RE_CURRENCY.match(s): return "CurrencyData"
    if _RE_EMAIL.match(s):    return "EmailData"
    return None


def aggregate_by_format(
    values: List[List[Any]],
    nfs_matrix: List[List[str]],
) -> Tuple[List[Tuple[Tuple[int, int, int, int], str]], List[List[bool]]]:
    n_rows = len(values)
    n_cols = len(values[0]) if n_rows > 0 else 0
    if n_rows == 0 or n_cols == 0:
        return [], [[False] * n_cols for _ in range(n_rows)]

    type_matrix: List[List[Optional[str]]] = [
        [detect_data_type(values[r][c], nfs_matrix[r][c]) for c in range(n_cols)]
        for r in range(n_rows)
    ]
    visited    = [[False] * n_cols for _ in range(n_rows)]
    aggregated = [[False] * n_cols for _ in range(n_rows)]
    regions: List[Tuple[Tuple[int, int, int, int], str]] = []

    for r0 in range(n_rows):
        for c0 in range(n_cols):
            if visited[r0][c0]:
                continue
            t = type_matrix[r0][c0]
            if t is None:
                visited[r0][c0] = True
                continue
            stack = [(r0, c0)]
            min_r = max_r = r0
            min_c = max_c = c0
            cells: List[Tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                if not (0 <= r < n_rows and 0 <= c < n_cols):
                    continue
                if visited[r][c]:
                    continue
                if type_matrix[r][c] != t:
                    continue
                visited[r][c] = True
                cells.append((r, c))
                min_r = min(min_r, r); max_r = max(max_r, r)
                min_c = min(min_c, c); max_c = max(max_c, c)
                stack.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
            for r, c in cells:
                aggregated[r][c] = True
            regions.append(((min_r, min_c, max_r, max_c), t))
    return regions, aggregated


# ============================================================
#  SHEETCOMPRESSOR — full pipeline
# ============================================================

def encode_sheet_compressor(
    data: SpreadsheetData,
    k: int = DELTA,
    use_extraction: bool = True,
    use_translation: bool = True,
    use_aggregation: bool = False,   # deactivated: paper Table 2 M1+M2 best
    use_nfs_as_label: bool = True,
    gt_ranges: Optional[List[str]] = None,
) -> Tuple[str, Dict[int, int], Dict[int, int]]:
    """Apply Modules 1, 2, (3) and return LLM-ready encoding + coordinate maps.

    Module 1 uses the faithful C# TableDetectionHybrid pipeline.
    Module 3 is off by default (paper Table 2: M1+M2 outperforms M1+M2+M3).
    gt_ranges: when provided (training mode), GT table corner coordinates are
        added as anchors so that ground-truth boundaries are never dropped by
        Module 1 extraction.
    """
    # ---- Module 1 ----
    if use_extraction:
        values, nfs_matrix, row_map, col_map = extract_anchors_original(
            data, k=k, gt_ranges=gt_ranges)
    else:
        values    = [list(r) for r in data.values]
        nfs_matrix = [list(r) for r in data.number_formats]
        row_map = {i: i for i in range(data.n_rows)}
        col_map = {j: j for j in range(data.n_cols)}

    n_rows = len(values)
    n_cols = len(values[0]) if n_rows > 0 else 0

    # ---- Module 3 ----
    if use_aggregation and n_rows > 0:
        regions, aggregated = aggregate_by_format(values, nfs_matrix)
    else:
        regions    = []
        aggregated = [[False] * n_cols for _ in range(n_rows)]

    tuples: List[str] = []

    # ---- Module 2 ----
    if use_translation:
        value_to_positions: Dict[str, List[Tuple[int, int]]] = {}
        for r in range(n_rows):
            for c in range(n_cols):
                if aggregated[r][c]:
                    continue
                val = values[r][c]
                if val is None or (isinstance(val, str) and val == ''):
                    continue
                value_to_positions.setdefault(str(val), []).append((r, c))

        for val, positions in value_to_positions.items():
            for (r0, c0), (r1, c1) in find_rectangles_with_value(positions):
                addr = (cell_address(r0, c0)
                        if (r0, c0) == (r1, c1)
                        else f"{cell_address(r0, c0)}:{cell_address(r1, c1)}")
                tuples.append(f"({val}|{addr})")
    else:
        lines: List[str] = []
        for r in range(n_rows):
            cells = []
            for c in range(n_cols):
                if aggregated[r][c]:
                    continue
                val = values[r][c]
                val_str = '' if val is None else str(val)
                cells.append(f"{cell_address(r, c)},{val_str}")
            if cells:
                lines.append("|" + "|".join(cells) + "|")
        grid_part = "\n".join(lines)

    # ---- Aggregated regions ----
    for (r0, c0, r1, c1), label in regions:
        if use_nfs_as_label:
            nfs_here = nfs_matrix[r0][c0]
            if nfs_here and nfs_here not in ('General', '@'):
                label = nfs_here
        addr = (cell_address(r0, c0)
                if (r0, c0) == (r1, c1)
                else f"{cell_address(r0, c0)}:{cell_address(r1, c1)}")
        tuples.append(f"({label}|{addr})")

    if use_translation:
        encoded = "\n".join(tuples)
    else:
        encoded = grid_part + ("\n" + "\n".join(tuples) if tuples else "")

    return encoded, row_map, col_map
