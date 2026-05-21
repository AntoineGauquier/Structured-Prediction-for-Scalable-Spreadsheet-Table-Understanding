from typing import Dict, List, Optional, Tuple

from .utils import parse_range


def _to_box(rng):
    """Accept either an A1-style string ('A1:F44') or an already parsed box
    ((r0,c0),(r1,c1)). Returns the normalized box tuple."""
    if rng is None:
        return None
    if isinstance(rng, str):
        return parse_range(rng)
    # Assume it is ((r0,c0),(r1,c1)) — normalize ordering for safety.
    (r0, c0), (r1, c1) = rng
    return ((min(r0, r1), min(c0, c1)), (max(r0, r1), max(c0, c1)))


def eob_0_match(predicted: str, ground_truth: str) -> bool:
    p = _to_box(predicted)
    g = _to_box(ground_truth)
    if p is None or g is None:
        return False
    return p == g


def evaluate_detection(predicted_ranges: List[str],
                       gt_ranges: List[str]) -> Dict[str, float]:
    """Per-spreadsheet precision/recall/F1 with EoB-0 matching.

    Each ground-truth range can match at most one predicted range and vice
    versa (one-to-one bipartite matching, greedy in input order)."""
    matched_gt = set()
    matched_pred = set()
    for i, p in enumerate(predicted_ranges):
        for j, g in enumerate(gt_ranges):
            if j in matched_gt:
                continue
            if eob_0_match(p, g):
                matched_gt.add(j)
                matched_pred.add(i)
                break

    tp = len(matched_pred)
    fp = max(0, len(predicted_ranges) - tp)
    fn = max(0, len(gt_ranges) - len(matched_gt))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'n_pred': len(predicted_ranges),
        'n_gt': len(gt_ranges),
    }


def _iou(p, g) -> float:
    """Intersection-over-Union for two axis-aligned bounding boxes."""
    (pr0, pc0), (pr1, pc1) = p
    (gr0, gc0), (gr1, gc1) = g
    ir0, ic0 = max(pr0, gr0), max(pc0, gc0)
    ir1, ic1 = min(pr1, gr1), min(pc1, gc1)
    inter = max(0, ir1 - ir0 + 1) * max(0, ic1 - ic0 + 1)
    if inter == 0:
        return 0.0
    area_p = (pr1 - pr0 + 1) * (pc1 - pc0 + 1)
    area_g = (gr1 - gr0 + 1) * (gc1 - gc0 + 1)
    return inter / (area_p + area_g - inter)


def evaluate_detection_iou(predicted_ranges: List[str],
                           gt_ranges: List[str],
                           thresholds: Tuple[float, ...] = (0.5, 0.75, 0.95),
                           ) -> Dict[str, float]:
    """Per-spreadsheet P/R/F1 at multiple IoU thresholds (greedy matching).

    Returns keys ``precision@<t>``, ``recall@<t>``, ``f1@<t>`` for each
    threshold *t*, plus ``tp@<t>``, ``fp@<t>``, ``fn@<t>`` for aggregation."""
    pred_boxes = [_to_box(r) for r in predicted_ranges]
    gt_boxes   = [_to_box(r) for r in gt_ranges]
    pred_boxes = [b for b in pred_boxes if b is not None]
    gt_boxes   = [b for b in gt_boxes   if b is not None]

    results: Dict[str, float] = {}
    for t in thresholds:
        matched_gt:   set = set()
        matched_pred: set = set()
        for i, pb in enumerate(pred_boxes):
            for j, gb in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                if _iou(pb, gb) >= t:
                    matched_gt.add(j)
                    matched_pred.add(i)
                    break
        tp = len(matched_pred)
        fp = max(0, len(pred_boxes) - tp)
        fn = max(0, len(gt_boxes)   - len(matched_gt))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        tag = str(t)
        results[f'precision@{tag}'] = prec
        results[f'recall@{tag}']    = rec
        results[f'f1@{tag}']        = f1
        results[f'tp@{tag}']        = tp
        results[f'fp@{tag}']        = fp
        results[f'fn@{tag}']        = fn
    results['n_pred'] = len(pred_boxes)
    results['n_gt']   = len(gt_boxes)
    return results


def aggregate_iou_metrics(per_sheet: List[Dict[str, float]],
                          thresholds: Tuple[float, ...] = (0.5, 0.75, 0.95),
                          ) -> Dict[str, float]:
    """Micro-averaged IoU P/R/F1 over a list of per-sheet evaluations."""
    results: Dict[str, float] = {}
    for t in thresholds:
        tag = str(t)
        tp = sum(d.get(f'tp@{tag}', 0) for d in per_sheet)
        fp = sum(d.get(f'fp@{tag}', 0) for d in per_sheet)
        fn = sum(d.get(f'fn@{tag}', 0) for d in per_sheet)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[f'precision@{tag}'] = prec
        results[f'recall@{tag}']    = rec
        results[f'f1@{tag}']        = f1
        results[f'tp@{tag}'] = tp
        results[f'fp@{tag}'] = fp
        results[f'fn@{tag}'] = fn
    results['n_sheets'] = len(per_sheet)
    return results


def aggregate_metrics(per_sheet: List[Dict[str, float]]) -> Dict[str, float]:
    """Micro-averaged P/R/F1 over a list of per-sheet evaluations."""
    tp = sum(d['tp'] for d in per_sheet)
    fp = sum(d['fp'] for d in per_sheet)
    fn = sum(d['fn'] for d in per_sheet)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'n_sheets': len(per_sheet),
    }
