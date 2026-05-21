"""Pairwise feature extraction for 4-neighboring cells in a sheet."""

import numpy as np

def compute_pairwise_features(unary_features):
    """
    Compute pairwise features for 4-connected neighbors.

    Args:
        unary_features: (H, W, F) array of unary features

    Returns:
        edges: (E, 2) array of edge indices (node_i, node_j)
        pairwise_features: (E, 30) array of pairwise features
    """
    H, W, F = unary_features.shape
    # Flatten to (N, F) for vectorised edge indexing
    U = unary_features.reshape(-1, F)
    ids = np.arange(H * W).reshape(H, W)

    # Horizontal edges (i → right neighbour j)
    e1 = np.stack([ids[:, :-1].ravel(), ids[:, 1:].ravel()], axis=1)
    # Vertical edges (i → bottom neighbour j)
    e2 = np.stack([ids[:-1, :].ravel(), ids[1:, :].ravel()], axis=1)

    edges = np.vstack([e1, e2])
    E = edges.shape[0]

    xi = U[edges[:, 0]]
    xj = U[edges[:, 1]]
    pf = np.zeros((E, 30), dtype=np.float32)

    # Unary feature indices (must match extract_unary_features):
    # 14 = normalised row position, 15 = normalised col position
    same_row = (xi[:, 14] == xj[:, 14])
    same_col = (xi[:, 15] == xj[:, 15])

    pf[:, 0] = same_row
    pf[:, 1] = same_col
    # is_after: j comes after i in reading order (left-to-right, top-to-bottom)
    is_after = (same_row & (xi[:, 15] < xj[:, 15])) | (same_col & (xi[:, 14] > xj[:, 14]))
    pf[:, 2] = is_after

    # Boolean unary features stored as floats — threshold at 0.5 to recover bool
    def B(a): return a > 0.5

    # pf 3-7: both cells share the same basic cell-type flag (both empty, both numeric, etc.)
    pf[:, 3] = B(xi[:, 0]) & B(xj[:, 0])  # both is_na
    pf[:, 4] = B(xi[:, 1]) & B(xj[:, 1])  # both is_number
    pf[:, 5] = B(xi[:, 2]) & B(xj[:, 2])  # both is_string
    pf[:, 6] = B(xi[:, 3]) & B(xj[:, 3])  # both is_date
    pf[:, 7] = B(xi[:, 4]) & B(xj[:, 4])  # both is_formula

    # pf 8-10: absolute difference in text-length (5), digit-count (6), and starts-with-letter (12)
    pf[:, 8] = np.abs(xi[:, 5] - xj[:, 5])
    pf[:, 9] = np.abs(xi[:, 6] - xj[:, 6])
    pf[:, 10] = np.abs(xi[:, 12] - xj[:, 12])

    # pf 11-13: cross-type transitions (one empty↔non-empty, one number↔string, one number↔date)
    pf[:, 11] = B(xi[:, 0]) ^ B(xj[:, 0])
    pf[:, 12] = (B(xi[:, 1]) & B(xj[:, 2])) | (B(xi[:, 2]) & B(xj[:, 1]))
    pf[:, 13] = (B(xi[:, 1]) & B(xj[:, 3])) | (B(xi[:, 3]) & B(xj[:, 1]))

    # Formatting features start at index 23 in the unary vector
    FMT = 23

    # pf 14-15: same font size / same font name hash
    pf[:, 14] = xi[:, FMT+4] == xj[:, FMT+4]
    pf[:, 15] = xi[:, FMT+3] == xj[:, FMT+3]

    # pf 16-17: both bold / both italic
    pf[:, 16] = B(xi[:, FMT+0]) & B(xj[:, FMT+0])
    pf[:, 17] = B(xi[:, FMT+1]) & B(xj[:, FMT+1])

    # pf 18: same foreground (font) colour
    pf[:, 18] = (
        (xi[:, FMT+5] == xj[:, FMT+5]) &
        (xi[:, FMT+6] == xj[:, FMT+6]) &
        (xi[:, FMT+7] == xj[:, FMT+7])
    )
    # pf 19-21: absolute RGB difference in foreground colour
    pf[:, 19] = np.abs(xi[:, FMT+5] - xj[:, FMT+5])
    pf[:, 20] = np.abs(xi[:, FMT+6] - xj[:, FMT+6])
    pf[:, 21] = np.abs(xi[:, FMT+7] - xj[:, FMT+7])

    # pf 22: same background colour
    pf[:, 22] = (
        (xi[:, FMT+8] == xj[:, FMT+8]) &
        (xi[:, FMT+9] == xj[:, FMT+9]) &
        (xi[:, FMT+10] == xj[:, FMT+10])
    )
    # pf 23-25: absolute RGB difference in background colour
    pf[:, 23] = np.abs(xi[:, FMT+8] - xj[:, FMT+8])
    pf[:, 24] = np.abs(xi[:, FMT+9] - xj[:, FMT+9])
    pf[:, 25] = np.abs(xi[:, FMT+10] - xj[:, FMT+10])

    # pf 26: a cell border exists between i and j (checks right/bottom border of i and left/top of j)
    hor_ij = same_row & is_after & ((xi[:, FMT+13]==1) | (xj[:, FMT+12]==1))
    ver_ij = same_col & is_after & ((xi[:, FMT+14]==1) | (xj[:, FMT+15]==1))
    hor_ji = same_row & (~is_after) & ((xj[:, FMT+13]==1) | (xi[:, FMT+12]==1))
    ver_ji = same_col & (~is_after) & ((xj[:, FMT+14]==1) | (xi[:, FMT+15]==1))
    pf[:, 26] = hor_ij | ver_ij | hor_ji | ver_ji

    # pf 27-28: different horizontal / vertical text alignment
    align_i = np.argmax(xi[:, FMT+16:FMT+23], axis=1)
    align_j = np.argmax(xj[:, FMT+16:FMT+23], axis=1)
    pf[:, 27] = align_i != align_j

    valign_i = np.argmax(xi[:, FMT+23:FMT+29], axis=1)
    valign_j = np.argmax(xj[:, FMT+23:FMT+29], axis=1)
    pf[:, 28] = valign_i != valign_j

    # pf 29: both cells are underlined
    pf[:, 29] = B(xi[:, FMT+11]) & B(xj[:, FMT+11])

    return edges, pf
