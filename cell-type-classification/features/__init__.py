from .unary_features import extract_unary_features
from .pairwise_features import compute_pairwise_features

def extract_features(path, file_format='auto', sheet_name=None):
    """
    Extract both unary and pairwise features from a spreadsheet.
    
    Args:
        path: Path to spreadsheet file
        file_format: File format ('ods', 'xls', 'xlsx', 'csv', or 'auto')
        sheet_name: Sheet name or index to load
        
    Returns:
        unary_features: (H, W, F_u) array of unary features
        edges: (E, 2) array of edge indices
        pairwise_features: (E, F_p) array of pairwise features
    """
    unary_features, row_mapping, col_mapping, original_shape = extract_unary_features(path, file_format, sheet_name)
    edges, pairwise_features = compute_pairwise_features(unary_features)
    
    return unary_features, edges, pairwise_features, row_mapping, col_mapping, original_shape

__all__ = ['extract_features', 'extract_unary_features', 'compute_pairwise_features']