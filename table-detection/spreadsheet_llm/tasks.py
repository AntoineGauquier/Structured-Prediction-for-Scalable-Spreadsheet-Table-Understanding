import re
from typing import Any, Dict, List, Optional, Tuple

from .encoding import encode_sheet_compressor
from .llm import LLMClient, PROMPT_COMPRESSOR_DETECTION
from .spreadsheet_io import SpreadsheetData
from .utils import cell_address, parse_address


# Match A1 or A1:B5, tolerating optional single-quotes around cell refs and
# around the colon (e.g. 'A5':'H49' or 'A1':B5 or A1:'B5').
# A valid cell ref requires at least one letter then at least one digit, so
# bare numbers like '1' or invalid forms like '1:D45' are never matched.
_RANGE_RE = re.compile(r"\b([A-Za-z]+\d+)\b'?(?::'?([A-Za-z]+\d+)\b)?")

# Detects whether the response contained at least one quoted-range pattern.
_QUOTED_RANGE_RE = re.compile(
    r"'[A-Za-z]+\d+'?:'?[A-Za-z]+\d+'?", re.IGNORECASE
)


def parse_llm_ranges(text: str) -> List[Tuple[str, str]]:
    """Extract all cell-range references from LLM output.

    Returns (top_left, bottom_right) pairs, both uppercased.
    top == bottom for single-cell references (treated as single-cell tables).

    Handles:
      - Standard ranges:       A1:B5
      - Quoted-colon ranges:   'A1':'B5', 'A1':B5, A1:'B5'
      - Single cells:          A1, 'A1'  (each counts as one table)
    Rejects invalid forms such as '1':'D45' (no column letter → not matched).
    """
    out = []
    for m in _RANGE_RE.finditer(text):
        a = m.group(1).upper()
        b = (m.group(2) or m.group(1)).upper()
        out.append((a, b))
    return out


def _lift_range(
    a: str, b: str,
    row_map: Dict[int, int],
    col_map: Dict[int, int],
) -> Optional[str]:
    """Map a range from SheetCompressor compressed coords to original sheet coords."""
    pa = parse_address(a)
    pb = parse_address(b)
    if pa is None or pb is None:
        return None
    ra, ca = pa
    rb, cb = pb

    def lift_row(r: int) -> int:
        if r in row_map:
            return row_map[r]
        keys = list(row_map.keys())
        return row_map[min(keys, key=lambda k: abs(k - r))] if keys else r

    def lift_col(c: int) -> int:
        if c in col_map:
            return col_map[c]
        keys = list(col_map.keys())
        return col_map[min(keys, key=lambda k: abs(k - c))] if keys else c

    r0, c0 = min(lift_row(ra), lift_row(rb)), min(lift_col(ca), lift_col(cb))
    r1, c1 = max(lift_row(ra), lift_row(rb)), max(lift_col(ca), lift_col(cb))
    if (r0, c0) == (r1, c1):
        return cell_address(r0, c0)
    return f"{cell_address(r0, c0)}:{cell_address(r1, c1)}"


def detect_tables(
    data: SpreadsheetData,
    llm: LLMClient,
    *,
    k: int = 4,
    use_system_role: bool = True,
) -> Dict[str, Any]:
    """Detect tables in a spreadsheet using the fine-tuned local Mistral model.

    Always uses SheetCompressor with Modules 1+2 only (no Aggregation), which
    is the paper's best configuration ('GPT4-compress -w/o Aggregation', F1=78.9%).

    Returns predicted ranges in original (uncompressed) sheet coordinates.
    """
    encoded, row_map, col_map = encode_sheet_compressor(
        data,
        k=k,
        use_extraction=True,
        use_translation=True,
        use_aggregation=False,  # fixed: best model configuration
    )

    record = llm.complete(
        PROMPT_COMPRESSOR_DETECTION,
        encoded,
        use_system_role=use_system_role,
        label='detection',
    )

    raw_ranges = parse_llm_ranges(record.response)

    # True when the response had quoted-range artefacts like 'A5':'H49' that
    # required quote-stripping to reconstruct proper ranges.
    corners_merged = bool(_QUOTED_RANGE_RE.search(record.response))

    predicted_compressed = [
        a if a == b else f"{a}:{b}" for a, b in raw_ranges
    ]
    predicted_original = []
    for a, b in raw_ranges:
        lifted = _lift_range(a, b, row_map, col_map)
        if lifted is not None:
            predicted_original.append(lifted)

    # True when the model produced no usable prediction: either nothing was
    # parsed or every parsed result is a single cell (not a valid table range).
    no_valid_prediction = not any(':' in r for r in predicted_original)

    return {
        'predicted_ranges':            predicted_original,
        'predicted_ranges_compressed': predicted_compressed,
        'range_corners_merged':        corners_merged,
        'no_valid_prediction':         no_valid_prediction,
        'encoded_input':               encoded,
        'encoded_input_tokens':        llm.count_tokens(encoded),
        'llm_record':                  record.__dict__,
        'row_map':                     row_map,
        'col_map':                     col_map,
    }
