"""Address utilities for converting between (row, col) tuples and A1 notation."""
import re
from typing import Optional, Tuple


def col_index_to_letter(idx: int) -> str:
    """0-indexed column to letter: 0 -> 'A', 25 -> 'Z', 26 -> 'AA', 701 -> 'ZZ'."""
    if idx < 0:
        raise ValueError(f"Negative column index: {idx}")
    letters = ""
    n = idx + 1  # work in 1-indexed
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def letter_to_col_index(letters: str) -> int:
    """Letter to 0-indexed column: 'A' -> 0, 'AA' -> 26."""
    n = 0
    for ch in letters.upper():
        if not ('A' <= ch <= 'Z'):
            raise ValueError(f"Invalid column letter: {letters!r}")
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def cell_address(row: int, col: int) -> str:
    """0-indexed (row, col) -> A1 notation. cell_address(0, 0) == 'A1'."""
    return f"{col_index_to_letter(col)}{row + 1}"


_ADDR_RE = re.compile(r'^([A-Z]+)(\d+)$', re.IGNORECASE)


def parse_address(addr: str) -> Optional[Tuple[int, int]]:
    """A1-notation -> 0-indexed (row, col). Returns None if malformed."""
    m = _ADDR_RE.match(addr.strip())
    if not m:
        return None
    col_str, row_str = m.group(1), m.group(2)
    try:
        return int(row_str) - 1, letter_to_col_index(col_str)
    except ValueError:
        return None


def parse_range(rng: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """'A1:B5' -> ((0,0),(4,1)). 'A1' -> ((0,0),(0,0))."""
    parts = rng.replace(' ', '').split(':')
    if len(parts) == 1:
        a = parse_address(parts[0])
        return (a, a) if a is not None else None
    if len(parts) == 2:
        a = parse_address(parts[0])
        b = parse_address(parts[1])
        if a is None or b is None:
            return None
        # Normalize order
        r0, c0 = a; r1, c1 = b
        return ((min(r0, r1), min(c0, c1)), (max(r0, r1), max(c0, c1)))
    return None


def format_range(top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> str:
    """((r0,c0),(r1,c1)) -> 'A1:B5' (or 'A1' when single cell)."""
    if top_left == bottom_right:
        return cell_address(*top_left)
    return f"{cell_address(*top_left)}:{cell_address(*bottom_right)}"
