"""Text processing utilities."""

import numpy as np
from dateutil.parser import parse

WHITES = {
    "\u00A0",  # NBSP
    "\u202F",  # narrow NBSP
    "\u2009",  # thin space
}

def is_date(s):
    """Check if string can be parsed as a date."""
    try:
        parse(str(s), fuzzy=False)
        return True
    except:
        return False

def text_stats(s):
    """Extract text statistics from a string."""
    s = str(s)
    return (
        len(s), 
        sum(c.isdigit() for c in s),
        sum(c.isalpha() for c in s),
        sum(c.isspace() for c in s),
        sum(not c.isalnum() and not c.isspace() for c in s)
    )


def text_starts_with(s):
    """Check whether the first character is a letter and is a digit."""
    s = str(s)
    if len(s) == 0:
        return False, False
    return (s[0].isalpha(), s[0].isnumeric())


def get_cell_type(x):
    """Assesses cell type from value.
    Returns: (is_number, is_int, is_float, is_string, is_date, is_formula)
    """

    if is_date(x):
        return False, False, False, False, True, False

    if isinstance(x, str) and x.strip().startswith("="):
        return False, False, False, False, False, True
    
    if isinstance(x, bool):
        return False, False, False, True, False, False 
    
    if isinstance(x, int):
        return True, True, False, False, False, False
    
    if isinstance(x, float):
        return True, False, True, False, False, False
    
    if isinstance(x, str):
        x_stripped = x.strip()
        
        if x_stripped == "":
            return False, False, False, False, False, False
        
        # Remove common thousands separators (space, comma) but keep decimal separators (. and ,)
        x_normalized = x_stripped.replace(" ", "")
        for c in WHITES:
            x_normalized = x_normalized.replace(c, "")
        
        # Try parsing as int (no decimal point)
        if "." not in x_normalized and "," not in x_normalized:
            try:
                _ = int(x_normalized)
                return True, True, False, False, False, False
            except ValueError:
                pass
        
        # Try parsing as float: handle both . and , as decimal separators
        # First try with . as decimal 
        x_for_float = x_normalized.replace(",", "") # Remove comma (thousands separator in English)
        try:
            _ = float(x_for_float)
            return True, False, True, False, False, False
        except ValueError:
            pass
        
        # Then try with , as decimal separator
        x_for_float_eu = x_normalized.replace(".", "").replace(",", ".") # Remove . (thousands), replace , with .
        try:
            _ = float(x_for_float_eu)
            return True, False, True, False, False, False
        except ValueError:
            pass
        
        return False, False, False, True, False, False
    return False, False, False, True, False, False