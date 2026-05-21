from .ods_loader import ODSLoader
from .xls_loader import XLSLoader
from .xlsx_loader import XLSXLoader
from .csv_loader import CSVLoader
import numpy as np

def load_spreadsheet(path, file_format, sheet_name=None):
    """Load a spreadsheet file automatically detecting the format."""
    path_lower = path.lower()
    
    if file_format == "ods":
        loader = ODSLoader()
    elif file_format == "xls":
        loader = XLSLoader()
    elif file_format == "xlsx":
        loader = XLSXLoader()
    elif file_format == "csv" or file_format == "tsv":
        loader = CSVLoader()
    else:
        raise ValueError(f"Unsupported file format: {path}")

    fmt = file_format
    df, sheet, workbook, merged_map, cell_map = loader.load(path, sheet_name)

    df = df.replace(r'^\s*$', np.nan, regex=True)

    if not df.empty:
        mask = df.notna()

        # Find last row containing a value
        valid_rows = mask.any(axis=1)
        if valid_rows.any():
            last_row = valid_rows[::-1].idxmax()
            df = df.loc[:last_row]

        # Find last column containing a value
        valid_cols = mask.any(axis=0)
        if valid_cols.any():
            last_col = valid_cols[::-1].idxmax()
            df = df.loc[:, :last_col]

    return df, sheet, workbook, merged_map, cell_map, fmt

__all__ = ['load_spreadsheet', 'ODSLoader', 'XLSLoader', 'XLSXLoader', 'CSVLoader']