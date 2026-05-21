"""CSV/TSV file loader."""

import pandas as pd
from .base import SpreadsheetLoader

class CSVLoader(SpreadsheetLoader):
    """Loader for CSV and TSV files."""
    
    def load(self, path: str, sheet_name=None):
        sep = '\t' if path.lower().endswith('.tsv') else None
        engine = 'python' if sep is None else None
        df = None

        for enc in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252'):
            try:
                df = pd.read_csv(path, sep=sep, engine=engine, dtype=str, keep_default_na=False, encoding=enc, header=None)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None: 
            raise ValueError(f"Could not decode CSV/TSV with any known encoding (utf-8-sig, utf-8, latin-1, cp1252): {path}")
        
        # CSV files don't have merged cells or formatting
        return df, None, None, {}, {}