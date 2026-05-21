"""XLS file loader."""

import xlrd
import pandas as pd
from .base import SpreadsheetLoader
from ..utils.compression import is_empty_value

def build_merged_cell_map_xls(sheet):
    """Build map of merged cells in XLS sheet."""
    merged_map = {}
    
    for rlo, rhi, clo, chi in sheet.merged_cells:
        for row in range(rlo, rhi):
            for col in range(clo, chi):
                merged_map[(row, col)] = (rlo, clo)
    
    return merged_map


def get_xls_cell_value(sheet, row, col):
    """Get value from XLS cell with appropriate type conversion."""
    cell = sheet.cell(row, col)

    if cell.ctype == xlrd.XL_CELL_EMPTY or cell.ctype == xlrd.XL_CELL_BLANK:
        return ""
    elif cell.ctype == xlrd.XL_CELL_TEXT:
        return cell.value
    elif cell.ctype == xlrd.XL_CELL_NUMBER:
        val = cell.value
        if val == int(val): # Differentiate int from float (important for later date detection)
            return int(val)
        else:
            return val
    elif cell.ctype == xlrd.XL_CELL_DATE:
        return cell.value
    elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return bool(cell.value)
    elif cell.ctype == xlrd.XL_CELL_ERROR:
        return ""
    else:
        return cell.value
    
class XLSLoader(SpreadsheetLoader):
    """Loader for XLS files."""

    MAX_CONSECUTIVE_EMPTY_ROWS = 200

    def load(self, path, sheet_name=None):
        try:
            book = xlrd.open_workbook(path, formatting_info=True)
        except Exception as e:
            print(f"Warning: Could not open with formatting_info=True: {e}")
            book = xlrd.open_workbook(path, formatting_info=False)

        if sheet_name == "Sheet1":
            try:
                sheet_idx = self._resolve_sheet_index(book.sheets(), sheet_name)
            except:
                sheet_idx = self._resolve_sheet_index(book.sheets(), "Sheet 1")
        else:
            sheet_idx = self._resolve_sheet_index(book.sheets(), sheet_name)
        
        sheet = book.sheet_by_index(sheet_idx)
        
        merged_map = build_merged_cell_map_xls(sheet)
        cell_map = {}
        data = []
        consecutive_empty = 0
        
        nrows = sheet.nrows
        ncols = sheet.ncols
        
        for row_idx in range(nrows):
            row_data = []
            has_content = False
            
            for col_idx in range(ncols):
                if (row_idx, col_idx) in merged_map:
                    anchor_row, anchor_col = merged_map[(row_idx, col_idx)]
                    cell_value = get_xls_cell_value(sheet, anchor_row, anchor_col)
                else:
                    cell_value = get_xls_cell_value(sheet, row_idx, col_idx)
                
                if not is_empty_value(cell_value):
                    has_content = True
                
                row_data.append(cell_value)
                cell_map[(row_idx, col_idx)] = (row_idx, col_idx)
            
            if not has_content:
                consecutive_empty += 1
                if consecutive_empty >= self.MAX_CONSECUTIVE_EMPTY_ROWS:
                    break
            else:
                consecutive_empty = 0
            
            data.append(row_data)
        
        df = pd.DataFrame(data)
        return df, sheet, book, merged_map, cell_map