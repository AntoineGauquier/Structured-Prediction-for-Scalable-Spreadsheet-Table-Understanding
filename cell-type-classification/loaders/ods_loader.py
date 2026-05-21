"""ODS file loader."""

import pandas as pd
import numpy as np
from time import time
import odf
from odf.opendocument import load
from odf.namespaces import TABLENS
from odf.teletype import extractText
from odf.table import Table, TableRow, TableCell

from .base import SpreadsheetLoader
from ..utils.compression import is_empty_value

def safe_cell_text(cell):
    """Extract text from ODS cell safely."""
    return extractText(cell)


def rectangles_to_merged_map(merged_rects):
    """Convert list of merged rectangles (ranges) to cell mapping."""
    merged_map = {}

    for rect in merged_rects:
        start_row = rect['start_row']
        start_col = rect['start_col']
        end_row = rect['end_row']
        end_col = rect['end_col']

        for i in range(start_row, end_row + 1):
            for j in range(start_col, end_col + 1):
                merged_map[(i, j)] = (start_row, start_col)

    return merged_map


def build_merged_cell_map_ods(doc, sheet_index=0):
    """Build map of merged cells for ODS sheet."""
    sheets = doc.spreadsheet.getElementsByType(odf.table.Table)
    target_sheet = sheets[sheet_index]
    
    merged_cells = []
    rows = target_sheet.getElementsByType(odf.table.TableRow)
    current_row = 0
    
    for row in rows:
        repeat = row.getAttribute('numberrowsrepeated')
        row_repeat = int(repeat) if repeat else 1
        current_col = 0
        
        for child in row.childNodes:
            is_cell = (hasattr(child, "qname") and child.qname == (TABLENS, "table-cell"))
            is_covered = (hasattr(child, "qname") and child.qname == (TABLENS, "covered-table-cell"))
            
            if is_cell or is_covered:
                repeat = child.getAttribute('numbercolumnsrepeated')
                cell_repeat = int(repeat) if repeat else 1
                
                if is_cell:
                    cols_spanned = child.getAttribute('numbercolumnsspanned')
                    rows_spanned = child.getAttribute('numberrowsspanned')
                    
                    if cols_spanned or rows_spanned:
                        cols = int(cols_spanned) if cols_spanned else 1
                        rows = int(rows_spanned) if rows_spanned else 1
                        
                        merged_cells.append({
                            'start_row': current_row,
                            'start_col': current_col,
                            'end_row': current_row + rows - 1,
                            'end_col': current_col + cols - 1
                        })
                
                current_col += cell_repeat
        
        current_row += row_repeat
    return rectangles_to_merged_map(merged_cells)


def build_props_map(doc):
    """Build properties map from ODS styles."""
    from odf.style import Style
    from odf.namespaces import FONS, STYLENS, TEXTNS, TABLENS
    
    ns_map = {
        FONS: 'fo',
        STYLENS: 'style',
        TEXTNS: 'text',
        TABLENS: 'table',
    }
    
    raw_styles = {}
    parent_map = {}
    
    for style_container in [doc.automaticstyles, doc.styles]:
        if style_container is None:
            continue
            
        for style in style_container.getElementsByType(Style):
            style_name = style.getAttribute("name")
            if not style_name:
                continue
            
            # Store parent relationship
            parent_name = style.getAttribute("parentstylename")
            if parent_name:
                parent_map[style_name] = parent_name
            
            # Extract properties
            props = {}
            for prop_element in style.childNodes:
                if not hasattr(prop_element, 'attributes'):
                    continue
                    
                for (ns_url, local_name), value in prop_element.attributes.items():
                    prefix = ns_map.get(ns_url, '')
                    key = f"{prefix}:{local_name}" if prefix else local_name
                    props[key] = value
            
            raw_styles[style_name] = props
    
    def resolve_style(style_name, visited=None):
        if visited is None:
            visited = set()
            
        if style_name in visited:
            return {}
            
        visited.add(style_name)
            
        if style_name not in raw_styles:
            return {}
            
        result = {}
        if style_name in parent_map:
            parent_props = resolve_style(parent_map[style_name], visited)
            result.update(parent_props)

        # Override with own properties 
        result.update(raw_styles[style_name])
        return result

    # Build final props_map with resolved inheritance
    props_map = {}
    for style_name in raw_styles:
        props_map[style_name] = resolve_style(style_name)
    return props_map


class ODSLoader(SpreadsheetLoader):
    """Loader for ODS files."""
    
    MAX_CONSECUTIVE_EMPTY_ROWS = 200
    
    def load(self, path, sheet_name=None):
        st_load = time()
        doc = load(path)
        
        sheets = doc.spreadsheet.getElementsByType(odf.table.Table)
        sheet_idx = self._resolve_sheet_index(sheets, sheet_name)
        sheet = sheets[sheet_idx]
        
        #print(f"Doc and sheet load alone: {time() - st_load}s.")
        
        merged_map = build_merged_cell_map_ods(doc, sheet_idx)
        rows = sheet.getElementsByType(odf.table.TableRow)
        
        cell_map = {}
        current_row = 0
        data = []
        content_cache = {}
        consecutive_empty = 0
        max_col_with_content = 0

        for physical_row_idx, row in enumerate(rows):
            repeat = row.getAttribute('numberrowsrepeated')
            row_repeat = int(repeat) if repeat else 1
            
            current_col = 0
            row_data = []
            col_to_child = {}
            has_content = False
            
            for child in row.childNodes:
                has_attr_qname = hasattr(child, "qname")
                is_cell = (has_attr_qname and child.qname == (TABLENS, "table-cell"))
                is_covered = (has_attr_qname and child.qname == (TABLENS, "covered-table-cell"))
                
                if is_cell or is_covered:
                    repeat = child.getAttribute('numbercolumnsrepeated')
                    cell_repeat = int(repeat) if repeat else 1
                    
                    if is_cell:
                        cols_spanned = int(child.getAttribute('numbercolumnsspanned') or 1)
                        rows_spanned = int(child.getAttribute('numberrowsspanned') or 1)
                        anchor = (current_row, current_col)

                        if anchor not in content_cache:
                            content_cache[anchor] = safe_cell_text(child)
                        
                        if content_cache[anchor] is not None and content_cache[anchor] != "":
                            has_content = True
                            max_col_with_content = max(max_col_with_content, current_col + cell_repeat - 1)
                        
                        # 1. Column repetition (cell_repeat) - the cell appears in multiple consecutive columns
                        # 2. Row/column spanning (for merged cells)
                        for col_offset in range(cell_repeat):
                            for dr in range(rows_spanned):
                                for dc in range(cols_spanned):
                                    target_row = current_row + dr
                                    target_col = current_col + col_offset + dc
                                    content_cache[(target_row, target_col)] = content_cache[anchor]
                    
                    for offset in range(cell_repeat):
                        col_to_child[current_col] = child
                        row_data.append(content_cache.get((current_row, current_col), None))
                        current_col += 1

            if not has_content:
                consecutive_empty += row_repeat
                if consecutive_empty >= self.MAX_CONSECUTIVE_EMPTY_ROWS:
                    break
            else:
                consecutive_empty = 0
            
            for repeat_idx in range(row_repeat):
                for col_idx, child_node in col_to_child.items():
                    cell_map[(current_row, col_idx)] = (physical_row_idx, child_node)
                
                data.append(list(row_data))
                current_row += 1

        df = pd.DataFrame(data)
        
        if max_col_with_content > 0 and max_col_with_content + 3 < df.shape[1]:
            df = df.iloc[:, :max_col_with_content + 3]
            #print(f"Trimmed columns from {len(data[0]) if data else 0} to {df.shape[1]}")
        
        return df, sheet, doc, merged_map, cell_map