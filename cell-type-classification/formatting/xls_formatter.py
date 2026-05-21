"""XLS formatting feature extraction."""

import numpy as np
from ..utils.color_utils import xls_color_to_rgb, rgb_normalized, stable_hash
from .ods_formatter import create_default_features # Same default for all formats


def get_xls_default_font(book):
    """Get default font (size and name) from XLS workbook."""
    try:
        if len(book.xf_list) > 0:
            default_xf = book.xf_list[0]
            if default_xf.font_index < len(book.font_list):
                font = book.font_list[default_xf.font_index]
                font_size = font.height / 20.0  # Convert twips to points (1 point = 20 twips, by def): see https://xlrd.readthedocs.io/en/latest/api.html#xlrd.formatting.Font.height
                font_name = font.name
                return font_size, font_name
    except:
        pass
    return 10.0, "Arial"  # Excel defaults

def extract_xls_format_features(book, sheet, df_row, df_col, df_to_logical_row, df_to_logical_col, cell_map, merged_map):
    default_font_size, default_font_name = get_xls_default_font(book)
    features = create_default_features(default_font_size, default_font_name)
    
    logical_row = df_to_logical_row.get(df_row)
    logical_col = df_to_logical_col.get(df_col)
    
    if logical_row is None or logical_col is None:
        return features
    
    anchor_row, anchor_col = merged_map.get((logical_row, logical_col), (logical_row, logical_col))
    physical_row, physical_col = cell_map.get((anchor_row, anchor_col), (None, None))
    
    if physical_row is None or physical_col is None:
        return features
    
    if physical_row >= sheet.nrows or physical_col >= sheet.ncols:
        return features
    
    try:
        xf_index = sheet.cell_xf_index(physical_row, physical_col)
        
        if xf_index >= len(book.xf_list):
            return features
        
        xf = book.xf_list[xf_index]
        
        if xf.font_index < len(book.font_list):
            font = book.font_list[xf.font_index]
            
            features[0] = 1.0 if font.weight >= 700 else 0.0
            features[1] = 1.0 if font.italic else 0.0
            features[2] = 1.0 if font.underline_type != 0 else 0.0
            
            features[3] = font.height / 20.0 # convert from twips to points
            features[4] = stable_hash(font.name)
            
            font_rgb = xls_color_to_rgb(book, font.colour_index)
            if font_rgb:
                features[5:8] = rgb_normalized(font_rgb)
        
        # XF has a background attribute with fill_pattern and colour indexes
        if hasattr(xf, 'background'):
            bg = xf.background
            if bg.fill_pattern != 0: # 0=no fill
                bg_rgb = xls_color_to_rgb(book, bg.pattern_colour_index)
                if bg_rgb:
                    features[8:11] = rgb_normalized(bg_rgb)
        
        features[11] = 1.0 if (logical_row, logical_col) in merged_map else 0.0
        
        if hasattr(xf, 'border'):
            border = xf.border
            # 0=no line
            features[12] = 1.0 if border.left_line_style != 0 else 0.0
            features[13] = 1.0 if border.right_line_style != 0 else 0.0
            features[14] = 1.0 if border.top_line_style != 0 else 0.0
            features[15] = 1.0 if border.bottom_line_style != 0 else 0.0
        
        features[16:23] = 0
        if hasattr(xf, 'alignment'):
            align = xf.alignment
            hor_align = align.hor_align
            
            if hor_align == 0 or hor_align == 1:  # GENERAL or LEFT
                features[16] = 1 
            elif hor_align == 3: 
                features[17] = 1
            elif hor_align == 2 or hor_align == 6:  # CENTER or CENTER_ACROSS
                features[18] = 1 
            elif hor_align == 5:
                features[19] = 1 
            elif hor_align == 4: 
                features[20] = 1
            elif hor_align == 7: 
                features[21] = 1
            else:
                features[22] = 1  # other/unknown
        else:
            features[16] = 1  # Default: start
        
        features[23:29] = 0
        if hasattr(xf, 'alignment'):
            align = xf.alignment
            vert_align = align.vert_align
            
            if vert_align == 0:
                features[23] = 1
            elif vert_align == 1:
                features[24] = 1
            elif vert_align == 2:
                features[25] = 1
            elif vert_align == 3 or vert_align == 4:
                features[26] = 1
            else:
                features[26] = 1 
        else:
            features[26] = 1
        
    except (IndexError, AttributeError, KeyError) as e:
        print(f"Error extracting XLS format at ({df_row}, {df_col}): {e}")
        pass
    return features