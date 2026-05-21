"""XLSX formatting feature extraction."""

import numpy as np
from ..utils.color_utils import xlsx_theme_color_to_rgb, rgb_normalized, stable_hash
from .ods_formatter import create_default_features

def get_xlsx_default_font(wb):
    """Get default font from XLSX workbook."""
    try:
        default_font = wb.style_names.get('Normal')
        if default_font:
            if hasattr(default_font, 'font'):
                font = default_font.font
                size = font.size if font.size else 11.0
                name = font.name if font.name else "Calibri"
                return size, name
    except:
        pass
    return 11.0, "Calibri" # Excel defaults

def extract_xlsx_format_features(wb, sheet, df_row, df_col, df_to_logical_row, df_to_logical_col, cell_map, merged_map):
    default_font_size, default_font_name = get_xlsx_default_font(wb)
    features = create_default_features(default_font_size, default_font_name)
    
    logical_row = df_to_logical_row.get(df_row)
    logical_col = df_to_logical_col.get(df_col)
    
    if logical_row is None or logical_col is None:
        return features
    
    anchor_row, anchor_col = merged_map.get((logical_row, logical_col), (logical_row, logical_col))
    physical_row, physical_col = cell_map.get((anchor_row, anchor_col), (None, None))
    
    if physical_row is None or physical_col is None:
        return features
    
    try:
        cell = sheet.cell(physical_row + 1, physical_col + 1)

        if cell.font:
            font = cell.font
            
            features[0] = 1.0 if font.bold else 0.0
            features[1] = 1.0 if font.italic else 0.0
            features[2] = 1.0 if font.underline else 0.0
            
            if font.size:
                features[3] = float(font.size) # in points already
            
            if font.name:
                features[4] = stable_hash(font.name)

            if font.color:
                font_rgb = xlsx_theme_color_to_rgb(wb, font.color)
                if font_rgb:
                    features[5:8] = rgb_normalized(font_rgb)
        
        if cell.fill:
            fill = cell.fill
            # PatternFill has fgColor (foreground) and bgColor (background)
            if fill.patternType and fill.patternType != 'none':
                if fill.fgColor:
                    bg_rgb = xlsx_theme_color_to_rgb(wb, fill.fgColor)
                    if bg_rgb:
                        features[8:11] = rgb_normalized(bg_rgb)
        
        features[11] = 1.0 if (logical_row, logical_col) in merged_map else 0.0
        
        if cell.border:
            border = cell.border
            
            features[12] = 1.0 if (border.left and border.left.style) else 0.0
            features[13] = 1.0 if (border.right and border.right.style) else 0.0
            features[14] = 1.0 if (border.top and border.top.style) else 0.0
            features[15] = 1.0 if (border.bottom and border.bottom.style) else 0.0
        
        features[16:23] = 0
        if cell.alignment and cell.alignment.horizontal:
            # openpyxl horizontal alignment values:
            # 'general', 'left', 'center', 'right', 'fill', 'justify', 
            # 'centerContinuous', 'distributed'
            
            hor_align = cell.alignment.horizontal
            
            if hor_align == 'general':
                features[16] = 1
            elif hor_align == 'right':
                features[17] = 1
            elif hor_align in ('center', 'centerContinuous'):
                features[18] = 1
            elif hor_align == 'justify':
                features[19] = 1
            elif hor_align == 'left':
                features[20] = 1
            elif hor_align == 'fill':
                features[21] = 1
            elif hor_align == 'distributed':
                features[22] = 1  
            else:
                features[22] = 1  # unknown
        else:
            features[16] = 1  # Default: general/start
        
        features[23:29] = 0
        if cell.alignment and cell.alignment.vertical:
            # openpyxl vertical alignment values:
            # 'top', 'center', 'bottom', 'justify', 'distributed'
            
            vert_align = cell.alignment.vertical
            
            if vert_align == 'top':
                features[23] = 1
            elif vert_align == 'center':
                features[24] = 1
            elif vert_align == 'bottom':
                features[25] = 1
            elif vert_align in ('justify', 'distributed'):
                features[26] = 1 
            else:
                features[26] = 1  # other/unknown
        else:
            features[26] = 1  # Default: automatic/bottom
        
    except (IndexError, AttributeError, KeyError) as e:
        pass
    
    return features