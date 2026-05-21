""" Color conversion utilities + font name hashing."""

import hashlib
import numpy as np

def stable_hash(s):
    """Encode font name into a single numerical value in [0, 1]."""
    if not s:
        return 0.0
    h = hashlib.md5(s.encode("utf8")).hexdigest()
    return (int(h, 16) % 1_000_000) / 1_000_000

def rgb_normalized(rgb_tuple):
    """Normalize RGB tuple from (0-255, 0-255, 0-255) to (0-1, 0-1, 0-1)."""
    if rgb_tuple is None:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([rgb_tuple[0]/255.0, rgb_tuple[1]/255.0, rgb_tuple[2]/255.0], dtype=np.float32)


def ods_color_to_rgb(color_value):
    """Convert ODS color value to RGB tuple.""" 
    if not color_value:
        return None
    
    color_value = color_value.strip().lower()
    
    if color_value == "transparent":
        return None
    
    if color_value.startswith("#"):
        if len(color_value) == 7:  # #RRGGBB
            try:
                r = int(color_value[1:3], 16)
                g = int(color_value[3:5], 16)
                b = int(color_value[5:7], 16)
                return (r, g, b)
            except ValueError:
                return None
        elif len(color_value) == 4:  # #RGB 
            try:
                r = int(color_value[1] * 2, 16)
                g = int(color_value[2] * 2, 16)
                b = int(color_value[3] * 2, 16)
                return (r, g, b)
            except ValueError:
                return None
    
    named_colors = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
    }
    
    return named_colors.get(color_value, None)


def xls_color_to_rgb(book, colour_index):
    """Convert XLS color index to RGB tuple."""
    if colour_index is None:
        return None
    
    if colour_index == 0x7FFF or colour_index == 0x40 or colour_index == 64 or colour_index == 65:
        return None

    try:
        if hasattr(book, 'colour_map') and colour_index in book.colour_map:
            rgb = book.colour_map[colour_index]
            if rgb is not None:
                return rgb
    except:
        pass
    
    default_palette = {
        0: (0, 0, 0),
        1: (255, 255, 255),
        2: (255, 0, 0),
        3: (0, 255, 0),
        4: (0, 0, 255),
        5: (255, 255, 0),
        6: (255, 0, 255),
        7: (0, 255, 255),
        8: (0, 0, 0),
        9: (255, 255, 255),
    }
    
    return default_palette.get(colour_index, None)

def xlsx_theme_color_to_rgb(wb, color):
    """Convert XLSX theme color to RGB tuple."""
    if color is None:
        return None
    
    if hasattr(color, 'rgb') and color.rgb:
        rgb_value = color.rgb
        
        if hasattr(rgb_value, '__iter__') and not isinstance(rgb_value, str):
            try:
                if len(rgb_value) == 3:
                    return tuple(rgb_value)
                elif len(rgb_value) == 4: # RGBA
                    return tuple(rgb_value[:3])
            except:
                pass
        
        if isinstance(rgb_value, str):
            rgb_str = rgb_value
            # RGB string is in format 'AARRGGBB' or 'RRGGBB'
            if len(rgb_str) == 8:   # ARGB format
                rgb_str = rgb_str[2:]
            elif len(rgb_str) != 6:
                return None
            
            try:
                r = int(rgb_str[0:2], 16)
                g = int(rgb_str[2:4], 16)
                b = int(rgb_str[4:6], 16)
                return (r, g, b)
            except ValueError:
                return None
    
    # May need to look up in workbook theme
    if hasattr(color, 'theme') and color.theme is not None:
        theme_colors = {
            0: (255, 255, 255),
            1: (0, 0, 0),
            2: (238, 236, 225),
            3: (31, 73, 125),
            4: (79, 129, 189),
            5: (192, 80, 77),
            6: (155, 187, 89),
            7: (128, 100, 162),
            8: (75, 172, 198),
            9: (247, 150, 70),
        } # Default palette for common themes
        
        base_color = theme_colors.get(color.theme, (0, 0, 0))
        
        if hasattr(color, 'tint') and color.tint:
            # if tint < 0, darken; if tint > 0, lighten
            tint = float(color.tint)
            if tint < 0:
                r = int(base_color[0] * (1 + tint))
                g = int(base_color[1] * (1 + tint))
                b = int(base_color[2] * (1 + tint))
            else:
                r = int(base_color[0] * (1 - tint) + 255 * tint)
                g = int(base_color[1] * (1 - tint) + 255 * tint)
                b = int(base_color[2] * (1 - tint) + 255 * tint)
            return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
        
        return base_color
    
    if hasattr(color, 'indexed') and color.indexed is not None:
        indexed_colors = {
            0: (0, 0, 0),
            1: (255, 255, 255),
            2: (255, 0, 0),
            3: (0, 255, 0),
            4: (0, 0, 255),
            5: (255, 255, 0),
            6: (255, 0, 255),
            7: (0, 255, 255),
            64: (0, 0, 0),
            65: (255, 255, 255),
        } # Similary to XLS with default palette
        return indexed_colors.get(color.indexed, None)
    
    if hasattr(color, 'type'):
        color_type = color.type
        if color_type == 'rgb' and hasattr(color, 'value'):
            try:
                value = str(color.value)
                if len(value) == 8:
                    value = value[2:]
                if len(value) == 6:
                    r = int(value[0:2], 16)
                    g = int(value[2:4], 16)
                    b = int(value[4:6], 16)
                    return (r, g, b)
            except:
                pass
    
    return None