"""ODS formatting feature extraction."""

import numpy as np
from odf.text import P, Span
from odf.style import Style, TextProperties
from odf.namespaces import FONS, STYLENS, OFFICENS, TABLENS

from ..utils.color_utils import ods_color_to_rgb, rgb_normalized, stable_hash

def parse_font_size(size_str):
    """Parse font size string to points."""
    if not size_str:
        return 0.0
    
    size_str = size_str.strip().lower()
    
    try:
        if size_str.endswith("pt"):
            return float(size_str[:-2])
        elif size_str.endswith("px"): # Convert pixels to points (96 DPI standard: 1pt = 96/72 px)
            return float(size_str[:-2]) * 0.75 
        elif size_str.endswith("em"): # 1em = 12pt (default font size)
            return float(size_str[:-2]) * 12.0
        elif size_str.endswith("%"): # 100% = 12pt (default font size)
            return float(size_str[:-1]) * 0.12
        else:
            return float(size_str) #  Direct numerical value
    except ValueError:
        return 0.0


def get_default_font_size(props_map):
    """Get default font size from document."""
    if "Default" in props_map:
        font_size_str = props_map["Default"].get("fo:font-size", "")
        if font_size_str:
            return parse_font_size(font_size_str)
    
    for style_name in ["Standard", "Table Contents", "Default"]:
        if style_name in props_map:
            font_size_str = props_map[style_name].get("fo:font-size", "")
            if font_size_str:
                return parse_font_size(font_size_str)
    
    return 10.0  # LibreOffice Calc default


def get_default_font_name(props_map):
    """Get default font name from document."""
    for style_name in ["Default", "Standard", "Table Contents"]:
        if style_name in props_map:
            font_name = props_map[style_name].get("style:font-name", "")
            if font_name:
                return font_name
    
    return "Liberation Sans"  # LibreOffice Calc default


def create_default_features(default_font_size, default_font_name):
    """Create default formatting features vector."""
    features = np.zeros(29, dtype=np.float32)
    
    features[0] = 0  # Not bold
    features[1] = 0  # Not italic
    features[2] = 0  # Not underlined
    features[3] = default_font_size
    features[4] = stable_hash(default_font_name)
    features[5:8] = rgb_normalized((0, 0, 0))  # Text color: default black
    features[8:11] = -1  # Background: use -1 for no background
    features[11] = 0
    features[12:16] = 0
    features[16] = 1  # start alignment
    features[17:23] = 0
    features[23:26] = 0
    features[26] = 1  # automatic vertical alignment
    features[27:29] = 0
    
    return features


def check_text_formatting_in_cell(cell, props_map, property_name, property_value, doc=None):
    """Check if any text within cell has specific formatting property (used when a formatting info is not found at cell level)."""
    paragraphs = cell.getElementsByType(P)
    
    for para in paragraphs: # Check paragraph style
        try:
            para_style_name = para.getAttribute("stylename")
        except (ValueError, KeyError):
            para_style_name = None
            
        if para_style_name:
            para_props = props_map.get(para_style_name)
            if not para_props and doc:
                para_props = extract_text_properties_from_style(doc, para_style_name)
            
            if para_props:
                para_property = para_props.get(property_name, "")
                if para_property == property_value:
                    return True
        
        spans = para.getElementsByType(Span) # Check all span elements in the paragraph
        for span in spans:
            try:
                span_style_name = span.getAttribute("stylename")
            except (ValueError, KeyError):
                span_style_name = None
                
            if span_style_name:
                span_props = props_map.get(span_style_name)
                if not span_props and doc:
                    span_props = extract_text_properties_from_style(doc, span_style_name)
                
                if span_props:
                    span_property = span_props.get(property_name, "")
                    if span_property == property_value:
                        return True
    return False


def get_ods_cell_value_type(cell):
    """Get cell value type from ODS cell."""
    try:
        value_type = cell.getAttribute("valuetype")
        if value_type:
            return value_type
    except (ValueError, KeyError):
        pass
    
    try: # Try with namespace using getAttrNS
        value_type = cell.getAttrNS(OFFICENS, "value-type")
        if value_type:
            return value_type
    except (ValueError, KeyError, AttributeError):
        pass
    return None


def infer_alignment_from_value_type(cell, align_source):
    """Infer horizontal alignment from cell value type."""
    if align_source != "value-type":
        return ""
    
    value_type = get_ods_cell_value_type(cell)
    
    if value_type in ['float', 'currency', 'percentage', 'time', 'date']:
        return "end"
    elif value_type in ['string', 'boolean']:
        return "start"
    else:
        has_value = False
        try:
            if cell.getAttribute("value"):
                has_value = True
        except (ValueError, KeyError):
            pass
        
        if not has_value:
            try:
                if (cell.getAttrNS(OFFICENS, "value") or 
                    cell.getAttrNS(OFFICENS, "date-value") or
                    cell.getAttrNS(OFFICENS, "time-value")):
                    has_value = True
            except (ValueError, KeyError, AttributeError):
                pass
        
        return "end" if has_value else "start"


def extract_text_properties_from_style(doc, style_name):
    """Extract text properties from a style definition."""
    auto_styles = doc.automaticstyles
    for style in auto_styles.getElementsByType(Style):
        try:
            name = style.getAttribute("name")
            if name == style_name:
                props = {}
                text_props = style.getElementsByType(TextProperties)
                if text_props:
                    tp = text_props[0]
                    for attr_name in ['font-weight', 'font-style', 'text-underline-style']:
                        try:
                            if attr_name in ['font-weight', 'font-style']:
                                val = tp.getAttrNS(FONS, attr_name)
                                if val:
                                    props[f'fo:{attr_name}'] = val
                            elif attr_name == 'text-underline-style':
                                val = tp.getAttrNS(STYLENS, attr_name)
                                if val:
                                    props[f'style:{attr_name}'] = val
                        except (ValueError, KeyError, AttributeError):
                            pass
                return props
        except (ValueError, KeyError):
            continue 
    return {}


def extract_ods_format_features(doc, df_row, df_col, df_to_logical_row, df_to_logical_col, cell_map, props_map, merged_map):
    """Extract formatting features from ODS cell."""
    default_font_size = get_default_font_size(props_map)
    default_font_name = get_default_font_name(props_map)

    features = create_default_features(default_font_size, default_font_name)

    logical_row = df_to_logical_row.get(df_row)
    logical_col = df_to_logical_col.get(df_col)

    if logical_row is None or logical_col is None:
        return features
    
    anchor_row, anchor_col = merged_map.get((logical_row, logical_col), (logical_row, logical_col))
    physical_row_idx, cell = cell_map.get((anchor_row, anchor_col), (None, None))

    if cell is None:
        return features

    try:
        style_name = cell.getAttribute("stylename")
    except (ValueError, KeyError):
        style_name = None

    if not style_name or style_name not in props_map:
        return features
    
    props = props_map[style_name]

    cell_bold = props.get("fo:font-weight") == "bold"
    text_bold = check_text_formatting_in_cell(cell, props_map, "fo:font-weight", "bold", doc)
    features[0] = cell_bold or text_bold

    cell_italic = props.get("fo:font-style") == "italic"
    text_italic = check_text_formatting_in_cell(cell, props_map, "fo:font-style", "italic", doc)
    features[1] = cell_italic or text_italic

    cell_underline = props.get("style:text-underline-style", "")
    cell_has_underline = cell_underline != "" and cell_underline != "none"
    
    text_has_underline = False
    paragraphs = cell.getElementsByType(P)
    for para in paragraphs:
        # Check paragraph style
        try:
            para_style_name = para.getAttribute("stylename")
        except (ValueError, KeyError):
            para_style_name = None
        
        if para_style_name:
            para_props = props_map.get(para_style_name)
            if not para_props:
                para_props = extract_text_properties_from_style(doc, para_style_name)
            
            if para_props:
                para_underline = para_props.get("style:text-underline-style", "")
                if para_underline != "" and para_underline != "none":
                    text_has_underline = True
                    break
        
        # Check spans
        spans = para.getElementsByType(Span)
        for span in spans:
            try:
                span_style_name = span.getAttribute("stylename")
            except (ValueError, KeyError):
                span_style_name = None
            
            if span_style_name:
                span_props = props_map.get(span_style_name)
                if not span_props:
                    span_props = extract_text_properties_from_style(doc, span_style_name)
                
                if span_props:
                    span_underline = span_props.get("style:text-underline-style", "")
                    if span_underline != "" and span_underline != "none":
                        text_has_underline = True
                        break
        if text_has_underline:
            break
    
    features[2] = cell_has_underline or text_has_underline

    font_size_str = props.get("fo:font-size", "")
    if font_size_str:
        features[3] = parse_font_size(font_size_str)
    
    font_name = props.get("style:font-name", "")
    if font_name:
        features[4] = stable_hash(font_name)

    
    if "fo:color" in props:
        rgb = ods_color_to_rgb(props["fo:color"])
        features[5:8] = rgb_normalized(rgb) if rgb else rgb_normalized((0, 0, 0))
    if "fo:background-color" in props:
        rgb = ods_color_to_rgb(props["fo:background-color"])
        if rgb is not None:
            features[8:11] = rgb_normalized(rgb)

    features[11] =  1 if (logical_row, logical_col) in merged_map else 0

    border_shorthand = props.get("fo:border", "none")
    has_border_shorthand = border_shorthand != "none" and border_shorthand != ""
    
    features[12] = (props.get("fo:border-left", "none") != "none") or has_border_shorthand
    features[13] = (props.get("fo:border-right", "none") != "none") or has_border_shorthand
    features[14] = (props.get("fo:border-top", "none") != "none") or has_border_shorthand
    features[15] = (props.get("fo:border-bottom", "none") != "none") or has_border_shorthand

    align = props.get("fo:text-align", "")
    align_source = props.get("style:text-align-source", "")
    
    # When text-align-source is 'value-type', try to infer alignment from cell content
    if align == "" and align_source == "value-type":
        align = infer_alignment_from_value_type(cell, align_source)

    features[16:23] = 0
    if align == "start" or align == "":
        features[16] = 1
    elif align == "end":
        features[17] = 1
    elif align == "center":
        features[18] = 1
    elif align == "justify":
        features[19] = 1
    elif align == "left":
        features[20] = 1
    elif align == "right":
        features[21] = 1
    else:
        features[22] = 1
    
    valign = props.get("style:vertical-align", "")
    features[23:29] = 0
    if valign == "top":
        features[23] = 1
    elif valign == "middle":
        features[24] = 1
    elif valign == "bottom":
        features[25] = 1
    elif valign == "automatic" or valign == "":
        features[26] = 1
    elif valign == "baseline":
        features[27] = 1
    else:
        features[28] = 1
    return features