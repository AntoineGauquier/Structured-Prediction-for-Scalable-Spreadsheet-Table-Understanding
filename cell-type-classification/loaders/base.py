"""Base loader (interface) for spreadsheets."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import pandas as pd
import odf
from odf.table import Table

class SpreadsheetLoader(ABC):
    @abstractmethod
    def load(self, path, sheet_name=None):
        """Load a spreadsheet file.
        
        Args:
            path: Path to the spreadsheet file
            sheet_name: Sheet to load (index, name, or None for default)
            
        Returns:
            Tuple of (dataframe, sheet_object, workbook_object, merged_map, cell_map)
        """
        pass
    
    @staticmethod
    def _resolve_sheet_index(workbook, sheet_name):
        """Resolve sheet name/index to an index, for non-CSV formats.
        
        Args:
            workbook: Workbook object
            sheet_name: Sheet identifier (int index, str name, or None)
        """
        if sheet_name is None: # Default first sheet
            return 0
        elif isinstance(sheet_name, int):
            return sheet_name
        elif isinstance(sheet_name, str):
            for idx, sheet in enumerate(workbook):
                if hasattr(sheet, 'name') and sheet.name == sheet_name:
                    return idx
                elif hasattr(sheet, 'title') and sheet.title == sheet_name:
                    return idx
                elif getattr(sheet, 'tagName', None) == 'table:table': # For ODS only
                    ods_name = sheet.attributes.get(
                        ("urn:oasis:names:tc:opendocument:xmlns:table:1.0", "name")
                    )
                    if ods_name == sheet_name:
                        return idx
            raise ValueError(f"Sheet '{sheet_name}' not found")
        else:
            raise ValueError(f"Invalid sheet_name type: {type(sheet_name)}")