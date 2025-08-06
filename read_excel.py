import pandas as pd
import logging

def read_excel_and_chunk(excel_path: str):
    """
    Reads an Excel file, converting each row of each sheet into a text chunk.
    """
    try:
        xls = pd.ExcelFile(excel_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None).astype(str)
            # Filter out empty rows
            df.dropna(how='all', inplace=True)
            for index, row in df.iterrows():
                # Join cell values with " | "
                row_text = " | ".join([val for val in row if pd.notna(val) and str(val).strip()])
                if row_text:
                    yield {
                        "text": row_text,
                        "source": f"Sheet '{sheet_name}', Row {index + 1}"
                    }
    except Exception as e:
        logging.error(f"Could not process Excel file: {e}")
        return