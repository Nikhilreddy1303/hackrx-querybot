import pandas as pd
import logging

def read_excel_and_chunk(excel_path: str):
    """
    Reads an Excel file, using the first row as headers and converting each 
    subsequent row into a descriptive "key: value" text chunk.
    """
    try:
        xls = pd.ExcelFile(excel_path)
        for sheet_name in xls.sheet_names:
            # Use the first row as the header (header=0)
            df = pd.read_excel(xls, sheet_name=sheet_name, header=0).astype(str)
            df.dropna(how='all', inplace=True)

            # Iterate through rows and create descriptive chunks
            for index, row in df.iterrows():
                # Create a "key: value" string for each cell in the row
                row_text = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip()])
                if row_text:
                    yield {
                        "text": row_text,
                        # +2 because of 0-based index and the header row
                        "source": f"Sheet '{sheet_name}', Row {index + 2}"
                    }
    except Exception as e:
        logging.error(f"Could not process Excel file: {e}")
        return