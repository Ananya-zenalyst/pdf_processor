import pdfplumber
import json

def extract_tables(pdf_path):
    """
    Extracts tables from each page of a PDF file using pdfplumber.
    This version is improved to handle multi-line row headers (metrics)
    by merging them before associating them with data.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: A JSON string containing the extracted tables per page.
    """
    extracted_data = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if not tables:
                    continue

                page_tables = []
                for table in tables:
                    # Filter out any potential None or empty rows from the raw table
                    clean_table = [row for row in table if row and any(cell is not None for cell in row)]
                    if not clean_table or len(clean_table) < 2:
                        continue

                    # The first valid row is assumed to be the header. Clean it up.
                    # We start from the second cell, as the first is usually empty or a label for the metrics column.
                    headers = [str(h).replace('\n', ' ').strip() for h in clean_table[0][1:] if h]
                    if not headers:
                        continue

                    table_data = []
                    metric_buffer = []

                    # Process the data rows (all rows after the header)
                    for row in clean_table[1:]:
                        metric = str(row[0]).replace('\n', ' ').strip() if row[0] else ""
                        values = row[1:]

                        # Determine if the row contains any actual data or if it's just a label part
                        has_data_values = any(v is not None and str(v).strip() != '' for v in values)

                        if metric and not has_data_values:
                            # This row has a label but no data. It's part of a multi-line metric.
                            # Add it to our buffer and continue to the next line.
                            metric_buffer.append(metric)
                            continue
                        
                        if has_data_values:
                            # This row contains data. It's time to construct the full metric and the data record.
                            metric_buffer.append(metric)
                            full_metric = " ".join(metric_buffer).strip()
                            metric_buffer = []  # Reset the buffer for the next metric

                            # Clean up the data values for this row
                            clean_values = [str(v).replace('\n', ' ').strip() if v is not None else "" for v in values]
                            
                            row_dict = {"Metric": full_metric}
                            # Zip headers and values to fill the dictionary
                            for k, v in zip(headers, clean_values):
                                row_dict[k] = v
                            
                            table_data.append(row_dict)

                    if table_data:
                        page_tables.append(table_data)

                if page_tables:
                    extracted_data[f'page_{i+1}'] = page_tables

    except Exception as e:
        return json.dumps({"error": f"An error occurred during table extraction: {e}"}, indent=4)

    return json.dumps(extracted_data, indent=4, ensure_ascii=False)

