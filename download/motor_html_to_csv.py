import pandas as pd
from bs4 import BeautifulSoup
import re

# HTML content is too long for direct parsing, simulate partial parsing
# We will focus on the <table> that contains motor performance data
import os

motor_files = os.listdir("motor_html")  # Assuming the HTML file is in this directory

for motor_file in motor_files:
    html_file_path = os.path.join("motor_html", motor_file)
    table_data = []

    with open(html_file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Locate the relevant table
    table = soup.find("table", class_="par-table01")

    # Extract headers
    headers = []
    header_rows = table.find_all("tr")[0:2]  # main and sub-header
    for th in header_rows[0].find_all("th"):
        col_span = th.get("colspan")
        if col_span and int(col_span) > 1:
            # For '決まり手分析', extract subheaders from second header row
            subheaders = header_rows[1].find_all("th")
            headers.extend([th.get_text(strip=True) for th in subheaders])
        else:
            headers.append(th.get_text(strip=True))

    # Extract table rows
    for row in table.find("tbody").find_all("tr"):
        cols = [td.get_text(strip=True).replace("'", "") for td in row.find_all("td")]
        if cols:
            table_data.append(cols)

    # Create DataFrame
    df = pd.DataFrame(table_data, columns=headers)

    # Save to CSV
    csv_file_path = os.path.join("motor_csv", f"{os.path.splitext(motor_file)[0]}.csv")
    df.to_csv(csv_file_path, index=False)
