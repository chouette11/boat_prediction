from bs4 import BeautifulSoup
import pandas as pd

# Load the HTML file
import os
boat_files = os.listdir("boat_html")  # Assuming the HTML file is in this directory
for boat_file in boat_files:
    file_path = os.path.join("boat_html", boat_file)
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Find the boat ranking table
    table = soup.find("table", class_="par-table01")

    # Extract headers
    headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]

    # Extract data rows
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells:
            rows.append(cells)

    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Save to CSV
    csv_path = os.path.join("boat_csv", f"{os.path.splitext(boat_file)[0]}.csv")
    df.to_csv(csv_path, index=False)
