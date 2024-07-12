import os 
import glob 
import pandas as pd

CSV_FILE = "/home/noam/Downloads/complete-dataset.csv"

df = pd.read_csv(CSV_FILE)

urls = df['url'].tolist()

for url in urls:
    file = url.split('/')[-1]
    breakpoint()
    os.system(f"sudo download_and_unzip.sh {url} {file}")
