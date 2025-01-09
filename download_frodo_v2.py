import os 
import glob 
import pandas as pd

CSV_FILE = "dataset_v2.csv"

df = pd.read_csv(CSV_FILE)

urls = df['url'].tolist()

for url in urls:
    file = url.split('/')[-1]
    print("Downloading", file)
    os.system(f"sh download_and_unzip.sh {url} {file}")
