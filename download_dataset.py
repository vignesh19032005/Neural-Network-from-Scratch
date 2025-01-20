import urllib.request
import gzip
import shutil
import os
from urllib.error import HTTPError

def download_and_extract(url, output_path):
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Extracting {output_path}...")
        with gzip.open(output_path, 'rb') as f_in:
            with open(output_path.replace('.gz', ''), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(output_path)
        print(f"Done with {output_path}")
    except HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason} for URL: {url}")

def main():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

    for file in files:
        download_and_extract(base_url + file, file)
    
    print("Dataset downloaded and extracted successfully.")

if __name__ == "__main__":
    main()
