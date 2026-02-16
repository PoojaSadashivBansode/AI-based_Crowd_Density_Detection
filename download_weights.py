import os
import requests
import sys

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    weights_url = "https://huggingface.co/muasifk/CSRNet/resolve/main/CSRNet.pth"
    output_file = "csrnet_weights.pth"
    
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping download.")
    else:
        download_file(weights_url, output_file)
