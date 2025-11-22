import os
import ssl
import urllib.request


def download_file(url, filepath):
    print(f"Downloading {url} to {filepath}...")
    # Create unverified context to avoid SSL errors on some Windows setups
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(url, context=ctx) as response, open(filepath, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading: {e}")
        exit(1)

if __name__ == "__main__":
    if not os.path.exists("tests/data"):
        os.makedirs("tests/data")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    filepath = "tests/data/adult.data"
    download_file(url, filepath)

