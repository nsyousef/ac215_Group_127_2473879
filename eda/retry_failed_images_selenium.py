import pandas as pd
import os
from google.cloud import storage
from tqdm import tqdm
from PIL import Image
from io import BytesIO, StringIO
import mimetypes
import concurrent.futures
import traceback

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# --- CONFIGURE THESE ---
BUCKET_NAME = 'derma-datasets'
BUCKET_PATH_BASE = 'raw/fitzpatrick17k/'
BUCKET_PATH_IMGS = os.path.join(BUCKET_PATH_BASE, 'images')
MASTER_CSV_BLOB = os.path.join(BUCKET_PATH_BASE, 'fitzpatrick17k.csv')
FAILURE_CSV_PATH = 'fitzpatrick_upload_failures.csv'
MAX_WORKERS = 2  # Adjust based on your machine's CPU/RAM (4-8 is typical for n1-standard-4 or better)

def is_valid_url(u):
    return isinstance(u, str) and u.startswith("http")

def get_image_extension(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes))
        format = img.format.lower()
        if format == 'jpeg':
            return '.jpg'
        elif format == 'png':
            return '.png'
        elif format == 'gif':
            return '.gif'
        elif format == 'webp':
            return '.webp'
        elif format == 'bmp':
            return '.bmp'
        elif format == 'tiff':
            return '.tif'
        elif format == 'ico':
            return '.ico'
        else:
            return f".{format}"
    except Exception as e:
        print(f"Error reading image bytes: {e}")
        return None

def get_content_type(image_bytes, extension):
    try:
        img = Image.open(BytesIO(image_bytes))
        content_type = Image.MIME[img.format]
    except Exception:
        content_type, _ = mimetypes.guess_type("file" + extension)
        if not content_type:
            content_type = 'application/octet-stream'
    return content_type

def blob_md5_set(bucket, img_path):
    md5_set = set()
    blobs = bucket.list_blobs(prefix=img_path)
    for blob in blobs:
        filename = os.path.basename(blob.name)
        if '.' in filename:
            md5 = filename.split('.')[0]
            md5_set.add(md5)
    return md5_set

def make_selenium_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--ignore-certificate-errors')
    service = Service('/usr/bin/chromedriver')
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(15)
    return driver

def selenium_get_image_bytes(url):
    driver = make_selenium_driver()
    try:
        driver.get(url)
        # For direct image URLs, a screenshot of the body will suffice.
        if url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tif', '.tiff')):
            image_bytes = driver.get_screenshot_as_png()
        else:
            img_els = driver.find_elements("tag name", "img")
            if img_els:
                try:
                    image_bytes = img_els[0].screenshot_as_png
                except Exception:
                    image_bytes = driver.get_screenshot_as_png()
            else:
                image_bytes = driver.get_screenshot_as_png()
    finally:
        driver.quit()
    return image_bytes

def upload_image(row, md5_set):
    url = row['url']
    md5 = row['md5hash']
    if not is_valid_url(url) or not isinstance(md5, str) or not md5:
        return (url, md5, 'skip', 'invalid url or md5hash')
    if md5 in md5_set:
        return (url, md5, 'skip', None)
    try:
        img_bytes = selenium_get_image_bytes(url)
        if not img_bytes or len(img_bytes) < 32:
            return (url, md5, 'fail', "empty image or download error")
        extension = get_image_extension(img_bytes)
        if not extension:
            return (url, md5, 'fail', "Could not determine extension")
        filename = md5 + extension
        content_type = get_content_type(img_bytes, extension)
        # storage API inside the thread: create a client per thread, safer
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(os.path.join(BUCKET_PATH_IMGS, filename))
        blob.upload_from_string(img_bytes, content_type=content_type)
        return (url, md5, 'ok', None)
    except (WebDriverException, TimeoutException) as e:
        return (url, md5, 'fail', f"WebDriverException: {str(e)}")
    except Exception as e:
        tb = traceback.format_exc()
        return (url, md5, 'fail', f"Exception: {str(e)}\n{tb}")

def main():
    # Read mapping
    print("Loading master mapping from bucket...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    master_blob = bucket.blob(MASTER_CSV_BLOB)
    master_csv_data = master_blob.download_as_text()
    master = pd.read_csv(StringIO(master_csv_data), usecols=['md5hash', 'url'])
    print(f"Read {len(master)} entries from master CSV.")

    # Read failures, merge w/ master for URL/md5hash pairing
    df_fail = pd.read_csv(FAILURE_CSV_PATH)
    if 'md5hash' not in df_fail.columns:
        df = pd.merge(df_fail, master, on='url', how='left')
    else:
        if 'url' not in df_fail.columns:
            df = pd.merge(df_fail, master, on='md5hash', how='left')
        else:
            df = df_fail

    df = df[df['url'].apply(is_valid_url) & df['md5hash'].notna() & (df['md5hash'] != "")]
    print(f"Found {len(df)} valid failed URL-md5hash pairs to retry.")

    md5_set = blob_md5_set(bucket, BUCKET_PATH_IMGS)
    print(f"Found {len(md5_set)} images already in bucket.")

    # Safe: create an iterable of dict per row (avoid pandas object serialization)
    jobs = df[['url', 'md5hash']].to_dict(orient='records')

    # Use ThreadPoolExecutor, at most MAX_WORKERS concurrent browsers
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # t is a tuple: (url, md5hash, status, reason)
        future_to_job = {
            executor.submit(upload_image, row, md5_set): row for row in jobs
        }
        for fut in tqdm(concurrent.futures.as_completed(future_to_job), total=len(jobs), desc="Processing"):
            try:
                results.append(fut.result())
            except Exception as exc:
                row = future_to_job[fut]
                results.append((row['url'], row['md5hash'], 'fail', f'Worker exception: {exc}'))

    num_ok = sum(1 for r in results if r[2] == 'ok')
    num_skip = sum(1 for r in results if r[2] == 'skip')
    num_fail = sum(1 for r in results if r[2] == 'fail')
    print(f"Done!\nUploaded: {num_ok}\nAlready existed/skipped: {num_skip}\nFailed: {num_fail}")

    fail_rows = [r for r in results if r[2] == 'fail']
    if fail_rows:
        df_fail = pd.DataFrame(fail_rows, columns=['url','md5hash','status','reason'])
        df_fail.to_csv('fitzpatrick_retry_failures.csv', index=False)
        print(f"New failures written to fitzpatrick_retry_failures.csv")

if __name__ == "__main__":
    main()