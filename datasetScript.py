# === OPTIONAL: SETUP VIRTUAL ENVIRONMENT (run once) ===
# To run this block manually in the terminal:
# py -m venv venv
# source venv/bin/activate       # On macOS/Linux
# .\venv\Scripts\activate        # On Windows
# pip install --upgrade pip
# pip install duckduckgo-search pillow tqdm

# === ACTUAL SCRIPT STARTS HERE ===

import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from duckduckgo_search import DDGImages

# === CONFIG ===
CATEGORIES = {
    "screenshots": ["phone screenshot", "app UI screenshot", "chat screenshot"],
    "memes": ["funny meme", "relatable meme", "reaction meme"],
    "people": ["portrait photography", "smiling person", "friends group photo"],
    "food": ["delicious food", "gourmet meal", "street food"],
    "places": ["beautiful landscape", "city skyline", "travel destination"],
    "animals": ["cute puppy", "wildlife", "funny cat"]
}
OUTPUT_DIR = "gallery_dataset"
IMAGES_PER_CATEGORY = 500
IMAGES_PER_QUERY = 100  # number per subquery

# === HELPER FUNCTIONS ===
def sanitize_filename(name):
    return "".join(c if c.isalnum() else "_" for c in name)

def download_image(url, path):
    try:
        response = requests.get(url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(path)
        return True
    except Exception:
        return False

def fetch_images(query, category_dir, start_index=0, max_images=100):
    search = DDGImages()
    results = search.search(query, max_results=max_images)
    downloaded = 0

    for result in tqdm(results, desc=f"  ↳ {query[:30]}...", leave=False):
        url = result.image
        filename = os.path.join(category_dir, sanitize_filename(query + "_" + str(downloaded)) + ".jpg")
        if download_image(url, filename):
            downloaded += 1
        if downloaded >= max_images:
            break

# === MAIN ===
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for category, queries in CATEGORIES.items():
        print(f"\nCategory: {category}")
        category_dir = os.path.join(OUTPUT_DIR, category)
        os.makedirs(category_dir, exist_ok=True)
        total_per_query = IMAGES_PER_CATEGORY // len(queries)
        for query in queries:
            fetch_images(query, category_dir, max_images=total_per_query)

if __name__ == "__main__":
    main()