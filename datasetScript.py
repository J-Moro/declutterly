# ==============================================================
# Reddit Image Dataset Scraper using Official Reddit API (PRAW)
#
# Setup Instructions:
# --------------------------------------------------------------
# 1. Create a Reddit app:
#    - Go to: https://www.reddit.com/prefs/apps
#    - Click "Create App"
#    - App type: script
#    - Redirect URI: http://localhost:8080
#    - Save the app to get your `client_id` and `client_secret`
#
# 2. Create a `config.ini` file in the same directory with this content:
#
# [REDDIT]
# client_id = your_client_id_here
# client_secret = your_client_secret_here
# username = your_reddit_username
# password = your_reddit_password
#
# 3. Install dependencies:
# pip install praw tqdm requests pandas
#
# 4. Run the script:
# python reddit_scraper.py
#
# Output: gallery_dataset/<category>/<subreddit>_<index>.jpg
# ==============================================================

import os
import requests
import configparser
import pandas as pd
from tqdm import tqdm
import praw
import shutil
import random
import time
import uuid

# === Load credentials from config.ini ===
config = configparser.ConfigParser()
config.read("config.ini")

REDDIT_CLIENT_ID = config["REDDIT"]["client_id"]
REDDIT_CLIENT_SECRET = config["REDDIT"]["client_secret"]
REDDIT_USERNAME = config["REDDIT"]["username"]
REDDIT_PASSWORD = config["REDDIT"]["password"]
USER_AGENT = f"ImageDatasetScript/0.2 by {REDDIT_USERNAME}"

# === Reddit client ===
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD,
    user_agent=USER_AGENT,
)

# === Categories and subreddits ===
CATEGORIES = {
    "screenshots": {
        "subreddits": ["screenshots", "softwaregore", "texts"],
        "allow_nsfw": True
    },
    "memes": {
        "subreddits": ["memes", "dankmemes", "wholesomememes", "me_irl"],
        "allow_nsfw": True
    },
    "people": {
        "subreddits": ["Portraits", "selfie", "OldSchoolCool"],
        "allow_nsfw": False  # Filter NSFW
    },
    "food": {
        "subreddits": ["foodporn", "food", "BudgetFood"],
        "allow_nsfw": True
    },
    "places": {
        "subreddits": ["EarthPorn", "cityporn", "travel"],
        "allow_nsfw": True
    },
    "animals": {
        "subreddits": ["aww", "cats", "dogs"],
        "allow_nsfw": True
    }
}

LIMIT = 1000  # Total images per category
OUTPUT_DIR = "gallery_dataset"
HEADERS = {"User-Agent": USER_AGENT}

# === Initialize metadata container ===
all_metadata = []


def download_image(url, path):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if "image" in resp.headers.get("Content-Type", ""):
            with open(path, "wb") as f:
                f.write(resp.content)
            return True
    except:
        pass
    return False


def scrape_subreddits(category, config, target_total):
    os.makedirs(os.path.join(OUTPUT_DIR, category), exist_ok=True)
    total_count = 0
    subreddits = config["subreddits"]
    allow_nsfw = config["allow_nsfw"]
    per_sub_limit = target_total // len(subreddits)

    for sub in subreddits:
        print(f"\nCategory: {category} — Subreddit: r/{sub}")
        count = 0
        subreddit = reddit.subreddit(sub)

        for post in subreddit.top(limit=per_sub_limit * 2):  # overfetch
            if post.over_18 and not allow_nsfw:
                continue

            ext = os.path.splitext(post.url)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png"]:
                continue

            filename = f"{sub}_{count}{ext}"
            filepath = os.path.join(OUTPUT_DIR, category, filename)

            if download_image(post.url, filepath):
                all_metadata.append({
                    "filename": filename,
                    "category": category,
                    "subreddit": sub,
                    "title": post.title,
                    "author": str(post.author),
                    "upvotes": post.score,
                    "url": post.url
                })
                count += 1
                total_count += 1
                tqdm.write(f"✔ {filepath}")

            if count >= per_sub_limit or total_count >= target_total:
                break

        if total_count >= target_total:
            break

    print(f"Finished {category}: {total_count} images\n")

def split_dataset(base_dir="gallery_dataset", splits=(0.7, 0.15, 0.15), seed=None):
    assert sum(splits) == 1.0, "Splits must sum to 1.0"
    if seed is None:
        seed = int(time.time()) % 100000
        print(f"[INFO] Using dynamic seed: {seed}")
    random.seed(seed)

    categories = [c for c in os.listdir(base_dir)
                  if os.path.isdir(os.path.join(base_dir, c)) and c not in {"train", "val", "test"}]

    for split_name in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dir, split_name), exist_ok=True)

    for category in categories:
        src_dir = os.path.join(base_dir, category)
        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * splits[0])
        n_val = int(n * splits[1])

        split_map = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split_name, files in split_map.items():
            if split_name == "test":
                dst_dir = os.path.join(base_dir, "test")
            else:
                dst_dir = os.path.join(base_dir, split_name, category)

            os.makedirs(dst_dir, exist_ok=True)

            for file in files:
                src_path = os.path.join(src_dir, file)
                ext = os.path.splitext(file)[1].lower()

                if split_name == "test":
                    # Generate a UUID-based name with the same file extension
                    new_name = f"{uuid.uuid4().hex}{ext}"
                    dst_path = os.path.join(dst_dir, new_name)
                else:
                    dst_path = os.path.join(dst_dir, file)

                shutil.copy2(src_path, dst_path)

    print("✅ Dataset split complete.")
    print("✅ Test images are flattened and randomly renamed.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for category, config in CATEGORIES.items():
        scrape_subreddits(category, config, LIMIT)

    # Save all metadata to a single CSV file
    df = pd.DataFrame(all_metadata)
    df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
    print(f"Saved metadata for all categories: {len(df)} images")


if __name__ == "__main__":
    main()
    split_dataset()