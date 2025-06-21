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
# pip install praw tqdm requests
#
# 4. Run the script:
# python reddit_scraper.py
#
# Output: gallery_dataset/<category>/<subreddit>_<index>.jpg
# ==============================================================

import os
import requests
import configparser
from tqdm import tqdm
import praw

# === Load credentials from config.ini ===
config = configparser.ConfigParser()
config.read("config.ini")

REDDIT_CLIENT_ID = config["REDDIT"]["client_id"]
REDDIT_CLIENT_SECRET = config["REDDIT"]["client_secret"]
REDDIT_USERNAME = config["REDDIT"]["username"]
REDDIT_PASSWORD = config["REDDIT"]["password"]
USER_AGENT = f"ImageDatasetScript/0.1 by {REDDIT_USERNAME}"

# === PRAW client ===
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD,
    user_agent=USER_AGENT,
)

# === Categories config with optional NSFW filtering ===
CATEGORIES = {
    "screenshots": {"subreddit": "screenshots", "allow_nsfw": True},
    "memes": {"subreddit": "memes", "allow_nsfw": True},
    "people": {"subreddit": "Portraits", "allow_nsfw": False},  # NSFW filtered
    "food": {"subreddit": "foodporn", "allow_nsfw": True},
    "places": {"subreddit": "EarthPorn", "allow_nsfw": True},
    "animals": {"subreddit": "aww", "allow_nsfw": True},
}

LIMIT = 500
OUTPUT_DIR = "gallery_dataset"
HEADERS = {"User-Agent": USER_AGENT}

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

def scrape_subreddit(subreddit_name, category_folder, allow_nsfw, limit):
    os.makedirs(category_folder, exist_ok=True)
    print(f"\nDownloading from r/{subreddit_name} (NSFW allowed: {allow_nsfw})")

    subreddit = reddit.subreddit(subreddit_name)
    count = 0

    for post in subreddit.top(limit=limit):
        if post.over_18 and not allow_nsfw:
            continue  # Skip NSFW posts if not allowed

        url = post.url
        ext = os.path.splitext(url)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            continue

        filename = os.path.join(category_folder, f"{subreddit_name}_{count}{ext}")
        if download_image(url, filename):
            count += 1
            tqdm.write(f"✔ Saved {filename}")
        if count >= limit:
            break

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for category, config in CATEGORIES.items():
        subreddit_name = config["subreddit"]
        allow_nsfw = config["allow_nsfw"]
        category_folder = os.path.join(OUTPUT_DIR, category)
        scrape_subreddit(subreddit_name, category_folder, allow_nsfw, LIMIT)

if __name__ == "__main__":
    main()