# 📂 Declutterly: Intelligent Image Categorization for Gallery Cleanup  
*A Computer Vision project with an academic focus on personal photo gallery management*

## 🎓 Project Overview

Declutterly is an academic project developed to explore how convolutional neural networks (CNNs) can assist in organizing and cleaning personal photo galleries. It automatically classifies images into categories such as `screenshots`, `memes`, `food`, `places`, `animals`, and `people`, helping users identify clutter or redundant content.

This solution addresses a common real-world problem, the exponential growth of poorly categorized photos on smartphones, by applying image classification techniques and offering insights into the feasibility of AI-powered gallery decluttering.

## 🧠 Motivation

- Users often accumulate thousands of unsorted images, consuming valuable storage.
- Manual sorting is time-consuming and error-prone.
- Traditional cleanup tools lack personalized or contextual image understanding.
- This project investigates whether deep learning can provide meaningful assistance in this context.

## 📥 How to Set Up and Run

### 1. Clone the repository

```bash
git clone https://github.com/J-Moro/declutterly.git
cd declutterly
```
### 2. Install dependencies

Ensure you have Python 3.9+ installed. The required libraries are listed as comments at the top of datasetScript.py, but you can install them quickly with:

```bash
pip install praw tqdm requests pandas torch torchvision matplotlib pillow
```
### 3. Create a `config.ini` file in the same directory with this content:

```bash
[REDDIT]
client_id = your_client_id_here
client_secret = your_client_secret_here
username = your_reddit_username
password = your_reddit_password
```
## 🧪 Dataset Generation

To build the dataset:

```bash
python datasetScript.py
```
This script:

- Downloads images from Reddit subreddits relevant to each category.
- Organizes them into subfolders under ./dataset/.
- Saves metadata like subreddit and title for future reference.
- You’ll end up with a structured dataset that mimics a real-world photo gallery, divided by type.

## 🚀 Running the Classifier App

Once the dataset is ready, run: 

```bash
python main.py
```
This will load the model saved locally or, if not found, attempt to download it from Github. Then, a simple UI will open, where you can:

- Select a folder from your local gallery.
- Automatically classify each image using the trained CNN.
- View predictions along with confidence scores.

## 📈 Model Architecture and Results

- Architecture: CNN with Transfer Learning
- Framework: PyTorch
- Training: Based on the curated Reddit dataset
- Evaluation:
- High accuracy for distinct categories (e.g., memes vs. people)
- Confusion may occur with visually similar classes (e.g., memes vs. screenshots)

## 📌 Limitations and Improvements

### Limitations

- Model is trained with fixed categories, limiting adaptability to individual preferences.
- Images from Reddit may not perfectly reflect a user’s personal gallery.
- Supervised learning requires ongoing dataset maintenance for new image types.

### Future Improvements

- Explore unsupervised learning to allow category discovery without labels.
- Incorporate user feedback to improve predictions over time.
- Investigate edge deployment for on-device photo management.

## 👩‍🎓 Academic Relevance

Presentation used in academic context: [View slides](https://www.canva.com/design/DAGrM_VDyNE/s-7n0VqFURTvXJaBFWTkjw)