# gallery_cleanup_app.py

"""
📦 Dependencies (install before running):

pip install torch torchvision matplotlib pillow pandas

🧠 This script:
1. Loads and preprocesses images (resized to 224x224)
2. Uses MobileNetV2 (pretrained on ImageNet) with PyTorch
3. Trains a classifier head for 6 gallery categories
4. Launches a Tkinter UI to classify new images
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Label, Button, Canvas
from PIL import Image, ImageTk
import urllib.request

# --- CONFIG ---
DATA_DIR = "gallery_dataset"
BATCH_SIZE = 32
NUM_CLASSES = 6
IMG_SIZE = 224
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['animals', 'food', 'memes', 'people', 'places', 'screenshots']
MODEL_PATH = "gallery_model.pth"
MODEL_URL = "https://github.com/J-Moro/declutterly/releases/download/v1.0/gallery_model.pth"

# --- TRANSFORMS ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# --- DATA LOADERS ---
image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ['train', 'val']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
    for x in ['train', 'val']
}

# --- MODEL SETUP ---
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model = model.to(DEVICE)

# --- TRAINING ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

def train_model():
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects.double() / len(image_datasets['train'])
        print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n")

    torch.save(model.state_dict(), "gallery_model.pth")
    print("✅ Training complete. Model saved as gallery_model.pth")

# Uncomment to train:
# train_model()

# --- INFERENCE UTIL ---
def predict_image(image_path):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    transform = data_transforms['val']
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    return CLASS_NAMES[predicted.item()], probs[predicted.item()].item()

# --- UI ---
def open_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    img = Image.open(file_path).resize((224, 224))
    tk_image = ImageTk.PhotoImage(img)
    canvas.create_image(112, 112, image=tk_image, anchor='center')
    canvas.image = tk_image
    label_result.config(text="Predicting...")
    label_result.update_idletasks()
    pred_class, confidence = predict_image(file_path)
    label_result.config(text=f"Prediction: {pred_class}\nConfidence: {confidence:.2f}")

# --- LOAD TRAINED MODEL ---

if not os.path.exists(MODEL_PATH):
    print("⚠️ Model not found locally. Attempting to download from GitHub...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Downloaded model from GitHub.")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("Please train the model manually by uncommenting train_model() and running the script.")
        MODEL_PATH = None

if MODEL_PATH and os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("✅ Loaded saved model.")

# --- LAUNCH UI ---
root = Tk()
root.title("PyTorch Gallery Classifier")

Label(root, text="Choose an image to classify:", font=("Arial", 14)).pack(pady=10)
canvas = Canvas(root, width=224, height=224)
canvas.pack()
Button(root, text="Choose Image", command=open_image).pack(pady=10)
label_result = Label(root, text="", font=("Arial", 12))
label_result.pack()

root.mainloop()
