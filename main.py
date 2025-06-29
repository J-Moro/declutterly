import os
import glob
from PIL import ImageTk
from tkinter import Tk, filedialog, Label, Button, Canvas

import cnn

# --- BATCH CLASSIFICATION UI ---
class BatchClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Declutterly")
        self.image_paths = []
        self.current_index = 0

        self.label_info = Label(root, text="Select a folder with images:", font=("Arial", 14))
        self.label_info.pack(pady=10)

        self.canvas = Canvas(root, width=224, height=224)
        self.canvas.pack()

        self.label_result = Label(root, text="", font=("Arial", 12))
        self.label_result.pack(pady=10)

        self.btn_select = Button(root, text="Select Folder", command=self.load_folder)
        self.btn_select.pack(pady=5)

        self.btn_next = Button(root, text="Next Image", command=self.next_image, state="disabled")
        self.btn_next.pack(pady=5)

        self.btn_delete = Button(root, text="Click here to delete", command=self.delete, state="disabled")
        self.btn_delete.pack(pady=5)

    def load_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.image_paths = glob.glob(os.path.join(folder, "*.jpg")) + \
                           glob.glob(os.path.join(folder, "*.jpeg")) + \
                           glob.glob(os.path.join(folder, "*.png"))
        self.image_paths.sort()
        self.current_index = 0
        if self.image_paths:
            self.btn_next.config(state="normal")
            self.show_image()
        else:
            self.label_result.config(text="No images found in folder.")

    def show_image(self):
        img_path = self.image_paths[self.current_index]
        category, confidence, image = cnn.predict_image(img_path)
        if category in ("screenshots", "memes"):
            self.btn_delete.config(state="normal")
        else: 
            self.btn_delete.config(state="disabled")

        image = image.resize((224, 224))
        tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(112, 112, image=tk_image, anchor='center')
        self.canvas.image = tk_image
        self.label_result.config(text=f"{os.path.basename(img_path)}\nPrediction: {category}\nConfidence: {confidence:.2f}")

    def next_image(self):
        self.current_index += 1
        if self.current_index < len(self.image_paths):
            self.show_image()
        else:
            self.label_result.config(text="Done! All images processed.")
            self.btn_next.config(state="disabled")

    def delete(self):
        if not self.image_paths:
            return
        
        # Get current image path
        img_path = self.image_paths[self.current_index]

        try:
            os.remove(img_path)
            print(f"Deleted: {img_path}")
        except Exception as e:
            print(f"Error deleting file: {e}")
            self.label_result.config(text=f"Error deleting file:\n{e}")
            return

        # Remove from list
        del self.image_paths[self.current_index]

        # Adjust index if needed
        if self.current_index >= len(self.image_paths):
            self.label_result.config(text="Done! All images processed.")
            self.btn_next.config(state="disabled")
            self.btn_delete.config(state="disabled")
        else:
            self.show_image()

# --- LAUNCH ---
if __name__ == "__main__":
    root = Tk()
    app = BatchClassifierApp(root)
    root.mainloop()