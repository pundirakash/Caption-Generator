# Import required libraries
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# Load the image captioning model, feature extractor, and tokenizer
model_name = "bipin/image-caption-generator"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Define a function to generate captions
def generate_captions(image_path, num_captions):
    # Load and preprocess the image
    img = Image.open(image_path).convert(mode="RGB")
    pixel_values = feature_extractor(images=[img], return_tensors="pt").pixel_values.to(device)

    # Generate and decode captions
    output_ids = model.generate(pixel_values, num_beams=4, max_length=128, num_return_sequences=num_captions)
    captions = []
    for i in range(num_captions):
        preds = tokenizer.decode(output_ids[i], skip_special_tokens=True)
        captions.append(f"Caption {i+1}: {preds}")
    return captions

# Define a function to handle button click
def handle_click():
    # Use filedialog to select an image and get the number of captions from the entry field
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    num_captions = int(num_captions_entry.get())

    # Generate captions and display them in the captions_text widget
    captions = generate_captions(image_path, num_captions)
    captions_text.delete("1.0", "end")
    captions_text.insert("1.0", "\n".join(captions))

# Create a GUI window
root = tk.Tk()
root.title("Image Caption Generator")
root.configure(bg="#F8F8F8")

# Create a label and entry for the number of captions
num_captions_label = tk.Label(root, text="Enter no. of captions you want to generate:", font=("Open Sans", 14), bg="#F8F8F8")
num_captions_label.pack(pady=10)
num_captions_entry = tk.Entry(root, font=("Open Sans", 14))
num_captions_entry.insert(0, "3")
num_captions_entry.pack(pady=10)

# Create a button to select an image and generate captions
select_image_button = tk.Button(root, text="Select Image", font=("Open Sans", 14), bg="#5E5E5E", fg="#FFFFFF", command=handle_click)
select_image_button.pack(pady=10)

# Create a text box to display captions
captions_text = tk.Text(root, font=("Open Sans", 14), height=10)
captions_text.pack(pady=10)

# Start the GUI event loop
root.mainloop()
