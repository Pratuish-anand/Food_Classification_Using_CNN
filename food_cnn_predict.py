# food_predictor_gui.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as image_utils
from PIL import Image, ImageTk
import requests
from io import BytesIO
from tkinter import Tk, Label, Entry, Button

# Load the trained model
model_path = './models/model.keras'
model = load_model(model_path)

# Create the GUI window
window = Tk()
window.title("🍕 Food Image Predictor") 
window.geometry('800x600')

# Label for the URL input
lbl = Label(window, text="Enter the URL of the image", font=("Helvetica", 16))
lbl.pack()

# Function to handle image prediction
def clicked():
    url = User_input.get()
    print(f"Image URL: {url}")

    try:
        # Fetch the image
        response = requests.get(url)
        test_image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print("Error loading image:", e)
        error_lbl = Label(window, text="❌ Failed to load image!", font=("Helvetica", 14), fg="red")
        error_lbl.pack()
        return

    # Resize for display and model
    put_image = test_image.resize((400, 400))
    test_image_resized = test_image.resize((128, 128))
    
    # Display the image
    img = ImageTk.PhotoImage(put_image)
    pic = Label(window, image=img)
    pic.pack()
    pic.image = img

    # Preprocess for prediction
    test_image_resized = image_utils.img_to_array(test_image_resized)
    test_image_resized = np.expand_dims(test_image_resized, axis=0)
    test_image_resized = test_image_resized / 255.0

    # Predict
    result = model.predict(test_image_resized)
    class_names = ['french fries', 'pizza', 'samosa']
    predicted_class = class_names[np.argmax(result)]

    # Show prediction
    out = Label(window, text=f'🍽️ Predicted: {predicted_class}', font=("Helvetica", 16))
    out.pack()

# Input field
User_input = Entry(window, width=100)
User_input.pack()

# Predict button
btn = Button(window, text="Detect Image", font=("Helvetica", 12), command=clicked)
btn.pack()

# Run the GUI
window.mainloop()

