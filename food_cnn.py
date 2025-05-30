# food_cnn.py
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Use 3 for three classes
])

# Compile the CNN
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training and test data
training_set = train_datagen.flow_from_directory(
    './dataset/training_set',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)
test_set = test_datagen.flow_from_directory(
    './dataset/test_set',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Train the model
model.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=20,
    validation_data=test_set,
    validation_steps=len(test_set)
)

# Save the model
model_dir = './models/'
os.makedirs(model_dir, exist_ok=True)

# Save in native Keras format
model.save(os.path.join(model_dir, 'model.keras'))

# Optional: save only weights (with proper extension)
model.save_weights(os.path.join(model_dir, 'weights.weights.h5'))

print("✅ Model training complete and saved successfully.")

