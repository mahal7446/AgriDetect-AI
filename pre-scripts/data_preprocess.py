"""This script was extracted from the Colab notebook
and contains all preprocessing steps
"""

# ================================================
from google.colab import drive
drive.mount('/content/drive')
# ================================================
import os

os.listdir('/content/drive/MyDrive')
# ================================================
DATASET_ROOT = '/content/drive/MyDrive/Kisan_Sathi'
os.listdir(DATASET_ROOT)
# ================================================
import os
from collections import Counter
from PIL import Image

DATASET_ROOT = '/content/drive/MyDrive/Kisan_Sathi'

class_counts = Counter()
corrupted = []

for crop in os.listdir(DATASET_ROOT):
    crop_path = os.path.join(DATASET_ROOT, crop)
    if not os.path.isdir(crop_path):
        continue

    for cls in os.listdir(crop_path):
        cls_path = os.path.join(crop_path, cls)
        if not os.path.isdir(cls_path):
            continue

        for img_name in os.listdir(cls_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(cls_path, img_name)
                class_counts[f"{crop}/{cls}"] += 1
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except:
                    corrupted.append(img_path)

print("Image count per class:")
for k, v in class_counts.items():
    print(f"{k}: {v}")

print("\nCorrupted images found:", len(corrupted))
# ================================================
import os
from PIL import Image

DATASET_ROOT = '/content/drive/MyDrive/Kisan_Sathi'

removed = 0

for crop in os.listdir(DATASET_ROOT):
    crop_path = os.path.join(DATASET_ROOT, crop)
    if not os.path.isdir(crop_path):
        continue

    for disease in os.listdir(crop_path):
        disease_path = os.path.join(crop_path, disease)
        if not os.path.isdir(disease_path):
            continue

        for img in os.listdir(disease_path):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(disease_path, img)
                try:
                    im = Image.open(img_path)
                    im.verify()
                except:
                    os.remove(img_path)
                    removed += 1

print("Total corrupted images removed:", removed)
# ================================================
import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE_DIR = '/content/drive/MyDrive/Kisan_Sathi'
TARGET_DIR = '/content/drive/MyDrive/Kisan_Sathi_Split'

splits = ['train', 'val', 'test']

for split in splits:
    os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

for crop in os.listdir(SOURCE_DIR):
    crop_path = os.path.join(SOURCE_DIR, crop)
    if not os.path.isdir(crop_path):
        continue

    for disease in os.listdir(crop_path):
        disease_path = os.path.join(crop_path, disease)
        if not os.path.isdir(disease_path):
            continue

        images = [img for img in os.listdir(disease_path)
                  if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        train_imgs, temp_imgs = train_test_split(
            images, test_size=0.30, random_state=42)

        val_imgs, test_imgs = train_test_split(
            temp_imgs, test_size=0.50, random_state=42)

        for split_name, img_list in zip(
            ['train', 'val', 'test'],
            [train_imgs, val_imgs, test_imgs]
        ):
            dest_dir = os.path.join(
                TARGET_DIR, split_name, crop, disease)
            os.makedirs(dest_dir, exist_ok=True)

            for img in img_list:
                shutil.copy(
                    os.path.join(disease_path, img),
                    os.path.join(dest_dir, img)
                )

print("Dataset split completed.")
# ================================================
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = '/content/drive/MyDrive/Kisan_Sathi_Split'

train_dir = BASE_DIR + '/train'
val_dir   = BASE_DIR + '/val'
test_dir  = BASE_DIR + '/test'

train_gen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1.0/255)
test_gen = ImageDataGenerator(rescale=1.0/255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
# ================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================================================
EPOCHS = 10

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ================================================
test_loss, test_accuracy = model.evaluate(test_data)

print("Test accuracy:", test_accuracy)
print("Test loss:", test_loss)
# ================================================
MODEL_PATH = '/content/drive/MyDrive/Kisan_Sathi_Model.h5'
model.save(MODEL_PATH)

print("Model saved at:", MODEL_PATH)

# ================================================
model.save('/content/drive/MyDrive/Kisan_Sathi_Model.keras')
print("Model saved in new Keras format")
