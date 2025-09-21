# src/train_audio.py
# Train CNN on spectrogram images using predefined splits (.txt files)

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tqdm.keras import TqdmCallback # <--- 1. Import TqdmCallback for training

from src.config import AUDIO_MODEL_PATH, MODELS_DIR, IMG_SIZE_AUDIO, SPLITS_DIR

BATCH_SIZE = 16
EPOCHS = 12

def build_audio_model(input_shape=(224,224,3), num_classes=2):
    # This function is unchanged
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(), MaxPool2D(),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(), MaxPool2D(),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(), MaxPool2D(),
        Flatten(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_split(split_file):
    # This function is unchanged
    items = []
    with open(split_file, "r") as f:
        for line in f:
            path, label = line.strip().split()
            label = 0 if label.lower() in ["real", "bonafide", "0"] else 1
            items.append((path, label))
    return items

def make_generator(items, target_size, batch_size, shuffle=True):
    # This function is unchanged
    n = len(items)
    while True:
        if shuffle:
            np.random.shuffle(items)
        for i in range(0, n, batch_size):
            batch = items[i:i+batch_size]
            X, y = [], []
            for path, label in batch:
                img = load_img(path, target_size=target_size)
                img_arr = img_to_array(img)
                X.append(img_arr / 255.0)
                onehot = np.zeros(2)
                onehot[label] = 1
                y.append(onehot)
            yield np.array(X), np.array(y)

def main():
    train_items = load_split(os.path.join(SPLITS_DIR, "audio_train.txt"))
    val_items   = load_split(os.path.join(SPLITS_DIR, "audio_val.txt"))
    test_items  = load_split(os.path.join(SPLITS_DIR, "audio_test.txt"))

    train_gen = make_generator(train_items, IMG_SIZE_AUDIO, BATCH_SIZE, shuffle=True)
    val_gen   = make_generator(val_items, IMG_SIZE_AUDIO, BATCH_SIZE, shuffle=False)
    test_gen  = make_generator(test_items, IMG_SIZE_AUDIO, BATCH_SIZE, shuffle=False)
    
    steps_train = len(train_items) // BATCH_SIZE
    steps_val   = len(val_items) // BATCH_SIZE
    steps_test  = len(test_items) // BATCH_SIZE

    model = build_audio_model(input_shape=(IMG_SIZE_AUDIO[0], IMG_SIZE_AUDIO[1], 3))
    os.makedirs(MODELS_DIR, exist_ok=True)

    chk = ModelCheckpoint(AUDIO_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=0) # Set verbose=0 to avoid duplicate output
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

    # --- 2. Add TqdmCallback to the list of callbacks for the training loop ---
    model.fit(train_gen,
              validation_data=val_gen,
              steps_per_epoch=steps_train,
              validation_steps=steps_val,
              epochs=EPOCHS,
              callbacks=[chk, rlr, TqdmCallback(verbose=2)],
              verbose=0) # Set verbose=0 to let TqdmCallback handle the output

    print("\n✅ Training finished. Best model saved to:", AUDIO_MODEL_PATH)

    print("\n🔎 Evaluating on test set...")
    # --- 3. Use verbose=1 in model.evaluate to show a built-in progress bar ---
    loss, acc = model.evaluate(test_gen, steps=steps_test, verbose=1)
    print(f"\nTest accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()