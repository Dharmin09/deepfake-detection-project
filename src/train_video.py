# src/train_video.py
# Fine-tune Xception on frames using predefined splits (.txt files)

import os
import numpy as np
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tqdm.keras import TqdmCallback
from tqdm import tqdm

from src.config import VIDEO_MODEL_PATH, MODELS_DIR, IMG_SIZE_VIDEO, SPLITS_DIR
from src.utils import preprocess_frames_for_xception

BATCH_SIZE = 8
EPOCHS = 6

def build_model(num_classes=2, lr=1e-4):
    base = Xception(weights='imagenet', include_top=False, pooling='avg',
                    input_shape=(IMG_SIZE_VIDEO[0], IMG_SIZE_VIDEO[1], 3))
    x = base.output
    x = Dropout(0.3)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=preds)

    for layer in base.layers[:-50]:
        layer.trainable = False
    for layer in base.layers[-50:]:
        layer.trainable = True

    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_split(split_file):
    items = []
    with open(split_file, "r") as f:
        for line in f:
            path, label = line.strip().split()
            label = 0 if label.lower() in ["real", "bonafide", "0"] else 1
            items.append((path, label))
    return items

def make_generator(items, target_size, batch_size, shuffle=True):
    n = len(items)
    while True:
        if shuffle:
            np.random.shuffle(items)
        for i in range(0, n, batch_size):
            batch_items = items[i:i+batch_size]
            X, y = [], []
            for path, label in batch_items:
                img = load_img(path, target_size=target_size)
                img_arr = img_to_array(img)
                X.append(img_arr)
                onehot = np.zeros(2)
                onehot[label] = 1
                y.append(onehot)
            X_processed = preprocess_frames_for_xception(np.array(X))
            yield X_processed, np.array(y)

def main():
    train_items = load_split(os.path.join(SPLITS_DIR, "video_train.txt"))
    val_items   = load_split(os.path.join(SPLITS_DIR, "video_val.txt"))
    test_items  = load_split(os.path.join(SPLITS_DIR, "video_test.txt"))

    train_gen = make_generator(train_items, IMG_SIZE_VIDEO, BATCH_SIZE, shuffle=True)
    val_gen   = make_generator(val_items, IMG_SIZE_VIDEO, BATCH_SIZE, shuffle=False)
    test_gen  = make_generator(test_items, IMG_SIZE_VIDEO, BATCH_SIZE, shuffle=False)
    
    steps_train = len(train_items) // BATCH_SIZE
    steps_val   = len(val_items) // BATCH_SIZE
    steps_test  = len(test_items) // BATCH_SIZE

    model = build_model()
    os.makedirs(MODELS_DIR, exist_ok=True)

    chk = ModelCheckpoint(VIDEO_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

    # Training with tqdm progress bar
    model.fit(train_gen,
              validation_data=val_gen,
              steps_per_epoch=steps_train,
              validation_steps=steps_val,
              epochs=EPOCHS,
              callbacks=[chk, rlr, TqdmCallback(verbose=1)])

    print("✅ Training finished. Best model saved to:", VIDEO_MODEL_PATH)

    print("\n🔎 Evaluating on test set...")
    for _ in tqdm(range(steps_test), desc="Testing", unit="batch"):
        next(test_gen)
    loss, acc = model.evaluate(test_gen, steps=steps_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
