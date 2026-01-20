"""
CAT EMOTION RECOGNITION SYSTEM - COMPLETE PROJECT
For VSCode/Local Python Environment

Prerequisites:
1. Python 3.8+
2. Dataset folders: catimage/ and cataudio/ with emotions: angry, fear, happy, sad
3. Install required packages (see below)
"""

# ============================================
# INSTALLATION REQUIREMENTS
# ============================================
"""
Run these commands in your terminal first:

pip install tensorflow
pip install librosa
pip install opencv-python
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install tqdm
pip install soundfile
"""

# ============================================
# MILESTONE 1: DATA PREPROCESSING
# ============================================

import os
import cv2
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf
from tensorflow.keras.models import load_model 

print("="*60)
print("MILESTONE 1: DATA PREPROCESSING")
print("="*60)

# ============================================
# 1.1 SET YOUR PATHS
# ============================================
# CHANGE THESE TO YOUR LOCAL PATHS
BASE_PATH = "C:/Users/sanka/OneDrive/Documents/CatEmotion_Ai_Projects"
  # ⭐ CHANGE THIS TO YOUR FOLDER
IMAGE_PATH = os.path.join(BASE_PATH, "catimage")
AUDIO_PATH = os.path.join(BASE_PATH, "cataudio")
PROCESSED_PATH = os.path.join(BASE_PATH, "preprocessed_data")

# Create output directory
os.makedirs(PROCESSED_PATH, exist_ok=True)

print(f"Base path: {BASE_PATH}")
print(f"Image path: {IMAGE_PATH}")
print(f"Audio path: {AUDIO_PATH}")

# Check if paths exist
if not os.path.exists(IMAGE_PATH):
    print(f"❌ ERROR: Image path not found: {IMAGE_PATH}")
    print("Please update IMAGE_PATH in the code")
    exit()

if not os.path.exists(AUDIO_PATH):
    print(f"❌ ERROR: Audio path not found: {AUDIO_PATH}")
    print("Please update AUDIO_PATH in the code")
    exit()

print(f"Image Classes: {os.listdir(IMAGE_PATH)}")
print(f"Audio Classes: {os.listdir(AUDIO_PATH)}")

# ============================================
# 1.2 PREPROCESS IMAGES
# ============================================
print("\n" + "="*60)
print("STEP 1: Preprocessing Images")
print("="*60)

IMG_SIZE = 128
emotion_folders = [f for f in os.listdir(IMAGE_PATH) if os.path.isdir(os.path.join(IMAGE_PATH, f))]
emotion_folders.sort()

print(f"Found emotion folders: {emotion_folders}")

# Create label mapping
label_map = {emotion.lower(): idx for idx, emotion in enumerate(emotion_folders)}
print(f"Label mapping: {label_map}")

X_images = []
y_images = []

for emotion in emotion_folders:
    emotion_folder = os.path.join(IMAGE_PATH, emotion)
    label = label_map[emotion.lower()]
    
    print(f"\nProcessing {emotion} (label={label})...")
    
    for img_name in tqdm(os.listdir(emotion_folder)):
        img_path = os.path.join(emotion_folder, img_name)
        
        # Read image
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Resize and convert to RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        X_images.append(img)
        y_images.append(label)

X_images = np.array(X_images, dtype='float32')
y_images = np.array(y_images, dtype='int32')

print(f"\n✅ Preprocessed Images: {X_images.shape}")
print(f"✅ Labels: {y_images.shape}")
print(f"✅ Label distribution: {np.bincount(y_images)}")

# Save preprocessed images
np.save(os.path.join(PROCESSED_PATH, "X_images.npy"), X_images)
np.save(os.path.join(PROCESSED_PATH, "y_images.npy"), y_images)
print(f"\n💾 Saved images to: {PROCESSED_PATH}")

# ============================================
# 1.3 PREPROCESS AUDIO (MFCC EXTRACTION)
# ============================================
print("\n" + "="*60)
print("STEP 2: Preprocessing Audio (MFCC)")
print("="*60)

MAX_LEN = 130
N_MFCC = 40

X_audio = []
y_audio = []

audio_emotions = [f for f in os.listdir(AUDIO_PATH) if os.path.isdir(os.path.join(AUDIO_PATH, f))]

for emotion in audio_emotions:
    folder = os.path.join(AUDIO_PATH, emotion)
    label = label_map[emotion.lower()]
    
    print(f"\nProcessing {emotion} audio (label={label})...")
    
    for file in tqdm(os.listdir(folder)):
        if not file.lower().endswith((".wav", ".mp3")):
            continue
        
        try:
            file_path = os.path.join(folder, file)
            
            # Load audio
            y_audio_sig, sr = librosa.load(file_path, sr=16000, duration=3)
            
            # Extract MFCC
            mfcc = librosa.feature.mfcc(y=y_audio_sig, sr=sr, n_mfcc=N_MFCC)
            
            # Pad or truncate
            if mfcc.shape[1] < MAX_LEN:
                mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])), mode='constant')
            else:
                mfcc = mfcc[:, :MAX_LEN]
            
            X_audio.append(mfcc.T)  # (time_steps, features)
            y_audio.append(label)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

X_audio = np.array(X_audio)
y_audio = np.array(y_audio)

print(f"\n✅ Audio MFCC shape: {X_audio.shape}")
print(f"✅ Labels: {y_audio.shape}")
print(f"✅ Label distribution: {np.bincount(y_audio)}")

# Save preprocessed audio
np.save(os.path.join(PROCESSED_PATH, "X_audio.npy"), X_audio)
np.save(os.path.join(PROCESSED_PATH, "y_audio.npy"), y_audio)
print(f"\n💾 Saved audio to: {PROCESSED_PATH}")

# ============================================
# 1.4 VISUALIZE SAMPLES
# ============================================
print("\n" + "="*60)
print("STEP 3: Visualizing Samples")
print("="*60)

# Show sample images
fig, axes = plt.subplots(1, len(emotion_folders), figsize=(15, 4))
fig.suptitle("Sample Images from Each Emotion Class")

for idx, emotion in enumerate(emotion_folders):
    class_indices = np.where(y_images == idx)[0]
    if len(class_indices) > 0:
        sample_img = X_images[class_indices[0]] / 255.0  # Normalize for display
        axes[idx].imshow(sample_img)
        axes[idx].set_title(f"{emotion}\n(Label: {idx})")
        axes[idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_PATH, "sample_images.png"))
print(f"✅ Saved visualization to: {os.path.join(PROCESSED_PATH, 'sample_images.png')}")
plt.show()

print("\n✅ MILESTONE 1 COMPLETE!")

# ============================================
# MILESTONE 2: MODEL TRAINING
# ============================================

print("\n" + "="*60)
print("MILESTONE 2: MODEL TRAINING")
print("="*60)

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ============================================
# 2.1 LOAD PREPROCESSED DATA
# ============================================
print("\n📂 Loading preprocessed data...")

X_images = np.load(os.path.join(PROCESSED_PATH, "X_images.npy"))
y_images = np.load(os.path.join(PROCESSED_PATH, "y_images.npy"))
X_audio = np.load(os.path.join(PROCESSED_PATH, "X_audio.npy"))
y_audio = np.load(os.path.join(PROCESSED_PATH, "y_audio.npy"))

print(f"Images: {X_images.shape}, Labels: {y_images.shape}")
print(f"Audio: {X_audio.shape}, Labels: {y_audio.shape}")

#from tensorflow.keras.models import load_model

image_model = load_model(os.path.join(PROCESSED_PATH, "best_image_model.keras"))
audio_model = load_model(os.path.join(PROCESSED_PATH, "best_audio_model.keras"))

print("✅ Pre-trained models loaded successfully")
# ============================================
# 2.2 PREPARE IMAGE DATA
# ============================================
print("\n🔧 Preparing image data...")

# Normalize
X_images = X_images / 255.0

# Split
X_img_train, X_img_test, y_img_train, y_img_test = train_test_split(
    X_images, y_images,
    test_size=0.2,
    random_state=42,
    stratify=y_images
)

NUM_CLASSES = len(np.unique(y_images))
y_img_train_cat = to_categorical(y_img_train, NUM_CLASSES)
y_img_test_cat = to_categorical(y_img_test, NUM_CLASSES)

print(f"✅ Train: {X_img_train.shape}, Test: {X_img_test.shape}")

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True
)

# ============================================
# 2.3 BUILD IMAGE MODEL
# ============================================
print("\n🖼️ Building Image Model (CNN + MobileNetV2)...")

base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

image_model = Model(inputs=base_model.input, outputs=output)

image_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("✅ Image model built")

# ============================================
# 2.4 TRAIN IMAGE MODEL - PHASE 1
# ============================================
print("\n🚀 Training Image Model - Phase 1 (frozen base)...")

callbacks_phase1 = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
]

history_phase1 = image_model.fit(
    train_datagen.flow(X_img_train, y_img_train_cat, batch_size=32),
    validation_data=(X_img_test, y_img_test_cat),
    epochs=15,
    callbacks=callbacks_phase1,
    verbose=1
)

# ============================================
# 2.5 FINE-TUNE IMAGE MODEL - PHASE 2
# ============================================
print("\n🔓 Fine-tuning Image Model - Phase 2...")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

image_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3),
    ModelCheckpoint(
        os.path.join(PROCESSED_PATH, "best_image_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

history_phase2 = image_model.fit(
    train_datagen.flow(X_img_train, y_img_train_cat, batch_size=32),
    validation_data=(X_img_test, y_img_test_cat),
    epochs=20,
    callbacks=callbacks_phase2,
    verbose=1
)

# ============================================
# 2.6 EVALUATE IMAGE MODEL
# ============================================
print("\n📊 Evaluating Image Model...")

loss, acc = image_model.evaluate(X_img_test, y_img_test_cat, verbose=0)
print(f"🎯 Image Model Test Accuracy: {acc*100:.2f}%")

y_pred_probs = image_model.predict(X_img_test)
y_pred = np.argmax(y_pred_probs, axis=1)

class_names = [emotion.capitalize() for emotion in emotion_folders]

print("\n📋 Classification Report:")
print(classification_report(y_img_test, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_img_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Image Model - Accuracy: {acc*100:.1f}%")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_PATH, "image_confusion_matrix.png"))
plt.show()

# ============================================
# 2.7 PREPARE AUDIO DATA
# ============================================
print("\n🔧 Preparing audio data...")

X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(
    X_audio, y_audio,
    test_size=0.2,
    random_state=42,
    stratify=y_audio
)

print(f"✅ Train: {X_a_train.shape}, Test: {X_a_test.shape}")

# ============================================
# 2.8 BUILD AUDIO MODEL
# ============================================
print("\n🎧 Building Audio Model (Bidirectional LSTM)...")

audio_model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(MAX_LEN, N_MFCC)),
    BatchNormalization(),
    Dropout(0.4),
    
    Bidirectional(LSTM(64)),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])

audio_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("✅ Audio model built")

# ============================================
# 2.9 TRAIN AUDIO MODEL
# ============================================
print("\n🚀 Training Audio Model...")

callbacks_audio = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ModelCheckpoint(
        os.path.join(PROCESSED_PATH, "best_audio_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

history_audio = audio_model.fit(
    X_a_train, y_a_train,
    validation_data=(X_a_test, y_a_test),
    epochs=40,
    batch_size=32,
    callbacks=callbacks_audio,
    verbose=1
)

# ============================================
# 2.10 EVALUATE AUDIO MODEL
# ============================================
print("\n📊 Evaluating Audio Model...")

loss_audio, acc_audio = audio_model.evaluate(X_a_test, y_a_test, verbose=0)
print(f"🎯 Audio Model Test Accuracy: {acc_audio*100:.2f}%")

y_pred_audio = audio_model.predict(X_a_test)
y_pred_audio_class = np.argmax(y_pred_audio, axis=1)

print("\n📋 Classification Report:")
print(classification_report(y_a_test, y_pred_audio_class, target_names=class_names))

# Confusion Matrix
cm_audio = confusion_matrix(y_a_test, y_pred_audio_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_audio, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Audio Model - Accuracy: {acc_audio*100:.1f}%")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_PATH, "audio_confusion_matrix.png"))
plt.show()

# ============================================
# 2.11 TRAINING HISTORY
# ============================================
print("\n📈 Plotting training history...")

def plot_history(history, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title(f'{title} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title(f'{title} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_PATH, filename))
    plt.show()

plot_history(history_phase2, "Image Model", "image_training_history.png")
plot_history(history_audio, "Audio Model", "audio_training_history.png")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"📊 Image Model Accuracy: {acc*100:.2f}%")
print(f"📊 Audio Model Accuracy: {acc_audio*100:.2f}%")
print("="*60)

if acc >= 0.70:
    print("✅ Image Model: PASSED (>70%)")
else:
    print("❌ Image Model: Need improvement")

if acc_audio >= 0.70:
    print("✅ Audio Model: PASSED (>70%)")
else:
    print("❌ Audio Model: Need improvement")

print("="*60)
print(f"\n💾 All results saved to: {PROCESSED_PATH}")
