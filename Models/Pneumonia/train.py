"""
Zatürre (Pneumonia) Tespit Modeli (Eğitim Betiği)

Göğüs röntgeni (X-Ray) görüntülerini "Normal" ve "Pneumonia" olmak üzere
ikili sınıflandırır. Omurga olarak U-Net (ResNet-34) kullanılır; U-Net'in
çıkışı tam bağlı (dense) katmanlarla sınıflandırma başlığına bağlanır.

Veri seti Kaggle üzerinden (vivek468/beginner-chest-xray-image-classification)
otomatik olarak indirilir.

Not: segmentation-models kütüphanesi gerektirir (`pip install segmentation-models`).
"""

import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

import matplotlib.pyplot as plt
import segmentation_models as sm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Sabitler -----------------------------------------------------------
DATASET_HANDLE = "vivek468/beginner-chest-xray-image-classification"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 20
BACKBONE = "resnet34"
CLASS_LABELS = ["Normal", "Pneumonia"]
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "Unet-Pneumonia.h5")


def download_dataset() -> tuple:
    """Kaggle'dan veri setini indirir; eğitim ve test klasör yollarını döndürür."""
    import kagglehub

    path = kagglehub.dataset_download(DATASET_HANDLE)
    train_dir = os.path.join(path, "chest_xray", "train")
    test_dir = os.path.join(path, "chest_xray", "test")
    print(f"Veri seti indirildi: {path}")
    return train_dir, test_dir


def build_generators(train_dir: str, test_dir: str):
    """Eğitim için tıbbi görüntülere uygun hafif veri artırma, test için
    yalnızca ölçekleme uygulayan generator'lar oluşturur."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=(0.95, 1.05),
        brightness_range=(0.9, 1.1),
        horizontal_flip=True,
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=True,
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=True,
    )
    return train_gen, test_gen


def create_model(input_shape=(256, 256, 3)) -> tf.keras.Model:
    """U-Net (ResNet-34) omurgasını ikili sınıflandırma başlığıyla birleştirir."""
    conv_base = sm.Unet(backbone_name=BACKBONE, input_shape=input_shape)

    model = Sequential([
        conv_base,
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dense(16, activation="relu"),
        BatchNormalization(),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adadelta(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, train_gen, val_gen, epochs=EPOCHS):
    early_stop = EarlyStopping(
        monitor="val_loss", mode="min", patience=5,
        restore_best_weights=True, verbose=1,
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
    )
    history = model.fit(
        train_gen, epochs=epochs, validation_data=val_gen,
        callbacks=[early_stop, lr_scheduler], verbose=1,
    )
    return history


def plot_training_history(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], "r", label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], "b", label="Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], "r", label="Training Loss")
    plt.plot(history.history["val_loss"], "b", label="Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.show()


def main():
    train_dir, test_dir = download_dataset()
    train_gen, val_gen = build_generators(train_dir, test_dir)

    print("Model oluşturuluyor...")
    model = create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.summary()

    print("\nModel eğitiliyor...")
    history = train_model(model, train_gen, val_gen)
    plot_training_history(history)

    model.save(MODEL_OUTPUT_PATH)
    print(f"\nModel kaydedildi: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
