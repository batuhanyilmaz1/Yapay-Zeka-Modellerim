"""
HemaDeep - Kan Kanseri Hücre Sınıflandırma Modeli (Eğitim Betiği)

Mikroskobik kan hücresi görüntülerini 5 sınıfa ayıran bir CNN modelini
eğitir: basophil, erythroblast, monocyte, myeloblast, seg_neutrophil.

Veri seti Kaggle üzerinden (sumithsingh/blood-cell-images-for-cancer-detection)
otomatik olarak indirilir.
"""

import os
import warnings

import kagglehub
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# --- Sabitler -----------------------------------------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
CLASS_LABELS = ["basophil", "erythroblast", "monocyte", "myeloblast", "seg_neutrophil"]
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "HemaDeep.h5")
RANDOM_STATE = 42


def download_dataset() -> str:
    """Kaggle'dan veri setini indirir ve yerel yolunu döndürür."""
    dataset_path = kagglehub.dataset_download(
        "sumithsingh/blood-cell-images-for-cancer-detection"
    )
    print(f"Veri seti indirildi: {dataset_path}")
    return dataset_path


def build_dataframe(dataset_path: str) -> pd.DataFrame:
    """Görüntü dosya yollarını ve sınıf etiketlerini içeren bir DataFrame oluşturur."""
    records = []
    for class_name in CLASS_LABELS:
        class_dir = os.path.join(dataset_path, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(".jpg"):
                records.append({
                    "Image": os.path.join(class_dir, filename),
                    "Target": class_name,
                })
    return pd.DataFrame(records)


def split_dataset(df: pd.DataFrame):
    """Her sınıftan 700 eğitim örneği alır, kalanı val/test olarak eşit böler."""
    splits = {}
    for target in df["Target"].unique():
        df_target = df[df["Target"] == target]
        train_df, temp_df = train_test_split(
            df_target, train_size=700, random_state=RANDOM_STATE, shuffle=True
        )
        val_df, test_df = train_test_split(
            temp_df, train_size=0.5, random_state=RANDOM_STATE, shuffle=True
        )
        splits[target] = {"train": train_df, "val": val_df, "test": test_df}

    train_df = pd.concat([splits[t]["train"] for t in splits])
    val_df = pd.concat([splits[t]["val"] for t in splits])
    test_df = pd.concat([splits[t]["test"] for t in splits])
    return train_df, val_df, test_df


def build_generators(train_df, val_df, test_df):
    """Eğitim için veri artırma (augmentation) uygulanmış, doğrulama/test için
    yalnızca ölçekleme yapan görüntü üreticileri (generator) oluşturur."""
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        rescale=1.0 / 255,
    )

    common_kwargs = dict(
        x_col="Image",
        y_col="Target",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="grayscale",
    )

    train_generator = datagen.flow_from_dataframe(dataframe=train_df, **common_kwargs)
    val_generator = datagen.flow_from_dataframe(dataframe=val_df, **common_kwargs)
    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df, shuffle=False, **common_kwargs
    )
    return train_generator, val_generator, test_generator


def create_model(input_shape=(128, 128, 1)) -> tf.keras.Model:
    """3 konvolüsyon bloğundan oluşan basit bir CNN sınıflandırma modeli kurar."""
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(CLASS_LABELS), activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, train_generator, val_generator, epochs=EPOCHS):
    """Erken durdurma ve öğrenme oranı azaltma callback'leri ile modeli eğitir."""
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
    ]
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def plot_training_history(history):
    """Eğitim/doğrulama doğruluk ve kayıp eğrilerini çizer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["accuracy"], label="Training")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history["loss"], label="Training")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """Sınıf etiketleriyle birlikte karışıklık matrisini çizer."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
    )
    plt.title("Confusion Matrix - Blood Cell Cancer")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_generator):
    """Modeli test verisi üzerinde değerlendirir ve tahmin sınıflarını döndürür."""
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    predictions = model.predict(test_generator)
    predicted_classes = predictions.argmax(axis=1)
    return test_loss, test_accuracy, predicted_classes


def main():
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    dataset_path = download_dataset()
    df = build_dataframe(dataset_path)
    train_df, val_df, test_df = split_dataset(df)
    train_generator, val_generator, test_generator = build_generators(train_df, val_df, test_df)

    print("Model oluşturuluyor...")
    model = create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
    model.summary()

    print("\nModel eğitiliyor...")
    history = train_model(model, train_generator, val_generator)
    plot_training_history(history)

    print("\nModel değerlendiriliyor...")
    test_loss, test_accuracy, predicted_classes = evaluate_model(model, test_generator)
    print(f"Test doğruluğu: {test_accuracy:.4f}")

    plot_confusion_matrix(test_generator.classes, predicted_classes)
    print("\nSınıflandırma Raporu:")
    print(classification_report(
        test_generator.classes, predicted_classes, target_names=CLASS_LABELS, digits=4
    ))

    model.save(MODEL_OUTPUT_PATH)
    print(f"\nModel kaydedildi: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
