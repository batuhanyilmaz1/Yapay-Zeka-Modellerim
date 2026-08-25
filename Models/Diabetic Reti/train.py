"""
Diyabetik Retinopati Tespit Modeli (Eğitim Betiği)

Göz dibi (fundus) fotoğraflarını "DR" (diyabetik retinopati var) ve
"No_DR" (yok) olmak üzere ikili sınıflandırır. Omurga olarak önceden
eğitilmiş MobileNetV2 kullanılır; üzerine çok-başlı dikkat (multi-head
attention) katmanı eklenerek özellik haritalarının önemli bölgelere
odaklanması sağlanır.

Veri seti Kaggle üzerinden (pkdarabi/diagnosis-of-diabetic-retinopathy)
otomatik olarak indirilir. Sınıflar dengesiz olduğu için RandomOverSampler
ile azınlık sınıfı çoğaltılır.
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, GaussianNoise,
    Input, MultiHeadAttention, Reshape,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

sns.set_style("darkgrid")

# --- Sabitler -----------------------------------------------------------
DATASET_HANDLE = "pkdarabi/diagnosis-of-diabetic-retinopathy"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
CLASS_LABELS = ["DR", "No_DR"]
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "MobileNetV2-VGG16-DiabeticReti.h5")
RANDOM_STATE = 42


def download_dataset() -> str:
    """Kaggle'dan veri setini indirir ve eğitim klasörünün yolunu döndürür."""
    import kagglehub

    path = kagglehub.dataset_download(DATASET_HANDLE)
    train_path = os.path.join(path, "Diagnosis of Diabetic Retinopathy", "train")
    print(f"Veri seti indirildi: {path}")
    return train_path


def build_dataframe(train_path: str) -> pd.DataFrame:
    """Görüntü dosya yollarını ve etiketlerini içeren bir DataFrame oluşturur."""
    records = []
    for category in CLASS_LABELS:
        category_path = os.path.join(train_path, category)
        for image_name in os.listdir(category_path):
            records.append({
                "image_path": os.path.join(category_path, image_name),
                "label": category,
            })
    return pd.DataFrame(records)


def oversample(df: pd.DataFrame) -> pd.DataFrame:
    """Azınlık sınıfını çoğaltarak sınıf dengesizliğini giderir."""
    df = df.copy()
    df["category_encoded"] = (df["label"] == "No_DR").astype(int)

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    x_resampled, y_resampled = ros.fit_resample(df[["image_path"]], df["category_encoded"])

    df_resampled = pd.DataFrame(x_resampled, columns=["image_path"])
    df_resampled["category_encoded"] = y_resampled.astype(str)
    print("Oversampling sonrası sınıf dağılımı:")
    print(df_resampled["category_encoded"].value_counts())
    return df_resampled


def split_dataset(df: pd.DataFrame):
    """Veriyi %80 eğitim, %10 doğrulama, %10 test olarak katmanlı (stratified) böler."""
    train_df, temp_df = train_test_split(
        df, train_size=0.8, shuffle=True, random_state=RANDOM_STATE,
        stratify=df["category_encoded"],
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, shuffle=True, random_state=RANDOM_STATE,
        stratify=temp_df["category_encoded"],
    )
    return train_df, val_df, test_df


def build_generators(train_df, val_df, test_df):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    common_kwargs = dict(
        x_col="image_path", y_col="category_encoded",
        target_size=IMG_SIZE, class_mode="binary",
        color_mode="rgb", batch_size=BATCH_SIZE,
    )

    train_gen = train_datagen.flow_from_dataframe(train_df, shuffle=True, **common_kwargs)
    val_gen = test_datagen.flow_from_dataframe(val_df, shuffle=True, **common_kwargs)
    test_gen = test_datagen.flow_from_dataframe(test_df, shuffle=False, **common_kwargs)
    return train_gen, val_gen, test_gen


def create_model(input_shape=(224, 224, 3)) -> tf.keras.Model:
    """MobileNetV2 omurgası + çok-başlı dikkat katmanı ile ikili sınıflandırma modeli kurar.

    MobileNetV2, VGG16 tabanlı alternatife kıyasla daha az parametreyle
    benzer/daha iyi doğruluk verdiği için (bkz. README) tercih edilmiştir.
    """
    inputs = Input(shape=input_shape)
    base_model = MobileNetV2(weights="imagenet", input_tensor=inputs, include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]
    x = Reshape((height * width, channels))(x)
    x = MultiHeadAttention(num_heads=4, key_dim=channels)(x, x)
    x = Reshape((height, width, channels))(x)
    x = GaussianNoise(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = GaussianNoise(0.2)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, train_gen, val_gen, epochs=EPOCHS):
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(
        train_gen, validation_data=val_gen, epochs=epochs,
        callbacks=[early_stopping], verbose=1,
    )
    return history


def plot_training_history(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()


def evaluate_model(model, test_gen):
    test_labels = test_gen.classes
    predictions = model.predict(test_gen)
    predicted_labels = (predictions > 0.5).astype(int).flatten()

    print(classification_report(test_labels, predicted_labels, target_names=list(test_gen.class_indices.keys())))

    cm = confusion_matrix(test_labels, predicted_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["DR", "No DR"], yticklabels=["DR", "No DR"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def main():
    train_path = download_dataset()
    df = build_dataframe(train_path)
    df_resampled = oversample(df)
    train_df, val_df, test_df = split_dataset(df_resampled)
    train_gen, val_gen, test_gen = build_generators(train_df, val_df, test_df)

    print("Model oluşturuluyor...")
    model = create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    print("\nModel eğitiliyor...")
    history = train_model(model, train_gen, val_gen)
    plot_training_history(history)

    print("\nModel değerlendiriliyor...")
    evaluate_model(model, test_gen)

    model.save(MODEL_OUTPUT_PATH)
    print(f"\nModel kaydedildi: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
