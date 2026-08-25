"""
Zatürre (Pneumonia) Tespit Modeli (Tahmin Betiği)

Eğitilmiş Unet-Pneumonia.h5 modelini yükler ve TEST klasöründeki
görüntüler üzerinde tahmin yapıp sonucu görselleştirir.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "Unet-Pneumonia.h5")
TEST_DIR = os.path.join(os.path.dirname(__file__), "TEST")
IMG_SIZE = (256, 256)
CLASS_LABELS = {0: "Normal", 1: "Pneumonia"}
NUM_SAMPLES = 4


def list_test_images(test_dir: str) -> list:
    return [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:NUM_SAMPLES]


def predict_single(model, img_path: str):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array / 255.0, axis=0)

    preds = model.predict(img_batch, verbose=0)
    if preds.shape[1] == 1:
        prob = preds[0][0]
        class_idx = 1 if prob > 0.5 else 0
        confidence = prob * 100 if class_idx == 1 else (1 - prob) * 100
    else:
        class_idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_idx] * 100)

    return img_array, CLASS_LABELS.get(class_idx, "Unknown"), confidence


def predict_and_plot(model, image_paths):
    plt.figure(figsize=(18, 5))
    for i, img_path in enumerate(image_paths):
        img_array, predicted_label, confidence = predict_single(model, img_path)

        plt.subplot(1, len(image_paths), i + 1)
        plt.imshow(img_array.astype("uint8"))
        plt.axis("off")

        title_color = "red" if predicted_label == "Pneumonia" else "green"
        plt.title(
            f"Pred: {predicted_label}\nConf: %{confidence:.2f}\nFile: {os.path.basename(img_path)}",
            fontsize=9, color=title_color, fontweight="bold",
        )
    plt.tight_layout()
    plt.show()


def main():
    model = load_model(MODEL_PATH)
    image_paths = list_test_images(TEST_DIR)
    predict_and_plot(model, image_paths)


if __name__ == "__main__":
    main()
