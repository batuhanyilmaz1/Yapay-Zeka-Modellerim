"""
Beyin Tümörü Sınıflandırma Modeli (Tahmin Betiği)

Eğitilmiş Unet-BrainTumor.h5 modelini yükler ve TEST klasöründeki
görüntüler üzerinde tahmin yapıp sonucu görselleştirir.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "Unet-BrainTumor.h5")
TEST_DIR = os.path.join(os.path.dirname(__file__), "TEST")
IMG_SIZE = (256, 256)
CLASS_INDICES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
NUM_SAMPLES = 4


def list_test_images(test_dir: str) -> list:
    return [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:NUM_SAMPLES]


def predict_and_plot(model, image_paths):
    plt.figure(figsize=(16, 4))
    for i, img_path in enumerate(image_paths):
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array / 255.0, axis=0)

        pred_probs = model.predict(img_batch, verbose=0)
        pred_class = int(np.argmax(pred_probs, axis=1)[0])
        confidence = float(np.max(pred_probs) * 100)
        predicted_name = CLASS_INDICES.get(pred_class, f"Sınıf {pred_class}")

        plt.subplot(1, len(image_paths), i + 1)
        plt.imshow(img_array.astype("uint8"))
        plt.axis("off")
        plt.title(
            f"Pred: {predicted_name}\nConf: %{confidence:.2f}\n{os.path.basename(img_path)}",
            fontsize=10, color="blue", fontweight="bold",
        )
    plt.tight_layout()
    plt.show()


def main():
    model = load_model(MODEL_PATH)
    image_paths = list_test_images(TEST_DIR)
    predict_and_plot(model, image_paths)
    print(f"Toplam {len(image_paths)} fotoğraf için tahmin tamamlandı.")


if __name__ == "__main__":
    main()
