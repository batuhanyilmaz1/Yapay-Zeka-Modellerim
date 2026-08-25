"""
HemaDeep - Kan Kanseri Hücre Sınıflandırma Modeli (Tahmin Betiği)

Eğitilmiş HemaDeep.h5 modelini yükler ve TEST klasöründeki örnek
görüntüler üzerinde rastgele seçilmiş birkaç tanesi için tahmin yapıp
sonucu görselleştirir.
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "HemaDeep.h5")
TEST_DIR = os.path.join(os.path.dirname(__file__), "TEST")
IMG_SIZE = (128, 128)
CLASS_INDICES = {
    0: "Basophil",
    1: "Erythroblast",
    2: "Monocyte",
    3: "Myeloblast",
    4: "Seg Neutrophil",
}
NUM_SAMPLES = 5


def list_test_images(test_dir: str) -> list:
    return [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def predict_and_plot(model, image_paths):
    plt.figure(figsize=(18, 5))
    for i, img_path in enumerate(image_paths):
        img = image.load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array / 255.0, axis=0)

        pred_probs = model.predict(img_batch, verbose=0)
        pred_class = int(np.argmax(pred_probs, axis=1)[0])
        confidence = float(np.max(pred_probs) * 100)
        predicted_name = CLASS_INDICES.get(pred_class, f"Sınıf {pred_class}")

        plt.subplot(1, len(image_paths), i + 1)
        plt.imshow(img_array.squeeze(), cmap="gray")
        plt.axis("off")
        plt.title(
            f"Pred: {predicted_name}\nConf: %{confidence:.2f}\n{os.path.basename(img_path)}",
            fontsize=9, color="blue", fontweight="bold",
        )
    plt.tight_layout()
    plt.show()


def main():
    model = load_model(MODEL_PATH)
    print(f"Modelin beklediği giriş boyutu: {model.input_shape}")

    all_images = list_test_images(TEST_DIR)
    sample_images = random.sample(all_images, min(NUM_SAMPLES, len(all_images)))
    predict_and_plot(model, sample_images)


if __name__ == "__main__":
    main()
