import tensorflow as tf
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb, gray2rgb
import os

model1 = tf.keras.models.load_model("model/model_huber.keras")
model2 = tf.keras.models.load_model("model/coloring_model.keras")


def preprocess_image(path, force_grayscale=False):
    image = Image.open(path).convert('RGB')
    if force_grayscale:
        image = image.convert('L').convert('RGB')
    image = image.resize((224, 224))
    img_np = np.array(image)/ 255.0

    lab = rgb2lab(img_np)
    L = lab[:, :, 0]
    L = L / 100.0

    L_tensor = L[np.newaxis, ..., np.newaxis]
    return L_tensor, L

def postprocess_and_save(L_original, ab_pred, save_path):
    ab_pred = ab_pred[0]
    ab_pred = ab_pred * 128

    L = L_original * 100
    lab = np.zeros((224, 224, 3))
    lab[:, :, 0] = L
    lab[:, :, 1:] = ab_pred

    rgb = lab2rgb(lab)
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    img_out = Image.fromarray(rgb_uint8)
    img_out.save(save_path)



if __name__ == "__main__":
    img_path = input("Podaj ścieżkę do obrazu: ").strip()
    if not os.path.exists(img_path):
        print("Nie znaleziono pliku.")
        exit()

    mode = input("Czy obraz jest czarno-biały (tak/nie)? ").strip().lower()
    force_grayscale = (mode == "tak")

    
    L_tensor, L_original = preprocess_image(img_path, force_grayscale)


    ab_pred = model1.predict(L_tensor) 
    output_path = os.path.splitext(img_path)[0] + "_colorized_model1.png"
    postprocess_and_save(L_original, ab_pred, output_path)

    ab_pred = model2.predict(L_tensor) 
    output_path = os.path.splitext(img_path)[0] + "_colorized_model2.png"
    postprocess_and_save(L_original, ab_pred, output_path)
        
                  
   