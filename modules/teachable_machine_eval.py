
import cv2
import os
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps  # Install pillow instead of PIL
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from modules.draw_chinese_text import put_chinese_text

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


def load_model(model_path):
    return tf.saved_model.load(model_path)


def load_image_for_model(img_path):

    # Replace this with the path to your image
    image = Image.open(img_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    RGB_data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    RGB_data[0] = normalized_image_array
    return RGB_data


def evaluate_model(model, test_folder):
    y_true = []
    y_pred = []
    class_names = sorted(os.listdir(test_folder))

    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(test_folder, class_name)
        if not os.path.isdir(class_folder):
            continue

        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img_array = load_image_for_model(img_path)
            predictions = model(img_array)
            predicted_class = np.argmax(predictions)
            confidence_score = predictions[0][predicted_class]

            # Show windows
            text_to_put = f'{class_idx}->{predicted_class}: score={confidence_score}%'
            show_image = cv2.imread(img_path)
            show_image = cv2.resize(
                show_image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            drawed_image = put_chinese_text(show_image, text_to_put)
            cv2.imshow("Show Test Image", drawed_image)
            keyboard_input = cv2.waitKey(1000)

            # 27 is the ASCII for the esc key on your keyboard.
            if keyboard_input == 27:
                exit()

            y_true.append(class_idx)
            y_pred.append(predicted_class)

    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    return accuracy


if __name__ == '__main__':

    # 設定模型與測試資料夾
    MODEL_PATH = "models\pue_tm_enhance_model/tm_meeting_camera"  # 修改為你的模型路徑
    TEST_IMAGES_FOLDER = 'datasets\pue_with_dot_dataset/test'  # 測試圖片資料夾
    IMAGE_SIZE = 224  # 根據 Teachable Machine 訓練時的輸入尺寸設置

    # 執行測試
    tm_model = load_model(MODEL_PATH)
    eval_score = evaluate_model(tm_model, TEST_IMAGES_FOLDER)
