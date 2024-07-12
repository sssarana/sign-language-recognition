import tensorflow as tf
import numpy as np
import cv2
import os

label_names = []
with open('data\\processed\\label_names.txt', 'r') as file:
    label_names = [line.strip() for line in file]

# Print label names for verification
print("Label Names:", label_names)

# Preprocess a single image
def preprocess_image(image_path, img_size=224):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {image_path}")
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype('float32') / 255.0 
    img = np.expand_dims(img, axis=0) 
    return img

# Predict the class of a single image
def predict_image_tflite(image_path, interpreter, input_details, output_details):
    img = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions)
    predicted_label = label_names[predicted_class]
    # Debugging: Print raw prediction probabilities
    #print(f"Predictions for {image_path}: {predictions}")
    return predicted_label, confidence_score

# Load the TFLite model and allocate tensors
tflite_model_path = 'data\\models\\sign_lang.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on a new image
test_image_path = 'data\\raw\\asl_alphabet_test\\A_test.jpg'
predicted_label, confidence_score = predict_image_tflite(test_image_path, interpreter, input_details, output_details)
print(f"Predicted label: {predicted_label} with confidence: {confidence_score:.2f}")

# Predict the class of multiple images
def predict_images_tflite(image_dir, interpreter, input_details, output_details):
    results = {}
    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            predicted_label, confidence_score = predict_image_tflite(image_path, interpreter, input_details, output_details)
            results[filename] = (predicted_label, confidence_score)
    return results

# Test the model on a directory of new images
test_image_dir = 'data\\raw\\asl_alphabet_test'
results = predict_images_tflite(test_image_dir, interpreter, input_details, output_details)
for filename, (predicted_label, confidence_score) in results.items():
    print(f"Image: {filename}, Predicted label: {predicted_label}, Confidence: {confidence_score:.2f}")
