import cv2
import numpy as np
import yaml
import tensorflow as tf

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def preprocess_frame(frame, input_shape):
    frame_resized = cv2.resize(frame, (input_shape[0], input_shape[1]))
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized.astype(np.float32), axis=0) 

if __name__ == "__main__":
    model_path = 'data\\models\\sign_lang.tflite'
    config_path = 'config\\camera_config.yaml'
    labels_path = 'data\\processed\\label_names.txt'

    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    config = load_config(config_path)
    input_shape = tuple(config['model']['input_shape'])

    cap = cv2.VideoCapture("tcp://112.168.132.120:34808")  
    
    if not cap.isOpened():
        print("Error: Unable to open camera")
        exit()
    
    with open(labels_path, 'r') as f:
        label_names = f.read().splitlines()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame")
            break
        
        processed_frame = preprocess_frame(frame, input_shape)
        
        interpreter.set_tensor(input_details[0]['index'], processed_frame)
        
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100
        
        label = label_names[predicted_class]
        cv2.putText(frame, f"Predicted: {label}, Confidence: {confidence:.2f}%", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
