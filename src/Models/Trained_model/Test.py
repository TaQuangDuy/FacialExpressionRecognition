import numpy as np
import joblib
import cv2

model_path = "xgboost_model.joblib"
xgb_classifier = joblib.load(model_path)
image_path = r"C:\Users\TGDD\Downloads\60798496_2438588776154361_803931806666588160_n.jpg"

image = cv2.imread(image_path, 0)

image = cv2.resize(image, (48, 48))

real_image = np.array(image, dtype=np.float32)
real_image = real_image.reshape(1, -1)
real_image /= 255.0
prediction = xgb_classifier.predict(real_image)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
predicted_emotion = emotion_labels[prediction[0]]
print("Predicted Emotion:", predicted_emotion)
