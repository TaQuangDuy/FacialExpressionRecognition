# FacialExpressionRecognition
## Overview
Facial expression recognition enables more intuitive human-machine interactions and has applications in diverse domains such as psychology and human-computer interaction, which brings a huge advantage to numerous fields such as health care, education... etc.  

In this project, we investigate the problem of improving prediction accuracy through the application of various advanced methods. Specifically, we explore the implementation of Random Forest (RF), K-Nearest Neighbors (KNN), and XGBoost Classifier as individual classifiers. 

To further enhance accuracy, we utilize ensemble learning algorithms: Bagging and boosting (AdaBoost) techniques. These approaches aim to reduce overfitting and combine diverse predictions for improved performance. Through empirical evaluations, we assess the effectiveness of these methods in achieving higher accuracy. 
## Setup
1, Get data:
Raw data (Images): - Fer 2013: https://www.kaggle.com/msambare/fer2013

2, Preprocess data:
- Package: OpenCV, NumPy, Pandas, and Albumentations
- Preprocess: src/preprocess/preprocess_data.py

3, Train models:
- Package: sklearn, matplotlib, joblib
- Train models: src/models/knn.py, src/models/XGBoost.py, src/models/random_forest.py
## Collaborators
| Name             | Student ID | Email                               |
|------------------|------------|-------------------------------------|
| Vu Tung Linh     | 20210523   | Linh.VT210523@sis.hust.edu.vn        |
| Ta Quang Duy     | 20214884   | Duy.TQ214884@sis.hust.edu.vn         |
| Dao Ha Xuan Mai  | 20210562   | Mai.DHX20210562@sis.hust.edu.vn      |
| Nguyen Viet Dung | 20214883   | Dung.NV20214883@sis.hust.edu.vn      |
