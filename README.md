#  Reliability Projects:
Fault Diagnosis and Predictive Maintenance are core topics in Reliability Engineering. These techniques aim to prevent unexpected failures and optimize maintenance schedules, reducing downtime and repair costs in industrial systems.

This repository contains several **vibration-based condition monitoring** projects that apply machine learning, deep learning, anomaly detection, and time-series forecasting to improve industrial reliability.

1. **Bearing Fault Diagnosis**
2. **Turbofan Engine Remaining Useful Life (RUL) Prediction**
3. **Circuit Breaker Vibration Signal Condition Monitoring**: Monitor short-duration, non-stationary vibration signals during circuit breaker operations. Signals are first transformed into Kurtogram images, then classified using Convolutional Neural Networks (CNNs). 

These projects apply **machine learning, deep learning, anomaly detection, and time-series forecasting** to improve industrial reliability.


## Project 1: NASA Bearing Fault Diagnosis 

### **Data**
- **Source**: NASA Prognostics Data Repository
- **Description**: Run-to-failure vibration dataset for 4 bearings under real operating conditions.
                   At the end of the experiment, an outer race failure occurred in Bearing 1. This dataset is ideal for fault diagnosis, anomaly detection, and failure prediction in rotating machinery.

### **Objective**
- Detect **early signs of degradation** using time-domain statistical features.
- Apply **PCA, anomaly detection, and exponential modeling** to predict failure.
- Train ML & DL models for **fault classification**.

### **Methods**
**Feature Extraction**: Extract time-domain features including *RMS*, *Standard Deviation*, *Kurtosis*, *P2P*, *clearance*, *entropy* etc. 12 features;

**Dimensionality Reduction**: PCA (Principal Component Analysis) 

**Anomaly Detection**: 3-sigma Method, Local Outlier Factor (LOF)

**Failure Prediction**: Exponential degradation modeling   

### 1. First, take a look at raw data 

The dataset contains vibration recordings from **4 bearings** over their entire lifespan. Below shows the **first** and **last** sensor readings:

- **Left:** The first 4 sensor recordings → Bearings in **normal condition**.
- **Right:** The last 4 sensor recordings → Bearings at the **end of life** (Bearing 1 failed).

<p align="center">
  <img src="https://github.com/user-attachments/assets/9b3eb4af-3218-439f-b5e3-6db67e697a11" width="48%">
  <img src="https://github.com/user-attachments/assets/be11f7b9-3d0c-46a3-b21a-cd2ddc6de2b2" width="48%">
</p>

### 2. Time-domain feature Extraction

To understand the degradation patterns, I extract **statistical time-domain features** from vibration signals. These features help **detect early degradation** and track the health of the bearings.

Below plots the **feature trends** for all four bearings throughout their operational lifetime:

The extracted features show clear degradation patterns, which can be used for predictive maintenance and early fault detection. By analyzing these trends, we can identify progressive wear before complete failure, allowing for timely interventions and reducing unplanned downtime.

<p align="center">
  <img src="https://github.com/user-attachments/assets/86b803be-184b-45bc-9e6a-336a8ded18101" width="48%">
  <img src="https://github.com/user-attachments/assets/047d2697-dac9-4a61-8444-8b7ef384a276" width="48%">
</p>

### 3. Anomaly Detection

To detect abnormal behaviour in bearings before failure, I apply two methods **Isolation Forest** and **Autoencoder**.

![image](https://github.com/user-attachments/assets/ae0bd008-a0a0-4196-8e87-6bad295ff082)
![image](https://github.com/user-attachments/assets/d21953ad-9628-47dd-bada-9eab961c7af4)




### 4. PCA-Based Health Indicator & Exponential Failure Prediction

Principal Component Analysis (PCA) is applied to extract a single health indicator (PC1) from multiple vibration features. 

The degradation trend is then modeled using an exponential model to predict failure.

- Blue Line: The smoothed PC1 degradation trend, derived from vibration signals.
- Red Line: The exponential model fit, predicting the remaining useful life (RUL).
- Dashed Line: The failure threshold, where the degradation crosses the critical limit.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c900108c-b60c-4b0d-ae34-5cbf20c5fabb" width="55%">
</p>

### 5. Apply this model to Bearing 2, 3, 4

Bearing 2: 
- Fitted parameters: a=0.000001, b=0.014899
- Predicted failure at cycle: 1008.45

Bearing 3: 
- Fitted parameters: a=0.000001, b=0.015650
- Predicted failure at cycle: 924.04
  
Bearing 4:
- No clear degradation trend
- no prediction


Prediction plots: 

<p align="center">
  <img src="https://github.com/user-attachments/assets/00070b7c-e3c6-4a51-818d-a0e86bd29f70" width="48%">
  <img src="https://github.com/user-attachments/assets/552ae586-0daf-4bee-a608-cf8cdc87ff66" width="48%">
</p>




---




##  Project 2: NASA Turbofan Engine RUL Prediction

### **Data**
- **Source**: [CMAPSS Turbofan Dataset](https://www.nasa.gov/content/prognostics-data-repository-cmaps)
- **Description**: Simulated turbofan engine degradation with multiple sensor readings.

### **Objective**
- Predict the **Remaining Useful Life (RUL)** of turbofan engines.
- Compare **traditional ML (Random Forest, Ridge Regression) vs. deep learning (LSTM, CNN)** models.

### **Methods**

 **Machine Learning**: Ridge Regression, Random Forest  
**Deep Learning**: LSTM, CNN with time-series sliding window  
**Evaluation Metrics**: RMSE, MAE, Score Function  

### **Key Results**
- **LSTM-based RUL model achieved high accuracy** in failure predictions.
- **Feature selection improved model interpretability**.
- **Sliding window approach enhanced time-series forecasting performance**.


### 1. Survive Analysis
![image](https://github.com/user-attachments/assets/d2edebfe-e652-4899-9171-f96f70f419c6)

### 2. Model Prediction Results

| Model                | RMSE ↓ (Lower is better) | Score ↓ (Lower is better) |
|----------------------|----------------------|----------------------|
| **LSTM**            | **13.69**            | **359.45**           |
| **CNN**             | **15.43**            | **474.34**           |
| **XGBoost**         | 21.93                | 2463.86              |
| **Random Forest**   | 22.12                | 3322.24              |
| **Ridge Regression**| 23.87                | 2573.98              |
| **Linear Regression**| 23.87               | 2574.11              |
| **Lasso Regression**| 23.88                | 2574.85              |

### Key Insights
- LSTM has the best performance, achieving the lowest RMSE (13.69) and the best score (359.45), indicating it captures sequential dependencies effectively in vibration signals.

- CNN is the second-best model, also performing significantly better than traditional ML models.
  
- XGBoost was the best ML model, but it's still far behind deep learning approaches.




### CNN model Prediction:
![image](https://github.com/user-attachments/assets/878aba30-5557-4d3c-8ffa-8dd0b0d89b74)

![image](https://github.com/user-attachments/assets/96f0c798-523d-4e72-b8c8-5aa1c4aa5ae9)


### LSTM model prediction:

![image](https://github.com/user-attachments/assets/40a242ad-2d27-445a-ae81-92ff0a16bdd2)

![image](https://github.com/user-attachments/assets/6f459d3a-3da2-4bb0-b82b-e9bdb0136d13)


## Project 3: Circuit Breaker Vibration Signal Condition Monitoring

### **Data**
**Source**: ETHZurich High Voltage Laboratory - Circuit Breaker Vibration Dataset

This dataset contains vibration signals recorded from circuit breakers during open/close operations, capturing non-stationary, transient behaviors critical for early fault detection.

### **Methodology**

- Vibration signals are transformed into Kurtogram images (frequency-based representations).

- A Convolutional Neural Network (CNN) is trained to classify healthy vs. faulty operation.

### **Results**
![image](https://github.com/user-attachments/assets/45c9292f-8d1b-40e0-ace3-a9fefe13115f)

Classification Report:
               
               precision    recall  f1-score   support

           0       1.00      0.74      0.85       200
           1       0.79      1.00      0.88       200

    accuracy                           0.87       400
   macro avg       0.90      0.87      0.87       400
weighted avg       0.90      0.87      0.87       400


- Accuracy: 87%
- Macro Avg F1-score: 0.87
- **Model successfully detects all faulty cases (Recall = 1.00)**
- A well-designed CNN can effectively learn discriminative patterns from kurtogram images, but generalization may be affected by signal variability across different circuit breakers. Fine-tuning or incorporating domain adaptation techniques may help improve healthy case recall and reduce false positives.


