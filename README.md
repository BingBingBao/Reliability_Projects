#  Reliability Projects:
Fault Diagnosis and Predictive Maintenance are core topics in Reliability Engineering. These techniques aim to prevent unexpected failures and optimize maintenance schedules, reducing downtime and repair costs in industrial systems.

This repository contains two predictive maintenance projects using NASA datasets: 
1. **Bearing Fault Diagnosis** → Analyzing vibration signals to detect faults in rotating machinery.
2. **Turbofan Engine Remaining Useful Life (RUL) Prediction** → Predicting the remaining time before engine failure.

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

To detect abnormal behavior in bearings before failure, I first apply **statistical anomaly detection techniques**.

####  **3-Sigma Method (Z-Score Based Anomaly Detection)**
One of the most common threshold-based anomaly detection methods is the **3-Sigma Rule**:

<p align="center">
  <img src="https://github.com/user-attachments/assets/11bbd5b5-d743-407f-b28c-c09b33923cca" width="48%">
  <img src="https://github.com/user-attachments/assets/4193e2db-0b01-434a-a366-3f286e5864d7" width="48%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/734f2086-0d7f-41c2-bb36-b6637f7bcae6" width="48%">
  <img src="https://github.com/user-attachments/assets/fa11144c-2a5a-48ce-92d0-a7dfb0667996" width="48%">
</p>

- Red dots indicate anomalies, which correspond to potential bearing failures.  
- The trend shows an increase in anomalies as the bearing degrades

####  **Box-Plot Method**

<p align="center">
  <img src="https://github.com/user-attachments/assets/c1d0e848-8c5e-4c24-975c-cef21d074692" width="48%">
  <img src="https://github.com/user-attachments/assets/31cd9c52-f1fb-4a32-bc5d-bcb5a0295967" width="48%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/8c77d97e-8a17-4350-a334-f7c0a708cddc" width="48%">
  <img src="https://github.com/user-attachments/assets/9b880c99-3e35-4abc-ad8c-dbd3b9b3aaf6" width="48%">
</p>


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


### 1. First, look at the sensor data
![image](https://github.com/user-attachments/assets/2422ca78-4f34-428b-8e6f-a113dbd4e7c4)

![image](https://github.com/user-attachments/assets/6441b0c0-6c96-4ccf-9ebf-063cc62eb536)

![image](https://github.com/user-attachments/assets/ebece7f1-df33-43bb-914a-bf38a4aa8ea9)

### 2. Survive Analysis
![image](https://github.com/user-attachments/assets/d2edebfe-e652-4899-9171-f96f70f419c6)

### 3. Results

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










