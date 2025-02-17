# Reliability_Projects

## Project1: Remaining Useful Life (RUL) Prediction – NASA Turbofan Engine Data
### Introduction
This project focuses on predicting the Remaining Useful Life (RUL) of aircraft engines using the NASA C-MAPSS dataset. 
The dataset provides time-series sensor readings from turbofan engines operating under various conditions, allowing for the development of predictive maintenance models.

The goal is to build machine learning (ML) and deep learning (CNN) models to estimate how many cycles an engine has before failure, enabling proactive maintenance and minimizing operational downtime.

### Data
In turbofan engine degradation monitoring, multiple sensors capture various physical properties of the engine over time. These sensor readings reflect engine wear and degradation trends, which are crucial for predicting the Remaining Useful Life (RUL).

Dataset Structure: 
- Training Set: Each engine is monitored from start until failure. The model learns patterns of degradation to predict failure trends.
- Test Set: Each engine is monitored without failure information. The final recorded state of the engine is used for model evaluation.
- RUL Set: Contains the actual Remaining Useful Life (RUL) for each engine in the test set, serving as the ground truth for performance assessment.

<p align="center">
    <img src="https://github.com/user-attachments/assets/f12668d5-3eb5-437f-b469-0df3d9729416" width="45%">
    <img src="https://github.com/user-attachments/assets/6aba93bc-b8f0-485c-b0ef-9e3e023b90d9" width="45%">
</p>

### Models:
#### traditional ML models:
- Linear Regression (Ridge/Lasso) for baseline analysis.
- Random Forest for capturing non-linear degradation trends.
- XGBoost for advanced feature interactions and boosting.
#### DL model:
- CNN model with a sliding window of size 30
- Sliding window is to capture spatial patterns in the sensor reading

## Some Results:

### Machine Learning Model Performance

The table below summarizes the RMSE and Score of different machine learning models used for RUL prediction.

| Model              | RMSE     | Score        |
|--------------------|---------|-------------|
| **XGBoost**       | 21.929   | 2463.859097 |
| **Random Forest** | 22.121   | 3322.241690 |
| **Ridge Regression** | 23.874 | 2573.890882 |
| **Linear Regression** | 23.874 | 2574.114833 |
| **Lasso Regression**  | 23.877 | 2574.848880 |

![image](https://github.com/user-attachments/assets/bf186083-386f-435e-93a6-49fdc11bd148)

The XGBoost model demonstrates a good ability to capture RUL trends, as seen by the alignment between predicted (orange) and actual (blue) values. However, it can be seen that it overestimates the RUL value in many cases. 

### CNN Model + sliding window Appraoch

The CNN model aims to learn degradation patterns from sequential sensor data, leveraging local feature extraction for better prediction accuracy.

Currently the work has not been finished, requiring further adjustment and fine-tuning

Preliminary Results:

Below is an initial plot comparing actual vs. predicted RUL for Engine 1 using the CNN model:

![image](https://github.com/user-attachments/assets/1e17a843-77b8-4fc7-8ba7-329529d9b4aa)

