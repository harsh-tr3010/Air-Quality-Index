# AQI-Project
Project Overview

This project focuses on analyzing and predicting the Air Quality Index (AQI) using multiple machine learning and deep learning regression techniques. The goal is to compare different models and evaluate their performance in predicting AQI based on environmental parameters.

The project includes data preprocessing, visualization, model training, evaluation, and comparison using both traditional ML algorithms and neural networks.

📂 Project Structure
├── Data/
│   └── (Dataset files used for AQI prediction)
│
├── ANN.ipynb
├── LinearRegression.ipynb
├── LassoRegression.ipynb
├── KNearestNeighborRegressor.ipynb
├── DecisionTreeRegressor.ipynb
├── RandomForestRegressor.ipynb
├── XgboostRegressor.ipynb
│
├── Extract_combine.py
├── Html_script.py
├── Plot_AQI.py
│
└── README.md

🧠 Techniques & Models Used

The following regression and learning techniques are implemented and compared:

🔹 Machine Learning Models

Linear Regression

Lasso Regression

K-Nearest Neighbors (KNN) Regressor

Decision Tree Regressor

Random Forest Regressor

XGBoost Regressor

🔹 Deep Learning

Artificial Neural Network (ANN)

⚙️ Supporting Scripts

Extract_combine.py
Used for data extraction, cleaning, and combining multiple datasets into a single usable format.

Plot_AQI.py
Generates visualizations for AQI trends and comparisons.

Html_script.py
Creates HTML-based output or visual representation of AQI data and predictions.

📊 Workflow

Data Collection & Preprocessing

Raw AQI data extracted and combined

Handling missing values and normalization

Exploratory Data Analysis

AQI visualization and trend analysis

Model Training

Training multiple ML and DL regression models

Hyperparameter tuning where applicable

Model Evaluation

Performance comparison using error metrics

Identification of best-performing model

Visualization & Reporting

Graphical AQI plots

HTML-based output generation

📈 Evaluation Metrics

The models are evaluated using standard regression metrics such as:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

🛠️ Technologies Used

Python

NumPy

Pandas

Matplotlib / Seaborn

Scikit-learn

TensorFlow / Keras

XGBoost

🚀How to Run the Project

Clone the repository:

git clone <repository-url>


Install required dependencies:

pip install -r requirements.txt


Run preprocessing scripts:

python Extract_combine.py


Execute model notebooks:

Open .ipynb files in Jupyter Notebook or VS Code

Run cells sequentially

Generate AQI plots:

python Plot_AQI.py

📌 Conclusion

This project demonstrates how different machine learning and deep learning regression models perform on AQI prediction tasks. It provides a comparative study to identify the most accurate and reliable model for air quality forecasting.
