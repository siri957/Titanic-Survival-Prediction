🚢 Titanic Survival Prediction Model
📌 Project Overview

This project builds a machine learning model to predict whether a passenger survived the Titanic disaster. The model is implemented using Python and Scikit-learn and follows a complete data science workflow including preprocessing, training, and evaluation.

📊 Dataset

Dataset: Titanic Dataset
Source: Kaggle

The dataset contains passenger information such as age, gender, ticket class, fare, and embarkation point.

⚙️ Approach
Data Loading
Loaded dataset using Pandas
Data Preprocessing
Handled missing values:
Age → Median imputation
Embarked → Mode imputation
Dropped unnecessary columns:
Cabin, Name, Ticket, PassengerId
Converted categorical variables:
Sex → Numerical encoding
Embarked → One-hot encoding
Feature Selection
Selected relevant features for prediction
Train-Test Split
80% training data
20% testing data
Model Training
Logistic Regression model was used
Prediction
Predicted survival on test dataset
Evaluation
Model evaluated using:
Accuracy
Mean Squared Error
R² Score
Confusion Matrix
Classification Report
📈 Model Performance
Accuracy: 81.01%
Mean Squared Error: 0.1899
R² Score: 0.2167
Confusion Matrix:
[[90 15]
 [19 55]]
Classification Report:
              precision    recall  f1-score   support

           0       0.83      0.86      0.84       105
           1       0.79      0.74      0.76        74

    accuracy                           0.81       179
   macro avg       0.81      0.80      0.80       179
weighted avg       0.81      0.81      0.81       179
🔍 Observations
Female passengers had higher survival probability
First-class passengers had better survival chances
Age and fare significantly influenced survival prediction
Logistic Regression performed well as a baseline model
✅ Conclusion

The Logistic Regression model achieved approximately 81% accuracy, demonstrating strong baseline performance for Titanic survival prediction. The model can be further improved using advanced algorithms such as Random Forest, Decision Trees, or Gradient Boosting.

🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
🚀 How to Run
python Titanic_Survival_Prediction.py

👨‍💻 Author
Sirichandana Kota
