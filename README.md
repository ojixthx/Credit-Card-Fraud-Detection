# Credit-Card-Fraud-Detection
Detect fraudulent transactions from imbalanced credit card data using machine learning and advanced sampling techniques like SMOTE, ADASYN, and RandomUnderSampler.


🛡️ Credit Card Fraud Detection
Detect fraudulent transactions from imbalanced credit card data using machine learning and advanced sampling techniques like SMOTE, ADASYN, and RandomUnderSampler.

📂 Project Structure
bash
Copy
Edit
.
├── data/
│   └── creditcard.csv           # Original dataset (highly imbalanced)
├── notebooks/
│   └── creditcard_fraud.ipynb   # Jupyter notebook for training & evaluation
├── models/
│   └── trained_model.pkl        # Saved model (optional)
├── README.md
├── requirements.txt
└── fraud_detection.py           # Main Python script (optional)
📊 Dataset
Source: Kaggle - Credit Card Fraud Detection

Details: European cardholders' transactions in September 2013.

Size: 284,807 transactions (492 frauds)

🧠 Models Used
Logistic Regression

Random Forest

XGBoost (optional)

Decision Tree

⚖️ Handling Imbalanced Data
This dataset is highly imbalanced. To improve model performance, we used:

SMOTE (Synthetic Minority Oversampling Technique)

ADASYN (Adaptive Synthetic Sampling)

Random Under Sampling (RUS)

These techniques help to create a balanced training dataset for better generalization.

🚀 How to Run
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Jupyter notebook

bash
Copy
Edit
jupyter notebook notebooks/creditcard_fraud.ipynb
Or run the script (if available):

bash
Copy
Edit
python fraud_detection.py
📈 Evaluation Metrics
Confusion Matrix

Precision, Recall, F1-score

ROC-AUC Score

PR-AUC Score

🧪 Sampling Techniques Code Snippet
python
Copy
Edit
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# Example usage:
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
📌 Requirements
nginx
Copy
Edit
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
📚 References
imblearn documentation

Scikit-learn docs

Kaggle dataset

🧑‍💻 Author
Ojith Adithya – LinkedIn | GitHub




