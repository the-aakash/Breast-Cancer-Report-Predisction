import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

try:
    df = pd.read_csv("cancer.csv") 
except:
    print("Error: 'cancer.csv' file not Found. Please check the Folder.")

# 2. Data Cleaning [cite: 933, 934, 945]
# 'id' aur 'Unnamed: 32' (NaN column) ko hatana
cols_to_drop = ['id', 'Unnamed: 32']
x = df.drop([c for c in cols_to_drop if c in df.columns] + ['diagnosis'], axis=1)
y = df['diagnosis'].map({'M': 1, 'B': 0}) # Malignant ko 1 aur Benign ko 0 banana [cite: 576]

# 3. Data Split (70% Training, 30% Testing) [cite: 711, 759]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print(f"Total Samples: {len(df)}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}\n")

# 4. Models Training aur Scoring
models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier()
}

results = {}

for name, model in models   .items():
    model.fit(X_train, y_train) # Machine ko sikhana [cite: 770, 784]
    predictions = model.predict(X_test) # Prediction lena [cite: 693]
    acc = accuracy_score(y_test, predictions)
    results[name] = acc
    print(f"{name} Accuracy: {acc*100:.2f}%")

# 5. Visualization (Result Graph) [cite: 857]
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.show()