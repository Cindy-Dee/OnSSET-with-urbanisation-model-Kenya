import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Load the data
urbanisation = pd.read_csv('urbanisationmarch.csv')

# Extract predictors and target
predictor_columns = [
    "Distance_grid", 
    "Electrified_Population", 
    "Distance_to_Urban", 
    "Elevation", 
    "Major_Road_Dist", 
    "Population_density", 
    "RiverLakes_Dist"
]

# Prepare X and y
X = urbanisation[predictor_columns]
y = urbanisation['IsUrban']

# Remove rows with missing values
X_clean = X.dropna()
y_clean = y[X_clean.index]

# Compute correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = X_clean.corr()
sns.heatmap(
    correlation_matrix.abs(), 
    annot=True, 
    cmap='coolwarm', 
    linewidths=0.5
)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_chart.png', dpi=600)
plt.close()

# Check class imbalance
print('Class distribution before balancing:')
print(y_clean.value_counts())
print(f'Imbalance ratio: {y_clean.value_counts()[0]/y_clean.value_counts()[1]}')

# Split data initially
X_train_initial, X_test_initial, y_train_initial, y_test_initial = train_test_split(
    X_clean, y_clean, test_size=0.2, stratify=y_clean, random_state=42
)

# Apply SMOTE for balancing
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train_initial, y_train_initial)

# Undersample majority class if still imbalanced
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_balanced, y_balanced = undersampler.fit_resample(X_balanced, y_balanced)

# Split balanced data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression(
    penalty='l2', 
    solver='lbfgs', 
    max_iter=1000, 
    class_weight='balanced'
)
lr_model.fit(X_train_scaled, y_train)

# Predict probabilities
y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Find optimal threshold using F1 score
from sklearn.metrics import f1_score

thresholds = np.linspace(0, 1, 100)
f1_scores = [f1_score(y_test, y_pred_proba >= t) for t in thresholds]
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f'Optimal threshold: {optimal_threshold}')

# Predictions using optimal threshold
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=['Rural', 'Urban'], 
    yticklabels=['Rural', 'Urban']
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=600)
plt.close()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=600)
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig('precision_recall_curve.png', dpi=600)
plt.close()

# Save model and results
import joblib
joblib.dump(lr_model, 'urbanisation_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Predict probabilities for entire dataset
urbanisation_new = pd.read_excel('missing del.xlsx')
urbanisation_filled = urbanisation_new.fillna(urbanisation_new.mean())

X_new = urbanisation_filled[predictor_columns]
X_new_scaled = scaler.transform(X_new)

y_pred_proba_new = lr_model.predict_proba(X_new_scaled)[:, 1]
urbanisation_new['PredictedProbability'] = y_pred_proba_new

urbanisation_new.to_csv('updated_urbanisation_with_probabilities.csv', index=False)
