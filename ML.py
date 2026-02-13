import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (replace 'Global_Dataset.xlsx' with your file path)
df = pd.read_excel("Global_Dataset.xlsx", sheet_name="Sheet1")

# Preprocess data
df_filtered = df[['CVSS-V2', 'SEVERITY', 'CWE-ID']].dropna()

# Encode categorical variables
label_enc_severity = LabelEncoder()
df_filtered['SEVERITY'] = label_enc_severity.fit_transform(df_filtered['SEVERITY'])

label_enc_cwe = LabelEncoder()
df_filtered['CWE-ID'] = label_enc_cwe.fit_transform(df_filtered['CWE-ID'])

# Convert target to numeric
df_filtered['CVSS-V2'] = pd.to_numeric(df_filtered['CVSS-V2'], errors='coerce')

# Remove any remaining NaN values
df_filtered = df_filtered.dropna()

# Define features and target
X = df_filtered[['SEVERITY', 'CWE-ID']]
y = df_filtered['CVSS-V2']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.bar(range(len(y_test[:50])), y_test[:50], label="Actual", alpha=0.6)
plt.bar(range(len(y_pred[:50])), y_pred[:50], label="Predicted", alpha=0.6)
plt.xlabel("Sample Index")
plt.ylabel("CVSS-V2 Score")
plt.title("Actual vs Predicted CVSS-V2 Scores using KNN")
plt.legend()
plt.show()

# Display results
print(f"Accuracy (R² Score): {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

