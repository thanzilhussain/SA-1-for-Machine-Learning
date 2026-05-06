# SA-1-for-Machine-Learning
# Fuel Consumption and CO₂ Emission Analysis
## *Program Developed by: THANZIL HUSSAIN A*
## *Register Number: 212225240169*

# program:
```python

# ==========================================
# Import necessary libraries
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==========================================
# Load dataset
# ==========================================
data = pd.read_csv("FuelConsumption.csv")

# Display first few rows
print("First 5 rows of dataset:")
print(data.head())

# ==========================================
# Q1. Scatter plot between Cylinder vs CO2Emission (green color)
# ==========================================
plt.figure(figsize=(6,4))
plt.scatter(data['CYLINDERS'], data['CO2EMISSIONS'], color='green')
plt.title("Cylinders vs CO2 Emission")
plt.xlabel("Number of Cylinders")
plt.ylabel("CO2 Emission")
plt.show()

# ==========================================
# Q2. Scatter plot: Compare Cylinder vs CO2Emission and EngineSize vs CO2Emission
# ==========================================
plt.figure(figsize=(6,4))
plt.scatter(data['CYLINDERS'], data['CO2EMISSIONS'], color='red', label='Cylinders')
plt.scatter(data['ENGINESIZE'], data['CO2EMISSIONS'], color='blue', label='Engine Size')
plt.title("Comparison: Cylinder & Engine Size vs CO2 Emission")
plt.xlabel("Feature Values")
plt.ylabel("CO2 Emission")
plt.legend()
plt.show()

# ==========================================
# Q3. Scatter plot: Compare Cylinder, EngineSize, and FuelConsumption_comb vs CO2Emission
# ==========================================
plt.figure(figsize=(6,4))
plt.scatter(data['CYLINDERS'], data['CO2EMISSIONS'], color='green', label='Cylinders')
plt.scatter(data['ENGINESIZE'], data['CO2EMISSIONS'], color='blue', label='Engine Size')
plt.scatter(data['FUELCONSUMPTION_COMB'], data['CO2EMISSIONS'], color='red', label='Fuel Consumption')
plt.title("Comparison: Multiple Features vs CO2 Emission")
plt.xlabel("Feature Values")
plt.ylabel("CO2 Emission")
plt.legend()
plt.show()

# ==========================================
# Q4. Model 1: Train model with 'CYLINDERS' as independent variable
# ==========================================
X1 = data[['CYLINDERS']]
y = data['CO2EMISSIONS']

X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
model1 = LinearRegression()
model1.fit(X1_train, y_train)

y_pred1 = model1.predict(X1_test)
acc1 = r2_score(y_test, y_pred1)
print("\nModel 1 Accuracy (Cylinders vs CO2Emission):", round(acc1, 3))

# ==========================================
# Q5. Model 2: Train model with 'FuelConsumption_comb' as independent variable
# ==========================================
X2 = data[['FUELCONSUMPTION_COMB']]
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)
model2 = LinearRegression()
model2.fit(X2_train, y_train)

y_pred2 = model2.predict(X2_test)
acc2 = r2_score(y_test, y_pred2)
print("Model 2 Accuracy (FuelConsumption_comb vs CO2Emission):", round(acc2, 3))

# ==========================================
# Q6. Train model on different train-test ratios
# ==========================================
ratios = [0.1, 0.2, 0.3, 0.4]
accuracies = []

for r in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=r, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    accuracies.append(score)
    print(f"Train-Test Split {int((1-r)*100)}:{int(r*100)} - Accuracy: {round(score, 3)}")

# ==========================================
# Plot accuracy vs train-test ratio
# ==========================================
plt.figure(figsize=(6,4))
plt.plot(ratios, accuracies, marker='o')
plt.title("Model Accuracy vs Train-Test Ratio")
plt.xlabel("Test Size Ratio")
plt.ylabel("R² Accuracy")
plt.grid(True)
plt.show()
```
# Output:
<img width="699" height="457" alt="Screenshot 2025-10-10 203851" src="https://github.com/user-attachments/assets/b463cd49-0313-4f86-a3bb-001878fe40d2" />
<img width="1004" height="738" alt="Screenshot 2025-10-10 203935" src="https://github.com/user-attachments/assets/cc72bd4a-85ff-46b2-b8a6-145dfce7d74d" />
<img width="1008" height="746" alt="Screenshot 2025-10-10 204001" src="https://github.com/user-attachments/assets/5eed334f-7b7d-461e-89fb-9247c481b353" />
<img width="1003" height="744" alt="Screenshot 2025-10-10 204026" src="https://github.com/user-attachments/assets/f65db900-8dbc-4894-86f6-2b36807c35c2" />
<img width="935" height="203" alt="Screenshot 2025-10-10 204048" src="https://github.com/user-attachments/assets/ca4f3133-f533-4aef-a42b-69a3fe69e7bc" />
<img width="1047" height="732" alt="Screenshot 2025-10-10 204118" src="https://github.com/user-attachments/assets/c85ee024-ac6f-4f96-a24d-16a5ae28075e" />
