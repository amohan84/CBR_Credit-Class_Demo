from ReliefF import ReliefF
import numpy as np
from CBR_model import CBR_model


file_path = '54_instance_raw_data_credit_score.xlsx'
cols = [
    'X1'
    ,'X2'
    ,'X3'
    ,'CLASS'
    # ,'saliency_matrix'
]

diff_fns = {
        'X1':
            lambda x, y: 0 if x == y else 1,
        'X2':
            lambda x, y: 0 if x == y else 1,
        'X3':
            lambda x, y: 0 if x == y else 1
    }

sim_fns = {
        'X1':
            lambda x, y: 1 if x == y else 0,
        'X2':
            lambda x, y: 1 if x == y else 0,
        'X3':
            lambda x, y: 1 if x == y else 0
    }

# Times Run
m = 150  

# Use k misses and hits
k = 3

relief = ReliefF(diff_fns,file_path, cols, m, k)
weights = relief.calculate_weights()
print(weights)


# Weight learned from ReliefF_function : 10000 iterations normalized

w = weights

#> 77.77%
#w = [0.3450572418958393, 0.3437714298686596, 0.3111713282355012]  
# Use k misses and hits
k = 3

cbr_model = CBR_model(sim_fns, file_path, cols,w,k)

cbr_model.test_accuracy()

# --- Interactive Prediction from Command Prompt ---
print("\n--- Credit Class Prediction ---")
print("Enter applicant details below (use exact categories):")
print("Income Level: Low / Medium / High")
print("Employment Type: Salaried / Self_Employed / Contract / Unemployed")
print("Loan Purpose: Home / Auto / Education / Personal / Medical")

# Get input from user
x1 = input("Enter Income Level (X1): ")
x2 = input("Enter Employment Type (X2): ")
x3 = input("Enter Loan Purpose (X3): ")

# Build query
query = {'X1': x1, 'X2': x2, 'X3': x3}

# Predict using the trained model
pred = cbr_model.predict_one(query)
label_map = {1: "Good Credit", 0: "Bad Credit"}
print("\nPredicted CLASS:", pred, f"({label_map.get(pred, pred)})")

print(w)