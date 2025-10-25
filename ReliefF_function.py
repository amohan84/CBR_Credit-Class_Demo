'''
ReliefF implementation from: https://doi.org/10.1023/A:1025667309714
'''

from pandas import read_excel
import numpy as np
from random import randrange

def find_nearest_hits(X, y, ins, k):
    hit_idx = []
    target = y[ins]
    for i in range(X.shape[0]):
        if y[i] == target and i != ins:
            hit_idx.append(i)
    
    diff_idx = [abs(x - ins) for x in hit_idx]
    nearest_hit_idx = np.argsort(diff_idx)[:k]
    
    #print(nearest_hit_idx)
    
    nearest_hits = [hit_idx[l] for l in nearest_hit_idx]
    return nearest_hits
    #print("Nearest Hits",nearest_hits)

def find_nearest_misses(X, y, ins, k, C):
    """
    Gets the nearest k hit indexes
    X: feature matrix
    y: value vector
    ins: the instance index
    k: number of k nearest hits
    C: Class which is not class of Ri
    """
    
    miss_idx = []
    target = y[ins]
    for i in range(X.shape[0]):
        if y[i] == C:
            miss_idx.append(i)
    #hit_idx.sort()
    
    diff_idx = [abs(x - ins) for x in miss_idx]
    
    nearest_miss_idx = np.argsort(diff_idx)[:k]
    
    #print(nearest_hit_idx)
    
    nearest_miss = [miss_idx[l] for l in nearest_miss_idx]
    return nearest_miss
    #print("Nearest Hits",nearest_hits)


def prior(C,Ri_class):
    """
    Gets the prior probability
    C: class 1
    Ri_class: class 2
    """
    y_list = y.tolist()
    p_C = y_list.count(C) / no_of_instances
    p_Ri_class = y_list.count(Ri_class) / no_of_instances
    prior = p_C / (1-p_Ri_class)
    return prior

def diff(A, I1, I2):
    """
    Gets the diff value based on lambdas defined in diff_fns
    """
    return diff_fns[A](I1,I2)

def normalize_weights(w_arr):
    sum_val = sum(w_arr)
    
    return [number / sum_val for number in w_arr]

file_path = 'test.xlsx'
cols = [
    'image_category', 
    'predicted_class',
    'frog_935_sim',
    'CLASS'
    # ,'saliency_matrix'
]

diff_fns = {
        'image_category':
            lambda x, y: 0 if x == y else 1,
        'predicted_class':
            lambda x, y: 0 if x == y else 1,
        'frog_935_sim':
            lambda x, y: abs(x-y)
        ,'saliency_matrix':
            lambda x,y: np.linalg.norm(x-y)
    }

# Times Run
m = 1000

# Use k misses and hits
k = 3
# ---------- Hyper Parameters End Here ----------

# --- Load from Excel File ---
excel = read_excel(file_path, header=0, usecols=range(5))
data = excel[cols].values

# X is all features, y is classified value
X = data[:,0:-1]
y = data[:,-1]

no_of_instances = X.shape[0]
no_of_features = X.shape[1]
# Initialize weights to 0
attrib_weights = [0] * no_of_features
attrib_weights

# Weight calculation
for i in range(m):
    nearest_misses_arr = []
    # ALl avaialble classes
    classes = set(y)
    classes = list(classes)
    
    ins = randrange(no_of_instances)
    Ri = X[ins]
    print(Ri)
    
    Ri_class = y[ins]
    # Find k nearest hits
    nearest_hits = find_nearest_hits(X,y,ins,k)
    
    # Remove the current class, need unmatched classes for misses
    classes.remove(y[ins])
    
    # Iterating through classes to find the misses
    for c in classes:
        # Find k nearest misses
        nearest_misses = find_nearest_misses(X,y,ins,k,c )
        
        nearest_misses_arr.append(nearest_misses)
        
    # Calculate nearest hits calculation
    for A in range(no_of_features):
        hit_calc = 0
        attrib = cols[A]        
        for j in range(k):
            Hj = X[nearest_hits[j]]
            hit_calc += diff(attrib,Ri[A],Hj[A]) / (m*k)
            #print(attrib,Ri[A],Hj[A],hit_calc)
            
     # Calculate nearest misses calculation
        miss_calc = 0
                        
        for idx, nearest_miss in enumerate(nearest_misses_arr):
            miss_class = classes[idx]
            #print(miss_class, nearest_miss)
            attrib = cols[A]        
            for j in range(k):
                Mj = X[nearest_miss[j]]
                miss_calc += diff(attrib,Ri[A],Mj[A]) / (m*k)
            
            #print("miss_class,Ri_class",miss_class,Ri_class)
            miss_calc = miss_calc * prior(miss_class,Ri_class)
            
        # Update weight of the atttribute A
        
        attrib_weights[A] = attrib_weights[A] - hit_calc + miss_calc
    
    print("attrib_weights",attrib_weights)

    normalized_weights = normalize_weights(attrib_weights)
    print(normalized_weights)