'''
ReliefF implementation from: https://doi.org/10.1023/A:1025667309714
'''

from pandas import read_excel
import pandas as pd
import numpy as np
from random import randrange
from math import exp
from random import randrange
from math import exp
from pathlib import Path

class ReliefF:

    def __init__(self, diff_fns, file_path, cols, m = None, k=3, range_num=100):
        self.diff_fns = diff_fns
        self.file_path = file_path
        self.cols = cols
        self.m = m
        self.k = k

        # Resolve path relative to this file (robust to VS Code working dir)
        p = Path(file_path)
        if not p.is_absolute():
            p = Path(__file__).parent / p

        # 1) Read the sheet ONCE and only the requested columns BY NAME
        df = pd.read_excel(p, header=0, usecols=self.cols, engine="openpyxl")

        # Optional sanity check: make sure all requested cols exist
        missing = [c for c in self.cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in sheet: {missing}. Present: {list(df.columns)}")

        # 2) Shuffle rows (if you want randomness)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # 3) Split features/label
        feature_cols = self.cols[:-1]      # ['X1','X2','X3']
        label_col    = self.cols[-1]       # 'CLASS'

        # If X1/X2/X3 are categorical like your diff_fn assumes (==/!=),
        # keep them as-is; if they are numeric, ensure numeric dtype:
        # df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="ignore")

        self.X = df[feature_cols].values
        self.y = df[label_col].values

        # If m is None, set it to number of rows (or another sensible default)
        if self.m is None:
            self.m = len(df)

        self.no_of_instances = self.X.shape[0]
        self.no_of_features = self.X.shape[1]
        self.attrib_weights = [0] * self.no_of_features

        # (Optional) quick peek
        # print("X shape:", self.X.shape, "y shape:", self.y.shape)

        range_num = min(range_num, 4)  # since your sheet has only 4 columns
        excel = read_excel(file_path, header=0, usecols=range(range_num))
        data = excel[cols].sample(frac=1).values

        # X is all features, y is classified value
        self.X = data[:, 0:-1]
        self.y = data[:, -1]
        
        print(self.X)
        print(self.y)

        self.no_of_instances = self.X.shape[0]
        self.no_of_features = self.X.shape[1]
        # Initialize weights to 0
        self.attrib_weights = [0] * self.no_of_features
        

    def find_nearest_hits(self,X, y, ins, k):
        """
        Gets the nearest k hit indexes
        X: feature matrix
        y: value vector
        ins: the instance index
        k: number of k nearest hits
        """
        hit_idx = []
        target = y[ins]

        for i in range(X.shape[0]):
            if y[i] == target and i != ins:
                hit_idx.append(i)
        
        diff_idx = [abs(x - ins) for x in hit_idx]
        nearest_hit_idx = np.argsort(diff_idx)[:k]
        
        nearest_hits = [hit_idx[l] for l in nearest_hit_idx]
        return nearest_hits

    def find_nearest_misses(self, X, y, ins, k, C):
        """
        Gets the nearest k miss indexes
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
    
    def prior(self, C, Ri_class):
        """
        Gets the prior probability
        C: class 1
        Ri_class: class 2
        """
        y_list = self.y.tolist()
        p_C = y_list.count(C) / self.no_of_instances
        p_Ri_class = y_list.count(Ri_class) / self.no_of_instances
        prior = p_C / (1-p_Ri_class)
        return prior
    
    def diff(self, A, I1, I2):
        """
        Gets the diff value based on lambdas defined in diff_fns
        """
        return self.diff_fns[A](I1,I2)

    def normalize_weights(self, w_arr):
        """
        Normalizes the weights calculated
        """
        # sum_val = sum(w_arr)
        # minx = min(w_arr)
        # maxx = max(w_arr)

        # norm = [(x-minx)/(maxx - minx) for x in w_arr]
        
        exp_list = [exp(x) for x in w_arr]
        sum_ = sum(exp_list)
        print(w_arr)
        print(exp_list)
        softmax = [x/sum_ for x in exp_list]

        return softmax
        #return [number / sum_val for number in w_arr]

    
    def calculate_weights(self):
        """
        Calculates the weights for each attributes from ReliefF method
        """
        # Weight calculation
        for i in range(self.m):
            nearest_misses_arr = []
            # ALl avaialble classes
            classes = set(self.y)
            classes = list(classes)
            
            ins = randrange(self.no_of_instances)
            Ri = self.X[ins]
            #print(Ri)
            
            Ri_class = self.y[ins]
            # Find k nearest hits
            nearest_hits = self.find_nearest_hits(self.X,self.y,ins,self.k)
            
            # Remove the current class, need unmatched classes for misses
            classes.remove(self.y[ins])
            
            # Iterating through classes to find the misses
            for c in classes:
                # Find k nearest misses
                nearest_misses = self.find_nearest_misses(self.X,self.y,ins,self.k,c )
                
                nearest_misses_arr.append(nearest_misses)
            
            # Calculate nearest hits calculation
            for A in range(self.no_of_features):
                hit_calc = 0
                attrib = self.cols[A]        
                for j in range(self.k):
                    #print('nearest_hits',nearest_hits)
                    Hj = self.X[nearest_hits[j]]
                    hit_calc += self.diff(attrib,Ri[A],Hj[A]) / (self.m*self.k)
                    #print(attrib,Ri[A],Hj[A],hit_calc)
                    
            # Calculate nearest misses calculation
                miss_calc = 0
                                
                for idx, nearest_miss in enumerate(nearest_misses_arr):
                    miss_class = classes[idx]
                    #print(miss_class, nearest_miss)
                    attrib = self.cols[A]        
                    for j in range(self.k):
                        #print('nearest_miss',nearest_miss)
                        Mj = self.X[nearest_miss[j]]
                        miss_calc += self.diff(attrib,Ri[A],Mj[A]) / (self.m*self.k)
                    
                    #print("miss_class,Ri_class",miss_class,Ri_class)
                    miss_calc = miss_calc * self.prior(miss_class,Ri_class)
                    
                # Update weight of the atttribute A
                
                self.attrib_weights[A] = self.attrib_weights[A] - hit_calc + miss_calc
            
            print("attrib_weights", self.attrib_weights)

        normalized_weights = self.normalize_weights(self.attrib_weights)
        #print(normalized_weights)
        return normalized_weights