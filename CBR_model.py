
"""
Case-Based Reasoning (CBR) model used with Weighted KNN-style retrieval.
- Reads only the named columns you pass in (e.g., ['X1','X2','X3','CLASS'])
- Uses provided per-feature similarity functions (sim_fns)
- Uses learned weights from ReliefF (weights must align with feature columns)
- Implements a simple leave-one-out test_accuracy() for classification

Replace your existing file with this one.
"""

from pathlib import Path
import pandas as pd
import numpy as np

class CBR_model:
    def __init__(self, sim_fns, file_path, cols, weights, k):
        """
        Parameters
        ----------
        sim_fns : dict[str, callable]
            e.g. {'X1': lambda x,y: 1 if x==y else 0, ...}
        file_path : str
            Path to the Excel file
        cols : list[str]
            Column names in order, last column is the class/label
            e.g. ['X1','X2','X3','CLASS']
        weights : list[float] or np.ndarray
            Feature weights aligned to cols[:-1]
        k : int
            Number of neighbors
        """
        self.sim_fns = sim_fns
        self.cols = cols
        self.feature_cols = cols[:-1]
        self.label_col = cols[-1]
        self.k = int(k)
        self.weights = np.array(weights, dtype=float)

        if len(self.weights) != len(self.feature_cols):
            raise ValueError(
                f"weights length {len(self.weights)} must equal number of features {len(self.feature_cols)}"
            )

        # ---------- NEW (robust, name-based, engine specified) ----------
        p = Path(file_path)
        if not p.is_absolute():
            p = Path(__file__).parent / p

        # Read only the requested columns by NAME (avoids out-of-bounds)
        self.df = pd.read_excel(p, header=0, usecols=self.cols, engine="openpyxl").copy()

        # ---------- OLD (positional indices; causes out-of-bounds on 4-col sheets) ----------
        # from pandas import read_excel
        # excel = read_excel(file_path, header=0, usecols=range(100))

        # Basic checks
        missing = [c for c in self.cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Columns not found in sheet: {missing}. Present: {list(self.df.columns)}")

        # Keep matrices for fast access
        self.X = self.df[self.feature_cols].values
        self.y = self.df[self.label_col].values

        # Normalize weights to sum to 1 (optional but common)
        wsum = self.weights.sum()
        if wsum > 0:
            self.weights = self.weights / wsum

    # ---------- Similarity helpers ----------

    def _case_similarity(self, row_a, row_b):
        """
        Weighted per-feature similarity using provided sim_fns.
        Assumes sim_fns[feat] returns a value in [0,1] (e.g., 1 if equal else 0 for categorical).
        """
        total = 0.0
        for i, feat in enumerate(self.feature_cols):
            sim = self.sim_fns[feat](row_a[i], row_b[i])
            total += self.weights[i] * sim
        return total  # higher is more similar

    def _knn_neighbors(self, idx):
        """
        Find indices of the top-k most similar cases to case idx (excluding itself).
        """
        target = self.X[idx]
        sims = []
        for j in range(self.X.shape[0]):
            if j == idx:
                continue
            s = self._case_similarity(target, self.X[j])
            sims.append((j, s))
        # sort by similarity desc, then take top-k
        sims.sort(key=lambda t: t[1], reverse=True)
        return sims[: self.k]

    def _knn_predict_label_for_idx(self, idx):
        """
        Predict label for case at index idx using weighted vote of its k nearest neighbors.
        Weight each neighbor's class by its similarity.
        """
        neighbors = self._knn_neighbors(idx)
        if not neighbors:
            # Edge case: dataset of size 1
            return self.y[idx]

        vote = {}
        for j, sim in neighbors:
            lab = self.y[j]
            vote[lab] = vote.get(lab, 0.0) + sim

        # choose class with largest accumulated similarity
        pred = max(vote.items(), key=lambda t: t[1])[0]
        return pred

    # ---------- Public API ----------

    def predict_one(self, feature_row_dict):
        """
        Predict label for a single external case passed as a dict of feature->value.
        Example:
            feature_row_dict = {'X1': 1, 'X2': 0, 'X3': 2}
        """
        # Build a row vector in the correct feature order
        row = np.array([feature_row_dict[f] for f in self.feature_cols], dtype=object)

        # Compute similarity vs all training rows
        sims = []
        for j in range(self.X.shape[0]):
            s = self._case_similarity(row, self.X[j])
            sims.append((j, s))

        sims.sort(key=lambda t: t[1], reverse=True)
        neighbors = sims[: self.k]

        vote = {}
        for j, sim in neighbors:
            lab = self.y[j]
            vote[lab] = vote.get(lab, 0.0) + sim

        pred = max(vote.items(), key=lambda t: t[1])[0]
        return pred

    def test_accuracy(self):
        """
        Leave-one-out evaluation:
        For each case i, predict its class using the remaining cases as the case base.
        Returns accuracy (0..1) and prints a small summary.
        """
        if self.X.shape[0] <= 1:
            print("Not enough cases to evaluate accuracy.")
            return 1.0

        correct = 0
        n = self.X.shape[0]

        # Precompute a full similarity matrix once (optional optimization)
        # For clarity, compute on the fly here.
        for i in range(n):
            pred = self._knn_predict_label_for_idx(i)
            if pred == self.y[i]:
                correct += 1

        acc = correct / n
        print(f"CBR leave-one-out accuracy: {acc:.4f}  ({correct}/{n})")
        return acc