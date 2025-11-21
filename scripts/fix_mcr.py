import os

def fix_mcr_indentation():
    filepath = "rashomon/rashomon_set.py"
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if "for j in range(self._d):" in line:
            # Find the one inside model_class_reliance
            # Check context: previous line has "Permutation importance"
            if i > 0 and "Permutation importance" in lines[i-1]:
                start_idx = i
        
        if "importance_matrix[s_idx, j] = float(np.mean(perm_scores))" in line:
            end_idx = i
            if start_idx != -1 and end_idx > start_idx:
                break
    
    if start_idx != -1 and end_idx != -1:
        print(f"Replacing lines {start_idx+1} to {end_idx+1}")
        
        new_block = """            for j in range(self._d):
                perm_scores = np.zeros(n_permutations, dtype=float)

                for p in range(n_permutations):
                    Xp = X.copy()

                    if perm_mode == "iid":
                        # Standard permutation
                        rng.shuffle(Xp[:, j])
                    elif perm_mode == "residual":
                        # Permute residuals from prediction without feature j
                        theta_minus_j = theta_s.copy()
                        theta_minus_j[j] = 0.0
                        pred_minus_j = X @ theta_minus_j
                        residual_j = X[:, j] - pred_minus_j / (theta_s[j] + 1e-12)
                        residual_j_perm = residual_j.copy()
                        rng.shuffle(residual_j_perm)
                        Xp[:, j] = residual_j_perm + pred_minus_j / (theta_s[j] + 1e-12)
                    elif perm_mode == "conditional":
                        # Conditional permutation (simplified: bin-based)
                        # Bin samples by correlated features and permute within bins
                        n_bins = max(3, int(np.sqrt(X.shape[0])))
                        # Use mean of other features for binning
                        other_mean = np.mean(np.delete(X, j, axis=1), axis=1)
                        bins = np.digitize(other_mean, np.linspace(other_mean.min(), other_mean.max(), n_bins))
                        for b in range(1, n_bins + 1):
                            mask = bins == b
                            if np.sum(mask) > 1:
                                Xp[mask, j] = rng.permutation(Xp[mask, j])
                    else:
                        raise ValueError(f"Unknown perm_mode: {perm_mode}")

                    # Score with permuted feature
                    if self.estimator == "logistic":
                        scores_p = Xp @ theta_s
                        preds_p = (scores_p > 0.0).astype(int)
                        score_p = float(np.mean((preds_p == y.astype(int)).astype(float)))
                    else:
                        preds_p = Xp @ theta_s
                        ss_res_p = float(np.sum((y - preds_p) ** 2))
                        score_p = 1.0 - ss_res_p / ss_tot if ss_tot > 0 else 1.0

                    perm_scores[p] = base - score_p

                importance_matrix[s_idx, j] = float(np.mean(perm_scores))
"""
        # Replace content
        # Convert new_block to list of lines
        new_lines = [l + "\n" for l in new_block.split("\n") if l]
        # Handle empty lines in split which might be lost? 
        # Actually split("\n") gives empty string for last newline.
        # Let's be careful.
        
        lines[start_idx : end_idx + 1] = [l + "\n" for l in new_block.splitlines()]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("File updated.")
    else:
        print("Block not found.")

if __name__ == "__main__":
    fix_mcr_indentation()

