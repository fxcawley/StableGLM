import os

def fix_all():
    filepath = "rashomon/rashomon_set.py"
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Fix 1: _cg_solve p=r.copy()
        # Context: else: \n p = r.copy()
        if "p = r.copy()" in line and "            p = r.copy()" not in line:
            # Check if previous line was else:
            if i > 0 and "else:" in lines[i-1]:
                # Check indentation of else
                else_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
                if else_indent == 8:
                    # Fix indentation to 12 spaces
                    new_lines.append("            p = r.copy()\n")
                    i += 1
                    # Next line rsold
                    if i < len(lines) and "rsold =" in lines[i]:
                        new_lines.append("            rsold = float(r @ r)\n")
                        i += 1
                    continue

        # Fix 2: _cg_solve rsnew=...
        if "rsnew = float(r @ r)" in line and "                rsnew" not in line:
             if i > 0 and "else:" in lines[i-1]:
                else_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
                if else_indent == 12:
                    new_lines.append("                rsnew = float(r @ r)\n")
                    i += 1
                    continue

        # Fix 3: model_class_reliance loop
        # Look for the specific broken loop start
        if "for j in range(self._d):" in line:
            # Check context: verify it's the one in model_class_reliance
            # (The one in compute_sample_diagnostics is fine)
            # In MCR, it usually follows "importance_matrix =" or similar
            # Let's assume if it's around line 1500-1600
            if i > 1500:
                print(f"Fixing MCR loop at line {i+1}")
                new_lines.append("        for j in range(self._d):\n")
                new_lines.append("            perm_scores = np.zeros(n_permutations, dtype=float)\n")
                new_lines.append("\n")
                new_lines.append("            for p in range(n_permutations):\n")
                new_lines.append("                Xp = X.copy()\n")
                new_lines.append("\n")
                new_lines.append("                if perm_mode == \"iid\":\n")
                new_lines.append("                    # Standard permutation\n")
                new_lines.append("                    rng.shuffle(Xp[:, j])\n")
                new_lines.append("                elif perm_mode == \"residual\":\n")
                new_lines.append("                    # Permute residuals from prediction without feature j\n")
                new_lines.append("                    theta_minus_j = theta_s.copy()\n")
                new_lines.append("                    theta_minus_j[j] = 0.0\n")
                new_lines.append("                    pred_minus_j = X @ theta_minus_j\n")
                new_lines.append("                    residual_j = X[:, j] - pred_minus_j / (theta_s[j] + 1e-12)\n")
                new_lines.append("                    residual_j_perm = residual_j.copy()\n")
                new_lines.append("                    rng.shuffle(residual_j_perm)\n")
                new_lines.append("                    Xp[:, j] = residual_j_perm + pred_minus_j / (theta_s[j] + 1e-12)\n")
                new_lines.append("                elif perm_mode == \"conditional\":\n")
                new_lines.append("                    # Conditional permutation (simplified: bin-based)\n")
                new_lines.append("                    # Bin samples by correlated features and permute within bins\n")
                new_lines.append("                    n_bins = max(3, int(np.sqrt(X.shape[0])))\n")
                new_lines.append("                    # Use mean of other features for binning\n")
                new_lines.append("                    other_mean = np.mean(np.delete(X, j, axis=1), axis=1)\n")
                new_lines.append("                    bins = np.digitize(other_mean, np.linspace(other_mean.min(), other_mean.max(), n_bins))\n")
                new_lines.append("                    for b in range(1, n_bins + 1):\n")
                new_lines.append("                        mask = bins == b\n")
                new_lines.append("                        if np.sum(mask) > 1:\n")
                new_lines.append("                            Xp[mask, j] = rng.permutation(Xp[mask, j])\n")
                new_lines.append("                else:\n")
                new_lines.append("                    raise ValueError(f\"Unknown perm_mode: {perm_mode}\")\n")
                new_lines.append("\n")
                new_lines.append("                # Score with permuted feature\n")
                new_lines.append("                if self.estimator == \"logistic\":\n")
                new_lines.append("                    scores_p = Xp @ theta_s\n")
                new_lines.append("                    preds_p = (scores_p > 0.0).astype(int)\n")
                new_lines.append("                    score_p = float(np.mean((preds_p == y.astype(int)).astype(float)))\n")
                new_lines.append("                else:\n")
                new_lines.append("                    preds_p = Xp @ theta_s\n")
                new_lines.append("                    ss_res_p = float(np.sum((y - preds_p) ** 2))\n")
                new_lines.append("                    score_p = 1.0 - ss_res_p / ss_tot if ss_tot > 0 else 1.0\n")
                new_lines.append("\n")
                new_lines.append("                perm_scores[p] = score_p\n")
                new_lines.append("\n")
                new_lines.append("            importance_matrix[s_idx, j] = base - np.mean(perm_scores)\n")
                
                # Skip lines until we find the end of the broken block
                # The broken block ends with "importance_matrix[s_idx, j] = ..." or similar
                # Let's skip until "Aggregate across samples"
                while i < len(lines) and "Aggregate across samples" not in lines[i]:
                    i += 1
                continue

        new_lines.append(line)
        i += 1

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("File updated.")

if __name__ == "__main__":
    fix_all()

