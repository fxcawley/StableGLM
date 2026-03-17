# When this toolkit is useful

## The problem it addresses

Most interpretability work treats the fitted model as given and asks what it has learned. The implicit assumption is that the model is, in some meaningful sense, *the* model. But for problems involving correlated or noisy features, many parameter vectors achieve nearly the same loss. If a feature importance ranking changes across these near-optimal models, the ranking is an artifact of the optimization trajectory, not a property of the data (Fisher, Rudin, & Dominici, 2019).

rashomon-py makes this concrete for L2-regularized logistic and linear regression by characterizing the $\varepsilon$-Rashomon set and computing metrics over it. The toolkit is useful in situations where this kind of explanation instability matters.

## Settings where it applies

The most natural use case is post-hoc auditing of a GLM that has already been fitted. The question is not "is this a good model?" but rather "would the conclusions change under a different, equally good model?"

This question arises in several contexts. In regulated domains (credit scoring, clinical decision support), auditors may need to demonstrate that a model's predictions do not depend on the arbitrary choice of one near-optimal parameter vector over another. In research settings where feature importance rankings are reported as findings, the Rashomon set provides a check on whether those rankings are robust or merely reflect one of many possible decompositions (Dong & Rudin, 2020).

The toolkit is also useful for understanding feature substitutability. When features are correlated, the loss surface has flat directions, and the optimizer's particular decomposition of prediction into coefficient weights is one of many valid solutions. The VIC and MCR outputs expose this directly: if several correlated features all have wide coefficient distributions, the data does not determine how much weight to assign each one individually.

For settings where a large Rashomon set is present, Rudin (2019) and Semenova, Rudin, & Parr (2022) argue that this creates an opportunity: if many equally accurate models exist, it may be possible to select one that is also interpretable, or fair, or aligned with domain constraints. rashomon-py does not perform this selection, but it quantifies the size and shape of the space within which such selection could occur.

## Limitations on applicability

The toolkit supports only L2-regularized logistic and linear regression. The mathematics rely on the convexity of the loss and the resulting structure of the sublevel set (ellipsoidal approximation via the Hessian). This does not extend to tree models, neural networks, or penalties other than L2.

The certificate-based estimates grow conservative in high dimensions. At $d = 61$, the tightness ratio is 4.4--6.2x; at $d = 104$, it exceeds 8x. Hit-and-run MCMC sampling provides tighter estimates but mixing degrades in high dimensions (ESS < 10 at $d = 104$ after 500 draws). For problems with $d > 50$ or so, dimensionality reduction before auditing may be necessary for credible results.

The toolkit does not compute fairness metrics (demographic parity, equalized odds, etc.), though the Rashomon set is relevant to fairness; see Rudin (2019). It does not perform model selection. It does not answer questions about statistical significance; bootstrap CIs and p-values address sampling uncertainty, while Rashomon intervals address model multiplicity, and these are different quantities.

## References

- Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful. *JMLR*, 20(177), 1--81.
- Dong, J. & Rudin, C. (2020). Exploring the cloud of variable importance for the set of all good models. *Nature Machine Intelligence*, 2, 810--824.
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206--215.
- Semenova, L., Rudin, C., & Parr, R. (2022). On the existence of simpler machine learning models. *FAccT*.
