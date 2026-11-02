# Graph Embedding and Marginal Fisher Analysis (MFA)

A from-scratch Python implementation of

> S. Yan, D. Xu, B. Zhang, H.-J. Zhang, Q. Yang and S. Lin,
> **"Graph Embedding and Extensions: A General Framework for Dimensionality Reduction,"**
> *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 29(1), 40–51, 2007.

Every method is written directly from the paper's formulas using NumPy/SciPy.
No dimensionality-reduction library routines are used for the methods themselves.

## What is implemented

**Part A — the framework.** One solver for the graph-embedding problem

$$\min_{\mathbf{y}^\top B\mathbf{y}=d}\ \mathbf{y}^\top L\mathbf{y},\qquad L = D - W,$$

in its direct, linear, kernel and tensor forms (Eqs. 3–10), solved as the generalised eigenproblem $\tilde L\mathbf v = \lambda \tilde B\mathbf v$.

**Part B — existing algorithms as special cases (Table 1).**
PCA, LDA (Fisherface, PCA+LDA), LPP, ISOMAP, LLE, Laplacian Eigenmap, LEA, KPCA, KDA, 2DPCA and DATER. Each is the Part A solver with a different graph $W$ and constraint $B$.

**Part C — the paper's proposal.**
Marginal Fisher Analysis (MFA), built from an intrinsic graph ($k_1$ same-class neighbours, Eq. 13) and a penalty graph ($k_2$ closest cross-class pairs, Eq. 14). Kernel MFA (Eqs. 16–17) and Tensor MFA (Eq. 18) follow from the same framework.

**Part D — experiments.** Face recognition on the ORL / Olivetti database (40 people × 10 images, resized to 56 × 46). It uses G*m*/P*n* splits (*m* training and *n* test images per person) over 5 random splits, with a 1-nearest-neighbour classifier. Every existing method is compared with its MFA counterpart.

**Part E — formula audit.** Each formula is checked against a literal (loop-based) computation and matches to floating-point precision. The audit also exposes a sign typo in Table 1's LLE entry (the paper's Appendix B has the correct $W = M + M^\top - M^\top M$).

## Repository layout

```
notebooks/
  GE_existing_vs_proposed.ipynb   full implementation: framework, existing methods, MFA, experiments, audit
  GE_MFA_demo.ipynb               short 5-minute demo version
requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/GE_existing_vs_proposed.ipynb
```

Then use **Kernel → Restart & Run All**.

- The ORL / Olivetti faces download automatically through `sklearn.datasets.fetch_olivetti_faces` on first run, so you need internet access the first time. If the download fails, the notebook falls back to scikit-learn's built-in digits data.
- Every figure is saved automatically to `notebooks/figures/` as a 300 dpi PNG and a vector SVG. Slide versions go to `notebooks/figures/slides/` as PDF and PNG. Set `SAVE_FIGURES = False` in the first cell to turn this off.

## Notes

- There is no official code release for this paper. This implementation follows the equations in the paper, so exact accuracies may differ from the published tables. Differences come from preprocessing, random splits and hyper-parameter grids.
- Hyper-parameters and output dimension are chosen by best test accuracy, following the paper's reporting protocol. That makes the reported numbers optimistic.
