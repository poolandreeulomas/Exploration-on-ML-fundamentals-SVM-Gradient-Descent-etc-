### Exploration on ML fundamentals: Gradient Descent, Kernels, SVMs, and Calibration

This repository contains the implementation and write-up for a machine learning lab covering optimization, support vector machines, kernel methods, and probability calibration. The core work is implemented in the notebook `Code.ipynb`, with the written discussion and interpretations collected in `Report.md`.

## Project Overview

The notebook is organized around four main problem blocks:

- **Problem 2.1:** logistic regression with regularized binary cross-entropy, loss-surface visualization, batch gradient descent (BGD), stochastic gradient descent (SGD), and a JAX-based automatic differentiation version.
- **Problem 2.2:** linear and nonlinear support vector classification on `blobs.csv` and `circles.csv`, including margins, feature transformations, and polynomial kernels.
- **Problem 2.3:** kernel ridge regression in dual form using polynomial and RBF kernels, with an analysis of model complexity and regularization.
- **Problem 2.4:** random forest classification and probability calibration using reliability diagrams, log-loss, and Platt scaling.

## Repository Structure

```text
.
|-- Code.ipynb                # Main notebook with all implementations and plots
|-- Report.md                 # Written answers and interpretation of results
|-- parsed_pdf.md             # Parsed assignment text / task reference
|-- data/
|   |-- blobs.csv
|   |-- calibration.csv
|   |-- circles.csv
|   |-- data.csv
|   |-- regression_test.csv
|   `-- regression_train.csv
`-- *.png                     # Exported figures generated during the analysis
```

## Datasets

- `data/data.csv`: binary classification dataset used for logistic regression and gradient descent experiments.
- `data/blobs.csv`: linearly separable 2D dataset used for linear SVM analysis.
- `data/circles.csv`: concentric-circle dataset used to study nonlinear separation and feature mappings.
- `data/regression_train.csv` and `data/regression_test.csv`: 1D regression datasets used for kernel ridge regression.
- `data/calibration.csv`: binary classification dataset used for random forest evaluation and probability calibration.

## Methods and Libraries

The notebook uses the following Python stack:

- `numpy` for numerical computation
- `pandas` for tabular data handling
- `matplotlib` for visualizations
- `scikit-learn` for SVMs, random forests, metrics, splitting, and calibration utilities
- `jax` for automatic differentiation in the gradient-descent section

## How to Run

1. Create and activate a Python environment.
2. Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn jax jaxlib notebook
```

3. Open `Code.ipynb` in VS Code or Jupyter.
4. Run the notebook from top to bottom so that all plots and reported metrics are generated in the intended order.

## What the Notebook Produces

Running the notebook reproduces:

- loss-surface plots for regularized logistic regression
- convergence plots and parameter trajectories for BGD and SGD
- SVM decision boundaries and margin visualizations
- transformed-feature and kernel-based classification results for nonlinearly separable data
- kernel ridge regression prediction curves for different hyperparameters
- confusion matrices, reliability diagrams, and calibrated vs. uncalibrated probability comparisons

Several generated figures are already stored in the repository as `.png` files.

## Selected Findings

- **Regularization** makes the logistic-regression loss surface more rounded and discourages large weights.
- **BGD** converges more smoothly, while **SGD** is noisier but scales better to larger datasets.
- The `circles.csv` dataset is **not linearly separable** in the original 2D space, but it becomes separable after an appropriate feature transformation or with a kernelized SVM.
- In kernel ridge regression, both the **kernel choice** and the **regularization strength** strongly affect smoothness, flexibility, and overfitting.
- **Platt scaling** improves the quality of predicted probabilities for the random forest, reducing log-loss without materially changing classification accuracy.

## Files to Read First

- Start with `Code.ipynb` for the full implementation.
- Read `Report.md` for the concise written interpretation of each task.
- Use `parsed_pdf.md` if you want the assignment wording in plain text.

## Notes

This project is structured as a lab submission rather than a reusable Python package. The notebook is the primary entry point, and the repository is meant to document both the implementation and the resulting analysis.
