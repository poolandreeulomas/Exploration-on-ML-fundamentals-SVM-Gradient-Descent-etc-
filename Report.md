## Problem 2.1 (25 %)

## 2.1.1 Download the dataset data.csv and generate a surface plot of the loss function J(w), with L(w) from Eq. (2.1.4). Briefly compare the plots for λ = 0 and λ = 10−2

For λ = 0, the loss surface shows a broader and more elongated minimum, which allows the model to favor larger weight values. For λ = 10^{-2}, the surface becomes more rounded and rises more quickly as the weights move away from the origin. This happens because the regularization term penalizes large parameter magnitudes. As a result, the minimum shifts slightly toward smaller weights, indicating a simpler and more stable model.

## 2.1.3 Use the result from Task 2.1.2 to implement BGD for the regressor from (2.1.4) and fit it to data.csv. Plot J(w) over the iterations k = [1, . . . , K]. Additionally provide a 2D scatter plot showcasing the history of the weight estimates ŵ(k) throughout the optimization process.

The first plot shows the value of the loss function J(w) over the iterations of batch gradient descent. Its decrease indicates that the algorithm successfully minimizes the objective and converges toward a stable solution. The second plot shows the trajectory of the weight vector in parameter space. It illustrates how the parameters move from the initial guess toward the optimum during the optimization process.

## 2.1.4 Repeat the Task 2.1.3 for SGD using the same configuration. What do you observe when comparing the plots for BGD and SGD? Further, specify the number of epochs denoting a full iteration over the training dataset required for BGD and SGD in this example. Which gradient descent variant would you choose for a large training dataset? Explain your answer.

When comparing BGD and SGD, the BGD plots are smoother because each update is computed using the full dataset. As a result, the loss decreases in a more regular way and the weight trajectory follows a more direct path toward the minimum. In contrast, the SGD plots are noisier because each update is based on a single sample. This introduces variance in the gradient estimate, which causes visible oscillations in both the loss curve and the parameter trajectory.

In this example, the dataset contains 100 samples and both methods use 1000 parameter updates. For BGD, each update processes the entire dataset, so 1000 iterations correspond to 1000 epochs. For SGD, each update processes only one sample, so 1000 iterations correspond to 10 epochs.

For a large training dataset, SGD would generally be the preferred choice. Its updates are much cheaper in terms of computation and memory, which makes it more scalable. Although SGD is less smooth and more stochastic than BGD, it is usually more practical for large-scale optimization problems and can reach a good solution faster in terms of overall computational cost.

## 2.1.5 Libraries for high-performance numerical computing such as JAX enable efficient implementation of gradient-based machine learning algorithms by providing automatic differentiation and just-in-time compilation. Study the tutorial in https://docs.jax.dev/en/latest/automatic-differentiation.html and adapt your code to compute gradients using jax.grad (or jax.value_and_grad) instead of manual derivatives. Visualize the optimization process for BGD and validate that the results are equivalent to the manual implementation from Task 2.1.3.

In this task, the manual gradient was replaced by automatic differentiation using `jax.grad`. The loss function was written directly in JAX, and the gradient with respect to the weight vector was obtained automatically from the computational graph. This removes the need to derive and implement the gradient expression by hand while preserving the same optimization procedure.

The resulting plots confirm that the JAX-based BGD implementation is equivalent to the manual implementation from Task 2.1.3. The loss decreases smoothly over the iterations and converges to approximately the same final value, while the weight trajectory follows the same overall path toward the optimum. Therefore, JAX provides the same numerical behavior as the manual gradient implementation, with the added benefit of simpler and less error-prone gradient computation.

## 2.2.2 Download the file blobs.csv from TUWEL and generate a scatter plot that highlights the two classes in color. Fit a SVC with a linear kernel and C = 1 to the dataset and extract the weights w and the bias b from the trained model.

The scatter plot shows two clearly separable clusters, which makes the dataset suitable for a linear support vector classifier. After fitting a linear SVC with C = 1, the learned parameters are approximately w = [0.8942, 1.0199] and b = -3.2415. These parameters define the separating hyperplane x^T w + b = 0.

## 2.2.3 Plot the separating hyperplane as well as the two equidistant translations running through the support vectors ontop of the scatter plot from Task 2.2.2. Calculate the distance of the margin of the separating hyperplane.

The separating hyperplane correctly divides the two classes, and the two dashed lines correspond to the margin boundaries defined by x^T w + b = \pm 1. The highlighted support vectors lie exactly on these two margin lines and determine the position of the classifier. Using the learned weight vector, the distance from the separating hyperplane to each margin is 1 / ||w|| \approx 0.7372. Therefore, the total width of the margin is 2 / ||w|| \approx 1.4745.

## 2.2.4 Repeat Tasks 2.2.2 and 2.2.3 for the dataset circles.csv. Is this dataset linearly separable in the given space?

When the same linear SVC is applied to circles.csv, the resulting decision boundary is not able to separate the two classes correctly. The scatter plot shows that the classes are arranged in a circular pattern rather than in two groups that can be split by a single straight line. This is also reflected in the training accuracy, which is only about 0.51. Therefore, the dataset is not linearly separable in the original input space. A nonlinear classifier or a feature transformation would be required to achieve good separation.

## 2.2.5 Find a transformation ϕ so that the circles.csv dataset is linearly separable in the new space x′.

One suitable transformation is

ϕ(x) = [x_0, x_1, r]^T, with r = \sqrt{x_0^2 + x_1^2}.

This mapping adds the radial distance from the origin as a third feature. In the original 2D space, the classes form concentric structures and cannot be separated by a straight line. In the transformed 3D space, however, the radius provides the missing information needed to distinguish the inner and outer circle, making the classes linearly separable.

After training a linear SVC with C = 100 on the transformed dataset, the classifier achieves a training accuracy of 1.0, which confirms perfect linear separability in the new feature space. The 2D visualization in the coordinates (x_0, x_2') shows that the two classes can now be separated by a linear boundary, and the corresponding margin lines can also be plotted consistently in this transformed representation.

## 2.2.6 Now fit a SVC with a polynomial kernel of degree 2 to the original 2D circles.csv dataset and plot a heatmap of the predicted labels y_hat for the interval x_0 in [-2, 2] and x_1 in [-2, 2]. Is the separating hyperplane a linear function of x?

Using a polynomial kernel of degree 2, the SVC is able to classify the original circles.csv dataset perfectly, achieving a training accuracy of 1.0. The heatmap of the predicted labels shows a nonlinear decision region that correctly separates the inner and outer circular classes. This confirms that the kernel method implicitly maps the data into a higher-dimensional feature space where linear separation becomes possible.

However, the separating boundary in the original input space is not a linear function of x. In the original 2D coordinates, the boundary is curved, which is clearly visible in the heatmap. Therefore, while the classifier is linear in the transformed feature space induced by the kernel, it is nonlinear with respect to the original variables x_0 and x_1.

## 2.3.3 Fit your regressor to the 1D dataset from Exercise 1.1 using m in [1, 2, 3, 10] and plot the predictions in the interval x in [0, 2]. Also analyse the influence of different regularization parameters.

The plots show a clear interaction between the polynomial degree m and the regularization parameter λ. For m = 1, the model behaves almost linearly and is too simple to capture the curvature of the data, which indicates underfitting. For m = 2 and m = 3, the regressor is flexible enough to follow the overall nonlinear trend while still remaining smooth, especially for moderate regularization. These degrees provide a better balance between bias and variance on this dataset.

For m = 10, the model becomes highly flexible. With very small regularization, especially λ = 10^-6, the prediction curve oscillates strongly and grows to extreme values near the boundary of the interval, which is a clear sign of overfitting. Increasing λ makes the solution much smoother and more stable. In general, larger λ reduces the variance of the estimator and prevents extreme behavior, but if λ becomes too large, the model can again underfit by becoming overly smooth. Therefore, the regularization parameter controls model complexity, while the kernel degree determines the expressive power of the regressor.

## 2.3.4 Repeat Task 2.3.3 with the RBF kernel given by k(x(i), x(j)) = exp(-1 / (2 ls^2) ||x_i - x_j||^2). Briefly discuss your interpretation of the role of the kernel parameter ls.

With the RBF kernel, the length-scale parameter ls controls how local or global the influence of each training point is. For very small ls, the kernel is highly localized, so the regressor can adapt strongly to individual data points. This makes the model very flexible, but it also increases the risk of overfitting, especially when the regularization is weak. In the plots, this appears as sharp local variations and unstable behavior.

For larger ls, each training point influences a wider region of the input space, which produces smoother and more global prediction curves. This reduces sensitivity to individual samples and typically improves stability, although very large ls can oversmooth the solution and lead to underfitting. As with the polynomial case, the regularization parameter λ further controls the trade-off between flexibility and stability. Therefore, ls mainly determines the locality of the kernel, while λ controls the strength of regularization.

## Problem 2.4

## 2.4.1 Download the file calibration.csv from TUWEL and use pandas to extract the data. Split the dataset into Ttrain_full and Ttest using a test size of 0.2. Subsequently, split Ttrain_full into Ttrain and Tcal using a test size of 0.2.

The dataset was loaded with pandas and then split in two stages using `train_test_split` with `random_state = 256` to make the result reproducible. First, the full dataset of 20000 samples was divided into `Ttrain_full = 16000` and `Ttest = 4000`. Then, `Ttrain_full` was split again into `Ttrain = 12800` and `Tcal = 3200`.

This structure is important for calibration. The classifier is trained on `Ttrain`, the calibration model is fitted on the separate set `Tcal`, and the final evaluation is performed on the untouched test set `Ttest`.

## 2.4.2 Fit a random forest classifier to Ttrain. What is the accuracy on Ttest? Calculate and plot the confusion matrix and provide the values for Precision, Recall, and F1-Score.

The random forest classifier achieves an accuracy of 0.9445 on `Ttest`, which indicates strong predictive performance. The confusion matrix is

$$
\begin{bmatrix}
1830 & 132 \\
90 & 1948
\end{bmatrix}
$$

This means that the classifier produces 1830 true negatives, 1948 true positives, 132 false positives, and 90 false negatives. The corresponding evaluation measures are Precision = 0.9365, Recall = 0.9558, and F1-score = 0.9461.

Overall, the classifier performs well on both classes, with slightly higher recall than precision. This means that it captures most positive samples while making a moderate number of false positive errors.

## 2.4.3 Predict the class probabilities on Ttest, build the reliability diagram manually, and compute the log-loss.

The reliability diagram was constructed by grouping the predicted probabilities of the positive class into 10 uniform bins over $[0,1]$. For each bin, the x-value is the mean predicted probability and the y-value is the fraction of positives, i.e. the empirical mean of the true class labels in that bin.

The resulting curve shows that the classifier is not perfectly calibrated. In the low and medium probability range, the curve lies clearly below the diagonal, which means that the classifier tends to overestimate the probability of the positive class in those bins. In the higher probability range, the curve moves above the diagonal, which indicates underestimation there. Therefore, the model does not show a uniform calibration error, but rather a change in behavior across the probability range.

The log-loss on `Ttest` is 0.2495. Since log-loss penalizes confident but wrong predictions strongly, this value confirms that the raw probabilities can still be improved even though the classification accuracy is already high.

## 2.4.4 Recreate the reliability diagram using CalibrationDisplay.from_estimator() and compare it with the manual version.

Using `CalibrationDisplay.from_estimator()` with `n_bins = 10` produces a reliability diagram with essentially the same shape as the manually computed one from Task 2.4.3. This confirms that the manual implementation was correct: both approaches apply the same underlying idea of binning predicted probabilities and comparing them to the observed fraction of positives.

The scikit-learn version is more compact and convenient, but it does not change the interpretation. The random forest remains noticeably miscalibrated, especially away from the highest-probability region.

## 2.4.5 Calibrate the random forest using Platt scaling. Compare the calibrated classifier with the uncalibrated one in terms of accuracy, log-loss, and reliability diagram.

Platt scaling was applied as a post-processing step on top of the already trained random forest, using the independent calibration set `Tcal`. The calibrated classifier achieves the same accuracy on `Ttest` as the uncalibrated model, namely 0.9445. This is expected, because calibration mainly changes the predicted probabilities and does not necessarily change the final class decisions when a fixed threshold of 0.5 is used.

However, the log-loss improves substantially from 0.2495 to 0.1502 after calibration. This shows that the calibrated probabilities are much better aligned with the actual empirical frequencies of the positive class. The reliability diagram confirms this improvement: the calibrated curve lies much closer to the diagonal of a perfectly calibrated classifier than the original random-forest curve.

Therefore, Platt scaling clearly improves the probabilistic quality of the model without changing its classification accuracy. This illustrates the main purpose of calibration: not to increase accuracy directly, but to make predicted probabilities more trustworthy and interpretable.
