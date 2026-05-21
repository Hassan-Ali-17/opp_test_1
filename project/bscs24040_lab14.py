# ============================================================
#  Linear SVM Implementation from Scratch
#  Student Lab Assignment
#  Only NumPy and Matplotlib are used (no sklearn for model)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles   # allowed ONLY for data generation in Task 5


# ============================================================
# TASK 1 - Data Synthesis and Preprocessing
# ============================================================

# Fix random seed so we get the same data every run
np.random.seed(42)

# Create two clusters
# Class +1 centered at (2, 2)
# Class -1 centered at (-2, -2)
n_samples = 100

X_pos = np.random.randn(n_samples, 2) + np.array([2, 2])    # positive class
X_neg = np.random.randn(n_samples, 2) + np.array([-2, -2])  # negative class

X = np.vstack([X_pos, X_neg])                  # stack into one array (200 x 2)
y = np.hstack([np.ones(n_samples),             # +1 for positive class
               -np.ones(n_samples)])            # -1 for negative class

# Manual Standard Scaling  (no sklearn allowed!)
# z = (x - mean) / std   done feature-by-feature
mean = X.mean(axis=0)   # mean of each column
std  = X.std(axis=0)    # std  of each column
X = (X - mean) / std    # normalized data


# ============================================================
# TASK 2 - The NumpySVM Class
# ============================================================

class NumpySVM:
    """
    A simple linear SVM trained with Stochastic Gradient Descent.
    Hinge Loss + L2 Regularization (no sklearn, just numpy).
    """

    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_epochs=1000):
        self.lr     = learning_rate   # eta  - step size
        self.lam    = lambda_param    # lambda - controls margin width
        self.epochs = n_epochs
        self.w      = None            # weight vector
        self.b      = None            # bias term
        self.losses = []              # store loss every epoch for plotting

    # ----------------------------------------------------------
    # TASK 3 - Training loop (SGD + gradient update rules)
    # ----------------------------------------------------------
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias to zero
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            # SGD: go through every sample one by one
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]

                # Margin check: yi * (w · xi - b) >= 1  means correctly classified outside margin
                margin = yi * (np.dot(self.w, xi) - self.b)

                if margin >= 1:
                    # Correctly classified outside the margin
                    # Only regularization gradient
                    self.w = self.w - self.lr * (2 * self.lam * self.w)
                    # b stays the same
                    hinge = 0.0

                else:
                    # Inside margin or misclassified
                    # Hinge loss applies
                    self.w = self.w - self.lr * (2 * self.lam * self.w - yi * xi)
                    self.b = self.b - self.lr * yi
                    hinge = 1 - margin

                epoch_loss += hinge

            # Average hinge loss for this epoch + regularization term
            total_loss = self.lam * np.dot(self.w, self.w) + (epoch_loss / n_samples)
            self.losses.append(total_loss)

    # ----------------------------------------------------------
    # TASK 2 - Predict method
    # ----------------------------------------------------------
    def predict(self, X):
        # sign(w · x - b)
        raw = np.dot(X, self.w) - self.b
        return np.sign(raw)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y) * 100   # percentage


# ============================================================
# TASK 4 - Geometry Visualization helper
# ============================================================

def plot_svm(ax, X, y, model, title="SVM Decision Boundary"):
    """
    Draws:
      - scatter plot of the two classes
      - decision boundary  (w·x - b = 0)
      - upper margin       (w·x - b = +1)
      - lower margin       (w·x - b = -1)
      - support vectors highlighted
    """

    # --- build a grid to draw the lines ---
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x_vals = np.linspace(x_min, x_max, 200)

    w = model.w
    b = model.b

    # avoid division by zero
    if w[1] == 0:
        ax.set_title(title + " (w[1]=0, cannot draw line)")
        return

    # Equation of lines: w[0]*x0 + w[1]*x1 - b = c  =>  x1 = (b + c - w[0]*x0) / w[1]
    decision  = (b + 0 - w[0] * x_vals) / w[1]
    margin_up  = (b + 1 - w[0] * x_vals) / w[1]
    margin_dn  = (b - 1 - w[0] * x_vals) / w[1]

    # --- scatter plot ---
    ax.scatter(X[y == 1, 0],  X[y == 1, 1],  color='royalblue', label='Class +1', edgecolors='k', zorder=3)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], color='tomato',    label='Class -1', edgecolors='k', zorder=3)

    # --- lines ---
    ax.plot(x_vals, decision,   'k-',  linewidth=2,   label='Decision Boundary')
    ax.plot(x_vals, margin_up,  'k--', linewidth=1.2, label='Margin +1')
    ax.plot(x_vals, margin_dn,  'k--', linewidth=1.2, label='Margin -1')

    # --- highlight support vectors ---
    # Support vectors are the points closest to / on the margin: |yi*(w·xi - b)| <= 1
    margins = y * (np.dot(X, w) - b)
    sv_mask = margins <= 1.0
    ax.scatter(X[sv_mask, 0], X[sv_mask, 1],
               s=150, facecolors='none', edgecolors='green',
               linewidths=2, label='Support Vectors', zorder=4)

    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_xlim(x_min, x_max)


# ============================================================
# --- MAIN TRAINING (default hyperparameters) ---
# ============================================================

svm = NumpySVM(learning_rate=0.01, lambda_param=0.01, n_epochs=1000)
svm.fit(X, y)
acc = svm.accuracy(X, y)
print(f"Default SVM Accuracy: {acc:.2f}%")


# ============================================================
# TASK 4 - Plot: Decision Boundary
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))
plot_svm(ax, X, y, svm, title=f"Linear SVM - Decision Boundary (λ=0.01)\nAccuracy: {acc:.1f}%")
plt.tight_layout()
plt.savefig("task4_decision_boundary.png", dpi=100)
plt.show()
print("Saved: task4_decision_boundary.png")


# ============================================================
# TASK 5a - Varying Lambda
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, lam, title_lam in zip(axes, [1.0, 0.001], ["λ=1.0  (High Reg → Narrow Margin)",
                                                     "λ=0.001 (Low Reg → Wide Margin)"]):
    svm_lam = NumpySVM(learning_rate=0.01, lambda_param=lam, n_epochs=1000)
    svm_lam.fit(X, y)
    acc_lam = svm_lam.accuracy(X, y)
    plot_svm(ax, X, y, svm_lam, title=f"SVM {title_lam}\nAccuracy: {acc_lam:.1f}%")

plt.suptitle("Task 5a: Effect of Lambda on Margin Width", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("task5a_lambda_comparison.png", dpi=100)
plt.show()
print("Saved: task5a_lambda_comparison.png")

# --- explanation printed to console ---
print("""
Task 5a Explanation:
  - λ = 1.0  → strong regularization → model penalizes large weights heavily
                → weights stay small → margin (2/||w||) becomes WIDER but more errors allowed
  - λ = 0.001 → weak regularization  → weights can grow larger
                → margin becomes NARROWER but fewer training errors
  The trade-off: higher λ = wider margin but possibly worse accuracy;
                 lower  λ = tighter fit to data but may overfit.
""")


# ============================================================
# TASK 5b - Linear Failure on Non-Linear Data
# ============================================================

# make_circles is ALLOWED for data generation only
X_circ, y_circ = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=42)

# convert labels from {0,1} to {-1,+1}
y_circ = np.where(y_circ == 0, -1, 1)

# normalize manually
mean_c = X_circ.mean(axis=0)
std_c  = X_circ.std(axis=0)
X_circ_norm = (X_circ - mean_c) / std_c

svm_circ = NumpySVM(learning_rate=0.01, lambda_param=0.01, n_epochs=1000)
svm_circ.fit(X_circ_norm, y_circ)
acc_circ = svm_circ.accuracy(X_circ_norm, y_circ)
print(f"Linear SVM on Circles Accuracy: {acc_circ:.2f}%")

fig, ax = plt.subplots(figsize=(7, 5))
plot_svm(ax, X_circ_norm, y_circ, svm_circ,
         title=f"Task 5b: Linear SVM on Circles Dataset\nAccuracy: {acc_circ:.1f}% (should be ~50%)")
plt.tight_layout()
plt.savefig("task5b_circles_failure.png", dpi=100)
plt.show()
print("Saved: task5b_circles_failure.png")

print("""
Task 5b Explanation:
  A linear SVM draws ONE straight line (hyperplane) to separate classes.
  The circles dataset is NOT linearly separable — the inner circle is
  completely surrounded by the outer circle.
  No straight line can split them, so accuracy hovers around 50% (random guess level).
  To solve this we need a non-linear kernel (e.g. RBF / polynomial).
""")


# ============================================================
# TASK 5c - Loss Convergence Plot
# ============================================================

# smooth the loss with a simple moving average so it is easier to read
def moving_average(data, window=20):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(np.mean(data[start:i+1]))
    return np.array(result)

epochs_range = np.arange(1, len(svm.losses) + 1)
smoothed     = moving_average(svm.losses, window=30)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs_range, svm.losses, color='lightblue', alpha=0.6, label='Raw Loss')
ax.plot(epochs_range, smoothed,   color='navy',      linewidth=2,  label='Smoothed (window=30)')
ax.set_title("Task 5c: Hinge Loss Convergence", fontsize=13)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("task5c_loss_convergence.png", dpi=100)
plt.show()
print("Saved: task5c_loss_convergence.png")

print("""
Task 5c Note:
  The raw loss looks jagged because SGD updates weights per-sample (noisy).
  The smoothed curve shows the overall downward trend — the model is learning.
  Loss should decrease and level off once the SVM finds the optimal hyperplane.
""")

print("\nAll tasks complete!")