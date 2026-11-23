"""Linear and polynomial ridge regression for oxide band-gap prediction.

Run from the project directory with: python Linear-Regression.py
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# Step 1: Load the provided training CSV relative to this script, so the
# script works regardless of the shell's current directory.
DATA_FILE = Path(__file__).resolve().parent / "features_training_reduced_gB.csv"
data = pd.read_csv(DATA_FILE)

# Step 2: Predict band_gap from numeric descriptors only. material_id and
# formula_pretty are identifiers/text, so they are not model inputs.
target = "band_gap"
X = data.drop(columns=[target, "material_id", "formula_pretty"], errors="ignore")
y = data[target]

# Fail early if the CSV contains missing/non-numeric feature values; this keeps
# the feature processing explicit and prevents silent row loss.
X = X.apply(pd.to_numeric, errors="raise")
if X.isna().any().any() or y.isna().any():
    raise ValueError("The training CSV contains missing feature or target values.")

# Step 3: Keep a final holdout set untouched during model/degree selection.
# The fixed seed makes the reported comparison reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Step 4: Compare polynomial bases using cross-validation
# NOTE: Placeholder code. The maximum polynomial order has not been decided
# yet; linear through cubic is used here for initial testing only.
# on training data only. Degree 3 is included as a check for added curvature;
# higher degrees rapidly create many correlated terms from 33 descriptors.
# Scaling occurs after polynomial expansion and inside the CV pipeline, so
# every fold learns its own scaling statistics without data leakage.
degrees = (1, 2, 3)
alphas = (0.01, 0.1, 1.0, 10.0, 100.0)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for degree in degrees:
    model = Pipeline(
        steps=[
            ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    search = GridSearchCV(
        model,
        param_grid={"ridge__alpha": alphas},
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    results.append((degree, search))

# Step 5: Report cross-validation and holdout performance. RMSE/MAE are in eV;
# lower is better. R-squared measures variance explained; closer to 1 is better.
print(f"Rows: {len(data)} | Features: {X.shape[1]} | Holdout: {len(X_test)}")
print("\nPolynomial degree comparison (best ridge alpha per degree):")
for degree, search in results:
    prediction = search.predict(X_test)
    cv_rmse = -search.best_score_
    test_rmse = mean_squared_error(y_test, prediction) ** 0.5
    test_mae = mean_absolute_error(y_test, prediction)
    test_r2 = r2_score(y_test, prediction)
    print(
        f"degree={degree}: alpha={search.best_params_['ridge__alpha']:g}, "
        f"CV RMSE={cv_rmse:.4f} eV, holdout RMSE={test_rmse:.4f} eV, "
        f"holdout MAE={test_mae:.4f} eV, holdout R2={test_r2:.4f}"
    )

# Step 6: Select the simplest degree whose CV error is within 1% of the best
# CV score. This avoids choosing a much larger basis for a negligible CV gain.
# Degree 3 is the maximum candidate: with 33 inputs, cubic expansion adds
# thousands of terms; degree 4 would add far more and raise overfitting risk.
best_cv = min(-search.best_score_ for _, search in results)
eligible = [
    (degree, search)
    for degree, search in results
    if -search.best_score_ <= best_cv * 1.01
]
selected_degree, selected_search = min(eligible, key=lambda item: item[0])

print(
    f"\nSelected degree: {selected_degree} "
    f"(simplest degree within 1% of best training CV RMSE)."
)
print(
    "Ridge effect: alpha=0 is ordinary least squares; positive alpha shrinks "
    "coefficients, usually reducing variance and sensitivity to correlated "
    "polynomial terms. Too much shrinkage can underfit. The selected alpha "
    "is chosen by cross-validation; compare the degree=1 and higher-degree "
    "holdout scores above to describe its observed effect on this dataset."
)

# One-feature visualization for a simple linear vs polynomial comparison.
# This is a 1D illustration using band center; the full model above still uses
# all five numeric descriptors.
visual_feature = "band center"
visual_X = data[[visual_feature]]
visual_y = data[target]
visual_X_train, visual_X_test, visual_y_train, visual_y_test = train_test_split(
    visual_X, visual_y, test_size=0.20, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(visual_X_train, visual_y_train)

polynomial_model = Pipeline(
    steps=[
        ("polynomial", PolynomialFeatures(degree=selected_degree, include_bias=False)),
        ("linear", LinearRegression()),
    ]
)
polynomial_model.fit(visual_X_train, visual_y_train)

x_plot = pd.DataFrame(
    {visual_feature: pd.Series(sorted(visual_X_test[visual_feature]))}
)
linear_prediction = linear_model.predict(x_plot)
polynomial_prediction = polynomial_model.predict(x_plot)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

axes[0].scatter(
    visual_X_test[visual_feature],
    visual_y_test,
    alpha=0.45,
    label="Holdout data",
)
axes[0].plot(
    x_plot[visual_feature],
    linear_prediction,
    color="red",
    linewidth=2,
    label="Linear regression",
)
axes[0].set_title("Simple linear regression")
axes[0].set_xlabel(visual_feature)
axes[0].set_ylabel("Band gap (eV)")
axes[0].legend()
axes[0].grid(alpha=0.25)

axes[1].scatter(
    visual_X_test[visual_feature],
    visual_y_test,
    alpha=0.45,
    label="Holdout data",
)
axes[1].plot(
    x_plot[visual_feature],
    polynomial_prediction,
    color="red",
    linewidth=2,
    label=f"Polynomial regression (degree={selected_degree})",
)
axes[1].set_title("Polynomial linear regression")
axes[1].set_xlabel(visual_feature)
axes[1].legend()
axes[1].grid(alpha=0.25)

fig.suptitle("Band gap vs. band center: linear model comparison")
fig.tight_layout()
figure_path = Path(__file__).resolve().parent / "linear_vs_polynomial_comparison.png"
plt.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.show()
