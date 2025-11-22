# Part 2. Modelling (ML & Statistics)

In this part, we model the data to determine the best method for predicting wine quality using the available chemical features.

# Regression

Let's start modelling the data by fitting a linear model, using `quality` as the outcome variable and the remaining chemical features as predictors. First, we need to split the dataset into a training set and a test set so that we can obtain a reliable assessment of the model’s predictive power.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

# Select the predictor matrix (excluding outliers)
X = df[df['outliers'] != -1].drop(['quality', 'quality_cat', 'outliers'], axis=1)
y = df[df['outliers'] != -1]['quality']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Standard-scale the predictors
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set and compute the error
pred = model.predict(X_test)
err = root_mean_squared_error(y_test, pred)

# Build a summary table containing each feature and its coefficient
summary = pd.DataFrame({
    'Feature': X.columns,
    'Coef': model.coef_
})
```

To add a statistical perspective, we can use this function:

```python
from scipy import stats

def compute_pvalues_linearly(X, y):
    """
    Fit a linear regression model and compute t-statistics and p-values
    for each coefficient.

    Parameters
    ----------
    X : pandas.DataFrame
        Design matrix with one column per predictor.
    y : array-like
        Target vector.

    Returns
    -------
    pandas.DataFrame
        Table with feature name, coefficient, and two-tailed p-value.
    """
    # Fit a linear regression model (using scikit-learn)
    model = LinearRegression().fit(X, y)

    # Combine intercept and coefficients into a single parameter vector
    params = np.append(model.intercept_, model.coef_)

    # Design matrix with an intercept column
    X_design = np.column_stack([np.ones(X.shape[0]), X.values])

    # Predictions and residuals
    pred = model.predict(X)
    residuals = y - pred

    # Residual sum of squares
    rss = np.sum(residuals ** 2)

    # Degrees of freedom: n - p - 1
    df_resid = X.shape[0] - X.shape[1] - 1

    # Estimated variance of the residuals
    var_res = rss / df_resid

    # Covariance matrix of the estimated parameters
    cov_matrix = var_res * np.linalg.inv(X_design.T @ X_design)

    # Standard errors (square root of the diagonal of the covariance matrix)
    std_err = np.sqrt(np.diag(cov_matrix))

    # t-statistics and two-tailed p-values
    t_stats = params / std_err
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))

    # Assemble results in a DataFrame (skip the intercept)
    p_data = pd.DataFrame({
        'Feature': X.columns,
        'Coef': params[1:],
        'p-value': p_values[1:]
    })

    return p_data

```

Using this approach, we fit a simple linear model and compute the p-values — a useful piece of information to complement our analysis:

```python
# Compute coefficients’ p‑values for the linear model
p_values = compute_pvalues_linearly(X, y)
```

Then we can visualise all the results in a single plot as follows:

```python
# Merge coefficient summary with p-values and create an effect label
summary_complete = summary.merge(p_values[['Feature', 'p-value']], on='Feature')
summary_complete['Effect'] = summary_complete['Coef'].apply(
    lambda x: 'Positive' if x >= 0 else 'Negative'
)

# Prepare dataframe for plotting: sort by absolute coefficient size
plot_df = summary_complete.sort_values(by='Coef', ascending=False).reset_index(drop=True)
plot_df['Coef'] = plot_df['Coef'].abs()

fig, ax = plt.subplots()

sns.barplot(
    data=plot_df,
    x='Feature',
    y='Coef',
    hue='Effect',
    ax=ax,
    palette={"Positive": "#44e271", "Negative": "#e81d0b"}
)

# Add significance stars for coefficients with p < 0.05
for i, row in plot_df.iterrows():
    if row['p-value'] < 0.05:
        ax.text(i, row['Coef'] + 0.004, '*', ha='center')

# Add a vertical reference line (e.g. to visually separate larger effects)
ax.vlines(4.5, 0, 0.3, color='black', linestyles='dashed', alpha=0.5)

# Display RMSE inside the same axis
ax.text(
    0.15, 0.95,
    f"RMSE: {round(err, 2)}",
    transform=ax.transAxes,
    ha='right',
    fontsize=9
)

# Rotate x-tick labels for readability
plt.xticks(rotation=90)

# Adjust layout and render the figure
plt.tight_layout()
plt.show()
```

![linear_model_summary.png](Part%202%20Modelling%20(ML%20&%20Statistics)/linear_model_summary.png)

The asterisks indicate the features that are statistically significant in their linear relationship with quality. As we observed earlier, some features have a positive impact (shown in green), while others contribute negatively (shown in red).

We could check whether a penalised regression yields more stable and reliable coefficients. Let’s therefore fit an Elastic Net model using cross‑validation.

```python
from sklearn.linear_model import ElasticNetCV

# Recreate train/test split from the original (unscaled) data
X = df[df['outliers'] != -1].drop(['quality', 'quality_cat', 'outliers'], axis=1)
y = df[df['outliers'] != -1]['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Standard-scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Elastic Net with built-in cross-validation
elastic_net_model = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.9],
    alphas=None,
    cv=5,
    n_jobs=-1
)

# Fit the model
elastic_net_model.fit(X_train_scaled, y_train)

# Predict and compute RMSE
pred = elastic_net_model.predict(X_test_scaled)
err = root_mean_squared_error(y_test, pred)

# Prepare summary table
summary = pd.DataFrame({
    'Feature': X.columns,
    'Coef': elastic_net_model.coef_
})

summary_complete = summary.merge(
    p_values[['Feature', 'p-value']], on='Feature'
)

summary_complete['Effect'] = summary_complete['Coef'].apply(
    lambda x: 'Positive' if x >= 0 else 'Negative'
)
```

Note that, because this is a penalised model, we must scale the training data to obtain more reliable and interpretable results. It is crucial to fit the scaler only on the training set and then use that fitted scaler to transform the test set, thereby avoiding data leakage.
Let us now generate the same type of plot as before.

```python
# Prepare dataframe for plotting: sort by coefficient size and use absolute values
plot_df = summary_complete.sort_values(by='Coef', ascending=False).reset_index(drop=True)
plot_df['Coef'] = plot_df['Coef'].abs()

fig, ax = plt.subplots(figsize=(7, 5))

sns.barplot(
    data=plot_df,
    x='Feature',
    y='Coef',
    hue='Effect',
    palette={"Positive": "#44e271", "Negative": "#e81d0b"},
    ax=ax
)

# Display RMSE in the axes
ax.text(
    0.15, 0.95,
    f"RMSE: {err:.2f}",
    transform=ax.transAxes,
    ha='right',
    fontsize=9
)

# Display Elastic Net hyperparameters
ax.text(
    0.98, 0.79,
    f"alpha: {elastic_net_model.alpha_:.2f}",
    transform=ax.transAxes,
    ha='right',
    fontsize=9
)

ax.text(
    0.98, 0.75,
    f"l1_ratio: {elastic_net_model.l1_ratio_:.2f}",
    transform=ax.transAxes,
    ha='right',
    fontsize=9
)

# Add significance stars for coefficients with p < 0.05
for i, row in plot_df.iterrows():
    if row['p-value'] < 0.05:
        ax.text(i, row['Coef'] + 0.004, '*', ha='center')

# Reference vertical line (e.g. to visually separate larger effects)
ax.vlines(4.5, 0, 0.3, color='black', linestyles='dashed', alpha=0.5)

# Rotate x-tick labels for readability
plt.xticks(rotation=90, fontsize=10)

plt.tight_layout()
plt.show()
```

![linea_model_summary_elastic.png](Part%202%20Modelling%20(ML%20&%20Statistics)/linea_model_summary_elastic.png)

The RMSE is 0.64—not terrible—but we don’t see any noticeable improvement in predictive performance. In fact, the characteristics of the penalisation are indicating that it is very soft, therefore the model does not change too much from the classical linear regression.

We can try to use polynomial regression in order to see if increasing the complexity of the model can help with the performance:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Reuse X and y defined previously (filtered to exclude outliers)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=42
)

list_of_rmse_test = []
list_of_rmse_train = []
degrees = [1, 2, 3, 4, 5]

# Evaluate polynomial models of increasing degree
for degree in degrees:
    model = Pipeline([
        ("scaling", StandardScaler()),
        ("poly", PolynomialFeatures(
            degree=degree,
            include_bias=False
        )),
        ("lin", LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    print(f"Fitted degree {degree}")
    
    pred_test = model.predict(X_test)
    pred_train = model.predict(X_train)
    
    rmse_test = root_mean_squared_error(y_test, pred_test)
    rmse_train = root_mean_squared_error(y_train, pred_train)
    
    list_of_rmse_test.append(rmse_test)
    list_of_rmse_train.append(rmse_train)
```

Then we can plot the results:

```python
plt.figure(figsize=(8, 5))

# Test RMSE
sns.lineplot(
    x=degrees, y=list_of_rmse_test,
    color="orange", label="Test RMSE"
)
sns.scatterplot(
    x=degrees, y=list_of_rmse_test,
    color="orange", s=60
)

# Train RMSE
sns.lineplot(
    x=degrees, y=list_of_rmse_train,
    color="steelblue", label="Train RMSE"
)
sns.scatterplot(
    x=degrees, y=list_of_rmse_train,
    color="steelblue", s=60
)

plt.xlabel("Polynomial degree", fontsize=11)
plt.ylabel("RMSE", fontsize=11)
plt.xticks(degrees)
plt.legend()
plt.tight_layout()
plt.show()
```

![poly_reg_elbow.png](Part%202%20Modelling%20(ML%20&%20Statistics)/poly_reg_elbow.png)

There is a very little improvement using degree 2 with about 0.62 as RMSE. Let’s try with elastic net using that degree just to be sure:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1) Polinomi sul train, poi applicati al test
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 2) Scaling dopo i polinomi
scaler = StandardScaler()
X_train_poly = scaler.fit_transform(X_train_poly)
X_test_poly = scaler.transform(X_test_poly)

# 3) Elastic Net con CV
elastic_net_model = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.9],
    alphas=None,
    cv=5,
    n_jobs=-1,
    random_state=42,
    max_iter=20000
)

elastic_net_model.fit(X_train_poly, y_train)

pred = elastic_net_model.predict(X_test_poly)
rmse = root_mean_squared_error(y_test, pred)
print("RMSE test:", rmse)
```

```python
RMSE test: 0.625023538631123
```

We can confidently say that this is the best result we can achieve with linear or polynomial regression. Let us now see what we can do using a non-parametric model:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Base random forest model
base_model = RandomForestRegressor(random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [200, 500, 800],
    'max_depth': [None, 10, 20, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.3, 0.5]
}

# Grid search with 5-fold cross-validation, optimising negative MSE
grid_model = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Fit the model on the training data
grid_model.fit(X_train, y_train)
```

Evaluating the best model performance:

```python
best_rf_reg = grid_model.best_estimator_

pred = best_rf_reg.predict(X_test)
rmse = root_mean_squared_error(y_test, pred)
print("RMSE test:", rmse)
```

```python
RMSE test: RMSE test: 0.57992387235099
```

It seems that we obtain an improvement of roughly 0.06 points in RMSE. Out of curiosity, we can now analyse the feature effects estimated by this model and check whether they are consistent with the coefficients of the linear model.  In the linear model, each coefficient can be interpreted as a partial derivative of the estimated function. Let $\hat{f}$ denote the model’s estimate of the true unknown function $f(X)$, where $y = f(X)$. If the model is linear, then we have:

$$
\hat{f}(x_1,x_2,\dots,x_p) = \sum_{i=1}^p \beta_ix_i \implies \frac{\partial\hat{f}}{\partial x_j} = \beta_j \ \forall j=1,2,\dots,n
$$

We can generalise this idea by recalling the definition of a partial derivative.
For a differentiable function $\hat{f}(X)$, the partial derivative with respect to feature $x_i$ is defined as:

$$
\frac{\partial\hat{f}}{\partial x_j} (X^0)= \lim_{h \to 0} \frac{\hat{f}(x_1^0,\dots,x_k^0 + h,\dots,x_p^0) - \hat{f}(x_1^0,\dots,x_k^0,\dots,x_p^0)}{h}
$$

that can be approximate discretely, even the function $\hat{f}$ is not smooth, as follows:

$$
\frac{\partial\hat{f}}{\partial x_j} (X^0) \approx \frac{\hat{f}(x_1^0,\dots,x_k^0 + h,\dots,x_p^0) - \hat{f}(x_1^0,\dots,x_k^0,\dots,x_p^0)}{h}
$$

From a coding perspective, we can implement this idea as follows:

```python
from scipy.stats import pearsonr

def compute_local_partial_effects_with_auto_eps(
    model,
    X,
    frac_eps_list=None,
    max_samples=300,
    corr_threshold=0.95,
    random_state=42
):
    """
    Estimate local partial effects (finite-difference approximations of
    partial derivatives) for each feature, and automatically select a
    stable step size frac_eps based on correlation between successive
    derivative estimates.

    Parameters
    ----------
    model : object
        Fitted model with a .predict(X) method.
    X : pandas.DataFrame or array-like
        Input data on which to estimate local partial effects.
    frac_eps_list : list of float, optional
        Grid of step sizes in z-space (standardised features).
        If None, a default grid is used.
    max_samples : int, optional
        Maximum number of samples used in the scan for frac_eps stability.
    corr_threshold : float, optional
        Minimum Pearson correlation between successive derivative estimates
        to consider frac_eps as "stable".
    random_state : int, optional
        Seed for the subsampling step.

    Returns
    -------
    D : pandas.DataFrame
        Matrix of local partial effects (n_samples x n_features).
    best_frac : float
        Selected step size frac_eps.
    results_df : pandas.DataFrame
        Diagnostics for each frac_eps (mean_abs, std_abs, corr_with_prev).
    """

    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X_values = X.values.astype(float)
    n, p = X_values.shape
    feature_names = X.columns

    # Default grid of relative step sizes (in z-space)
    if frac_eps_list is None:
        frac_eps_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

    # Subsample rows for the frac_eps scan (to save computation)
    rng = np.random.RandomState(random_state)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X_sub = X.iloc[idx].copy()
    else:
        X_sub = X.copy()

    X_sub_values = X_sub.values.astype(float)

    # Column-wise standard deviations (over the full dataset)
    col_std_full = X_values.std(axis=0, ddof=0)
    col_std_full[col_std_full == 0] = 1.0  # avoid division by zero

    def compute_D_given_frac_eps(X_mat, frac_eps, col_std):
        """
        Compute local finite-difference derivatives for a given frac_eps
        on a matrix X_mat (np.array), with respect to z_j = x_j / sd_j,
        using Δz = frac_eps.
        """
        n_local, p_local = X_mat.shape
        f_base_local = model.predict(X_mat)
        derivatives_local = np.zeros((n_local, p_local), dtype=float)

        # Step in x corresponding to a step frac_eps in z
        eps_vec = frac_eps * col_std

        for j in range(p_local):
            X_eps = X_mat.copy()
            eps_j = eps_vec[j]

            X_eps[:, j] += eps_j
            f_eps = model.predict(X_eps)

            # (f(x + Δx_j) - f(x)) / Δz, with Δz = frac_eps
            derivatives_local[:, j] = (f_eps - f_base_local) / frac_eps

        return derivatives_local

    # Scan over frac_eps_list to assess stability of the derivative estimates
    results = []
    D_prev = None

    for fe in frac_eps_list:
        D_local = compute_D_given_frac_eps(X_sub_values, fe, col_std_full)

        mean_abs = np.mean(np.abs(D_local))
        std_abs = np.std(np.abs(D_local))

        if D_prev is not None:
            # Guard against constant arrays to avoid pearsonr errors
            if np.std(D_local) == 0 or np.std(D_prev) == 0:
                r = np.nan
            else:
                r, _ = pearsonr(D_local.ravel(), D_prev.ravel())
        else:
            r = np.nan

        results.append({
            "frac_eps": fe,
            "mean_abs": mean_abs,
            "std_abs": std_abs,
            "corr_with_prev": r
        })

        D_prev = D_local

    results_df = pd.DataFrame(results)

    # Automatic selection of best_frac:
    # choose the first frac_eps whose correlation with the previous one
    # exceeds the chosen threshold
    stable_mask = results_df["corr_with_prev"] >= corr_threshold
    candidates = results_df[stable_mask].dropna(subset=["corr_with_prev"])

    if len(candidates) > 0:
        best_frac = candidates.iloc[0]["frac_eps"]
    else:
        # Fallback: use the frac_eps with highest corr_with_prev
        valid = results_df.dropna(subset=["corr_with_prev"])
        if len(valid) > 0:
            best_frac = valid.loc[valid["corr_with_prev"].idxmax(), "frac_eps"]
        else:
            # Extreme case: no correlation could be computed
            best_frac = frac_eps_list[len(frac_eps_list) // 2]

    # Final computation of D on the full X using best_frac
    D_full = compute_D_given_frac_eps(X_values, best_frac, col_std_full)
    D = pd.DataFrame(D_full, index=X.index, columns=feature_names)

    return D, best_frac, results_df
```

```python
D, best_frac, res = compute_local_partial_effects_with_auto_eps(
    best_rf_reg,
    pd.DataFrame(X_train, columns=X.columns),
    frac_eps_list=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
    max_samples=300,
    corr_threshold=0.95
)
```

`D` is essentially a matrix where the entry $(i, j)$ represents the partial derivative with respect to feature $j$, evaluated at sample $i$.

To visualise the average effect of each feature — in a way that is comparable to the coefficients of the linear model — we can simply take the mean across all samples using `D`:

```python
# Prepare a dataframe summarising the average partial effects
rf_df = pd.DataFrame({
    "Feature": D.columns,
    "Effect": np.where(D.mean() >= 0, "Positive", "Negative"),
    "Coef": D.mean().values,          # absolute magnitude for plotting
    "RawEffect": D.mean().values,     # signed effect (not used in the plot)
    "SD": D.std().values              # standard deviation of the local effects
})

# Sort by (signed) effect, then take absolute value for the bar plot
rf_df = rf_df.sort_values(by="Coef", ascending=False).reset_index(drop=True)
rf_df["Coef"] = rf_df["Coef"].abs()

# ---- Plot ----
fig, ax = plt.subplots(figsize=(7, 5))

sns.barplot(
    data=rf_df,
    x="Feature",
    y="Coef",
    hue="Effect",
    palette={"Positive": "#44e271", "Negative": "#e81d0b"},
    ax=ax
)

# Display the RMSE of the Random Forest model
ax.text(
    0.15, 0.95,
    f"RMSE: {rmse:.2f}",
    transform=ax.transAxes,
    ha="right",
    fontsize=9
)

plt.xticks(rotation=90, fontsize=10)
ax.legend(fontsize=9)

# Optional reference vertical line
ax.vlines(6.5, 0, ax.get_ylim()[1], color="black", linestyles="dashed", alpha=0.5)

plt.tight_layout()
plt.show()
```

![effect_rf.png](Part%202%20Modelling%20(ML%20&%20Statistics)/effect_rf.png)

The average effects are broadly consistent with the previous results, but now they are more precise because the model captures non-linear relationships as well.
This representation is still an approximation, however. We can inspect the full distribution of the local derivatives by focusing on one feature at a time.
In particular, we can use the following code:

```python
from scipy.interpolate import griddata  # needed for interpolation on the PCA plane
from matplotlib.colors import LinearSegmentedColormap

bad_color  = "#e81d0b"    # rosso
neutral    = "#ffffff"    # bianco
good_color = "#44e271"    # verde

# colormap a tre punti: negativo → bianco → positivo
cmap = LinearSegmentedColormap.from_list(
    "my_cmap_centered",
    [bad_color, neutral, good_color]
)

# Standardise features and compute PCA on the training set
scaler_pca = StandardScaler()
X_train_scaled = scaler_pca.fit_transform(X_train)

pca = PCA(n_components=2, random_state=42)
scores = pca.fit_transform(X_train_scaled)

# DataFrame with the first two principal components
pca_df = pd.DataFrame(scores, columns=["PCA1", "PCA2"])

def plot_feature_partial_pca(feature_name, D, pca_df, cmap,
                             save=False, name_to_save_with='default', levels=200):

    vals = D[feature_name].values   # true local derivatives

    P1 = pca_df["PCA1"].values
    P2 = pca_df["PCA2"].values

    points = np.column_stack([P1, P2])

    # regular grid on the PCA plane
    nx, ny = 80, 80
    x_grid = np.linspace(P1.min(), P1.max(), nx)
    y_grid = np.linspace(P2.min(), P2.max(), ny)
    Xg, Yg = np.meshgrid(x_grid, y_grid)

    # interpolation
    Z = griddata(points, vals, (Xg, Yg), method='cubic')
    Z = np.nan_to_num(Z, nan=np.nanmedian(vals))

    # REAL limits (from D)
    vmin = vals.min()
    vmax = vals.max()

    # clip Z to the interval [vmin, vmax] to avoid interpolation outliers
    Z_clipped = np.clip(Z, vmin, vmax)

    # norm centered at 0
    norm = mcolors.TwoSlopeNorm(
        vmin=vmin,
        vcenter=0.0,
        vmax=vmax
    )

    # uniform levels between vmin and vmax → colorbar consistent with D
    level_values = np.linspace(vmin, vmax, levels)

    plt.figure(figsize=(7, 5))

    # original points
    plt.scatter(P1, P2, s=8, alpha=1, color="black", label="data points")

    # interpolated map
    cs = plt.contourf(
        Xg, Yg, Z_clipped,
        levels=level_values,
        cmap=cmap,
        norm=norm,
        alpha=0.85
    )

    cbar = plt.colorbar(cs)
    cbar.set_label(f"Local ∂f/∂{feature_name}")

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"Local effect of '{feature_name}' in PCA space")
    plt.legend(loc="upper right")
    plt.tight_layout()
    if save:
        plt.savefig(name_to_save_with, bbox_inches='tight', dpi=300)
    plt.show()
```

We can analyse the effect of some features more precisely through the command:

```python
plot_feature_partial_pca("col_to_explore", D, pca_df, cmap)
```

Some of them are represented below:

![alchool_effect.png](Part%202%20Modelling%20(ML%20&%20Statistics)/alchool_effect.png)

![density.png](Part%202%20Modelling%20(ML%20&%20Statistics)/density.png)

![vol_acidity.png](Part%202%20Modelling%20(ML%20&%20Statistics)/vol_acidity.png)

![sulphates_effect.png](Part%202%20Modelling%20(ML%20&%20Statistics)/sulphates_effect.png)

`alcohol` has a positive average effect, but by examining the distribution of:

$$
\frac{\partial \texttt{quality}}{\partial \texttt{alcohol}}
$$

we obtain a much clearer picture of its impact on wine quality.
Conversely, even `density`—which shows a negative average effect—exhibits several regions where its local effect becomes positive. 

A particularly interesting phenomenon emerges in the bottom-left region of the PCA plane: here, sulphates and `alcohol` display completely opposite effects. This pattern aligns with established chemical and sensory evidence. Ethanol ($(\ce{C2H5OH})$) enhances body and aromatic volatility only up to a certain threshold, beyond which it produces excessive warmth and heaviness (King et al., 2013).
In contrast, `sulphates`—closely related to free and total ($\ce{SO2}$)—contribute to microbial stability, oxidative protection, and freshness, especially when ethanol levels are already high (Chandra et al., 2015; Cisilotto et al., 2021).

These interactions plausibly explain why the Random Forest identifies regions where the local partial derivative with respect to `alcohol` is negative while the derivative for sulphates is positive. This highlights the inherently **local, nonlinear, and context-dependent behaviour** captured by non-parametric models.

# Classification

Let’s try to classify whether a wine, given a specific set of chemical features, is “good” or “bad” according to the definition introduced in Part 1.

Next, we’ll define a function that evaluates the p‑value of each coefficient, just as we did before, but this time using a classical logistic regression model.

```python
from sklearn.linear_model import LogisticRegression

def compute_pvalues_logistic(X, y):
    """
    Fit a classical (unpenalised) logistic regression model and compute
    z-statistics and p-values for each coefficient.

    Parameters
    ----------
    X : pandas.DataFrame
        Predictor matrix.
    y : array-like (categorical)
        Target vector, containing 'good' or 'bad' labels.

    Returns
    -------
    pandas.DataFrame
        Table with feature names, coefficients, and two-tailed p-values.
    """

    # Encode the target: 'good' → 1, everything else → 0
    y_bin = (y == "good").astype(int)

    # Fit the logistic regression model (no regularisation)
    model = LogisticRegression(
        penalty=None,
        solver='lbfgs',
        max_iter=1000
    ).fit(X, y_bin)

    # Extract parameters
    intercept = model.intercept_[0]
    coefs = model.coef_[0]
    params = np.append(intercept, coefs)

    # Build design matrix (intercept + X)
    X_mat = X.to_numpy()
    X_design = np.column_stack([np.ones(X_mat.shape[0]), X_mat])

    # Predicted probabilities
    p = model.predict_proba(X)[:, 1]

    # Weight vector W = p * (1 - p)
    # Clip to avoid exact zeros (which cause singular matrices)
    W = np.clip(p * (1 - p), 1e-10, None)

    # Compute Xᵀ W X
    XTWX = X_design.T @ (W[:, None] * X_design)

    # Invert to obtain asymptotic covariance matrix
    cov_matrix = np.linalg.inv(XTWX)

    # Standard errors
    std_err = np.sqrt(np.diag(cov_matrix))

    # z-statistics and two-tailed p-values
    z_stats = params / std_err
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))

    # Build output dataframe (skip intercept)
    p_data = pd.DataFrame({
        "Feature": X.columns,
        "Coef": params[1:],
        "p-value": p_values[1:]
    })

    return p_data
```

Next, we fit two models side by side: an Elastic Net–penalised logistic regression (with cross-validation to select the optimal hyperparameters) and a classical logistic regression, for which we compute p-values.

```python
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Use only non-outlier samples and drop the original quality score
X = df[df["outliers"] != -1].drop(["quality", "quality_cat", "outliers"], axis=1)
y = df[df["outliers"] != -1]["quality_cat"]

# Compute p-values for the classical (unpenalised) logistic regression
p_values = compute_pvalues_logistic(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=42
)

# Standardise predictors (required for penalised logistic regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Binary encoding of the target: good -> 1, bad -> 0
y_train_bin = (y_train == "good").astype(int)
y_test_bin = (y_test == "good").astype(int)

# Base Elastic Net logistic regression model
logreg = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    max_iter=10000
)

# Hyperparameter grid (C = inverse regularisation strength, l1_ratio controls L1/L2 mix)
param_grid = {
    "C": np.logspace(-3, 3, 10),
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
}

# Grid search with 5-fold CV, optimising F1-score
grid = GridSearchCV(
    estimator=logreg,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# Fit the penalised logistic regression
grid.fit(X_train, y_train_bin)
```

From which we can extract the best combination of regularisation strength (`C`) and L1/L2 mixing (`l1_ratio`) that maximises the F1‑score, retrieve the corresponding trained Elastic‑Net logistic model, and then compare its performance (accuracy, precision, recall, ROC‑AUC, etc.) against the classical logistic regression whose coefficients and p‑values we computed with our function.

```python
	            precision    recall  f1-score   support

           0       0.72      0.70      0.71       226
           1       0.75      0.76      0.75       257

    accuracy                           0.73       483
   macro avg       0.73      0.73      0.73       483
weighted avg       0.73      0.73      0.73       483

```

Not so bad — clearly better than a random classifier.Now let us visualise the results graphically.

```python
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Sort features by absolute coefficient size and take absolute values
plot_df = summary_complete.sort_values(by='Coef', ascending=False).reset_index(drop=True)
plot_df['Coef'] = plot_df['Coef'].apply(lambda x: abs(x))

# Main figure and axis for the bar plot
fig, ax = plt.subplots(figsize=(7, 5))

# Bar plot of coefficient magnitudes, colored by effect direction
sns.barplot(
    data=plot_df,
    x='Feature',
    y='Coef',
    hue='Effect',
    palette={"Positive": "#44e271", "Negative": "#e81d0b"},
    ax=ax
)
plt.xticks(rotation=90, fontsize=10)

# Create an inset axis (top‑right) for the normalized confusion matrix
ax_inset = inset_axes(
    ax,
    width="35%", height="35%", loc="upper right",
    bbox_to_anchor=(0.25, 0.15, 0.8, 0.8),
    bbox_transform=ax.transAxes,
    borderpad=1
)

# Compute and display the normalized confusion matrix
cm = confusion_matrix(y_test_bin, pred, normalize='true')
disp = ConfusionMatrixDisplay(cm, display_labels=['bad', 'good'])
disp.plot(ax=ax_inset, colorbar=False)

# Remove titles/labels from the inset for a cleaner look
ax_inset.set_title("")
ax_inset.set_xlabel("")
ax_inset.set_ylabel("")
# Reduce font size of tick labels inside the inset
for label in ax_inset.get_xticklabels() + ax_inset.get_yticklabels():
    label.set_fontsize(7)

# ----------------------------------------------
# Position the legend (outside the bar plot, top‑left)
# ---------------------------------------------
ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.94), fontsize=9)

# ----------------------------------------
# Add asterisk (*) above bars whose p‑value < 0.05 and coefficient ≤ 5
# -------------------------------------------
for i, row in plot_df.iterrows():
    if row['p-value'] < 0.05 and row['Coef'] <= 5:
      ax.text(i, row['Coef'] + 0.004, '*', ha='center')

plt.show()
```

![plot_logistic.png](Part%202%20Modelling%20(ML%20&%20Statistics)/plot_logistic.png)

By penalising the model, we can identify four features that are both statistically significant and useful for building a classifier with reasonably good performance, especially considering its simplicity.

To visualise the power of this classifier, we can project the decision boundary and the predicted class probabilities onto the principal component space.

```python
# -------------------------------------------------------------------
# 1) Data preparation: remove outliers and select feature matrix / labels
# -------------------------------------------------------------------
X = df[df["outliers"] != -1].drop(["quality", "quality_cat", "outliers"], axis=1)
y = df[df["outliers"] != -1]["quality_cat"]

# -------------------------------------------------------------------
# 2) Standardise the features using the SAME scaler as in the logistic model
#    (we assume 'scaler' was already fitted on the training data)
# -------------------------------------------------------------------
X_scaled = scaler.transform(X)

# -------------------------------------------------------------------
# 3) PCA on the scaled features
# -------------------------------------------------------------------
pca = PCA(n_components=2)
scores = pca.fit_transform(X_scaled)
pca_data = pd.DataFrame(scores, columns=["PCA1", "PCA2"])

# Add labels for plotting
pca_data_plot = pca_data.copy()
pca_data_plot["quality_cat"] = y.values

# -------------------------------------------------------------------
# 4) Create main figure + single axis
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

# PCA loadings (directions of maximal variance)
ldngs = pca.components_
features = X.columns

# -------------------------------------------------------------------
# 5) Plot normalised loadings (arrows)
# -------------------------------------------------------------------
for i, feature in enumerate(features):

    # Loading coordinates (PC1, PC2)
    x_arrow = ldngs[0, i]
    y_arrow = ldngs[1, i]

    # Normalise the arrow (so all arrows have comparable length)
    norm = np.sqrt(x_arrow**2 + y_arrow**2)
    x_arrow_norm = x_arrow / norm
    y_arrow_norm = y_arrow / norm

    # Draw arrow
    ax.arrow(
        0, 0,
        x_arrow_norm * 3,
        y_arrow_norm * 3,
        head_width=0.2,
        head_length=0.2,
        length_includes_head=True,
        color="black",
        zorder=3
    )

    # Feature name
    ax.text(
        x_arrow_norm * 3 * 1.05,
        y_arrow_norm * 3 * 1.05,
        feature,
        fontsize=10,
        zorder=4
    )

# -------------------------------------------------------------------
# 6) Scatter plot of PCA scores, coloured by class ("good"/"bad")
# -------------------------------------------------------------------
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    data=pca_data_plot,
    hue="quality_cat",
    palette={"good": "#44e271", "bad": "#e81d0b"},
    alpha=0.4,
    edgecolor="black",
    ax=ax,
    zorder=1
)

ax.legend(loc="upper right", title="quality_cat", frameon=True)

# -------------------------------------------------------------------
# 7) Build a grid in PCA space for the decision map
# -------------------------------------------------------------------
x_min_plot = pca_data_plot.PCA1.min() - 0.05
x_max_plot = pca_data_plot.PCA1.max() + 0.05

y_min_plot = pca_data_plot.PCA2.min() - 0.05
y_max_plot = pca_data_plot.PCA2.max() + 0.05

nx, ny = 30, 30
xs_plot = np.linspace(x_min_plot, x_max_plot, nx)
ys_plot = np.linspace(y_min_plot, y_max_plot, ny)

XX_plot, YY_plot = np.meshgrid(xs_plot, ys_plot)

# Flatten the PCA grid into a list of points
grid_pca = np.column_stack([XX_plot.ravel(), YY_plot.ravel()])

# -------------------------------------------------------------------
# 8) Inverse PCA to obtain points in the (scaled) feature space
# -------------------------------------------------------------------
X_for_model_scaled = pca.inverse_transform(grid_pca)

# -------------------------------------------------------------------
# 9) Predicted probability for each grid point (P(y = 'good'))
# -------------------------------------------------------------------
grid_probs = best_log_clf.predict_proba(X_for_model_scaled)[:, 1]
Z = grid_probs.reshape(YY_plot.shape)

# -------------------------------------------------------------------
# 10) Smooth decision map (background heatmap)
# -------------------------------------------------------------------
# We reuse bad_color / good_color defined earlier
cmap_prob = LinearSegmentedColormap.from_list("prob_map", [bad_color, good_color])

im = ax.imshow(
    Z,
    extent=[x_min_plot, x_max_plot, y_min_plot, y_max_plot],
    origin="lower",
    cmap=cmap_prob,
    interpolation="bicubic",
    alpha=0.6,
    zorder=0
)

# -------------------------------------------------------------------
# 11) Decision boundary: contour where P(good) = 0.5
# -------------------------------------------------------------------
ax.contour(
    XX_plot,
    YY_plot,
    Z,
    levels=[0.5],
    colors="black",
    linewidths=2,
    linestyles="dashed",
    zorder=2
)

# -------------------------------------------------------------------
# 12) Colourbar with the same height as the main plot (inset_axes)
# -------------------------------------------------------------------
cax = inset_axes(
    ax,
    width="3%",       # relative width
    height="100%",    # full height of the main axis
    loc="center left",
    bbox_to_anchor=(1.02, 0, 1, 1),  # slightly outside to the right
    bbox_transform=ax.transAxes,
    borderpad=0
)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label("P(good)")

ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")

plt.tight_layout()
plt.show()
```

![pca_decision_map.png](Part%202%20Modelling%20(ML%20&%20Statistics)/pca_decision_map.png)

The decision boundary is linear, since the classifier we used is logistic regression and this is a linear model in the feature space. To classify wine quality as accurately as possible, we would probably need a more flexible model than a linear one.

Let us therefore try a tree-based approach, specifically a Random Forest classifier.

```python
from sklearn.ensemble import RandomForestClassifier

# Use only non-outlier samples and drop target/label columns
X = df[df["outliers"] != -1].drop(["quality", "quality_cat", "outliers"], axis=1)
y = df[df["outliers"] != -1]["quality_cat"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=42
)

# Binary encoding: good -> 1, bad -> 0
y_train_bin = (y_train == "good").astype(int)
y_test_bin = (y_test == "good").astype(int)

# Base Random Forest classifier
base_model = RandomForestClassifier(random_state=42)

# Hyperparameter grid
param_grid = {
    "n_estimators": [100, 300, 500, 800],
    "max_depth": [None, 5, 10, 20, 40],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
}

# Grid search with 5-fold CV, optimising F1-score
grid_model = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_model.fit(X_train, y_train_bin)

```

The model selected by cross-validation is a Random Forest with 100 trees.
Let us now evaluate its performance.

```python
# Retrieve the best Random Forest model selected by GridSearchCV
best_rf_clf = grid_model.best_estimator_

# Predict the test-set labels with the tuned model
pred = best_rf_clf.predict(X_test)

# Full classification report (precision, recall, F1-score, support)
print(classification_report(y_test_bin, pred, target_names=["bad", "good"]))

# Normalised confusion matrix
cm = confusion_matrix(y_test_bin, pred, normalize="true")

disp = ConfusionMatrixDisplay(
    cm,
    display_labels=["bad", "good"]
)
disp.plot(colorbar=False)
plt.show()
```

```
              precision    recall  f1-score   support

           0       0.78      0.77      0.77       226
           1       0.80      0.81      0.80       257

    accuracy                           0.79       483
   macro avg       0.79      0.79      0.79       483
weighted avg       0.79      0.79      0.79       483
```

![cm_rf.png](Part%202%20Modelling%20(ML%20&%20Statistics)/8b73bab1-1ac0-4965-8001-e60f0fc8ee59.png)

The performance is clearly better than that of the logistic model, even if it is still far from perfect.
Let us now take a closer look at the decision boundary of this non-linear model.

![pca_decision_map_rf.png](Part%202%20Modelling%20(ML%20&%20Statistics)/pca_decision_map_rf.png)

Of course, the decision boundary here is far from linear — as expected from a tree-based model.

After experimenting with other classifiers, such as Gradient Boosting or Support Vector Machines, the performance does not appear to improve substantially. This suggests that the Random Forest classifier is likely among the best options available for this dataset, given its size and the complexity of the underlying relationships.

In summary, this analysis shows that while linear models provide valuable interpretability, capturing the non-linear structure of the data requires more flexible approaches. Among the models tested, Random Forests offered the best balance between performance and robustness, suggesting that moderate complexity is sufficient to model the chemical factors driving wine quality in this dataset.
