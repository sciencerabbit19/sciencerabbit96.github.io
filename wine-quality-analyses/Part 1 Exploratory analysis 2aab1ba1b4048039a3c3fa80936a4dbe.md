# Part 1. Exploratory analysis

Let's start by importing the data we want to analyse:

```python
# Import fundamental libraries
import numpy as np               
import pandas as pd              
import matplotlib.pyplot as plt  
import seaborn as sns            

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
```

First, let's take a look at the data:

```python
df.head()
```

![image.png](Part%201%20Exploratory%20analysis/image.png)

Let's examine the numerical summary of this dataset:

```python
df.describe().transpose()
```

![image.png](Part%201%20Exploratory%20analysis/image%201.png)

To understand the relationships between these variables, let's compute the correlation matrix and focus on the strongest correlations:

```python
# Compute the full correlation matrix for the DataFrame
cor_greater_than_30 = df.corr()

# Keep only the correlations whose absolute value exceeds 0.3
cor_greater_than_30 = (
    cor_greater_than_30[abs(cor_greater_than_30) > 0.3]
    .map(lambda x: "" if pd.isna(x) or x == 1 else str(round(x, 2)))
)
```

This captures correlations with a value of $|\rho| > 0.3$, which we can visualise with the following code:

```python
# Set the size of the figure (width = 7.5 inches, height = 6 inches)
plt.figure(figsize=(7.5, 6))

# Draw a heatmap of the full correlation matrix:
sns.heatmap(
    df.corr(),
    cmap='viridis',
    annot=np.array(cor_greater_than_30),
    fmt=""
)

# Render the plot on screen
plt.show()
```

![image.png](Part%201%20Exploratory%20analysis/image%202.png)

We can see some important correlations. If we're interested in wine quality, we can gather useful clues here. For instance, alcohol content correlates positively with wine quality ($\rho = 0.48$), whereas volatile acidity correlates negatively with it ($\rho = -0.39$).

To identify outliers, we can use the Isolation Forest algorithm. This algorithm identifies outliers in a non-parametric way:

```python
# Import necessary libraries
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Features matrix (remove the target column 'quality')
X = df.drop(['quality'], axis=1)

# Standardize features to zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest model for outlier detection
clf = IsolationForest(
    random_state=0,
    contamination='auto',
    n_estimators=300,
    max_samples=256,
    n_jobs=-1
)

# Fit the model on the scaled data
clf.fit(X_scaled)

# Store the prediction (-1 = outlier, 1 = inlier) in a new column
df['outliers'] = clf.predict(X_scaled)

# Show the unique labels and their counts
np.unique(df['outliers'], return_counts=True)
```

```python
(array([-1,  1]), array([  80, 1519])) # Distribution of outliers and inliers
```

To visualise the quality distribution more easily across other variables, let's categorise the quality score in a compact way:

```python
def my_func(a, b):
    """
    Classify a wine sample based on its quality score (a) and outlier flag (b).

    - If the Isolation Forest marked it as an outlier (b == -1) → 'outlier'
    - If the quality score is 3, 4, or 5 → 'bad'
    - Otherwise → 'good'
    """
    if b == -1:
        return 'outlier'
    elif a in {3, 4, 5}:
        return 'bad'
    else:
        return 'good'

# Apply the classification function row‑wise and store the result in a new column
df['quality_cat'] = df.apply(lambda row: my_func(row['quality'], row['outliers']), axis=1)
```

Since the variables have different scales, we should standardise them:

```python
# Scale the numeric features (exclude target and label columns)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Prepare a dataframe for plotting: drop the target and label columns
df_to_plot = df.drop(['quality', 'outliers', 'quality_cat'], axis=1)

# Apply standard scaling and rebuild the dataframe with original column names
df_to_plot = pd.DataFrame(scaler.fit_transform(df_to_plot))
df_to_plot.columns = df.drop(['quality', 'outliers', 'quality_cat'], axis=1).columns

# Re‑attach the categorical columns for later use in plots
df_to_plot['quality_cat'] = df['quality_cat']
df_to_plot['outliers']    = df['outliers']
```

Let's create a pair plot coloured by `quality_cat` and mark the outliers with purple crosses:

```python
def scatter_custom(x, y, alpha=0.3, alpha_out=0.6, **kwargs):
    """
    Custom scatter plot that draws in‑lier points semi‑transparent
    and outliers with higher opacity.
    """
    # Extract the DataFrame passed via the `data` argument
    df = kwargs.pop("data")

    # Boolean masks for outliers (‑1) and in‑liers (1)
    mask_out = df["outliers"] == -1
    mask_in  = df["outliers"] == 1

    # In‑lier points (transparent)
    sns.scatterplot(
        x=x[mask_in],
        y=y[mask_in],
        alpha=alpha,
        style=df["outliers"][mask_in],
        markers={1: "o", -1: "X"},
        s=50,
        legend=False,          # important: avoid duplicate legends
        **kwargs
    )

    # Outlier points (more opaque)
    sns.scatterplot(
        x=x[mask_out],
        y=y[mask_out],
        alpha=alpha_out,
        style=df["outliers"][mask_out],
        markers={1: "o", -1: "X"},
        s=50,
        legend=False,          # important: avoid duplicate legends
        **kwargs
    )

# ----------------------------------------------------------------------
# Create a PairGrid visualising all feature pairs, coloured by quality category
# ----------------------------------------------------------------------
plt.figure(dpi=300)

g = sns.PairGrid(
    df_to_plot.drop("outliers", axis=1),          # exclude the raw outlier flag
    hue="quality_cat",
    palette={'good': '#44e271', 'bad': '#e81d0b', 'outlier': 'purple'}
)

# Diagonal: KDE plots of each variable
g.map_diag(sns.kdeplot)

# Off‑diagonal: custom scatter showing in‑liers/outliers
g.map_offdiag(scatter_custom, data=df_to_plot)

# Add the legend for the quality categories
g.add_legend()

```

![pairplot_prova.png](Part%201%20Exploratory%20analysis/pairplot_prova.png)

This plot supports our hypothesis that alcohol improves wine quality while volatile acidity worsens it. The outliers appear spread throughout the data, which might suggest they aren't truly extreme—but this is simply due to our limited ability to perceive multidimensional data.

When we examine the Principal Components, the picture becomes clearer. Here's the code to calculate them:

```python
# Perform Principal Component Analysis (retain the first two components)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

# Standard‑scale the numeric features (exclude target and label columns)
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(['quality', 'outliers', 'quality_cat'], axis=1))

# Project the scaled data onto the two principal components
pca_data = pca.fit_transform(X)

# Convert the result to a DataFrame and name the component columns
pca_data = pd.DataFrame(pca_data)
pca_data.columns = ['PCA1', 'PCA2']

# Append the categorical labels for later visualisation
pca_data['quality_cat'] = df['quality_cat']
pca_data['outliers']    = df['outliers']

```

Let’s visualise the scatterplot of the first two principal components in relation the wine quality and being an outlier:

```python
# Principal component vectors (loadings) for the two components
PC1 = pca_data.iloc[:, 0]                     # First principal component scores
PC2 = pca_data.iloc[:, 1]                     # Second principal component scores

# Normalisation factors to bring the component scores into the [0, 1] range
scalePC1 = 1 / (PC1.max() - PC1.min())
scalePC2 = 1 / (PC2.max() - PC2.min())

# Loadings (coefficients) of the original features on the two PCs
ldngs = pca.components_

# Names of the original features (used for labeling the arrows)
features = df.drop(['quality', 'outliers', 'quality_cat'], axis=1).columns

# Create a square figure for the biplot
fig, ax = plt.subplots(figsize=(7, 7))

# Plot an arrow for each original feature, starting at the origin
for i, feature in enumerate(features):
    ax.arrow(
        0, 0,                         # start point (origin)
        ldngs[0, i], ldngs[1, i],     # end point (loading on PC1, loading on PC2)
        head_width=0.03,
        head_length=0.03,
        length_includes_head=True,
        color='black'
    )
    # Place the feature name slightly beyond the tip of the arrow
    ax.text(ldngs[0, i] * 1.10, ldngs[1, i] * 1.10, feature, fontsize=10)

# Scatter plot of the normalized PC scores, using the custom scatter function
scatter_custom(
    x=pca_data["PCA1"] * scalePC1,      # normalized PC1 values
    y=pca_data["PCA2"] * scalePC2,      # normalized PC2 values
    data=pca_data,
    hue=pca_data["quality_cat"],
    palette={"good": "#44e271", "bad": "#e81d0b", "outlier": "purple"},
    axes=ax
)

# Axis labels
ax.set_xlabel('PC1', fontsize=10)
ax.set_ylabel('PC2', fontsize=10)

plt.show()
```

![biplot_prova.png](Part%201%20Exploratory%20analysis/5caf2e36-4b21-42c6-80c3-35fdf01037f1.png)

The outliers are now better positioned—far from the main cloud of data points. The arrows clearly show which variables contribute positively or negatively to wine quality.

To create a plot that shows a quality gradient, we can first define a palette:

```python
import matplotlib.colors as mcolors

bad_color  = "#e81d0b"   # lowest quality
good_color = "#44e271"   # highest quality

# Build a linear colour map that interpolates between the two extremes
cmap = mcolors.LinearSegmentedColormap.from_list(
    "quality_map",
    [bad_color, good_color],
    N=6                     # number of discrete steps in the map
)

# Possible quality scores in the dataset
quality_values = [3, 4, 5, 6, 7, 8]

# Sample six colours from the colour map and convert them to hexadecimal strings
quality_palette = {
    q: mcolors.to_hex(cmap(i / (len(quality_values) - 1)))
    for i, q in enumerate(quality_values)
}

```

Then, create the plot just as we did earlier:

![biplot_prova_detailed.png](Part%201%20Exploratory%20analysis/1b13e7ce-4559-4e40-b52c-235340afe4d8.png)

It would be very interesting to examine the direction of maximal increase in `quality`. At first glance, this direction appears to be expressible as a combination of `alcohol` and `citric acid` within the space spanned by the principal components.

Treating `quality` as a function of the first two principal components:

$$
\texttt{quality}=f(\texttt{PCA}_1,\texttt{PCA}_2)
$$

the vector that yields the greatest increase in quality would be:

$$
v = \left(\frac{\partial \texttt{quality}}{\partial\texttt{PCA}_1},\frac{\partial \texttt{quality}}{\partial\texttt{PCA}_2}\right)
$$

To estimate the partial derivatives we approximate quality with a simple linear model in the two‑component space:

$$
\texttt{quality} \approx \beta_1\cdot \texttt{PCA}_1 + \beta_2 \cdot \texttt{PCA}_2
$$

Because the model is linear, the partial derivatives are constant and equal to the regression coefficients:

$$
\frac{\partial \texttt{quality}}{\partial\texttt{PCA}_1} \approx \beta_1 \text{ and } \frac{\partial \texttt{quality}}{\partial\texttt{PCA}_2} \approx \beta_2

$$

Through code we can compute the gradient easily:

```python
# Prepare the feature matrix X and target vector y
X = pca_data_no_out[["PCA1", "PCA2"]]                 # principal‑component coordinates (no outliers)
y = pca_data_no_out["quality"].astype(float)        # wine quality as a float

# Fit a simple linear regression model in the PCA space
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# Regression coefficients give the gradient of quality with respect to the PCs
grad = model.coef_

# Compute the lengths of the loading vectors (to later scale the gradient)
load_lengths = np.sqrt(ldngs[0, :]**2 + ldngs[1, :]**2)
mean_loading_len = load_lengths.mean()

# Normalize the gradient and bring it to the same scale as the loadings
grad_norm = np.linalg.norm(grad)
grad_dir = grad / grad_norm if grad_norm > 0 else grad
grad_vec_plot = grad_dir * mean_loading_len

```

Let’s create the same type of plot we made earlier:

```python
fig, ax = plt.subplots(figsize=(7, 7))

# Plot the points without outliers, colored by wine quality
scatter_custom(
    x=pca_data_no_out["PCA1"],
    y=pca_data_no_out["PCA2"],
    alpha=0.8,
    data=pca_data_no_out,
    hue=pca_data_no_out["quality"],
    palette=quality_palette,
    axes=ax,
)

# Plot the outlier points in purple
scatter_custom(
    x=pca_data_out["PCA1"],
    y=pca_data_out["PCA2"],
    data=pca_data_out,
    color='purple',
    axes=ax,
)

# Draw arrows for each original feature (loadings) and label them
for i, feature in enumerate(features):
    ax.arrow(
        0,
        0,
        ldngs[0, i],
        ldngs[1, i],
        head_width=0.03,         
        head_length=0.03,
        length_includes_head=True,
        color='black',
    )
    ax.text(
        ldngs[0, i] * 1.10,
        ldngs[1, i] * 1.10,
        feature,
        fontsize=8,
    )

# Arrow representing the direction of increasing quality
ax.arrow(
    0,
    0,
    grad_vec_plot[0],
    grad_vec_plot[1],
    head_width=0.03,
    head_length=0.03,
    length_includes_head=True,
    color=good_color,            
    linestyle='--',
)

# Label for the quality gradient vector
ax.text(
    grad_vec_plot[0] * 1.10,
    grad_vec_plot[1] * 1.10,
    "quality (grad)",
    fontsize=8,
    color=good_color,
)

# Create a color bar for the quality values
norm = mcolors.Normalize(vmin=3, vmax=8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

plt.colorbar(sm, label="quality", ax=ax)

# Axis labels
ax.set_xlabel('PC1', fontsize=10)
ax.set_ylabel('PC2', fontsize=10)

# Display the figure
plt.show()
```

![image.png](Part%201%20Exploratory%20analysis/65b8e42e-aa11-45b7-a8c3-a56916124001.png)

We can also assess how each feature relates to the gradient by calculating its scalar product with the gradient vector and the angle between them:

```python
# Prepare a DataFrame with the loadings (ldngs) for each original feature
features_component = pd.DataFrame(ldngs).T
features_component.index = features                     
features_component.columns = ['x', 'y']                

# Compute the angle (in degrees) between the quality‑gradient and direction and each feature’s loading vector
angle_between = lambda x: np.arccos(np.dot(grad_dir, x) / np.linalg.norm(x))

list_angles = []
for f in features:
    list_angles.append(angle_between(features_component.loc[f].values))

feat_angles = pd.DataFrame({'angle': list_angles, 'feature': features})
feat_angles['angle'] = (feat_angles['angle'] / np.pi) * 180   # convert rad to deg

# Define a custom colormap for low/high quality
from matplotlib.colors import LinearSegmentedColormap

# Continuous palette from good to bad
cmap = LinearSegmentedColormap.from_list("quality_cmap", [good_color, bad_color])

# Bar plot: angle between the gradient vector and each feature
plt.figure(figsize=(10, 10))
sns.barplot(
    data=feat_angles.sort_values(by='angle'),
    x='feature',
    y='angle',
    hue='angle',
    palette=cmap,
)

plt.xticks(rotation=90, fontsize=7)
plt.yticks([0, 45, 90, 135, 180])
plt.xlabel('')                                  
plt.ylabel('Angle between gradient and features (°)')
plt.show()
```

![barplot_angle.png](Part%201%20Exploratory%20analysis/7024db74-7156-43a7-862b-c35cffb6fcaf.png)

Among the variables, `alcohol` aligns most closely with the gradient (i.e., it is the most parallel vector), whereas **free sulfur** `dioxide` aligns the least (i.e., it is the least parallel).