---
layout: page
title: "Analysing Wine Quality Data"
permalink: /projects/wine-quality/
---

# Analysing wine quality data

Let's start by importing the data we want to analyse:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')
```

First, let's take a look at the data:

```python
df.head()
```

![image.png](Analysing%20wine%20quality%20data/image.png)

Let's examine the numerical summary of this dataset:

```python
df.describe().transpose()
```

![image.png](Analysing%20wine%20quality%20data/image%201.png)

To understand the relationships between these variables, let's compute the correlation matrix and focus on the strongest correlations:

```python
cor_greater_than_30 = df.corr()[abs(df.corr()) > 0.3].map(lambda x: "" if pd.isna(x) or x == 1 else str(round(x,2)))
```

This captures correlations with a value of $\rho > 0.3$, which we can visualise with the following code:

 

```python
plt.figure(figsize = (7.5,6))
sns.heatmap(df.corr(),cmap = 'viridis', 
            annot = np.array(cor_greater_than_30),
            fmt = "")
plt.show()
```

![image.png](Analysing%20wine%20quality%20data/image%202.png)

We can see some important correlations. If we're interested in wine quality, we can gather useful clues here. For instance, alcohol content correlates positively with wine quality ($\rho = 0.48$), whereas volatile acidity correlates negatively with it ($\rho = -0.39$).

# Outliers and relationship insights

To identify outliers, we can use the Isolation Forest algorithm instead of classical statistical methods. This algorithm identifies outliers in a non-parametric way:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

X = df.drop(['quality'], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = IsolationForest(
    random_state=0,
    contamination=0.05,
    n_estimators=300,
    max_samples=256,
    n_jobs=-1
)

clf.fit(X_scaled)
df['outliers'] = clf.predict(X_scaled)
np.unique(df['outliers'],return_counts=True)
```

```python
(array([-1,  1]), array([  80, 1519])) #distribution of outliers and inliers
```

To visualise the quality distribution more easily across other variables, let's categorize the quality score in a compact way:

```python
def my_func(a,b):
    if b == -1:
        return 'outlier'
    elif a in {3,4,5}:
        return 'bad'
    else:
        return 'good'

df['quality_cat'] = df.apply(lambda row: my_func(row['quality'],row['outliers']), axis = 1)
```

Since the variables have different scales, we should standardise them:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_to_plot = df.drop(['quality','outliers','quality_cat'], axis=1)

df_to_plot = pd.DataFrame(scaler.fit_transform(df_to_plot))

df_to_plot.columns = df.drop(['quality','outliers','quality_cat'], axis=1).columns

df_to_plot['quality_cat'] = df['quality_cat']

df_to_plot['outliers'] = df['outliers']
```

Let's create a pair plot coloured by `quality_cat` and mark the outliers with purple crosses:

![pairplot_prova.png](Analysing%20wine%20quality%20data/pairplot_prova.png)

This plot supports our hypothesis that alcohol improves wine quality while volatile acidity worsens it. The outliers appear spread throughout the data, which might suggest they aren't truly extreme—but this is simply due to our limited ability to perceive multidimensional data.

When we examine the Principal Components, the picture becomes clearer. Here's the code to calculate them:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(['quality', 'outliers', 'quality_cat'], axis = 1))
pca_data = pca.fit_transform(X)
pca_data = pd.DataFrame(pca_data)
pca_data.columns = ['PCA1','PCA2']
pca_data['quality_cat'] = df['quality_cat']
pca_data['outliers'] = df['outliers']
```

Let’s visualise the scatterplot of the first two principal components in relation the wine quality and being an outlier:

```python
PC1 = pca_data.iloc[:,0]
PC2 = pca_data.iloc[:,1]

scalePC1 = 1/(PC1.max() - PC1.min())
scalePC2 = 1/(PC2.max() - PC2.min())

ldngs = pca.components_

features = df.drop(['quality', 'outliers', 'quality_cat'], axis = 1).columns

fig, ax = plt.subplots(figsize=(7, 7))
 
for i, feature in enumerate(features):
    ax.arrow(0, 0, ldngs[0, i], 
             ldngs[1, i],
             head_width=0.03,     
             head_length=0.03,
             length_includes_head=True, color = 'black')
    ax.text(ldngs[0, i] * 1.10, 
            ldngs[1, i] * 1.10, 
            feature, fontsize=10)
 
scatter_custom(
    x = pca_data["PCA1"] * 1/(pca_data["PCA1"].max() - pca_data["PCA1"].min()),
    y = pca_data["PCA2"] * 1/(pca_data["PCA2"].max() - pca_data["PCA2"].min()),
    data = pca_data,                         
    hue = pca_data["quality_cat"],
    palette = {
        "good": "#44e271",
        "bad": "#e81d0b",
        "outlier": "purple"
    },
    axes = ax
)
ax.set_xlabel('PC1', fontsize=10)
ax.set_ylabel('PC2', fontsize=10)

plt.show()
```

![biplot_prova.png](Analysing%20wine%20quality%20data/18a021a3-127d-4463-b74b-bf4e992afd07.png)

The outliers are now better positioned—far from the main cluster of data points. The arrows clearly show which variables contribute positively or negatively to wine quality.
