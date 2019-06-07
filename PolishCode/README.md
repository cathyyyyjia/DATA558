# DATA 558 Polished Code

In this assignment, I implemented k-means clustering algorithm. I have tested my code on both simulated dataset and real-world dataset and compared with sklearn built-in KMeans function.

## K-Means Clustering Implemention: mykmeans

```python
class mykmeans(k=3, max_iter=200, eps=0.01)
```

A class implements k-means clustering algorithm using Eucliean distance for distance computing.

### Parameters

- `k`: *int*, *optional*

Number of clusters

- `max_iter`: *int*, *optional*

Maximum number of iterations

- `eps`: *float*, *optional*

Tolerance for clustering convergence

### Methods

```python
__init__(self, k=3, max_iter=200, eps=0.01)
```

```python
get_k(self)
```
Return number of clusters

```python
get_max_iter(self)
```
Return maximum number of iterations

```python
get_eps(self)
```
Return tolerance for clustering convergence

```python
euclidean_dist(self, X, Y)
```

Compute Euclidean distance between arrays

Parameters:
- X: *(1,d) array or (n,d) array*

A point or points

- Y: *(1,d) array or (n,d) array*

A point or points

```python
fit(self, X)
```

Apply k-means clustering algorithm

Parameters:
- X: *(n,d) array*

A (n, d) array need to be clustered

### Attributes

- `clusters`: A dictionary {label: [data points]}
- `centers`: A (k, ) array of centroids
- `num_iter`: Number of iterations

## Simulated Dataset Demo

This is a demo using simulated dataset with size of 300 and dimension of 20, assigned with 3 classes (0, 1 and 2).

To start the demo, run the following code in command line. You will see a report with clustering plots.

```
python demo_simulated_data.py
```

## Real-World Dataset Demo

This is a demo using iris dataset which is an example datasets used by scikit-learn. The dataset can be loaded using the following code:

```python
from sklearn import datasets
iris = datasets.load_iris()
```

To start the demo, run the following code in command line. You will see a report with clustering plots.

```
python demo_real_world_data.py
```

## Experimental Comparison

This file is designed for comparing my implemented k-means clustering with sklearn's k-means clustering. The simulated dataset with size of 300 and dimension of 20, assigned with 3 classes (0, 1 and 2). The real-world dataset is the iris dataset which is an example datasets used by scikit-learn. The dataset can be loaded using the following code:

```python
from sklearn import datasets
iris = datasets.load_iris()
```

To start the experimental comparison, run the following code in command line. You will see a report with clustering plots.

```
python demo_real_world_data.py
```
