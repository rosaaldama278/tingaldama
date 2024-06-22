---
title: "Unsupervised Learning and Clustering"
date: 2022-12-15
summary: This post provides an in-depth look at various regression techniques, including parametric and non-parametric regression, linear regression, Lasso and Ridge regression, logistic regression, and kernel regression.
categories:
- k-means 
- Hierarchicall Clustering
- Gaussian Mixture Models
---
## Unsupervised Learning 

 **Data** : x1, x2, ... in X

 **Assumption** : there is an underlying structure in X

 **Learning task** : discover the structure given n examples from the data

 **Goal** : come up with the summary of the data using the discovered structure

### **Clustering**

#### k-means 

 **Given** : data $\vec{x_1}, \vec{x_2}, \ldots, \vec{x_n} \in \mathbb{R}^d$, and intended number of groupings $k$

 **Idea** :
find a set of representatives $\vec{c_1}, \vec{c_2}, \ldots, \vec{c_k}$ such that data is close to some representative

 **Optimization** :
$\displaystyle \min_{c_1, \ldots, c_k} \left[\sum_{i=1}^n \min_{j=1, \ldots, k} \lVert\vec{x_i} - \vec{c_j}\rVert^2\right]$

The k-means problem has been proven to be NP-hard even for small dimensions and specific cases. This means there is no known polynomial-time algorithm that can solve the problem exactly for all instances.

#### Approximate K-means Algorithm

 **Given** : data $\vec{x_1}, \vec{x_2}, \ldots, \vec{x_n} \in \mathbb{R}^d$, and intended number of groupings $k$

 **Alternating optimization algorithm** :

* Initialize cluster centers $\vec{c_1}, \vec{c_2}, \ldots, \vec{c_k}$ (say randomly)
* Repeat till no more changes occur
  * Assign data to its closest center (this creates a partition) (assume centers are fixed)
  * Find the optimal centers $\vec{c_1}, \vec{c_2}, \ldots, \vec{c_k}$ (assuming the data partition is fixed)

The approximation can be arbitrarily bad, compared to the best cluster assignment. Performance quality is heavily dependent on the initialization. 

### Hierarchical Clustering 

There are two different approaches: top down, basically we partition data into two groups, then recurse on each part, then stop when cannot partition data anymore; bottom up, we start by each data samples as its own cluster, repeatdly merge 'closest' pair of clusters and stop when only one cluster is left. 

#### Clustering via Probabilistic Mixture Modeling 

Probabilistic mixture models assume that the data is generated from a mixture of several probability distributions. Each distribution represents a different cluster. The overall model is a weighted sum of these distributions, where the weights correspond to the probability of each distribution (cluster) generating a given data point.

Given: $\vec{x}_1, \vec{x}_2, \ldots \vec{x}_n \in \mathbb{R}^d$ and number of intended number of clusters $k$.
Assume a joint probability distribution $(X, C)$ over the joint space $\mathbb{R}^d \times [k]$

$$
C \sim \begin{pmatrix}
\pi_1 \
\vdots \
\pi_k
\end{pmatrix}
$$

Discrete distribution over the clusters $P[C=i] = \pi_i$

$X|C = i \sim$ Some multivariate distribution, e.g. $N(\vec{\mu}_i, \Sigma_i)$

Parameters: $\theta = (\pi_1, \vec{\mu}_1, \Sigma_1, \ldots, \pi_k, \vec{\mu}_k, \Sigma_k)$ d

Modeling assumption data $(x_1,c_1),..., (x_n,c_n)$ i.i.d. from $\mathbb{R}^d \times [k]$
BUT only get to see partial information: $x_1, x_2, ..., x_n$ $(c_1, ..., c_n \text{ hidden!})$

#### Gaussian Mixture Modeling (GMM)

GMMs are a specific type of probabilistic mixture model where each cluster is modeled by a Gaussian (normal) distribution.

Given: $\vec{x}_1, \vec{x}_2, \ldots \vec{x}_n \in \mathbb{R}^d$ and $k$.
Assume a joint probability distribution $(X, C)$ over the joint space $\mathbb{R}^d \times [k]$

$$
C \sim \begin{pmatrix}
\pi_1 \
\vdots \
\pi_k
\end{pmatrix}, \quad X|C = i \sim N(\vec{\mu}_i, \Sigma_i) \text{ Gaussian Mixture Model}
$$

$\theta = (\pi_1, \vec{\mu}_1, \Sigma_1, \ldots, \pi_k, \vec{\mu}_k, \Sigma_k)$

$$
P[\vec{x} | \theta] = \sum_{j=1}^k \pi_j \frac{1}{\sqrt{(2\pi)^d \det(\Sigma_j)}} e^{ -\frac{1}{2} (\vec{x} - \vec{\mu}_j)^T \Sigma_j^{-1} (\vec{x} - \vec{\mu}_j) }
$$

**π**j: The mixing coefficient for the j-th Gaussian component

**How do we learn the parameters $\theta$ then? **

#### Maximum Likelihood Estimation 

MLE for Mixture modeling (like GMMs) is NOT a convex optimization problem. Even though a global MLE maximizer is not appropriate, several

local maximizers are desirable!


#### Expectatino Maximization (EM)

Similar in spirit to the alternating update for k-means algorithm

Idea:

* Initialize the parameters arbitrarily
* Given the current setting of parameters find the best (soft) assignment of
* data samples to the clusters (Expectation-step)
* Update all the parameters with respect to the current (soft) assignment that maximizes the likelihood (Maximization-step)
* Repeat until no more progress is made.






### Dimensionality Reduction

Find a low-dimensional representation that retains important information, and suppresses irrelevant/noise information (`Dimensionality reduction`)
