---
title: "Nearest Neighbors & Decision Tree"
summary: This post explores the concepts of Nearest Neighbors (k-NN) and Decision Tree algorithms, including their pros and cons, and how to measure uncertainty using Gini impurity and entropy. A detailed example is provided to illustrate the decision tree splitting process.
date: 2022-11-09
categories: 
- Nearnest Neighbors
- Decision Tree
---
## Nearest Neighbors

In my last post, i talked about doing classification via probablistic model. But a lot of time,  we don't know how to correctly model P[x|y].

In this post, we take a look at a different apporach, which is to use disriminative models:

### Nearest Neighbor (NN) classification

The idea is to assign the label to the same label as its 'closest neighbor' for new test example. But the question is: How to measure 'closeness' in X?

* use distances: Euclidean distance, Manhattan distance, cityblock distance
* compare similarity: cosine similarity
* domain expertise: minimum number of insertions, deletions, mutations needed

### The Pro of k-NN

1. A straightforward approach
2. Don't deal with probability modeling

### The Cons of  k-NN

1. **Finding the k closest neighbor takes time.** This involves calculating the distance between the test point and each point in the training dataset. If there are nn**n** training points and each point is d-dimensional, a naive approach requires O(nd) time to compute the distance from the test point to all training points. And sorting them takes O(nlogn). The overall complexity is **O**(**n**d+**n**lo**g**n) per test point, which is very slow for large datasets.
2. The 'closeness' in raw measurement space is not good. Noisy data can distorts the computation. Features have different scales.
3. Need to keep all the training data around during test. k-NN is a lazy learning algorithm, meaning it does not build a model during training but rather stores the entire training dataset. During testing, it uses the stored data to make predictions.

## Decision Tree

A decision tree classifier works by recursively partitioning the data into subsets that are more **homogeneous** in terms of the target variable.

**Recursive Partitioning:**

* A decision tree splits the data into subsets based on the value of input features.
* At each node in the tree, the algorithm selects the feature and threshold that results in the best split, meaning the split that most reduces uncertainty or impurity.

**Uncertainty Reduction:**

* The goal is to partition the data such that the resulting subsets (nodes) are as pure as possible, meaning they contain instances predominantly of a single class.

**The metrics:**

Gini Impurity: It measures the probability of a randomly chosen element being incorrectly classified if it was labeled according to the distribution of labels in the dataset.

$$
u(C) = 1 - \sum_{y \in \mathcal{Y}} p_y^2
$$

Entropy: It measures the average amount of information needed to identify the class of an element in the dataset.

$$
u(C) = - \sum_{y \in \mathcal{Y}} p_y \log_2(p_y)
$$

Information Gain:

$$
IG(S, A) = H(S) - \sum_{t \in T} p(t) H(t) = H(S) - H(S|A)
$$

Classification error:

$$
u(C) = 1 - \max_{y} p_y
$$

Here we need to maximally reduce the uncertainty to find feature F and threshold T

$$
\arg \max_{F, T} \left[ u(C) - \left( p_L \cdot u(C_L) + p_R \cdot u(C_R) \right) \right]
$$

### Purning in Decision Tree

Pruning is a technique used in decision tree algorithms to reduce overfitting and improve the model's generalization to unseen data. Overfitting occurs when a decision tree model becomes too complex and captures noise in the training data instead of the underlying pattern. Pruning simplifies the tree by removing parts that do not provide significant power to classify instances.

Two tyeps of pruning:

* **Pre-pruning (Early Stopping)**
* **Post-pruning (Pruning after Tree Construction)**

### **ID3 (Iterative Dichotomiser 3)**

ID3 (Iterative Dichotomiser 3) is an algorithm used to create decision trees. ID3 is the precursor to the [C4.5 algorithm](https://en.wikipedia.org/wiki/C4.5_algorithm "C4.5 algorithm"), and is typically used in the [machine learning](https://en.wikipedia.org/wiki/Machine_learning "Machine learning") and [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing "Natural language processing") domain.

#### How it works:

**Initialization**:

- Start with the entire dataset as the root of the tree.

**Recursive Splitting**:

- For each node, calculate the entropy of the dataset.
- For each attribute, calculate the information gain if the dataset is split based on that attribute.
- Select the attribute that provides the highest information gain and split the dataset based on this attribute.
- Create child nodes for each subset of the dataset resulting from the split.
- Repeat the process recursively for each child node, treating the subset of the dataset at each node as the new dataset.

**Stopping Criteria**:

- The recursion stops when one of the following conditions is met:
  - All instances in the dataset at a node belong to the same class.
  - There are no more attributes to split on.
  - The dataset at a node is empty.

ID3 follows a greedy approach, making locally optimal choices at each step by selecting the attribute that provides the highest information gain. This approach is computationally efficient and works well for many practical problems.

---

An example to illustate how it works if we split left and right cell of each iteration:

| Feature 1 | Feature 2 | Class |
| :-------: | --------- | ----- |
|    2.5    | 2.4       | 0     |
|    0.5    | 0.7       | 1     |
|    2.2    | 2.9       | 0     |
|    1.9    | 2.2       | 0     |
|    3.1    | 3.0       | 1     |

Initial Gini Impurity Calculation for the Entire Dataset:

$$
u(C) = 1 - \sum_{y \in \mathcal{Y}} p_y^2 = 1 - (p(y=0)^2 + p(y=1)^2) \approx0.48
$$

Splitting the Dataset on Feature 1

Threshold 2.4:

left cell:

| Feature 1 | Feature 2 | Class |
| --------- | --------- | ----- |
| 1.9       | 2.2       | 0     |
| 0.5       | 0.7       | 1     |
| 2.2       | 2.9       | 0     |

Class distribution:

* Class 0: 2 instance
* Class 1: 1 instance

$$
u(CL) = 1 - \sum_{y \in \mathcal{Y}} p_y^2 = 1 - (p(y=0)^2 + p(y=1)^2) \approx 0.44
$$

right cell:

| Feature 1 | Feature 2 | Class |
| --------- | --------- | ----- |
| 2.5       | 2.4       | 0     |
| 3.1       | 3.0       | 1     |

$$
u(CR) = 1 - \sum_{y \in \mathcal{Y}} p_y^2 = 1 - (p(y=0)^2 + p(y=1)^2) \approx 0.5
$$

Wighted Gini impurity for the split:

$$
P_L = 3/5 \space P_R = 2/5
$$

$$
u(C) - \left( p_L \cdot u(C_L) + p_R \cdot u(C_R) \right) \approx 0.014
$$

Threshold 2.0:

left cell:

| Feature 1 | Feature 2 | Class |
| --------- | --------- | ----- |
| 1.9       | 2.2       | 0     |
| 0.5       | 0.7       | 1     |

right cell:

| Feature 1 | Feature 2 | Class |
| --------- | --------- | ----- |
| 2.5       | 2.4       | 0     |
| 3.1       | 3.0       | 1     |
| 2.2       | 2.9       | 0     |

$$
u(CL)  =0.5 \space  u(CR)  \approx 0.44
$$

$$
u(C) - \left( p_L \cdot u(C_L) + p_R \cdot u(C_R) \right) \approx 0.016
$$

**If we split the dataset with Feature 2:**

Threshold 1.5:

left cell:

| Feature 1 | Feature 2 | Class |
| --------- | --------- | ----- |
| 0.5       | 0.7       | 1     |

right cell:

| Feature 1 | Feature 2 | Class |
| --------- | --------- | ----- |
| 1.9       | 2.2       | 0     |
| 2.5       | 2.4       | 0     |
| 3.1       | 3.0       | 1     |
| 2.2       | 2.9       | 0     |

$$
u(CL) =0  u(CR) = 0.375
$$

$$
u(C) - \left( p_L \cdot u(C_L) + p_R \cdot u(C_R) \right) = 0.18
$$

Threshold 2.5:

left cell:

| Feature 1 | Feature 2 | Class |
| --------- | --------- | ----- |
| 0.5       | 0.7       | 1     |
| 1.9       | 2.2       | 0     |
| 2.5       | 2.4       | 0     |

right cell:

| Feature 1 | Feature 2 | Class |
| --------- | --------- | ----- |
| 3.1       | 3.0       | 1     |
| 2.2       | 2.9       | 0     |

$$
u(C) - \left( p_L \cdot u(C_L) + p_R \cdot u(C_R) \right) \approx 0.014
$$

$$
\arg \max_{F, T} \left[ u(C) - \left( p_L \cdot u(C_L) + p_R \cdot u(C_R) \right) \right] = 0.18
$$

The optimal split is with  **Feature 2 and threshold 1.5** , which provides the highest reduction in impurity.

---

### The observations of Decision Tree

* The decision tree construction is via a greedy approach
* Finding the optimal decision tree is NP-hard!
* You quickly run out of training data as you go down the tree, so uncertainty estimates become very unstable
* Tree complexity is highly dependent on data geometry in the feature space
* Popular instantiations that are used in real-world: ID3, C4.5, CART
