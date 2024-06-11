---
title: "NN & Decisoin Tree"
summary: Machine Learning is the study of making machines learn a concept without explicitly programming it. It involves building algorithms that can learn from input data to make predictions or find patterns in the data.
date: 2022-11-09
categories: 
- Nearnest Neighbors
- Decision Tree
---
## Nearest Neighbors

In my last post, i talked about doing classification via probablistic model. But a lot of time,  we don't know how to correctly model P[x|y].

In this most, we take a look at a different apporach, which is to use disriminative models:

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

Compared to the naive tree algorithm , the goal here is to select the feature and threshold that maximally reduces label uncertainty. Typically measured by criteria like Gini impurity, Entropy

Criterias to measure uncertainty in cell C:

Gini Impurity: It measures the probability of a randomly chosen element being incorrectly classified if it was labeled according to the distribution of labels in the dataset.

$$
u(C) = 1 - \sum_{y \in \mathcal{Y}} p_y^2
$$

Entropy: It measures the average amount of information needed to identify the class of an element in the dataset.

$$
u(C) = - \sum_{y \in \mathcal{Y}} p_y \log_2(p_y)
$$

Classification error:

$$
u(C) = 1 - \max_{y} p_y
$$

Here we need to maximally reduce the uncertainty to find feature F and threshold T

$$
\arg \max_{F, T} \left[ u(C) - \left( p_L \cdot u(C_L) + p_R \cdot u(C_R) \right) \right]
$$

---

Lets take a look at the example to see how it works:

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
