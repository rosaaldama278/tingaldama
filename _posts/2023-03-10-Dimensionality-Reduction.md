---
title: "Dimensionality Reduction"
date: 2023-03-10
summary: Tokenization is the process of breaking down text into smaller units called tokens. In the context of the Byte Pair Encoding (BPE) algorithm, tokenization involves splitting words into subword units based on a learned vocabulary. The BPE tokenizer aims to find a balance between representing the text with a limited vocabulary size while still capturing meaningful subword units.
categories:
   - tokenization
   - BPE
---
## PCA (Principle Components Analysis)

PCA (Principal Component Analysis) is a common data analysis method, often used for dimensionality reduction of high-dimensional data, and can be used to extract the main feature components of data.

**Linear transformation:**

Transformations are easier to understand by visualizing points on a grid moving to new locations.

matrix-vector multiplication: Linear transformations in 2D can be described using matrices. The columns of a matrix represent where the basis vectors (ii**i**-hat and jj**j**-hat) land after transformation.To find where any vector lands after transformation, express it as a combination of the basis vectors and track where those basis vectors land.

Matrices represent linear transformations by showing where the basis vectors land. Those are very important perspective of thinking linear transformation which helps us understand PCA.

**Maximum Seperability:**

From the previous persepctive of linear transformation, choosing different bases can give different representations for the same set of data. If the number of bases is less than the data's original dimension, then we achive dimension reduction. To reduce N-dimensional vectors to K-dimensional (where K < N) while retaining the most original information, select the K bases (principal components) that capture the most variance. This ensures that the new basis maximizes the retention of the original information.

**Variance & Covariance**

We know that the degree of dispersion of values can be described by the mathematical variance. The variance of a variable can be seen as the average of the sum of squares of the difference between each element and the mean of the variable, that is:

$\operatorname{var}(x)=\frac{\sum_{i=1}^n (x_i-\bar{x})^2}{n}$

For ease of handling, we normalize the mean of each variable to 0, so the variance can be directly represented by the sum of squares of each element divided by the number of elements:

$\operatorname{var}(x)=\frac{\sum_{i=1}^n x_i^2}{n}$

So the above question is formally described as: find a one-dimensional basis so that the variance value is maximum after all data is transformed into the coordinate representation on this basis.

In one-dimensional space, variance represents data dispersion. For high-dimensional data, covariance shows the correlation between variables. To maximize original information, we aim for no linear correlation between variables, as correlation indicates redundancy and lack of independence.

The covariance formula is:

$\operatorname{cov}(x,y)=\frac{\sum_{i=1}^n (x_i-\bar{x})(y_i-\bar{y})}{n}$

When the covariance is 0, it means that the two variables are linearly uncorrelated. In order to make the covariance 0, when we choose the second basis, we can only choose it in the direction orthogonal to the first basis, so the finally selected two directions must be orthogonal.

**Covariance Matrix**

Suppose we only have two variables a and b, then we arrange them into matrix X by row:

$$
\mathbf{X}=\begin{bmatrix}
a_1 & a_2 & \cdots & a_m \\
b_1 & b_2 & \cdots & b_m
\end{bmatrix}
$$

Then:

$$
C = \frac{1}{m} \mathbf{X}\mathbf{X}^\top = \begin{pmatrix}
\frac{1}{m} \sum_{i=1}^{m} a_i^2 & \frac{1}{m} \sum_{i=1}^{m} a_i b_i \\
\frac{1}{m} \sum_{i=1}^{m} a_i b_i & \frac{1}{m} \sum_{i=1}^{m} b_i^2
\end{pmatrix} = \begin{pmatrix}
\text{Cov}(a, a) & \text{Cov}(a, b) \\
\text{Cov}(b, a) & \text{Cov}(b, b)
\end{pmatrix}
$$

From the covariance matrix, we can generalize the situaton. Suppose we have m n-dimensional data records, arrange them into matrix $\mathbf{X}$, and let $\mathbf{C}=\frac{1}{m}\mathbf{X}\mathbf{X}^T$, then C is a symmetric matrix, whose diagonal corresponds to the variances of each variable, and the elements in the i-th row and j-th column and the j-th row and i-th column are the same, representing the covariance of variables i and j.


***Calculate Eigenvalues and Eigenvectors*:**

- Solve the eigenvalue problem for the covariance matrix $C$:
  $$
  C = Q \Lambda Q^T
  $$
- Here, $Q$ is the matrix of eigenvectors, and $\Lambda$ is the diagonal matrix of eigenvalues.

3. **Form the Transformation Matrix (P)**:

- The matrix $P$ is constructed using the eigenvectors of $C$:
  $$
  P = Q^T
  $$

4. **Transform the Data (Y)**:

- Transform the original data $X$ using the matrix $P$:
  $$
  Y = PX = Q^T X
  $$

5. **Covariance Matrix of Transformed Data (D)**:

- Compute the covariance matrix of the transformed data $Y$:
  $$
  D = \frac{1}{m} YY^T = \frac{1}{m} (Q^T X)(Q^T X)^T = Q^T \left(\frac{1}{m} XX^T\right) Q = Q^T C Q = Q^T (Q \Lambda Q^T) Q = \Lambda
  $$
- The resulting covariance matrix $D$ is diagonal, with the eigenvalues of $C$ as its diagonal elements.

**Eigenvalues and Eigenvectors**

1. **Maximizing Variance**:

   - Eigenvectors corresponding to the largest eigenvalues represent directions in the data that have the most variance.
   - By projecting the data onto these eigenvectors, PCA captures the most significant patterns and variations in the data with fewer dimensions.
2. **Decorrelation**:

   - The principal components (eigenvectors) are orthogonal, meaning they are uncorrelated.
   - This orthogonality ensures that the new basis vectors are independent, simplifying the structure of the data and making it easier to analyze.
3. **Dimensionality Reduction**:

   - By selecting the top \( k \) eigenvectors (those with the largest eigenvalues), PCA reduces the dimensionality of the data while retaining most of its variance.
   - This reduction simplifies the data, making it more manageable and reducing computational costs for further analysis.
4. **Optimal Representation**:

   - When the data is projected onto the new basis formed by the eigenvectors, the reconstruction error (the difference between the original data and its projection) is minimized.
   - This optimal representation ensures that the reduced-dimensional data retains as much information as possible from the original data.
5. **Geometric Interpretation**:

   - Geometrically, eigenvectors are directions along which the data is stretched or compressed the most by the linear transformation (covariance matrix).
   - Choosing these directions as the new basis aligns the data along its principal axes of variation, providing a clearer and more meaningful representation.

### THe main steps in PCA:

Step 1: Calculate the covariance matrix

- Center the data by subtracting the mean of each feature:
  $\mathbf{X}_{centered} = \mathbf{X} - \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i$
- Calculate the covariance matrix:
  $\mathbf{C} = \frac{1}{n-1} \mathbf{X}_{centered}^T \mathbf{X}_{centered}$
  where $\mathbf{C}$ is an m x m symmetric matrix.

Step 2: Find the eigenvectors and eigenvalues

- Solve the eigenvalue equation:
  $\mathbf{C} \mathbf{v}_i = \lambda_i \mathbf{v}_i$
  where $\mathbf{v}_i$ is the i-th eigenvector and $\lambda_i$ is the corresponding eigenvalue.
- This step results in m eigenvectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_m$ and their corresponding eigenvalues $\lambda_1, \lambda_2, \ldots, \lambda_m$.

Step 3: Sort the eigenvectors

- Sort the eigenvectors in descending order based on their corresponding eigenvalues:
  $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_m$
- The sorted eigenvectors are denoted as $\mathbf{v}_{(1)}, \mathbf{v}_{(2)}, \ldots, \mathbf{v}_{(m)}$.

Step 4: Select the top k eigenvectors

- Choose the top k eigenvectors that capture the desired amount of variance:
  $\mathbf{P} = [\mathbf{v}_{(1)}, \mathbf{v}_{(2)}, \ldots, \mathbf{v}_{(k)}]$
  where $\mathbf{P}$ is an m x k matrix.

Step 5: Project the data

- Project the centered data onto the new basis formed by the selected eigenvectors:
  $\mathbf{Y} = \mathbf{X}_{centered} \mathbf{P}$
  where $\mathbf{Y}$ is an n x k matrix representing the transformed dataset in the new basis.

The resulting matrix $\mathbf{Y}$ contains the transformed dataset with n samples and k features, where each feature corresponds to a principal component. The first principal component (first column of $\mathbf{Y}$) captures the most variance, the second principal component (second column of $\mathbf{Y}$) captures the second-most variance, and so on.

## LDA

## PCA vs LDA

Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) are both linear dimensionality reduction techniques, but they serve different purposes and operate based on different principles. Here's a comparison between PCA and LDA:

PCA:

1. **Objective**:

   - PCA aims to reduce the dimensionality of the data by transforming it into a new set of variables, the principal components, which capture the maximum variance in the data.
2. **Method**:

   - PCA does not consider any class labels. It is an unsupervised method.
   - It finds the directions (principal components) that maximize the variance of the data.
   - The principal components are orthogonal (uncorrelated) to each other.
3. **Computation**:

   - Compute the covariance matrix of the data.
   - Calculate the eigenvalues and eigenvectors of the covariance matrix.
   - Sort the eigenvectors by descending eigenvalues.
   - Select the top \( k \) eigenvectors to form a new feature space.
4. **Applications**:

   - Data visualization (e.g., reducing high-dimensional data to 2D or 3D).
   - Noise reduction.
   - Preprocessing for machine learning algorithms.
   - Feature extraction.
5. **Key Feature**:

   - PCA focuses on capturing the global variance structure of the data.

LDA:

1. **Objective**:

   - LDA aims to reduce the dimensionality of the data while preserving as much of the class discriminatory information as possible. It maximizes the separation between multiple classes.
2. **Method**:

   - LDA considers class labels. It is a supervised method.
   - It finds the directions (linear discriminants) that maximize the ratio of the between-class variance to the within-class variance.
   - The linear discriminants are not necessarily orthogonal.
3. **Computation**:

   - Compute the within-class scatter matrix and the between-class scatter matrix.
   - Calculate the eigenvalues and eigenvectors of the scatter matrices.
   - Sort the eigenvectors by descending eigenvalues.
   - Select the top \( k \) eigenvectors to form a new feature space.
4. **Applications**:

   - Classification problems (e.g., reducing the dimensionality of the data before applying a classifier).
   - Feature extraction for improving classifier performance.
   - Face recognition and other pattern recognition tasks.
5. **Key Feature**:

   - LDA focuses on maximizing class separability.

## Summary of Differences

| Feature               | PCA                                          | LDA                                             |
| --------------------- | -------------------------------------------- | ----------------------------------------------- |
| **Type**              | Unsupervised                                 | Supervised                                      |
| **Objective**         | Maximize variance                            | Maximize class separability                     |
| **Uses class labels** | No                                           | Yes                                             |
| **Main application**  | Dimensionality reduction, data visualization | Classification, feature extraction              |
| **Computes**          | Covariance matrix                            | Within-class and between-class scatter matrices |
| **Focus**             | Captures global variance structure           | Maximizes class separation                      |
