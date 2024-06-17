---
title: "Regression"
date: 2022-12-11
summary: Tokenization is the process of breaking down text into smaller units called tokens. In the context of the Byte Pair Encoding (BPE) algorithm, tokenization involves splitting words into subword units based on a learned vocabulary. The BPE tokenizer aims to find a balance between representing the text with a limited vocabulary size while still capturing meaningful subword units.
categories:
   - tokenization
   - BPE
---
## Parametric vs Nonparametric Regressioin

**Parametric regression**

If we assume a particular form of the regressor:

Goal: to learn the parameters which yield the minimum error/loss

**Non-parametric regression**

If no specific form of regressor is assumed:

Goal: to learn the predictor directly from the input data that yields the minimum error/loss

### Linear Regression

Want to find a **linear predictor** \( f \), i.e., \( w \) (intercept \( w_0 \) absorbed via lifting):

$$
hat{f}(\vec{x}) := \vec{w} \cdot \vec{x}
$$

which minimizes the prediction loss over the population.

$$
min_{\vec{w}} \mathbb{E}_{\vec{x}, y} \left[ L(\hat{f}(\vec{x}), y) \right]
$$

We estimate the parameters by minimizing the corresponding loss on the training data:

$$
\arg \min_{\vec{w}} \frac{1}{n} \sum_{i=1}^n \left[ L(\vec{w} \cdot \vec{x}_i, y_i) \right] = \arg \min_{\vec{w}} \frac{1}{n} \sum_{i=1}^n \left( \vec{w} \cdot \vec{x}_i - y_i \right)^2
$$

#### Learning the Parameters

Unconstraint problem:

$$
= \arg \min_{\vec{w}} \left\|
\begin{pmatrix}
... & \vec{x}_1 & ... \\
... & \vec{x}_i & ... \\
... & \vec{x}_n & ...
\end{pmatrix}
\vec{w} -
\begin{pmatrix}
y_1 \\
y_i \\
y_n
\end{pmatrix}
\right\|_2^2
$$

$$
= \arg \min_{\vec{w}} \| X \vec{w} - \vec{y} \|_2^2
$$

Thus best fitting w:

$$
\frac{\partial}{\partial \mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|^2 = 2X^T (X\mathbf{w} - \mathbf{y})
$$

At stationarity, this results in the equation:

$$
(X^T X)\mathbf{w} = X^T \mathbf{y}
$$

This system is always consistent: 

$$
\mathbf{\vec{y}} = \mathbf{\vec{y}}_{\text{col}(X)} + \mathbf{\vec{y}}_{\text{null}(X^T)}
$$

Thus,

$$
X^T \mathbf{y} = X^T \mathbf{\vec{y}}_{\text{col}(X)}
$$

Since,

$$
{\vec{y}}_{\text{col}(X)} = \sum_i w_i \mathbf{x}_i \quad \text{(for some coefficients \(w_i\), where \(\mathbf{x}_i\) are columns of \(X\))}
$$

Define,

$$
{\vec{w}} := \begin{bmatrix}
w_1 \\
\vdots \\
w_d
\end{bmatrix}
$$

Then,

$$
(X^T X)\mathbf{\vec{w}} = X^T (X\mathbf{\vec{w}}) = X^T \left(\sum_i w_i \mathbf{x}_i \right) = X^T \mathbf{y}_{\text{col}(X)} = X^T \mathbf{y}
$$

$$
\mathbf{\vec{w}}_{\text{ols}} = (X^T X)^{\dagger} X^T \mathbf{y}
$$

Also called the Ordinary Least Squares (OLS). The solution is unique and stable when $(X^T X)$ is invertible

### Linear Regression in Statisticall Modeling View 

Let's assume that data is generated from the following process:

* A example $x_i$ is draw independently from the data space $X$
  $$
  x_i \sim D_X
  $$
* $y_i^{clean}$ is computed as $(w \cdot x_i)$, from a fixed unknown $w$
  $$
  y_i^{clean} := w \cdot x_i
  $$
* $y_i$ is corrupted from $y_i^{clean}$ by adding independent Gaussian noise $N(0,\sigma^2)$
  $$
  y_i := y_i^{clean} + \epsilon_i = w \cdot x_i + \epsilon_i \quad\quad \epsilon_i \sim N(0, \sigma^2)
  $$
* $(x_i, y_i)$ is revealed as the $i$-th sample
  $$
  (x_1, y_1), ..., (x_n, y_n) =: S
  $$

How can we determine $w$, from Gaussian noise corrupted observations?

$$
S = (x_1, y_1), ..., (x_n, y_n)
$$

Observation:

$$
y_i \sim w \cdot x_i + N(0, \sigma^2) = N(w \cdot x_i, \sigma^2)
$$

parameter: w (Ignored terms indepdent of w)

$$
\log L(w|S) = \sum_{i=1}^n \log p(y_i|w)
$$

$$
\propto \sum_{i=1}^n \frac{-(w \cdot x_i - y_i)^2}{2\sigma^2}
$$

optimizing for $w$ yields the same OLS result!

#### Lasso Regression & Ridge Regression 

Previously we looked at Ordinary Least Square (OLS)

$$
minimize\| X \vec{w} - \vec{y} \|_2^2
$$

$$
\mathbf{\vec{w}}_{\text{ols}} = (X^T X)^{\dagger} X^T \mathbf{y}
$$

Which is  poorly behaved (due to overfitting) when we have limited data. We can incorporate application dependent piror knowledge.

**Lasso regression:** 

Objective

minimize $|X\vec{w} - \vec{y}|^2 + \lambda\|\vec{w}\|^2$

reconstruction error 'regularization' parameter

$\vec{w}_{ridge} = (X^TX + \lambda I)^{-1}X^T\vec{y}$

The 'regularization' helps avoid overfitting, and always resulting in a unique solution. Equivalent to the following optimization problem:

minimize $|X\vec{w} - \vec{y}|^2$

such that $|\vec{w}|^2 \leq B$

**Ridge Regression:** 

Objective

minimize $|X\vec{w} - \vec{y}|^2 + \lambda\|\vec{w}\|_1$

'lasso' penalty

$\vec{w}_{lasso} = ?$ no closed form solution

Lasso regularization encourages sparse solutions. Equivalent to the following optimization problem:

minimize $|X\vec{w} - \vec{y}|^2$

such that $|\vec{w}|_1 \leq B$

### Logistic Regression

Linear regression for classification: 

Although its name contains "regression," it is actually used for classification tasks. For a binary classification problem, given input x, how likely is it that it has label 1?Let this be denoted by P, ie, P is the chance that a given x the associated label y = 1, P = P(Y=1|X=x) ranges between 0 and 1, hence cannot be modelled appropriately via linear regression. If we look at the 'odds' of getting y=1 (for a given x) $odds(P) := \frac{P}{1-P}$

For an event with P=0.9, odds = 9
But, for an event P=0.1, odds = 0.11

Consider the "log" of the odds (very asymmetric)

$log(odds(P)) := logit(P) := log(\frac{P}{1-P})$

$logit(P) = -logit(1-P)$ Symmetric! Can model logit as a linear function!!

Model the log-odds or logit with linear function!
Given an input x

$logit(P(Y=1|X=x)) = logit(P) = log(\frac{P}{1-P}) \stackrel{\text{modeling assumption}}{=} w \cdot x$

$\frac{P}{1-P} = e^{w \cdot x}$

$P(Y=1|X=x) = P = \frac{e^{w \cdot x}}{1+e^{w \cdot x}} = \frac{1}{1+e^{-w \cdot x}}$    Sigmoid!

How do we learn the prameters?

Given samples $S = (x_1, y_1), ..., (x_n, y_n)$ yi is binary)

$\mathcal{L}(w|S) = \prod_{i=1}^{n} P((x_i, y_i)|w) \propto \prod_{i=1}^{n} P(y_i = 1|x_i, w)= \prod_{i=1}^{n} P(y_i = 1|x_i, w)^{y_i} (1 - P(y_i = 1|x_i, w))^{1-y_i}$ (Binomial MLE)

$log \mathcal{L}(w|S) \propto \sum_{i=1}^{n} y_i log P_{x_i} + (1 - y_i) log(1 - P_{x_i}) = \sum_{i=1}^{n} y_i log \frac{P_{x_i}}{1 - P_{x_i}} + \sum_{i=1}^{n} log(1 - P_{x_i})$

Now, use logistic model!

$= \sum_{i=1}^{n} y_i (w \cdot x_i) + \sum_{i=1}^{n} - log(1 + e^{w \cdot x_i})$

No closed form solution
Can use iterative methods like gradient 'ascent' to find the solution

### Non-parametric Regression 

What if we don’t know parametric form of the relationship between the independent and dependent variables? How can we predict value of a

new test point  x without model assumptions?

#### Kernel Regression

$y = f_n(x) := \sum_{i=1}^{n} w_i(x)y_i$

Want weights that emphasize local observations

Localization functions:

Gaussian Kernel: $K_h(x, x') = e^{-||x-x'||^2/h}$

Box Kernel: $1[||x - x'|| \leq h]$

$$
K_h(x, x') = e^{-||x-x'||^2/h}$$ Gaussian kernel
$$= 1[||x - x'|| \leq h]$$ Box kernel
$$= [1 - (1/h)||x - x'||]_+$$ Triangle kernel
       
Then define:
$$w_i(x) := \frac{K_h(x,x_i)}{\sum_{j=1}^n K_h(x,x_j)}$$ Weighted average
$$
