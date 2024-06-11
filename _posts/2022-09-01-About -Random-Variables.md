title: "Random Variables"
summary: Discrete and continous random variables, joint distribution and independence
date: 2022-09-01
categories: random variables

## Random Variables

Suppose X represents some unknown quantity of interest, such as which way a dice will land when
we roll it, or the temperature outside your house at the current time. If the value of X is unknown
and/or could change, we call it a random variable or rv. The set of possible values, denoted X, is
known as the sample space or state space. An event is a set of outcomes from a given sample
space. For example, if X represents the face of a dice that is rolled, so X = {1, 2, . . . , 6}, the event
of “seeing a 1” is denoted X = 1, the event of “seeing an odd number” is denoted X ∈ {1, 3, 5}, the
event of “seeing a number between 1 and 3” is denoted 1 ≤ X ≤ 3

### Discrete and Continuous Random Variables

If the sample space X is finite or countably infinite, then X is called a discrete random variable.
In this case, we denote the probability of the event that X has value x by Pr(X = x). Probability mass function or pmf as a function which computes the probability of events which correspond to setting the rv to each possible value. If X ∈ R is a real-valued quantity, it is called a continuous random variable. In this case, we
can no longer create a finite (or countable) set of distinct possible values it can take on. However,
there are a countable number of intervals which we can partition the real line into.

### Joint Distribution

The joint distribution of two random variables \( X \) and \( Y \), denoted as \( p(x, y) = p(X = x, Y = y) \), describes the probability that \( X \) takes on value \( x \) and \( Y \) takes on value \( y \) simultaneously. This joint distribution accounts for the relationship between the two random variables.

Marginal Distribution

The marginal distribution of a random variable is obtained by summing (or integrating, in the case of continuous variables) the joint distribution over all possible values of the other variable. For example, the marginal distribution of \( X \) is obtained by summing the joint distribution over all possible values of \( Y \):

$$
p(x) = \sum_y p(x, y)
$$

which is also called sum rule

The product rule:

$$
p(X = x, Y = y) = p(X = x) \cdot p(Y = y | X = x)
$$

By extending the product rule to \( D \) variables, we get the chain rule of probability:

$$
p(x_{1:D}) = p(x_1) p(x_2 \mid x_1) p(x_3 \mid x_1, x_2) p(x_4 \mid x_1, x_2, x_3) \ldots p(x_D \mid x_{1:D-1})
$$

---

### Independence and Conditional Independence

Independence:

$$
P(X₁, ..., Xₙ) = ∏ᵢ₌₁ⁿ P(Xᵢ)
$$

Indendence is rare, the conditional independence:

$$
X ⊥ Y | Z ⇐⇒ p(X, Y | Z) = p(X | Z)p(Y | Z)
$$

Pairwise independence doesn't imply mutual independence

### Pairwise Independence vs. Mutual Independence

Pairwise independence doesn't imply mutual independence

- **Pairwise Independence**: A set of random variables \$(X_1, X_2, \ldots, X_n\)$ is pairwise independent if every pair of random variables is independent. That is, for all \(i \neq j\),
- $$
  P(X_i \cap X_j) = P(X_i) P(X_j).
  $$
- **Mutual Independence**: A set of random variables \(X_1, X_2, \ldots, X_n\) is mutually independent if every subset of the random variables is independent. That is, for any subset \(\{i_1, i_2, \ldots, i_k\}\) of \(\{1, 2, \ldots, n\}\),
  $$
  P(X_{i_1} \cap X_{i_2} \cap \cdots \cap X_{i_k}) = \prod_{j=1}^{k} P(X_{i_j})
  $$
