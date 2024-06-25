---
title: "Attention Mechanism"
date: 2024-01-22
summary: This post provides brief timeline of the development of attention mechanism and how is applied in LLMs
categories:
   - attention mechanism 
   - LLMs
---
## Timeline of attention mechanism

1. Early 1990s:

   - The concept of "attention" was introduced in the context of computer vision and neural network research.
   - Researchers explored the idea of selectively focusing on specific parts of an input image or sequence.
2. 2014:

   - Bahdanau et al. introduced the concept of "attention" in the context of neural machine translation (NMT) in their paper "Neural Machine Translation by Jointly Learning to Align and Translate."
   - They proposed an attention mechanism that allowed the model to selectively focus on different parts of the source sentence during the decoding process, improving the translation quality.
3. 2015:

   - Luong et al. further explored attention mechanisms in their paper "Effective Approaches to Attention-based Neural Machine Translation."
   - They introduced different variations of attention, such as global attention and local attention, and demonstrated their effectiveness in NMT tasks.
4. 2015-2016:

   - Attention mechanisms gained popularity and were applied to various NLP tasks beyond machine translation, such as text summarization, question answering, and sentiment analysis.
   - Researchers explored different architectures and variants of attention, such as hierarchical attention, self-attention, and multi-head attention.
5. 2017:

   - Vaswani et al. introduced the Transformer architecture in their paper "Attention Is All You Need."
   - The Transformer relied solely on attention mechanisms, particularly self-attention, and dispensed with recurrent or convolutional layers commonly used in previous architectures.
   - This marked a significant milestone in the development of attention mechanisms and their application in NLP tasks.
6. 2018-present:

   - The Transformer architecture and its variants, such as BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer), have become the dominant approach in NLP.
   - Attention mechanisms have been further refined and applied to a wide range of tasks, including language understanding, language generation, and multi-modal learning.
   - Researchers continue to explore new variations and applications of attention mechanisms, pushing the boundaries of NLP and machine learning.

## Queries, Keys, and Values

The query, key, value concept in attention mechanisms might remind people of qeury, key and value in database retrieveal. However they serve different purposes and are used in different ways. Database retrieval focuses on exact matching and retrieving relevant records, while LLMs use attention mechanisms to determine the importance of different parts of the input sequence when generating an output. Let $\mathcal{D} \equiv {(k_1, v_1), \ldots, (k_m, v_m)}$ be a database of $m$ key-value pairs, and $\mathbf{q}$ be a query. The attention over $\mathcal{D}$ is defined as:

$$
\text{Attention}(\mathbf{q}, \mathcal{D}) \equiv \sum_{i=1}^{m} \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i,
$$

where $\alpha(\mathbf{q}, \mathbf{k}_i) \in \mathbb{R}$ $(i = 1, \ldots, m)$ are scalar attention weights. This operation, called attention pooling, generates a linear combination of values in the database, paying more attention to terms with significant weights.

To ensure weights sum to 1, they can be normalized:

$$
\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}.
$$

For non-negative weights, the softmax operation can be applied:

$$
\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(\alpha(\mathbf{q}, \mathbf{k}_i))}{\sum_j \exp(\alpha(\mathbf{q}, \mathbf{k}_j))}.
$$

This function is differentiable with non-vanishing gradients, making it suitable for use in models.

### Similarity Measure

Now that we have introduced the primary components of the attention mechanism, let's use them in a classical setting: regression and classification via kernel density estimation ([Nadaraya, 1964](https://link.springer.com/article/10.1007/BF02310518), [Watson, 1964](https://www.tandfonline.com/doi/abs/10.1080/00401706.1964.10490196)).

At their core, Nadaraya-Watson estimators rely on some similarity kernel $\alpha(\mathbf{q}, \mathbf{k})$ relating queries $\mathbf{q}$ to keys $\mathbf{k}$. Common kernels are:

$$
\alpha(\mathbf{q}, \mathbf{k}) = \exp \left( -\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2 \right) \quad \text{Gaussian};
$$

$$
\alpha(\mathbf{q}, \mathbf{k}) = 1 \text{ if } \|\mathbf{q} - \mathbf{k}\| \leq 1 \quad \text{Boxcar};
$$

$$
\alpha(\mathbf{q}, \mathbf{k}) = \max \left( 0, 1 - \|\mathbf{q} - \mathbf{k}\| \right) \quad \text{Epanechnikov}.
$$

There are many more choices for kernels. All kernels are heuristic and can be tuned. We can adjust the width globally or on a per-coordinate basis.

Regardless of the kernel, they all lead to the following equation for regression and classification alike:

$$
f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}.
$$

In the case of a (scalar) regression with observations $(x_i, y_i)$, $y_i$ are scalars, $\mathbf{k}_i = \mathbf{x}_i$ are vectors, and the query $\mathbf{q}$ denotes the new location where $f$ should be evaluated. In the case of multiclass classification, we use one-hot encoding of $y_i$ to obtain $\mathbf{v}_i$. One of the convenient properties of this estimator is that it requires no training. If we narrow the kernel with increasing data (consistent with [Mack and Silverman, 1982](https://link.springer.com/article/10.1007/BF00058627)), it will converge to some statistically optimal solution.

### Types of Attention Scoring Functions in Deep Learning

Attention mechanisms use scoring functions to calculate the relevance between the query and key vectors. Different scoring functions have been proposed, each with its own characteristics and applications. Here, we introduce some common types of attention scoring functions used in neural networks.

1. **Dot-Product (Scaled Dot-Product) Attention**

The dot-product attention computes the attention score as the dot product of the query and key vectors. For stability, the score is often scaled by the square root of the dimension of the key vectors.

$$
e_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}
$$

where $\mathbf{q}_i$ is the query vector, $\mathbf{k}_j$ is the key vector, and $d_k$ is the dimension of the key vectors.

2. **Additive (Bahdanau) Attention**

Introduced by Bahdanau et al., this scoring function uses a feedforward neural network to compute the score.

$$
e_{ij} = \mathbf{v}^\top \tanh(\mathbf{W}_1 \mathbf{q}_i + \mathbf{W}_2 \mathbf{k}_j)
$$

where $\mathbf{W}_1$ and $\mathbf{W}_2$ are weight matrices, and $\mathbf{v}$ is a parameter vector.

3. **General Attention**

This is a variant of dot-product attention where the query vector is first multiplied by a learned weight matrix before computing the dot product.

$$
se_{ij} = \mathbf{q}_i^\top \mathbf{W} \mathbf{k}_j
$$

where $\mathbf{W}$ is a learned weight matrix.

4. **Cosine Similarity Attention**

This scoring function computes the cosine similarity between the query and key vectors.

$$
e_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\|\mathbf{q}_i\| \|\mathbf{k}_j\|}
$$

**Attention Weight Calculation**

For all scoring functions, the attention weights $\alpha_{ij}$ are calculated by applying the softmax function to the scores:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k} \exp(e_{ik})}
$$

Different attention scoring functions offer various ways to measure the relevance between query and key vectors. Dot-product attention is computationally efficient and widely used in transformer models, while additive attention can be more expressive but computationally intensive. General attention introduces flexibility with learned weight matrices, and cosine similarity is useful for tasks where angle-based similarity is important. Each type of scoring function has its strengths and is chosen based on the specific needs of the application.

### Bahdanau Attention Mechanism

The Bahdanau attention mechanism, also known as additive attention, was introduced by Bahdanau et al. in 2014 to improve the performance of neural machine translation models. It allows the model to focus on different parts of the input sequence when generating each element of the output sequence, enhancing its ability to capture relevant context. In traditional sequence-to-sequence models, the encoder compresses the entire input sequence into a single fixed-length context vector, which can lead to information loss, especially for long sequences. Bahdanau attention addresses this issue by dynamically creating a context vector for each output token, based on a weighted sum of the encoder's hidden states. The weights are determined by an attention mechanism that scores the relevance of each hidden state with respect to the current decoder state.

**Mathematical Formulation**

Given the encoder hidden states $\{\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_T\}$ and the decoder hidden state at time step $t$, $\mathbf{s}_t$, the attention score for each encoder hidden state is computed as:

$$
e_{t,i} = \mathbf{v}^\top \tanh(\mathbf{W}_1 \mathbf{h}_i + \mathbf{W}_2 \mathbf{s}_t),
$$

where $\mathbf{W}_1$ and $\mathbf{W}_2$ are learned weight matrices, and $\mathbf{v}$ is a learned parameter vector.

The attention weights $\alpha_{t,i}$ are obtained by applying the softmax function to the attention scores:

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}.
$$

The context vector $\mathbf{c}_t$ is then computed as the weighted sum of the encoder hidden states:

$$
\mathbf{c}_t = \sum_{i=1}^{T} \alpha_{t,i} \mathbf{h}_i.
$$

Finally, the context vector $\mathbf{c}_t$ is combined with the decoder hidden state $\mathbf{s}_t$ to generate the output at time step $t$. This combination can be done in various ways, but a common approach is to concatenate them and pass the result through a feedforward layer.

- **Dynamic Context**: By creating a different context vector for each output token, the model can attend to different parts of the input sequence as needed, leading to better handling of long-range dependencies.
- **Improved Translation Quality**: The ability to focus on relevant parts of the input sequence helps in generating more accurate and contextually appropriate translations.

### Multi-Head Attention Mechanism

In multi-head attention, the high-dimensional embeddings of tokens are split into multiple lower-dimensional subspaces. Each subspace, corresponding to an attention head, processes the tokens independently. The outputs from all heads are then concatenated and linearly transformed to produce the final representation. This enables the model to capture diverse contextual information.

**Mathematical Formulation**

Given a set of queries $\mathbf{Q}$, keys $\mathbf{K}$, and values $\mathbf{V}$, the attention function for a single head is computed as:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^\top}{\sqrt{d_k}}\right)\mathbf{V},
$$

where $d_k$ is the dimensionality of the keys.

In multi-head attention, we project the queries, keys, and values into $h$ different subspaces using learned linear projections. For each head $i$:

$$
\mathbf{Q}_i = \mathbf{Q}\mathbf{W}_i^Q, \quad \mathbf{K}_i = \mathbf{K}\mathbf{W}_i^K, \quad \mathbf{V}_i = \mathbf{V}\mathbf{W}_i^V,
$$

where $\mathbf{W}_i^Q$, $\mathbf{W}_i^K$, and $\mathbf{W}_i^V$ are the projection matrices for the $i$-th head.

The attention output for each head is computed as:

$$
\text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i).
$$

The outputs of all heads are then concatenated and projected back to the original dimension:

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}^O,
$$

where $\mathbf{W}^O$ is the output projection matrix.

By attending to information from different representation subspaces, multi-head attention allows the model to capture more nuanced and diverse aspects of the input data, leading to improved performance in various natural language processing tasks.
