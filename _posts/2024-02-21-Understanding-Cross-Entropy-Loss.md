---
title: "Understanding Cross Entropy Loss"
date: 2024-02-21
summary: How does cross entropy loss work with an language model example
categories: 
- cross-entropy-loss
- language-model
---
Entropy originated in the field of thermodynamics and was later adapted to information theory by Claude Shannon. Its used to measure **uncertainty or unpredictability**in a system.

Before this, there was no formal way to **quantify the information** needed to describe a system’s state. You might wonder if this ‘uncertainty’ is the same as the uncertainty we experience in daily life, like being unsure if someone will come to dinner. It’s similar but not the same…

In everyday use, ‘uncertainty’ means having doubts or a lack of confidence.  **In information theory, ‘uncertainty’ is a precise measure of how much information or surprise an outcome holds, based on the probability distribution of a random variable** . Let’s dive into an example.

Lets simulate a fair coin toss for 1000 times:

```python
import random

# Function to simulate a fair coin toss
def coin_toss():
    return "Heads" if random.choice([True, False]) else "Tails"

# Simulate 1000 coin tosses
tosses = [coin_toss() for _ in range(1000)]

# Count the number of heads and tails
heads_count = tosses.count("Heads")
tails_count = tosses.count("Tails")

# Print the results
print(f"Heads: {heads_count}")
print(f"Tails: {tails_count}")

--------------
Heads: 488
Tails: 512

```

The probability:

Getting a heads is: P(H) = 488 / 1000 = 0.488

Getting a tail is: P(T) = 512 / 1000 = 0.512

Now lets simulate a unfair coin toss for 100 times:

```python
import random

# Function to simulate an unfair coin toss
def unfair_coin_toss():
    return "Tails" if random.random() < 5/6 else "Heads"

# Simulate 1000 coin tosses
tosses = [unfair_coin_toss() for _ in range(1000)]

# Count the number of heads and tails
heads_count = tosses.count("Heads")
tails_count = tosses.count("Tails")

# Print the results
print(f"Heads: {heads_count}")
print(f"Tails: {tails_count}")

----------
Heads: 177
Tails: 823
```

The probability:

Getting a heads is: P(H) = 177 / 1000 = 0.177

Getting a tail is: P(T) = 823 / 1000 = 0.823

Calculate Entropy in both scenarios using this Entropy formula:

$$
H(X)=−∑iP(xi)logP(xi)
$$

Fair coin:

$$
H(X)=−∑iP(xi)logP(xi) = -(0.488*log(0.488) + 0.512 * log(0.512)) = 0.9996
$$

Unfair coin:

$$
H(X)=−∑iP(xi)logP(xi) = -(0.177*log(0.177) + 0.823 * log(0.823)) = 0.6735
$$

In the scenario with a fair coin, the entropy is higher, reflecting greater uncertainty compared to the biased coin scenario. This aligns with our intuition: when two events are equally likely, uncertainty is at its maximum.

---

### **Cross-Entropy Loss in Machine Learning**

In the context of machine learning, entropy is used to measure the uncertainty in predictions. For a perfect model, the predicted probability distribution matches the actual distribution, resulting in low entropy. Conversely, higher entropy indicates a higher degree of uncertainty in the predictions. This Cross-Entropy loss function is widely used in classification problems because it quantifies how well the predicted probabilities match the true class labels. **Cross-entropy loss builds upon the concept of entropy. It measures the difference between two probability distributions.**

Consider a binary classification problem where we predict whether an email is spam (1) or not spam (0). Suppose we have the following true labels and predicted probabilities:

- True label: 1 (spam)
- Predicted probability: 0.8 (spam)

The cross-entropy loss for this single prediction is:

$$
H(P, Q) = -[1 \cdot \log(0.8) + 0 \cdot \log(0.2)] = -\log(0.8) \approx 0.22
$$

This value indicates the level of uncertainty in our prediction. If our model were perfect, the predicted probability would be 1, and the loss would be zero, indicating no uncertainty. Binary cross-entropy loss penalizes incorrect predictions, encouraging the model to output probabilities closer to the true labels.

### Cross-Entropy Loss in Categorical Classification

In multi-class classification problems, cross-entropy loss is extended to categorical cross-entropy loss. Here, the loss function compares the predicted probability distribution over multiple classes with the true distribution. For a classification problem with K classes, the categorical cross-entropy loss is:

For example, in a three-class classification problem (A, B, C) with true label B and predicted probabilities [0.1, 0.7, 0.2], the categorical cross-entropy loss is:

$$
L = -[0 \cdot \log(0.1) + 1 \cdot \log(0.7) + 0 \cdot \log(0.2)] = -\log(0.7) \approx 0.36
$$

### **Cross-Entropy Loss in language model**

Let’s take a look at an example of how Cross-Entropy loss is used in an encoder-decoder translation model. We’ll start by using a trained translation model to generate the first token in the target language.

```python

Python
# encode the src setence and predict the first token in tgt language 
src_sentence = 'Tell me something about yourself'

encoded_src = torch.LongTensor(tokenizer.encode_as_ids(src_sentence)).unsqueeze(0

#the output of the encoder
memory = model.encoder(src_sentence, padding_mask) 

#the first token in the tgt sentence (bos token)
x = tokenizer.bos_id()
tgt_tensor = torch.LongTensor([x]).unsqueeze(0)

#feed both memory and x into decoder, the output of decoder are logits with shape(1, 1, vocab_size)
with torch.no_grad():
    logits = model.decoder(tgt_tenosr, memory, padding_mask, target_mask)
```

The shape of output logits is (batch_size, sequence_length, vocab_size):

```
torch.Size([1, 1, 10000])
```

Then apply the softmax function to the logits generated by the model. This **converts the logits into a probability distribution over the entire vocabulary.** **Find the index of the token in the vocabulary that has the highest probability.**

```python
import torch.nn as nn 
import torch

sm = nn.Softmax(dim = -1)
prob = sm(logits)
print(prob)

print(torch.argmax(prob))
----tensor(900)
print(prob[0][0][900])
-----tensor(0.5717)

print(tokenizer.decode_ids(900))
----'告诉' 
```

The predicted token is ‘告诉‘, whcih aligns with our the first token of the tgt sentence.

```python
tgt_sen = '告诉我关于你的事情'
```

However, since our model only assigns a probability of 0.5717 to the correct token, which indicates a relatively low confidence, we can calculate the loss using the CrossEntropyLoss equation. The loss is given by -log(0.5717). Let’s compute the loss using PyTorch’s CrossEntropyLoss() function.

```python
# now we calculate the loss for the 'correct' prediction with lower confidence
criterion = nn.CrossEntropyLoss()
print(torch.LongTensor([tgt_ids[0]]).view(-1))
loss = criterion(logits.view(1,-1), torch.LongTensor([tgt_ids[0]]).view(-1))
print(loss)

--- tensor(0.5591)
```

Let’s explore two different scenarios: one where our model makes an incorrect prediction and another where it makes a correct prediction with high confidence. For the incorrect prediction, the loss is calculated as -log(4.6965e-07) ≈ 16.2224. For the correct prediction with high confidence, the loss is -log(1.0) ≈ 0

```python
#change the logits on index 900 to make the predicted token not the token 900
logits[0][0][900] /= 100
sm = nn.Softmax(dim = -1)
prob = sm(logits) 
print(torch.argmax(prob)) 
____1114 #notice that the predicted token is 1114 instead of token 900 now 

#now we calculate the loss for this "wrong" prediction 
criterion = nn.CrossEntropyLoss()
loss = criterion(logits.view(1,-1), torch.LongTensor([tgt_ids[0]]).view(-1))
print(loss)
------16.2224
```

```python
logits[0][0][900] *= 10
sm = nn.Softmax(dim = -1)
prob = sm(logits) 
print(torch.argmax(prob)) 
----900
print(prob[0][0][900]
----tensor(1.)

#now we calculate the loss for correct prediction with absolute confidence
criterion = nn.CrossEntropyLoss()
loss = criterion(logits.view(1,-1), torch.LongTensor([tgt_ids[0]]).view(-1))
print(loss)

----tensor(0.)

```

For the entire source code of this translation model: reference my Github repo: [https://github.com/tingaldama278/English-Chinese-Translation-App](https://github.com/tingaldama278/English-Chinese-Translation-App)

### Conclusion

Cross-entropy loss is a powerful tool in machine learning, leveraging the concept of entropy from Shannon’s Information Theory to measure the uncertainty in predictions. By penalizing incorrect predictions and rewarding correct ones, it guides models to improve their accuracy in both binary and categorical classification tasks. Understanding and effectively implementing cross-entropy loss is crucial for developing robust and reliable predictive models
