---
title: "How Does BPE Tokenization Work"
date: 2024-05-28
categories:
   - tokenization
summary: Before we dive into BPE algorithm, lets take a look at Tokenization. Tokenization is the process of breaking down text into smaller units called tokens. In the context of the Byte Pair Encoding (BPE) algorithm, tokenization involves splitting words into subword units based on a learned vocabulary. The BPE tokenizer aims to find a balance between representing the text with a limited vocabulary size while still capturing meaningful subword units.
---

Before we dive into BPE algorithm, lets take a look at Tokenization. Tokenization is the process of breaking down text into smaller units called tokens. In the context of the Byte Pair Encoding (BPE) algorithm, tokenization involves splitting words into subword units based on a learned vocabulary. The BPE tokenizer aims to find a balance between representing the text with a limited vocabulary size while still capturing meaningful subword units.

Other types of tokenization based on the level at which the text is split: 

1. Word Tokenization
2. Character Tokenization
3. N-gram Tokenizaiton
4. Sentence Tokenization

---

**Byte pair encoding**  (also known as  **digram coding** ) is an algorithm, first described in 1994 by Philip Gage for encoding strings of text into tabular form for use in downstream modeling. Its modification is notable as the [large language model](https://en.wikipedia.org/wiki/Large_language_model "Large language model") tokenizer with an ability to combine both tokens that encode single characters (including single digits or single punctuation marks) and those that encode whole words (even the longest compound words). This modification, in the first step, assumes all unique characters to be an initial set of 1-character long [n-grams](https://en.wikipedia.org/wiki/N-grams "N-grams") (i.e. initial "tokens"). Then, successively the most frequent pair of adjacent characters is merged into a new, 2-character long n-gram and all instances of the pair are replaced by this new token. This is repeated until a vocabulary of prescribed size is obtained. Note that new words can always be constructed from final vocabulary tokens and initial-set characters (Wikipedia)

Heres a demonstration that how BPE works:

### The initial tokens and vocab

```python
sentence = 'Today is such a great day'
#split words into characters and mark the end of the words 
char_tokens = [list(word[:-1]) + [word[-1]] for word in sentence.split()]
#Initialize the vocab which consists of the unique chars in the sentence
vocab = list(set(char for word in char_tokens for char in word))
```

The results:

```python
#The initial tokens
[['T', 'o', 'd', 'a', 'y'], ['i', 's'], ['s', 'u', 'c', 'h'], ['a'], ['g', 'r', 'e', 'a', 't'], ['d', 'a', 'y']]

#The initial vocab
['w', 'd', 'e', 'c', 'w', 'a', 'r', 'T', 'i', 'o', 's', 'g', 'u']

```

*Note: nested list was used to sepearete words or can add '/w' to mark the end of the words*

### Interative merging:

* The tokenizer iteratively merges the most frequent character pairs to form new subword units.
* In each iteration, the tokenizer identifies the most frequent pair of adjacent characters and merges them into a single token.
* The merging process continues until the desired vocabulary size is reached or a maximum number of iterations is performed.

Lets go through an example: 

In the previous intial tokens, we find that the most frequent pair: ('d', 'a')

Then we replace 'd' and 'a' in the initial tokens with 'da', and add 'da' to the vocab

We also update the token frequncy at the end

```python
#The first merge
[['T', 'o', 'da', 'y'], ['i', 's/w'], ['s', 'u', 'c', 'h'], ['a'], ['g', 'r', 'e', 'a', 't'], ['da', 'y']]

#The updated vocab 
[ 'd', 'e', 'c', 'a', 'h', 'r', 'T', 'y', 'i', 'o', 's', 'g', 'u', 'da']

```

### Tokenization:

How does the trained model tokenize new sentences

* Word-level tokenization:

  * The input text is split into individual words.
  * Each word is checked against the learned vocabulary to see if it exists as a complete token.
* Subword tokenization:

  * If a word is not found in the vocabulary, it is split into subword units.
  * The tokenizer finds the longest subword that exists in the vocabulary and adds it to the list of tokens.
  * The process is repeated for the remaining part of the word until the entire word is tokenized.
  * Handling single characters:

    * If a single character is encountered that is not part of any subword unit, it is added as an individual token.

    ```python
    def tokenize(self, text):
            tokens = []
            for word in text.split():
                # Step 1: Check if the word exists in the vocabulary
                if word in self.vocab:
                    tokens.append(word)
                else:
                    # Step 2: Split the word into subword tokens
                    word_tokens = []
                    while len(word) > 0:
                        longest_subword = self._find_longest_subword(word)
                        if longest_subword:
                            word_tokens.append(longest_subword)
                            word = word[len(longest_subword):]
                        else:
                            # Step 3: Handle remaining single characters
                            word_tokens.append(word[0])
                            word = word[1:]
                    tokens.extend(word_tokens)
            return tokens


    def _find_longest_subword(self, word):
            for i in range(len(word), 0, -1):
                subword = word[:i]
                if subword in self.vocab:
                    return subword
            return ''
    ```

One example: 

In the previous example, the final vocab is: `[ 'day', 'e', 'c', 'a', 'h', 'r', 'T', 'y', 'is', 'o', 's', 'g', 'u', 'da']`

The sentence to be tokenized is: 'Today is such a good day'

The tokens would be: `['T', 'o', 'day', 'is', 'such', 'a', 'g', 'o', 'o', 'day']`

In summary, the Byte Pair Encoding (BPE) algorithm is a subword tokenization technique that aims to create a compact and efficient representation of text by iteratively merging the most frequent character pairs. The BPE tokenizer strikes a balance between word-level and character-level tokenization, creating meaningful subword units that can handle out-of-vocabulary words effectively. By leveraging frequency information during training and tokenization, BPE achieves a compact representation of text while capturing important patterns and reducing the vocabulary size. Overall, the BPE tokenizer is a powerful and widely used technique for subword tokenization, offering a balance between efficiency and expressiveness in representing text for various natural language processing applications.
