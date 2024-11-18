# NLP Project 1

A simple project demonstrating unigram and bigram language models, including sentence probability, perplexity calculation, and next-word prediction.

---

## Features

- **Text Preprocessing**: Tokenization and lemmatization.  
- **Unigram & Bigram Models**: Train, predict, and compute probabilities.  
- **Perplexity**: Evaluate the model's quality.  
- **Interpolation**: Combine unigram and bigram probabilities.  

---

## Installation

```bash
pip install spacy datasets
python -m spacy download en_core_web_sm
```

---

## Usage

- **Train Models**:  
  ```python
  unigram_probs = train_unigram(train_data)
  bigram_probs = train_bigram(train_data)
  ```

- **Next-Word Prediction**:  
  ```python
  predict_next_word(bigram_probs, "word")
  ```

- **Sentence Probability**:  
  ```python
  compute_sentence_probability("Sentence", bigram_probs)
  ```

- **Perplexity**:  
  ```python
  compute_perplexity(sentences, bigram_probs)
  ```

- **Interpolation**:  
  ```python
  interpolate_models(unigram_probs, bigram_probs, sentence)
  ```

---

- **Answers to questions**:
**************************************************
Question 2; predict the next word in the sentence
I have house in... the
**************************************************
Question 3.a; Compute the probability of a sentence using the bigram model
The probability of the sentence  Brad Pitt was born in Oklahoma  is: -inf
The probability of the sentence  The actor was born in USA  is: -29.729887174236524
**************************************************
Question 3.b; Compute the perplexity of the two sentences using the bigram model
The perplexity of the two sentences is: inf
**************************************************
Question 4; Compute the interpolated probability of the sentence, using the unigram with lambda=1/3 and bigram with lambda=2/3
The interpolated probability of the sentence  Brad Pitt was born in Oklahoma  is: -inf
The interpolated probability of the sentence  The actor was born in USA  is: -inf

---