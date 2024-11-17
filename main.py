import spacy
from datasets import load_dataset
from collections import Counter, defaultdict
import math

# Preprocess the text data
def preprocess(text):
    '''Preprocess the text data
    Args:
        text: List of dictionaries with 'text' key
    Returns:
        List of lists of lemmatized tokens
    '''
    for line in text:
        docs = [nlp(line['text'])]
    processed_text = []
    for doc in docs:
        tokens = []
        for token in doc:
            if token.is_alpha:
                token = token.lemma_.lower()
                tokens.append(token)
        processed_text.append(tokens)
    return processed_text

# Train a unigram model
def train_unigram(train_data):
    '''Train a unigram model on the given data.
    Args:
        train_data: List of lists of tokens
    Returns:
        Dictionary with unigram probabilities
    '''
    unigram_counts = Counter()
    total_tokens = 0

    for doc in train_data:
        unigram_counts.update(doc)
        total_tokens += len(doc)

    unigram_probs = {word: count / total_tokens for word, count in unigram_counts.items()}

    return unigram_probs

# Train a bigram model
def train_bigram(train_data):
    '''Train a bigram model on the given data.
    Args:
        train_data: List of lists of tokens
    Returns:
        Dictionary with bigram probabilities
    '''
     bigram_counts = defaultdict(Counter)
    unigram_counts = Counter()
    total_bigrams = 0

    for doc in train_data:
        doc = ['<START>'] + doc
        for i in range(len(doc) - 1):
            bigram_counts[doc[i]][doc[i+1]] += 1
            unigram_counts[doc[i]] += 1
            total_bigrams += 1

    bigram_probs = {
        w1: {w2: count / unigram_counts[w1] for w2, count in bigrams.items()}
        for w1, bigrams in bigram_counts.items()
    }
    return bigram_probs

# Predict the next word given a context
def predict_next_word(bigram_probs, context):
    '''Predict the next word given a context.
    Args:
        bigram_probs: Dictionary with bigram probabilities
        context: The context word
    Returns:
        The most likely next word
    '''
    next_word_probs = bigram_probs.get(context)
    if next_word_probs:
        return max(next_word_probs, key=next_word_probs.get)
    return None

# Compute the probability of a sentence given a bigram model
def compute_sentence_probability(sentence, bigram_probs):
    '''Compute the probability of a sentence given a bigram model.
    Args:
        sentence: The sentence to compute the probability of
        bigram_probs: Dictionary with bigram probabilities
    Returns:
        The log probability of the sentence
    '''
    sentence = ['<START>'] + sentence.lower().split()
    prob = 0
    for i in range(len(sentence) - 1):
        p = bigram_probs.get(sentence[i], {}).get(sentence[i+1], 1e-10)
        prob += math.log(p)
    return prob

# Compute the perplexity of a list of sentences given a bigram model
def compute_perplexity(sentences, bigram_probs):
    '''Compute the perplexity of a list of sentences given a bigram model.
    Args:
        sentences: List of sentences
        bigram_probs: Dictionary with bigram probabilities
    Returns:
        The perplexity of the sentences
    '''
    total_log_prob = sum(compute_sentence_probability(sent, bigram_probs) for sent in sentences)
    n = sum(len(sent.split()) + 1 for sent in sentences)  # +1 for START token
    return math.exp(-total_log_prob / n)

# Interpolate unigram and bigram models
def interpolate_models(unigram_probs, bigram_probs, sentence, lambda_bigram=2/3, lambda_unigram=1/3):
    '''Compute the interpolated probability of a sentence given unigram and bigram models.
    Args:
        unigram_probs: Dictionary with unigram probabilities
        bigram_probs: Dictionary with bigram probabilities
        sentence: The sentence to compute the probability of
        lambda_bigram: Weight for bigram model
        lambda_unigram: Weight for unigram model
    Returns:
        The log probability of the sentence
    '''
    sentence = ['<START>'] + sentence.lower().split()
    prob = 0
    for i in range(len(sentence) - 1):
        unigram_prob = unigram_probs.get(sentence[i+1], 1e-10)
        bigram_prob = bigram_probs.get(sentence[i], {}).get(sentence[i+1], 1e-10)
        combined_prob = lambda_bigram * bigram_prob + lambda_unigram * unigram_prob
        prob += math.log(combined_prob)
    return prob

if __name__ == '__main__':
    # Load the dataset
    nlp = spacy.load("en_core_web_sm")
    text = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    train_data = preprocess(text) # Preprocess the text data we loaded

    unigram_probs = train_unigram(train_data) # Train a unigram model on the training data
    bigram_probs = train_bigram(train_data) # Train a bigram model on the training data

    print(predict_next_word(bigram_probs, 'in')) # Predict the next word given the context 'in'

    Sentences = ["Brad Pitt was born in Oklahoma", "The actor was born in USA"]

    sentence_1_prob = compute_sentence_probability(Sentences[0], bigram_probs) # Compute the probability of a sentence given a bigram model
    sentence_2_prob = compute_sentence_probability(Sentences[1], bigram_probs) # Compute the probability of a sentence given a bigram model


    perplexity_both = compute_perplexity([Sentences[0], Sentences[1]], bigram_probs) # Compute the perplexity of a list of sentences given a bigram model

    interpolated_prob = interpolate_models(unigram_probs, bigram_probs, Sentences[0]) # Interpolate unigram and bigram models

    print(sentence_1_prob)
    print(sentence_2_prob)

    print(interpolated_prob)
    print(perplexity_both)


