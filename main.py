import spacy
from datasets import load_dataset
from collections import defaultdict, Counter
import math

# Load SpaCy and dataset
nlp = spacy.load("en_core_web_sm")
text = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")

# Preprocessing: Get lemmas, ignoring punctuation and numbers
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
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
        processed_text.append(tokens)
    return processed_text

train_data = preprocess(text)

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

def predict_next_word(bigram_probs, context):
    '''Predict the next word given a context.
    Args:
        bigram_probs: Dictionary with bigram probabilities
        context: The context word
    Returns:
        The most likely next word
    '''
    next_word_probs = bigram_probs.get(context, {})
    if next_word_probs:
        return max(next_word_probs, key=next_word_probs.get)
    return None

bigram_probs = train_bigram(train_data)
print(predict_next_word(bigram_probs, 'in'))

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

interpolated_prob = interpolate_models(unigram_probs, bigram_probs, "Brad Pitt was born in Oklahoma")
perplexity = compute_perplexity(["Brad Pitt was born in Oklahoma", "The actor was born in USA"], bigram_probs)

if __name__ == '__main__':
    print(interpolated_prob)
    print(perplexity)
