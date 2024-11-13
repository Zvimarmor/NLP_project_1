import spacy
from datasets import load_dataset
from collections import defaultdict, Counter
import math

# Load SpaCy and dataset
nlp = spacy.load("en_core_web_sm")
text = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")

# Preprocessing: Get lemmas, ignoring punctuation and numbers
def preprocess(text):
    docs = [nlp(line['text']) for line in text]
    processed_text = []
    for doc in docs:
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
        processed_text.append(tokens)
    return processed_text

train_data = preprocess(text)

def train_unigram(train_data):
    unigram_counts = Counter()
    total_tokens = 0

    for doc in train_data:
        unigram_counts.update(doc)
        total_tokens += len(doc)

    unigram_probs = {word: count / total_tokens for word, count in unigram_counts.items()}
    return unigram_probs

def train_bigram(train_data):
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
    next_word_probs = bigram_probs.get(context, {})
    return max(next_word_probs, key=next_word_probs.get) if next_word_probs else None

bigram_probs = train_bigram(train_data)
print(predict_next_word(bigram_probs, 'in'))

def compute_sentence_probability(sentence, bigram_probs):
    sentence = ['<START>'] + sentence.lower().split()
    prob = 0
    for i in range(len(sentence) - 1):
        p = bigram_probs.get(sentence[i], {}).get(sentence[i+1], 1e-10)
        prob += math.log(p)
    return prob

def compute_perplexity(sentences, bigram_probs):
    total_log_prob = sum(compute_sentence_probability(sent, bigram_probs) for sent in sentences)
    n = sum(len(sent.split()) + 1 for sent in sentences)  # +1 for START token
    return math.exp(-total_log_prob / n)

def interpolate_models(unigram_probs, bigram_probs, sentence, lambda_bigram=2/3, lambda_unigram=1/3):
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
