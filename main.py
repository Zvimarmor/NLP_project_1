import spacy
from datasets import load_dataset
from collections import Counter, defaultdict
import math

# Preprocess the text data
def preprocess(text, nlp):
    '''Preprocess the text data
    Args:
        text: List of dictionaries with 'text' key
    Returns:
        List of lists of lemmatized tokens
    '''
    # Extract the 'text' field from each dictionary
    tokens = [doc for doc in text]

    for i in range(len(tokens)):
        tokens[i] = nlp(tokens[i]['text'])
        tokens[i] = [token.lemma_.lower() for token in tokens[i] if token.is_alpha]

    return tokens


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

    # Compute unigram probabilities by dividing the count of each word by the total number of tokens
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

    # Add <START> token to each document
    for doc in train_data:
        doc = ['<START>'] + doc
        for i in range(len(doc) - 1):
            bigram_counts[doc[i]][doc[i + 1]] += 1
            unigram_counts[doc[i]] += 1
    
    # Calculate bigram probabilities
    bigram_probs = {
        word: {next_word: count / unigram_counts[word] for next_word, count in next_words.items()}
        for word, next_words in bigram_counts.items()
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
    context = context.lower()
    next_word_probs = bigram_probs.get(context)
    if next_word_probs:
        return max(next_word_probs, key=next_word_probs.get)  # Return the word with the highest probability
    else:
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
    doc = nlp(sentence.lower())
    sentence = ['<START>'] + [token.lemma_ for token in doc if token.is_alpha]
    log_prob = 0

    for i in range(len(sentence) - 1):
        prev_word, next_word = sentence[i], sentence[i + 1]

        if next_word not in bigram_probs.get(prev_word, {}):
            return float('-inf') # Return negative infinity if the bigram is not in the model

        log_prob += math.log(bigram_probs[prev_word][next_word])
    
    return log_prob


# Compute the perplexity of a list of sentences given a bigram model
def compute_perplexity(sentences, bigram_probs):
    '''Compute the perplexity of a list of sentences given a bigram model.
    Args:
        sentences: List of sentences
        bigram_probs: Dictionary with bigram probabilities
    Returns:
        The perplexity of the sentences
    ''' 
    #calculate the total log probability of the sentences
    total_log_prob = 0
    total_words = 0
    
    for sentence in sentences:
        sentence_prob = compute_sentence_probability(sentence, bigram_probs)

        if sentence_prob == float('-inf'):  # Handle unshown sentences
            return float('inf')

        total_log_prob += sentence_prob
        total_words += len(sentence.split()) + 1  # +1 for the <START> token
    
    avg_log_prob = total_log_prob / total_words
    perplexity = math.exp(-avg_log_prob)
    return perplexity

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
    log_prob = 0
    
    for i in range(len(sentence) - 1):
        unigram_prob = unigram_probs.get(sentence[i+1], 0) #zero if the word is not in the unigram model
        bigram_prob = bigram_probs.get(sentence[i], {}).get(sentence[i+1], 0) #same as above
        
        if unigram_prob == 0 and bigram_prob == 0: #if the word is not in either model, return negative infinity
            return float('-inf')
        
        combined_prob = lambda_bigram * bigram_prob + lambda_unigram * unigram_prob
        if combined_prob == 0:
            return float('-inf')
        log_prob += math.log(combined_prob)
    
    return log_prob

if __name__ == '__main__':
    #Question 1: Load the dataset and train the unigram and bigram models
    nlp = spacy.load("en_core_web_sm")
    text = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")

    train_data = preprocess(text, nlp)

    unigram_probs = train_unigram(train_data)
    bigram_probs = train_bigram(train_data)

    #Question 2: Predict the next word in the sentence 'I have house in...'
    print('**************************************************')
    print('Question 2; predict the next word in the sentence')
    print('I have house in...', predict_next_word(bigram_probs, 'in'))

    #Question 3: Compute the probability of a sentence given a bigram model
    print('**************************************************')
    print('Question 3.a; Compute the probability of a sentence using the bigram model')
    Sentences = ["Brad Pitt was born in Oklahoma", "The actor was born in USA"]

    sentence_1_prob = compute_sentence_probability(Sentences[0], bigram_probs)
    sentence_2_prob = compute_sentence_probability(Sentences[1], bigram_probs)
    print('The probability of the sentence ', Sentences[0], ' is:', sentence_1_prob)
    print('The probability of the sentence ', Sentences[1], ' is:', sentence_2_prob)

    print('**************************************************')
    print('Question 3.b; Compute the perplexity of the two sentences using the bigram model')
    perplexity_both = compute_perplexity([Sentences[0], Sentences[1]], bigram_probs) # Compute the perplexity of a list of sentences given a bigram model
    print('The perplexity of the two sentences is:', perplexity_both)

    print('**************************************************')
    print('Question 4; Compute the interpolated probability of the sentence, using the unigram with lambda=1/3 and bigram with lambda=2/3')
    interpolated_prob_1 = interpolate_models(unigram_probs, bigram_probs, Sentences[0]) # Interpolate unigram and bigram models
    interpolated_prob_2 = interpolate_models(unigram_probs, bigram_probs, Sentences[1]) # Interpolate unigram and bigram models
    print('The interpolated probability of the sentence ', Sentences[0], ' is:', interpolated_prob_1)
    print('The interpolated probability of the sentence ', Sentences[1], ' is:', interpolated_prob_2)



