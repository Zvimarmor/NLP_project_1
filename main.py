import spacy
from datasets import load_dataset

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
text = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
doc = nlp(text[0]['text'])

# Print the named entities in the doc
for ent in doc.ents:
    print(ent.text, ent.label_)