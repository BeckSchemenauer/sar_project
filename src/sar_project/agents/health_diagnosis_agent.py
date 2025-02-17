import re
import pandas as pd
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from rapidfuzz import fuzz
from nltk.stem import PorterStemmer
ps = PorterStemmer()

def normalize_phrase(phrase):
    tokens = phrase.split()
    # Stem each token and then rejoin
    normalized = " ".join(ps.stem(token) for token in tokens)
    return normalized


# Download required NLTK resources
#nltk.download("punkt")
#nltk.download("wordnet")
#nltk.download("stopwords")  # Download stop words list

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

# Load CSV and extract symptom names
csv_path = "src/sar_project/knowledge/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
df = pd.read_csv(csv_path)

# Symptoms are all columns except the first (assumed to be the disease column)
symptoms = list(df.columns[1:])

# Create a symptom-to-synonyms dictionary
def expand_symptoms(symptom_list):
    """Generates a dictionary mapping each symptom to its synonyms."""
    symptom_dict = {}
    for symptom in symptom_list:
        synonyms = set()
        for syn in wordnet.synsets(symptom.replace(" ", "_")):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace("_", " "))
        symptom_dict[symptom] = list(synonyms) if synonyms else [symptom]
    return symptom_dict


symptom_synonyms = expand_symptoms(symptoms)


# Remove stop words from the text
def preprocess_text(text):
    """Preprocess text by tokenizing and removing stop words."""

    tokens = word_tokenize(text.lower())  # Tokenize input text
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return filtered_tokens


class SymptomMatch:
    """Custom class to represent a symptom match."""

    def __init__(self, phrase, phrase_stem=None, matched_symptom=None, synonym=None, match_type=""):
        self.phrase = phrase
        self.phrase_stem = phrase_stem
        self.matched_symptom = matched_symptom
        self.synonym = synonym
        self.match_type = match_type  # "phrase" or "synonym"

    def __repr__(self):
        if self.match_type == "phrase":
            return f"'{self.phrase}' matched with symptom: {self.matched_symptom} (whole phrase match)"
        elif self.match_type == "dependency":
            return f"'{self.phrase}' matched with symptom: {self.matched_symptom} (via dependency parsing) (Stemmed: {self.phrase_stem})"
        elif self.synonym:
            return f"'{self.phrase}' matched with symptom: {self.matched_symptom} (via synonym: {self.synonym})"
        else:
            return f"'{self.phrase}' matched with symptom: {self.matched_symptom}"


def match_phrase(phrase, symptom_dict, threshold=80, match_type="dependency"):
    """Matches a given phrase to symptoms using fuzzy matching."""
    detected = []
    normalized_phrase = normalize_phrase(phrase)

    for symptom, synonyms in symptom_dict.items():
        for synonym in synonyms:
            normalized_synonym = normalize_phrase(synonym)
            score = fuzz.token_set_ratio(normalized_phrase, normalized_synonym)
            if score >= threshold:
                detected.append(
                    SymptomMatch(
                        phrase=phrase,
                        phrase_stem=normalized_phrase,
                        matched_symptom=symptom,
                        synonym=synonym,
                        match_type=match_type
                    )
                )
    return detected


def extract_symptoms(text, symptom_dict, threshold=80):
    """Matches input text to symptoms using fuzzy matching and dependency parsing."""
    detected_symptoms = []
    matched_words = set()  # Tracks words already matched to avoid redundancy
    doc = nlp(text.lower())

    # Process dependency relations for adjectives and verbs
    for token in doc:
        if token.text in stop_words:
            continue

        subject = None
        if token.dep_ == "acomp":  # Adjective complement (e.g., "swollen")
            for child in token.head.children:
                if child.dep_ == "nsubj":  # Find subject noun
                    subject = child.text
                    break
            if subject:
                phrase = f"{token.text} {subject}"  # e.g., "swollen arm"
                detected_symptoms.extend(match_phrase(phrase, symptom_dict, threshold))
                matched_words.update([token.text, subject])

        elif token.pos_ == "VERB" and token.tag_ == "VBG":  # Verb in -ing form (e.g., "swelling")
            for child in token.children:
                if child.dep_ == "nsubj":  # Find subject noun
                    subject = child.text
                    break
            if subject:
                phrase = f"{subject} {token.text}"  # e.g., "arm swelling"
                detected_symptoms.extend(match_phrase(phrase, symptom_dict, threshold))
                matched_words.update([subject, token.text])

    # Match multi-word symptoms from the entire text
    joined_text = " ".join(token.text for token in doc)
    phrase_matches = match_phrase(joined_text, symptom_dict, threshold, match_type="phrase")
    detected_symptoms.extend(phrase_matches)

    # Update matched_words with tokens from each matched symptom phrase
    for phrase_match in phrase_matches:
        # Here, match.matched_symptom is assumed to be something like "arm pain" or "knee pain"
        tokens_in_match = phrase_match.matched_symptom.lower().split()
        matched_words.update(tokens_in_match)

    # Process individual tokens if not already matched
    for token in doc:
        if token.text in stop_words or token.text in matched_words:
            continue
        detected_symptoms.extend(match_phrase(token.text, symptom_dict, threshold, match_type="single-word"))
        matched_words.add(token.text)

    return detected_symptoms


# Interactive loop
print("Type a symptom description (or 'exit' to quit):")
while True:
    input_text = input("\n> ")
    #input_text = "My arm is swollen."
    if input_text.lower() == "exit":
        print("Exiting...")
        break
    elif "$" in input_text:
        print(symptom_synonyms[re.sub('[$]', '', input_text)])
        continue

    matches = extract_symptoms(input_text, symptom_synonyms)

    if matches:
        print("\nDetected Symptoms:")
        for match in matches:
            print(match)  # Automatically calls the __repr__ method of the SymptomMatch class

    else:
        print("\nNo symptoms detected.")

    #exit(0)