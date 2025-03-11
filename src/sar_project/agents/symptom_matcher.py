from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from rapidfuzz import fuzz
import spacy
import subprocess
import sys
import pandas as pd
from nltk.corpus import wordnet

ps = PorterStemmer()
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")  # Download stop words list

model_name = "en_core_web_sm"

try:
    nlp = spacy.load(model_name)
    print(f"{model_name} model already loaded.")
except OSError:
    print(f"{model_name} model not found. Downloading...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        nlp = spacy.load(model_name)
        print(f"{model_name} model downloaded and loaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {model_name}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

stop_words = set(stopwords.words("english"))

df = pd.read_csv("src/sar_project/knowledge/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
symptoms = list(df.columns[1:])


class SymptomMatch:

    def __init__(self, phrase, phrase_stem=None, matched_symptom=None, synonym=None, match_type=""):
        self.phrase = phrase
        self.phrase_stem = phrase_stem
        self.matched_symptom = matched_symptom
        self.synonym = synonym
        self.match_type = match_type

    def __repr__(self):
        if self.match_type == "phrase":
            return f"'{self.phrase}' matched with symptom: {self.matched_symptom} (whole phrase match)"
        elif self.match_type == "dependency":
            return f"'{self.phrase}' matched with symptom: {self.matched_symptom} (via dependency parsing) (Stemmed: {self.phrase_stem})"
        elif self.synonym:
            return f"'{self.phrase}' matched with symptom: {self.matched_symptom} (via synonym: {self.synonym})"
        else:
            return f"'{self.phrase}' matched with symptom: {self.matched_symptom}"


# symptom-to-synonyms dictionary
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


symptom_dict = expand_symptoms(symptoms)


def normalize_phrase(phrase):
    tokens = phrase.split()
    # Stem each token and then rejoin
    normalized = " ".join(ps.stem(token) for token in tokens)
    return normalized


def match_phrase(phrase, threshold=80, match_type="dependency"):
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


def extract_symptoms(text, threshold=80):
    """Matches input text to symptoms using fuzzy matching and dependency parsing."""
    detected_symptoms = []  # a list of SymptomMatch objects
    matched_words = set()  # a set of words used to match symptoms
    matched_symptoms = set()  # a set of symptoms that have already been matched
    doc = nlp(text.lower())

    # process dependency relations for adjectives and verbs
    for token in doc:
        if token.text in stop_words:
            continue

        subject = None
        if token.dep_ == "acomp":
            for child in token.head.children:
                if child.dep_ == "nsubj":
                    subject = child.text
                    break
            if subject:
                phrase = f"{token.text} {subject}"
                matches = match_phrase(phrase, threshold)
                for match in matches:
                    if match.matched_symptom not in matched_symptoms:
                        detected_symptoms.append(match)
                        matched_symptoms.add(match.matched_symptom)

                matched_words.update([token.text, subject])

        elif token.pos_ == "VERB" and token.tag_ == "VBG":
            for child in token.children:
                if child.dep_ == "nsubj":
                    subject = child.text
                    break
            if subject:
                phrase = f"{subject} {token.text}"
                matches = match_phrase(phrase, threshold)
                for match in matches:
                    if match.matched_symptom not in matched_symptoms:
                        detected_symptoms.append(match)
                        matched_symptoms.add(match.matched_symptom)

                matched_words.update([subject, token.text])

    # match multi-word symptoms from the entire text
    joined_text = " ".join(token.text for token in doc)
    phrase_matches = match_phrase(joined_text, threshold, match_type="phrase")
    for match in phrase_matches:
        if match.matched_symptom not in matched_symptoms:
            detected_symptoms.append(match)
            matched_symptoms.add(match.matched_symptom)

    # update matched_words with tokens from each matched symptom phrase
    for phrase_match in phrase_matches:
        tokens_in_match = phrase_match.matched_symptom.lower().split()
        matched_words.update(tokens_in_match)

    # process individual tokens if not already matched
    for token in doc:
        if token.text in stop_words or token.text in matched_words:
            continue
        token_matches = match_phrase(token.text, threshold, match_type="single-word")
        for match in token_matches:
            if match.matched_symptom not in matched_symptoms:
                detected_symptoms.append(match)
                matched_symptoms.add(match.matched_symptom)
        matched_words.add(token.text)

    return detected_symptoms
