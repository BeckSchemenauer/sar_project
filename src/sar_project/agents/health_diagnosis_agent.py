import re
import joblib
import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import wordnet, stopwords
from rapidfuzz import fuzz
from nltk.stem import PorterStemmer

ps = PorterStemmer()
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")  # Download stop words list

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

csv_path = "src/sar_project/knowledge/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
df = pd.read_csv(csv_path)
symptoms = list(df.columns[1:])


def load_model(model_path='src/sar_project/models/best_random_forest_model.pkl'):
    return joblib.load(model_path)


def predict_disease(model, symptom_list, all_symptoms):
    # input  vector (1 for present symptoms, 0 otherwise)
    input_vector = {symptom: 1 if symptom in symptom_list else 0 for symptom in all_symptoms}

    input_df = pd.DataFrame([input_vector])

    predicted_disease = model.predict(input_df)[0]
    return predicted_disease


def normalize_phrase(phrase):
    tokens = phrase.split()
    # Stem each token and then rejoin
    normalized = " ".join(ps.stem(token) for token in tokens)
    return normalized


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


symptom_synonyms = expand_symptoms(symptoms)


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
    matched_words = set()
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
                detected_symptoms.extend(match_phrase(phrase, symptom_dict, threshold))
                matched_words.update([token.text, subject])

        elif token.pos_ == "VERB" and token.tag_ == "VBG":
            for child in token.children:
                if child.dep_ == "nsubj":
                    subject = child.text
                    break
            if subject:
                phrase = f"{subject} {token.text}"
                detected_symptoms.extend(match_phrase(phrase, symptom_dict, threshold))
                matched_words.update([subject, token.text])

    # match multi-word symptoms from the entire text
    joined_text = " ".join(token.text for token in doc)
    phrase_matches = match_phrase(joined_text, symptom_dict, threshold, match_type="phrase")
    detected_symptoms.extend(phrase_matches)

    # update matched_words with tokens from each matched symptom phrase
    for phrase_match in phrase_matches:
        tokens_in_match = phrase_match.matched_symptom.lower().split()
        matched_words.update(tokens_in_match)

    # process individual tokens if not already matched
    for token in doc:
        if token.text in stop_words or token.text in matched_words:
            continue
        detected_symptoms.extend(match_phrase(token.text, symptom_dict, threshold, match_type="single-word"))
        matched_words.add(token.text)

    return detected_symptoms


rf_model = load_model()
print("Type a symptom description (or 'exit' to quit):")
while True:
    input_text = input("\n> ")

    if input_text.lower() == "exit":
        print("Exiting...")
        break
    elif "$" in input_text:
        print(symptom_synonyms[re.sub('[$]', '', input_text)])
        continue

    matches = extract_symptoms(input_text, symptom_synonyms)

    if matches:
        detected_symptoms = [obj.matched_symptom for obj in matches]
        print("\nDetected Symptoms:", detected_symptoms)
        print("Predicted Disease:", predict_disease(rf_model, detected_symptoms, symptoms))

    else:
        print("\nNo symptoms detected.")

    #exit(0)
