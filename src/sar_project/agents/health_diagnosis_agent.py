import re
import joblib
import nltk
import pandas as pd
import spacy
from nltk.corpus import wordnet, stopwords
from rapidfuzz import fuzz
from nltk.stem import PorterStemmer
import subprocess
import sys
from sklearn.feature_extraction.text import TfidfTransformer

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

csv_path = "src/sar_project/knowledge/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
df = pd.read_csv(csv_path)
symptoms = list(df.columns[1:])


def load_model(model_path='src/sar_project/models/best_random_forest_model.pkl'):
    return joblib.load(model_path)


def get_top_symptoms_for_condition(condition, condition_col="diseases", top_n=5):
    # aggregate symptom presence for each condition
    symptom_cols = [col for col in df.columns if col != condition_col]

    # count occurrences of each symptom per condition
    condition_symptom_counts = df.groupby(condition_col)[symptom_cols].sum()

    # check if the condition exists
    if condition not in condition_symptom_counts.index:
        raise ValueError(f"Condition '{condition}' not found in dataset.")

    # calculate frequency-based ranking
    sorted_frequent = condition_symptom_counts.loc[condition].sort_values(ascending=False)
    most_frequent_symptom = sorted_frequent.index[0]  # highest occurrence symptom
    top_frequent = sorted_frequent.head(top_n).index.tolist()

    # calculate TF-IDF scores
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(condition_symptom_counts)

    # convert TF-IDF matrix to df
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                            index=condition_symptom_counts.index,
                            columns=symptom_cols)

    # get top TF-IDF symptoms for the given condition
    sorted_tfidf = tfidf_df.loc[condition].sort_values(ascending=False)
    highest_tfidf_symptom = sorted_tfidf.index[0]  # highest TF-IDF scored symptom
    top_tfidf = sorted_tfidf.head(top_n).index.tolist()

    # compute the first 3 from the union of both lists (excluding most frequent & most unique)
    combined_symptoms = list(dict.fromkeys(top_frequent[1:] + top_tfidf[1:]))[:3]

    return most_frequent_symptom, highest_tfidf_symptom, combined_symptoms


def predict_disease(model, symptom_list, all_symptoms):
    # input  vector (1 for present symptoms, 0 otherwise)
    input_vector = {symptom: 1 if symptom in symptom_list else 0 for symptom in all_symptoms}
    input_df = pd.DataFrame([input_vector])

    # get probabilities for all classes
    probabilities = model.predict_proba(input_df)[0]

    class_labels = model.classes_
    disease_probabilities = list(zip(class_labels, probabilities))
    sorted_probabilities = sorted(disease_probabilities, key=lambda x: x[1], reverse=True)

    # return top 3
    return sorted_probabilities[:3]


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
        for match in matches:
            print(match)
        predicted_diseases = predict_disease(rf_model, detected_symptoms, symptoms)
        top_prediction = predicted_diseases[0][0]
        most_frequent_symptom, highest_tfidf_symptom, combined_symptoms = get_top_symptoms_for_condition(top_prediction)
        top_symptoms = [most_frequent_symptom, highest_tfidf_symptom]
        top_symptoms.extend(combined_symptoms)
        print("Predicted Disease:", predicted_diseases)
        print(f"Top symptoms for predicted disease: {top_symptoms}")

    else:
        print("\nNo symptoms detected.")

    #exit(0)
