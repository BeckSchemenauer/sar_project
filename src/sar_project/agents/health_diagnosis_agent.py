import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from rapidfuzz import fuzz

# Download required NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")  # Download stop words list

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


def preprocess_text(text):
    """Preprocess text by tokenizing and removing stop words."""
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())  # Tokenize input text
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return filtered_tokens


# Symptom matching function
def extract_symptoms(text, symptom_dict, threshold=100):
    """Matches input text to symptoms using fuzzy matching."""
    tokens = preprocess_text(text)
    detected_symptoms = []
    matched_words = set()  # To keep track of matched words to avoid redundancy

    # Match multi-word symptoms first
    for symptom, synonyms in symptom_dict.items():
        # Match the entire symptom as a phrase first
        score = fuzz.token_set_ratio(" ".join(tokens), symptom.lower())
        if score >= threshold:
            detected_symptoms.append((symptom, symptom))  # Ensure correct structure
            matched_words.update(symptom.lower().split())

    # Now match individual words for remaining tokens
    for word in tokens:
        if word in matched_words:
            continue  # Skip if this word is already part of a multi-word symptom
        for symptom, synonyms in symptom_dict.items():
            for synonym in synonyms:
                score = fuzz.token_set_ratio(word, synonym)
                if score >= threshold:
                    detected_symptoms.append((word, symptom, synonym))

    return detected_symptoms


# Interactive loop
print("Type a symptom description (or 'exit' to quit):")
while True:
    #input_text = input("\n> ")
    input_text = "I have a sore throat and cough."
    if input_text.lower() == "exit":
        print("Exiting...")
        break

    matches = extract_symptoms(input_text, symptom_synonyms)

    if matches:
        print("\nDetected Symptoms:")
        for match in matches:
            if len(match) == 2:  # Multi-word symptom match
                word, symptom = match
                print(f"- '{word}' matched with symptom: {symptom}")
            elif len(match) == 3:  # Single word matching with synonym
                word, symptom, synonym = match
                print(f"- '{word}' matched with symptom: {symptom} (via synonym: {synonym})")
    else:
        print("\nNo symptoms detected.")

    exit(0)
