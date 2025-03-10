import joblib
import pandas as pd
import re
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfTransformer
from src.sar_project.agents import SymptomMatcher

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

    symptom_matches = SymptomMatcher.extract_symptoms(input_text, symptom_synonyms)

    if symptom_matches:
        detected_symptoms = [obj.matched_symptom for obj in symptom_matches]
        print("\nDetected Symptoms:", detected_symptoms)
        for symptom_match in symptom_matches:
            print(symptom_match)
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
