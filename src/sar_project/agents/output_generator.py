from rich.console import Console
from rich.layout import Layout
from rich.table import Table
from rich.text import Text
from src.sar_project.agents import symptom_matcher
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import joblib

condition_description_df = pd.read_csv("src/sar_project/knowledge/disease_descriptions_v2.csv")
df = pd.read_csv("src/sar_project/knowledge/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
all_symptoms = list(df.columns[1:])


def display_diagnosis(detected_symptoms, predicted_diseases, top_symptoms, description):
    console = Console(width=250)  # Set a wider console width

    # Create the layout with spacing
    layout = Layout()
    layout.split_row(Layout(name="left", ratio=5), Layout(name="gap", size=2), Layout(name="middle", ratio=6),
                     Layout(name="right", ratio=6))

    # Left Panel (Diagnosis Summary) with added spacing
    left_text = Text()
    left_text.append(f"Predicted Disease: {predicted_diseases[0][0]}\n\n")
    left_text.append(f"Disease Description:\n{description}\n\n")
    left_text.append(f"Top Symptoms:\n" + "\n".join(f" • {symptom}" for symptom in top_symptoms))

    layout["left"].update(left_text)

    # Add a gap between left panel and tables
    layout["gap"].update("    ")

    # Table 1: Detected Symptoms
    symptom_table = Table(title="Detected Symptoms", show_lines=True)
    symptom_table.add_column("Symptom", justify="left")
    symptom_table.add_column("Method of Detection", justify="right")

    for symptom, method in detected_symptoms:
        symptom_table.add_row(symptom, method)

    # Table 2: Predicted Diseases
    disease_table = Table(title="Top Predicted Diseases", show_lines=True)
    disease_table.add_column("Disease", justify="left")
    disease_table.add_column("Confidence", justify="right")

    for disease, confidence in predicted_diseases:
        disease_table.add_row(disease, f"{confidence}%")

    # Update Middle and Right Panels to display tables side by side
    layout["middle"].update(symptom_table)
    layout["gap"].update("    ")
    layout["right"].update(disease_table)

    # Print the layout
    console.print(layout)


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


def predict_disease(model, symptom_list):
    # input  vector (1 for present symptoms, 0 otherwise)
    input_vector = {symptom: 1 if symptom in symptom_list else 0 for symptom in all_symptoms}
    input_df = pd.DataFrame([input_vector])

    # get probabilities for all classes
    probabilities = model.predict_proba(input_df)[0]

    class_labels = model.classes_
    disease_probabilities = list(zip(class_labels, probabilities))
    sorted_probabilities = sorted(disease_probabilities, key=lambda x: x[1], reverse=True)

    # normalize and round values
    total = sum(score for _, score in sorted_probabilities[:5])
    normalized_symptoms = [(symptom, round((score / total) * 100, 2)) for symptom, score in sorted_probabilities[:5]]

    # return top 3
    return normalized_symptoms


def generate_output(input_text, model):
    print()
    symptom_matches = symptom_matcher.extract_symptoms(input_text)

    if symptom_matches:
        # detected_symptoms
        detected_symptoms = [(obj.matched_symptom, obj.__repr__()) for obj in symptom_matches]

        # predicted diseases
        predicted_diseases = predict_disease(model, detected_symptoms)
        top_prediction = predicted_diseases[0][0]

        # top_symptoms
        most_frequent_symptom, highest_tfidf_symptom, combined_symptoms = get_top_symptoms_for_condition(
            top_prediction)
        top_symptoms = [most_frequent_symptom, highest_tfidf_symptom]
        top_symptoms.extend(combined_symptoms)

        # description
        try:
            description = condition_description_df.set_index('condition').loc[top_prediction, 'description']
        except KeyError:
            description = "No description found"

        display_diagnosis(detected_symptoms, predicted_diseases, top_symptoms, description)

    else:
        print("\nNo symptoms detected. Please expand on your description or take a different approach.")

def load_model(model_path='src/sar_project/models/best_random_forest_model.pkl'):
    return joblib.load(model_path)


rf_model = load_model()

generate_output("My ribs hurt and it hurts to breathe.", rf_model)