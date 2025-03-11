import joblib
from src.sar_project.agents import output_generator


def load_model(model_path='src/sar_project/models/best_random_forest_model.pkl'):
    return joblib.load(model_path)


rf_model = load_model()
print("Type a symptom description (or 'exit' to quit):")
while True:
    input_text = input("\n> ")

    if input_text.lower() == "exit":
        print("Exiting...")
        break

    output_generator.generate_output(input_text, rf_model)

    #exit(0)
