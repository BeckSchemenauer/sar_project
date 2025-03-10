import pandas as pd
import requests
from tqdm import tqdm

# Function to search Wikipedia and get the best matching page
def search_wikipedia(disease):
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": disease
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        search_results = response.json().get("query", {}).get("search", [])
        if search_results:
            return search_results[0]["title"]  # Return best match
    return None


# Function to get disease summary from Wikipedia
def get_wikipedia_summary(disease):
    page_title = search_wikipedia(disease) or disease.replace(' ', '_')  # Use best match if available
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"

    response = requests.get(summary_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("extract", "No summary available.")
    else:
        return None  # Mark as failed


# Load the CSV file
df = pd.read_csv("src/sar_project/knowledge/Final_Augmented_dataset_Diseases_and_Symptoms.csv")

# Get unique disease names
unique_diseases = df["diseases"].dropna().unique()

# Create a list to store results
results = []
failed_diseases = []

# Loop through diseases and fetch descriptions
for disease in tqdm(unique_diseases, desc="Processing Diseases"):
    summary = get_wikipedia_summary(disease)
    if summary:
        results.append({"condition": disease, "description": summary})
    else:
        failed_diseases.append(disease)

# Convert results to a DataFrame and save as CSV
results_df = pd.DataFrame(results)
results_df.to_csv("src/sar_project/knowledge/disease_descriptions_v2.csv", index=False)

# Print failed diseases
print("Diseases that failed to get a Wikipedia summary:")
print(failed_diseases)
