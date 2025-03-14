# Search and Rescue (SAR) Agent - CSC 581

*Health Assessor Agent*

Symptom extraction and disease prediction. This agent takes text (natural language) as input, and uses NLP techniques to match symptoms to those represented in a dataset. A random forest model then predicts a disease / condition on the detected symptoms.

Diseases and Symptoms Dataset: Compiled by Dhivyesh R.K. Available at [https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset]. Data source: The World Bank.

Steps to run:
1. Install requirements with "pip install -r requirements.txt"

2. Unzip:
    
    *best_random_forest_model.zip*
    
    *Final_Augmented_dataset_Diseases_and_Symptoms.zip*

3. From root, run with "python src/sar_project/agents/health_diagnosis_agent.py"
