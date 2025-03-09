# Search and Rescue (SAR) Agent - CSC 581

*Health Assessor Agent*

Symptom extraction and disease prediction. This agent takes text (natural language) as input, and uses NLP techniques to match symptoms to those represented in a dataset. A random forest model then predicts a disease / condition on the detected symptoms.

Steps to run:

1. Install requirements with "pip install -r ./requirements.txt"

   You may get an error saying, "Package requirement 'scikit_learn == 1.4.2' is not satisfied," the code should run fine anyway and this is safe to ignore.

2. Unzip:
    
    *best_random_forest_model.zip*
    
    *Final_Augmented_dataset_Diseases_and_Symptoms.zip*

3. From root, run with "python src/sar_project/agents/health_diagnosis_agent.py"