
# UniBioPAN: Bioactive Peptide Analysis and Prediction Tool

UniBioPAN is a versatile tool for the analysis and prediction of bioactive peptide activity. It provides a user-friendly interface and flexible options for both web-based and local execution.

## Web Server
The UniBioPAN web server allows for easy prediction of bioactive peptides:
1. Sequence Input: Enter FASTA-formatted peptide sequences (up to 10).
2. Threshold Selection: Adjust the threshold to control the prediction stringency.
3. Activity Selection: Choose specific bioactivities to predict or "all activity" for a comprehensive analysis.
4. Task Submission: Submit the task and wait for results to appear.

## Google Colab Integration
For users with limited resources or those who prefer a cloud-based solution, UniBioPAN integrates seamlessly with Google Colab.

1. Download Files: Obtain the necessary files from the GitHub repository.
2. Environment Setup: Execute provided code to install the required environment within Colab.
3. Parameter Configuration: Adjust parameters as needed and train your model.
4. Prediction: Use the best-performing model to predict bioactivity on your input data.

## Python Script (UniBioPANscript)
For maximum flexibility and local execution, UniBioPANscript offers a powerful Python interface.

1. Environment Setup: Create a conda environment using the provided environment.yml file.
2. Download Script: Obtain the UniBioPAN script files from GitHub.
3. Command-Line Usage:
Use python unibiopan.py -h to display the help information.

Train a model:
   > python unibiopan.py -t -tf Dataset/train.xlsx -ef Dataset/test.xlsx
Make predictions:
   > python unibiopan.py -p -lp model/best_model.h5 -pf predict/input.xlsx

## Input/Output Formats
1. Prediction Input: FASTA, CSV, or XLSX
2. Training/Evaluation Input: XLSX (with 0 for positive and 1 for negative samples)
3. Prediction Output: Table format, with 1 for predicted bioactive peptides, 0 for predicted inactive peptides, and 'x' for sequences that exceed the model's length limit or contain non-standard amino acids.

## Important Note
&emsp&emspUniBioPAN predictions are based on computational models. Experimental validation is crucial to confirm the bioactivity of any peptide.
