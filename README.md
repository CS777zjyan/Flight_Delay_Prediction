# Flight Delay Prediction Model

This project aims to predict flight delays using a logistic regression model. The model is trained on a preprocessed dataset that includes various features related to flight details.

## Project Structure

### T1: Data Preprocessing

- `T1.py`: The script for data preprocessing, which loads raw flight data, processes it, and outputs a preprocessed CSV file.
- `Raw_Flight_Data.csv`: The raw dataset containing flight information.
- `Project_combinedflight.csv`: The output preprocessed dataset used for training and testing the model in T2.

### T2: Model Training and Evaluation

- `T2.py`: The main script for training the model, making predictions, and evaluating the model performance.
- `Project_combinedflight.csv`: The preprocessed dataset used for training and testing the model.
- `evaluation_results.txt`: The output file containing the evaluation results of the model.

## Dependencies

The project relies on the following dependencies:
- Python 3.x
- Apache Spark 3.x
- pandas

### Running the Project

- T1: Data Preprocessing
To run the data preprocessing script, follow these steps:

Prepare the Raw Dataset:
Ensure that the raw dataset Raw_Flight_Data.csv is in the correct path.

Run the Script:
Use the following command to run the script:



