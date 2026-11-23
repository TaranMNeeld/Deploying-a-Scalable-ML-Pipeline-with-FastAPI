# Model Card

## Model Details
This model is a Random Forest classifier using scikit-learn. It uses default hyperparameters with random_state = 50. The model predicts whether an individual's annual income is greater than $50,000.

## Intended Use
This model is intended for educational purposes. The census.csv data is used to classify individuals into two income categories (<=50K and >50K). This model should not be used for real-world decisions.

## Training Data
The model was trained on the UC Irvine Adult Census Income dataset (https://archive.ics.uci.edu/dataset/20/census+income). A 80/20 train-test split was performed using random_state = 50. Categorical features were one-hot encoded and salaries were converted into binary values.

## Evaluation Data
The model was evaluated on the 20% of test data excluded from the training data. The test data was transformed using the categorical encoder fitted on the training data to ensure consistent preprocessing. Performance was also evaluated for each unique values of every categorical feature. These metrics are in slice_output.txt.

## Metrics
The model was evaluated using precision, recall, and F1 score.
- Precision: 0.7369
- Recall: 0.6288
- F1 Score: 0.6786

Additional performance metrics for individual categorical slices are in slice_output.txt.

## Ethical Considerations
The dataset contains sensitive demographic data. The data may reflect historical income inequalities. The model should not be used to make consequential decisions about individuals.

## Caveats and Recommendations
The model was trained on historic census data and may not accurately generalize to current economic conditions. Further validation, bias analysis, and testing on more recent data are recommended before considering real-world applications.
