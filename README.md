# Transaction-Fraudulent-Detection
The dataset consist of credit card transactions made over a 2-day period by European cardholders. It contains 284,807 transactions, each 
transaction has 30 features, which are all numerical. The features V1, V2, ..., V28 are the result of a PCA transformation. The objective 
is to perform data cleaning, data featuring, data visualization and build preditive models to detect whether a credit card transaction is 
fraudulent or not.

# Predictive Model
Split the data into train, validation and test sets with train and validation sets size of 20% each.
Performed Hyperparameter tuning method either to increase the predictive power of the model or to make it easier to train the model.
Implemented grid search with cross validation technique on validation set to attain best parameters then train the models with these parameters and evaluate its performance on the test set.

•	Trained a logistic regression to identify whether a new transaction is fraudulent or not and attain precision up to 75%

•	Reduce error rate by 30% and improved Matthew’s Correlation Coefficient by 18% using Decision Tree and Random Forest


