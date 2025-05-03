# Building and Optimizing a Machine Learning Pipeline in Azure

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Overview](#dataset-overview)
- [Methodology and Pipeline Design](#methodology-and-pipeline-design)
- [HyperDrive Pipeline](#hyperdrive-pipeline)
- [AutoML Pipeline](#automl-pipeline)
- [Feature Insights and Explanations](#feature-insights-and-explanations)
- [Performance Analysis](#performance-analysis)
- [Pipeline Comparison](#pipeline-comparison)
- [Future Improvements](#future-improvements)
- [Resource Cleanup Confirmation](#resource-cleanup-confirmation)

---

## Project Overview

This project, completed as part of the Udacity Nanodegree, focuses on developing and optimizing a machine learning pipeline using Azure Machine Learning. Two distinct approaches are implemented to build the pipeline:

1. **HyperDrive**: Utilizes the Azure Python SDK to train a Scikit-learn logistic regression model with hyperparameter tuning.
2. **AutoML**: Employs Azure's Automated Machine Learning to train and evaluate multiple models and select the best performer.

The performance of both pipelines is compared based on accuracy, with the goal of predicting whether a client will subscribe to a bank term deposit.

---

## Dataset Overview

### Dataset Description
The dataset pertains to direct marketing campaigns conducted by a Portuguese banking institution via phone calls. Multiple contacts with the same client were often necessary to determine if the client would subscribe to a term deposit (the target variable, `y`, with values `yes` or `no`). The objective is to develop a classification model to predict this subscription outcome.

The dataset is available at: [Bank Marketing Dataset](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv).

### Features
The dataset includes 20 input features and 1 target variable:

#### Client Information
- `age`: Age of the client (numeric).
- `job`: Occupation type (categorical: e.g., 'admin', 'blue-collar', 'student').
- `marital`: Marital status (categorical: 'divorced', 'married', 'single').
- `education`: Education level (categorical: e.g., 'basic.4y', 'university.degree').
- `default`: Has credit in default? (categorical: 'no', 'yes').
- `housing`: Has a housing loan? (categorical: 'no', 'yes').
- `loan`: Has a personal loan? (categorical: 'no', 'yes').

#### Campaign Details
- `contact`: Communication method (categorical: 'cellular', 'telephone').
- `month`: Last contact month (categorical: 'jan', 'feb', ..., 'dec').
- `day_of_week`: Last contact day (categorical: 'mon', 'tue', ..., 'fri').
- `duration`: Duration of the last contact in seconds (numeric; note: this feature is not realistic for predictive modeling as it is known only after the call).

#### Additional Attributes
- `campaign`: Number of contacts during this campaign (numeric).
- `pdays`: Days since last contact from a previous campaign (numeric; 999 indicates no prior contact).
- `previous`: Number of prior contacts before this campaign (numeric).
- `poutcome`: Outcome of the previous campaign (categorical: 'failure', 'nonexistent', 'success').

#### Socioeconomic Context
- `emp.var.rate`: Employment variation rate (numeric, quarterly).
- `cons.price.idx`: Consumer price index (numeric, monthly).
- `cons.conf.idx`: Consumer confidence index (numeric, monthly).
- `euribor3m`: Euribor 3-month rate (numeric, daily).
- `nr.employed`: Number of employees (numeric, quarterly).

#### Target Variable
- `y`: Subscription to a term deposit (binary: 'yes', 'no').

---

## Methodology and Pipeline Design

The project employs two pipelines to address the classification task:

1. **HyperDrive Pipeline**: Uses a Scikit-learn logistic regression model with hyperparameter tuning.
2. **AutoML Pipeline**: Leverages Azure AutoML to train and evaluate various models automatically.

The best-performing model, based on accuracy, is identified and analyzed.

### Best Model
The top-performing model was a `VotingEnsemble` model from the AutoML pipeline, achieving an accuracy of **91.72%**.

---

## HyperDrive Pipeline

### Pipeline Structure
The HyperDrive pipeline, implemented in the `udacity-project.ipynb` notebook, includes the following steps:

#### Training Script (`train.py`)
1. **Data Loading**: Imported the dataset using `TabularDatasetFactory` from the provided URL.
2. **Data Preprocessing**: Cleaned the data by dropping irrelevant columns, removing missing entries, one-hot encoding categorical features, and performing feature engineering.
3. **Data Splitting**: Divided the dataset into training and test sets.
4. **Model Training**: Trained a logistic regression model using hyperparameters specified by HyperDrive.
5. **Evaluation**: Computed the accuracy score on the test set.
6. **Model Saving**: Saved the best model as an output file.

#### Notebook Steps (`udacity-project.ipynb`)
1. **Compute Target**: Assigned a compute cluster (`Standard_D2_V2`) for training.
2. **Parameter Sampling**: Used `RandomParameterSampling` to define hyperparameter ranges:
   - `C`: Inverse of regularization strength.
   - `max_iter`: Maximum number of iterations.
3. **Early Stopping Policy**: Applied a `BanditPolicy` to terminate underperforming runs based on a predefined accuracy threshold.
4. **Estimator Creation**: Configured a `SKLearn` estimator for the logistic regression model.
5. **HyperDrive Configuration**: Set up the `HyperDriveConfig` with the estimator, sampler, and policy.
6. **Best Model Selection**: Retrieved the best run and its metrics.
7. **Model Persistence**: Saved the best model for future use.

### Benefits of Random Parameter Sampling
- Enables efficient exploration of the hyperparameter space by randomly sampling values for `C` and `max_iter`.
- Reduces the computational burden compared to exhaustive search methods like grid sampling.

### Benefits of Early Stopping
- The `BanditPolicy` halts training runs that do not meet the accuracy threshold, conserving computational resources.
- Enhances efficiency by focusing on promising hyperparameter configurations.

---

## AutoML Pipeline

### Configuration
The AutoML pipeline was configured with the following settings:
- `experiment_timeout_minutes`: 30 minutes to limit resource usage.
- `task`: 'classification', reflecting the binary prediction goal.
- `compute_target`: The `Standard_D2_V2` compute cluster with specified max nodes.
- `training_data`: The bank marketing dataset.
- `label_column_name`: `y`, the target variable.
- `n_cross_validations`: 3, for robust validation without user-specified validation data.
- `primary_metric`: 'accuracy', the optimization metric.
- `enable_early_stopping`: Enabled to terminate runs with no improvement.

### Execution
AutoML trained a variety of models, including LightGBM, XGBoost, Logistic Regression, and VotingEnsemble, in a no-code environment, allowing rapid experimentation.

### Results
The best model was a `VotingEnsemble`, achieving an accuracy of **91.72%**.

---

## Feature Insights and Explanations

AutoML provided insights into feature importance and data exploration:
- Highlighted key features influencing predictions, such as `duration`, `poutcome`, and socioeconomic indicators.
- Offered visualizations of data distributions and feature correlations, aiding in understanding the dataset’s characteristics.

---

## Performance Analysis

The AutoML pipeline generated detailed performance metrics for the `VotingEnsemble` model, including accuracy, precision, recall, and other classification metrics, providing a comprehensive evaluation of its effectiveness.

---

## Pipeline Comparison

### Accuracy Comparison
- **HyperDrive**: Achieved an accuracy of **90.97%**.
- **AutoML**: Achieved an accuracy of **91.72%**.

### Analysis
- **Performance Difference**: AutoML outperformed HyperDrive by 0.75%, a modest but notable improvement.
- **Efficiency and Flexibility**: AutoML requires minimal coding and enables training of diverse models (e.g., RandomForests, BoostedTrees) efficiently, whereas HyperDrive involves more manual configuration but offers granular control over the training process.
- **Scalability**: AutoML’s architecture supports rapid experimentation with multiple algorithms, making it more scalable for complex tasks.

---

## Future Improvements

To enhance the project, the following strategies could be explored:
1. **Addressing Data Imbalance**: Implement techniques like oversampling or undersampling to handle imbalanced classes in the dataset.
2. **Alternative Parameter Sampling**: Use a Bayesian Parameter Sampler in HyperDrive to potentially improve hyperparameter optimization.
3. **Advanced Models**: Experiment with neural networks or tree-based models (e.g., decision trees, random forests) in HyperDrive to boost performance.

---

## Resource Cleanup Confirmation

The compute cluster was properly terminated after the experiments to avoid unnecessary costs, as confirmed by the cleanup proof provided in the project documentation.