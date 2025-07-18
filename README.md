# Engie's Data Science Summer Learning 2025

Explainable Artificial Intelligence (XAI) refers to a set of techniques and methods designed to make the predictions and inner workings of AI models more transparent and understandable to humans. Its main goals are to enhance model interpretability, support debugging and validation, and ultimately foster trust in AI systems by helping users understand how and why decisions are made.​

However, XAI is only one component of the broader field of Responsible AI, which encompasses not only technical transparency but also ethical, social, legal, and environmental considerations. ​
In this training, we aim to position XAI within this wider framework — showing how explainability contributes to the development of AI systems that are not only functional but also fair, trustworthy, and sustainable.

​Key concepts to explore:
1. Explainability (XAI) — Making model predictions understandable to humans.​
2. Interpretability (XAI) — Ensuring that users can grasp the cause behind a model’s decisions.​
3. Fairness & Bias Assessment (Responsible AI) — Evaluating and mitigating biases to ensure equitable outcomes.​
4. Model Transparency (XAI / Responsible AI) — Openness about how models work and what data they use.​
5. Trustworthiness (Responsible AI) — Ensuring models are reliable, ethical, and aligned with expectations.​
6. Causal Inference (Responsible AI) — Understanding cause-and-effect relationships beyond correlation.​
7. Human-Centered Design (Responsible AI) — Building AI with a focus on user needs and usability.​
8. Model Debugging & Diagnostics (XAI) — Identifying errors or unexpected behavior within models.​
9. Carbon Footprint Evaluation (Responsible AI) — Measuring and reducing the environmental impact of AI.​
10. Human-AI Collaboration (Responsible AI) — Designing systems that enhance and complement human decision-making.​


## Challenge
The goal is to use xAI on 2 use cases:
* __UC1 : Predicting Income Levels Using Census Data__: Binary classification.  
* __UC2: Predicting Customer Churn Using Telecom Data__: build a model that flags customers likely to leave (the “Churn” target) and explains why each prediction was made. 

### UC1: Predicting Income Levels Using Census Data
In many socio-economic studies, accurately predicting an individual's income level based on demographic and employment attributes is crucial for understanding labor market trends, economic inequality, and policy-making. ​ The Adult dataset (also known as the Census Income dataset), derived from the 1994 U.S. Census, provides a set of attributes such as age, education, occupation, and hours worked per week, which can be used to classify whether an individual's income exceeds $50K per year.​

#### Challenges
* Imbalanced Data​
* Handling Missing Values​
* Bias and Fairness​
* Feature Engineering​

#### Expected outcome
* Train machine learning models and evaluate them using traditional metrics like ​accuracy, precision, recall, F2-score, and AUC-ROC but also XAI and carbon footprint ​methods. ​
* While fairness metrics won’t be considered in this session, fairness and explainability ​remain key to model transparency.​


### UC2: Predicting Customer Churn Using Telecom Data
In subscription-based telecom businesses, churn directly erodes recurring revenue; therefore, being able to anticipate which customers are at risk and understanding the drivers behind that risk is vital for retention teams. Using the public Telco-Customer-Churn dataset, which combines behavioural variables such as tenure, MonthlyCharges, and TotalCharges with service-usage indicators like Internet, Phone, and Contract type, we aim to train a transparent model that can both predict churn and provide actionable insights. ​

#### Challenges
* Imbalanced Data​
* Handling Missing Values​
* High cardinality categoricals ​
* Model interpretability​

#### Expected outcome
* Trained MLP and Random Forest models that score churn risk. ​
* Evaluation with ROC curves, confusion matrices, precision/recall. ​
* Explainability assets: permutation importance, SHAP (global ⬆ & local waterfall ⬇), PDPs, gauge charts. ​
* Actionable insights for retention campaigns and model transparency.​

## Notebook plan
1. Exploratory Data Analysis (EDA)​
    * Data distribution, missing values, outliers​
    * Income distribution across sensitive attributes (gender, race)​
    * Initial bias assessment​

2. Data Preprocessing​
    * Handle missing values​
    * Encode categorical variables​
    * Normalize/scale numerical features​
    * Address class imbalance​

3. Feature engineering
    * Select and transform meaningful features​
    * Remove proxy variables contributing to bias​
    * Create fairness-aware engineered features​

4. XAI Packages​
5. Carbon Footprint Monitoring Packages
6. Data Modeling​
    * Model training with carbon footprint monitoring​

7. Evaluation & Results Analysis​
    * Traditional metrics (Accuracy, F1-score, AUC-ROC)​

8. XAI (Explainable AI)​
    * Global explanation (Feature importance, Permutation importance, SHAP, LIME, SHAPASH)​
    * Local explanation (LIME, SHAP, SHAPASH)

## Model-agnostic methods
These methods can be applied to any machine learning model, regardless of its structure or type. They focus on analyzing the features’ input-output pair. This section will introduce and discuss LIME and SHAP, two widely-used surrogate models.

### SHAP
It stands for SHapley Additive exPlanations. This method aims to explain the prediction of an instance/observation by computing the contribution of each feature to the prediction.

```python
import shap
import matplotlib.pyplot as plt

# load JS visualization code to notebook
shap.initjs()

# Create the explainer
explainer = shap.TreeExplainer(rf_clf)

shap_values = explainer.shap_values(X_test)
```

SHAP offers an array of visualization tools for enhancing model interpretability, and the next section will discuss two of them: (1) variable importance with the summary plot, (2) summary plot of a specific target, and (3) dependence plot.

#### Variable Importance with Summary Plot
In this plot, features are ranked by their average SHAP values showing the most important features at the top and the least important ones at the bottom using the summary_plot() function. This helps to understand the impact of each feature on the model’s predictions.

```python
print("Variable Importance Plot - Global Interpretation")
figure = plt.figure()
shap.summary_plot(shap_values, X_test)
```

#### Summary Plot on a Specific Label
Using this approach can provide a more granular overview of the impact of each feature on a specific outcome (label).

In the example below, shap_values[1] is used to represent the SHAP values for instances classified as label 1 (having diabetes).

```python
shap.summary_plot(shap_values[1], X_test)
```

### Partial Dependence Plot
Unlike summary plots, dependence plots show the relationship between a specific feature and the predicted outcome for each instance within the data. This analysis is performed for multiple reasons and is not limited to gaining more granular information and validating the importance of the feature being analyzed by confirming or challenging the findings from the summary plots or other global feature importance measures. 

### LIME
Local Interpretable Model-agnostic Explanations (LIME for short). Instead of providing a global understanding of the model on the entire dataset, LIME focuses on explaining the model’s prediction for individual instances.

LIME explainer can be set up using two main steps: (1) import the lime module, and (2) fit the explainer using the training data and the targets. During this phase, the mode is set to classification, which corresponds to the task being performed.

```python
# Import the LimeTabularExplainer module
from lime.lime_tabular import LimeTabularExplainer

# Get the class names
class_names = ['Has diabetes', 'No diabetes']

# Get the feature names
feature_names = list(X_train.columns)

# Fit the Explainer on the training data set using the LimeTabularExplainer
explainer = LimeTabularExplainer(X_train.values, feature_names =     
                                 feature_names,
                                 class_names = class_names, 
                                 mode = 'classification')
```

The code snippet below generates and displays a LIME explanation for the 8th instance in the test data using the random forest classifier and presenting the final feature contribution in a tabular format. 

## Model-specific methods
As opposed to model-agnostic methods, these methods can only be applied to a limited category of models. Some of those models include linear regression, decision trees, and neural network interpretability. Different technics such as DeepLIFT, Grad-CAM, or Integrated Gradients can be leveraged to explain deep-learning models.

When using a decision-tree model, a graphical tree can be generated with the plot_tree function from scikit-learn to explain the decision-making process of the model from top-to-bottom.

# Links
* SHAP
    * Documentation: [Welcome to the SHAP documentation — SHAP latest documentation](https://shap.readthedocs.io/en/latest/index.html)
    * GitHub repo: [shap/shap: A game theoretic approach to explain the output of any machine learning model](https://github.com/shap/shap)
* LIME
  * Documentation: [Local Interpretable Model-Agnostic Explanations documentation](https://lime-ml.readthedocs.io/en/latest/)
  * GitHub repo: [marcotcr/lime: Lime: Explaining the predictions of any machine learning classifier](https://github.com/marcotcr/lime)

