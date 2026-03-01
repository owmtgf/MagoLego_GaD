# Problem Formulation and Description

## Decision-Making Problem

In practical machine learning workflows, selecting an appropriate classification algorithm represents a complex decision-making problem involving multiple competing objectives. Different models may achieve similar predictive accuracy while significantly differing in computational efficiency, stability, interpretability, and deployment cost. Consequently, model selection cannot be reduced to optimizing a single performance metric.

The objective of this study is to select the **most suitable classification model** using a multi-criteria decision-making framework. The decision problem consists of choosing the best alternative from a set of trained classification models evaluated according to several quantitative criteria derived from empirical experiments.

This problem reflects a realistic professional scenario faced by data scientists and deep learning engineers when preparing models for deployment. In real systems, models must satisfy not only predictive performance requirements but also operational constraints such as latency, scalability, and maintainability.

Therefore, the decision task is formulated as:

> **Selecting the optimal classification model that provides the best overall trade-off between predictive quality, computational efficiency, robustness, and interpretability.**

---

## Importance of Solving the Problem

In industrial and research environments, relying solely on accuracy or F1-score may lead to suboptimal decisions. For example:
- highly accurate models may require excessive computational resources
- complex ensemble models may introduce unacceptable inference latency
- black-box models may violate interpretability requirements in regulated domains
- unstable models may produce inconsistent performance across datasets

Modern machine learning systems must balance multiple objectives simultaneously. Multi-criteria decision-making methods provide a structured mathematical framework allowing objective comparison between alternatives and transparent justification of the final choice.

Applying social choice theory and Pareto optimality enables transforming model selection into a formal decision process rather than an intuitive or subjective preference.

Thus, solving this problem demonstrates how decision theory principles can enhance real-world machine learning engineering practices.

---

## Dataset and Data Source

The experimental evaluation is performed using the *Mushroom Classification Dataset*, publicly available through the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/73/mushroom) and distributed via [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification).

The dataset contains categorical descriptions of mushroom specimens with the task of predicting whether a mushroom is edible or poisonous.

Although the dataset domain concerns biological classification, its thematic content is **not central to this study**. The dataset serves as a representative binary classification problem analogous to applications such as:
- disease diagnosis
- fraud detection
- anomaly detection
- safety risk assessment

The primary purpose of the dataset in this work is to provide real empirical performance measurements for multiple classification algorithms under identical experimental conditions.

The dataset was selected because it:
- is publicly accessible and reproducible
- contains real observational data
- allows fair benchmarking across many algorithms
- enables consistent comparison using multiple evaluation metrics

---

## Alternatives (Evaluated Models)

Each alternative corresponds to a trained classification model evaluated using identical preprocessing and validation procedures. The experimental setup includes models from different methodological families to ensure diversity of approaches.

The evaluated alternatives include:

**Tree-Based Models**
- DecisionTree_default
- DecisionTree_shallow
- DecisionTree_pruned

**Ensemble Methods**
- RandomForest_small
- RandomForest_large
- ExtraTrees_fast
- ExtraTrees_large
- GradientBoosting_fast
- GradientBoosting_slow
- AdaBoost_light
- AdaBoost_heavy

**Kernel and Distance-Based Methods**
- SVC_linear
- SVC_rbf
- SVC_poly
- KNN_3
- KNN_15

**Linear Models**
- LogReg_weak_reg
- LogReg_strong_reg
- Ridge_alpha_small
- Ridge_alpha_large
- LDA_svd
- LDA_lsqr

**Probabilistic Models**
- GaussianNB_default
- GaussianNB_smoothed

> *You can check methods' extended details in `./decision_metrics_collection/alternatives.py` in `MODELS` variable and refer to [scikit-learn documentation](https://scikit-learn.org/stable/user_guide.html).* 

In total, more than 10 alternatives are considered, satisfying the assignment requirement while ensuring meaningful methodological diversity.

---

## Evaluation Criteria

The alternatives are assessed using experimentally measured metrics reflecting multiple aspects of model performance and practical usability.


### Predictive Performance Criteria

To evaluate classification quality, several standard performance metrics derived from the confusion matrix are used. These metrics quantify different aspects of predictive behavior and together provide a comprehensive assessment of model performance.

Let a binary classification problem consist of observations belonging to two classes: positive and negative. After prediction, outcomes can be summarized using a confusion matrix:

| | Predicted Positive | Predicted Negative |
|-|-|-|
| Actual Positive | True Positive ( $TP$ ) | False Negative ( $FN$ ) |
| Actual Negative | False Positive ( $FP$ ) | True Negative ( $TN$ ) |

1. **Accuracy**

    Accuracy measures the overall proportion of correctly classified observations.

    $\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$

2. **Precision**

    Precision evaluates the reliability of positive predictions, i.e., how many predicted positives are actually correct. High precision indicates a low number of false positive errors.

    $\mathrm{Precision} = \frac{TP}{TP + FP}$

3. **Recall**

    Indicates the model’s ability to detect positive instances. A high recall value indicates that few positive samples are missed.

    $\mathrm{Recall} = \frac{TP}{TP + FN}$

4. **F1-score**

    The F1-score combines precision and recall using their harmonic mean, providing a balanced evaluation when both false positives and false negatives are important. The harmonic mean penalizes extreme imbalance between precision and recall.

    $\mathrm{F_1} = 2 \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}$

5. **ROC-AUC**

    The Receiver Operating Characteristic (ROC) curve represents the relationship between:
    - True Positive Rate ( $TPR$ )
    - False Positive Rate ( $FPR$ )

    where

    $TPR = \frac{TP}{TP + FN}$ **($\mathrm{Recall}$)* \
    $FPR = \frac{FP}{FP + TN}$

    The ROC-AUC metric corresponds to the area under this curve (AUC):

    $\mathrm{ROC-AUC} = \int^1_0 TPR(FPR) d(FPR)$

    ROC-AUC measures the probability that a randomly chosen positive instance receives a higher predicted score than a randomly chosen negative instance.

6. **Cross-validation Mean Score** (cv_mean)

    To estimate model generalization ability, $k$-fold cross-validation is performed. The dataset is partitioned into
    $k$ disjoint subsets. Each subset is used once as validation data while the remaining folds are used for training.

    Let $s_i$ denote the evaluation score obtained in fold $i$, where $i = 1, 2, \dots, k$. 

    The cross-validation mean score is defined as: \
    $cv_{mean} = \frac{1}{k} \sum^{k}_{i=1} s_i$

    This metric approximates expected performance on unseen data.

7. **Cross-validation Standard Deviation** (cv_std)

    Model stability is evaluated using the variability of cross-validation scores: \
    $cv_{std} = \sqrt{\frac{1}{k} \sum^{k}_{i=1} (s_i - cv_{mean})^2}$

    Lower values indicate more stable and consistent performance across different data splits.


### Computational Efficiency Criteria

1. **Training Time** (train_time_sec)

    Training time ($T_{train}$) represents the total computational time required to fit a model using the training dataset. Training time is an important factor during model development and hyperparameter optimization, especially when repeated retraining is required.

    Since shorter training time is preferable, this criterion is naturally a minimization criterion.

2. **Inference Time** (inference_time_sec)

    Inference time ($T_{inf}$) measures the time required for a trained model to generate predictions. Denotes the total prediction time for $N$ samples.

    Lower inference time is essential for real-time and large-scale applications. This criterion is also minimized.

3. **Latency** 

    Latency represents the response delay between receiving an input and producing a prediction output (inference time per sample). It reflects operational responsiveness in deployment environments. 

    $T_{lat} = \frac{T_{inf}}{N}$

    Lower latency indicates faster response capability. 

4. **Throughput**

    Throughput quantifies the processing capacity of a model, defined as the number of predictions that can be generated per unit time.

    $\mathrm{Throughput} = \frac{N}{T_{inf}}$ or $\mathrm{Throughput} = \frac{1}{T_{lat}}$

    Unlike latency, throughput is a maximization criterion.

5. **Model Size** (model_size_kb)

   Model size represents the memory footprint required to store the trained model.

    Model size influences:
    - deployment feasibility
    - memory consumption
    - model transfer cost
    - edge-device compatibility

    Smaller models are preferred; therefore, this criterion is minimized.


### Model Complexity and Usability Criteria

**Concept of Interpretability:**

In addition to predictive performance and computational efficiency, modern machine learning systems must often satisfy requirements related to transparency and explainability. Interpretability refers to the degree to which a human can understand the internal logic of a model and explain how input variables influence predictions.

Formally, interpretability can be described as the extent to which a model allows a human observer to establish a mapping:

$f : X \to Y$

between input features $X$ and predicted outcomes $Y$ in a comprehensible and explainable manner.

Unlike accuracy or execution time, interpretability is not directly measurable through physical observation. Instead, it represents a qualitative property reflecting human cognitive accessibility of model behavior.

---

**Importance of Interpretability in Practical Applications:** 

Interpretability plays a critical role in many real-world domains, including:
- medical decision support systems
- financial risk assessment
- legal and regulatory environments
- safety-critical autonomous systems

In such contexts, stakeholders must be able to:
- justify decisions
- detect model biases
- verify logical consistency
- build trust in automated systems

Highly complex models may achieve strong predictive performance but can behave as *black boxes*, limiting their practical usability despite superior accuracy.

---

**Expert-Based Interpretability Scale:**

To incorporate interpretability into the decision model, an ordinal scoring system is introduced based on widely accepted machine learning interpretability literature (e.g. [Interpretable Machine Learning in Healthcare](https://www.researchgate.net/publication/328416903_Interpretable_Machine_Learning_in_Healthcare)) and practitioner consensus.

Each model is assigned an interpretability score:

$I \in \{1, 2, 3, 4, 5\}$

where higher values indicate greater interpretability.

| Score | Interpretability Level | Description |
|-|-|-|
| 5 | Very High | Model decisions easily understandable; transparent structure |
| 4 | High | Mostly interpretable with moderate complexity |
| 3 | Medium | Partial interpretability due to aggregated decision mechanisms |
| 2 | Low | Complex nonlinear relations difficult to explain |
| 1 | Very Low | Black-box behavior |

---

**Assignment Principles:**

Scores are assigned according to structural properties of model classes:

- Highly Interpretable Models (Score = 5)
    - Logistic Regression
    - Linear Models
    - Simple Decision Trees

    Characteristics:
    - explicit coefficients
    - direct feature influence interpretation
    - rule-based reasoning

- Moderately Interpretable Models (Scores = 3-4)
    - Random Forest
    - Extra Trees
    - Gradient Boosting
    - Linear Discriminant Analysis

    Characteristics:
    - aggregate decision mechanisms
    - feature importance available but global logic less transparent

- Low Interpretability Models (Scores = 1-2)
    - Support Vector Machines with nonlinear kernels
    - k-Nearest Neighbors
    - Neural networks
    - Complex ensemble methods

    Characteristics:
    - nonlinear decision boundaries
    - lack of explicit reasoning structure

> *You can find the assigned interpretability scores in `./decision_data_collection/alternatives.py` as `INTERPRETABILITY` variable.*

---

**Role in the Decision Framework:** \
Within the overall evaluation process, interpretability:
- complements performance and efficiency metrics
- introduces human-centered considerations
- influences social choice aggregation rules

---

## Direction of Optimization

Not all criteria share the same optimization direction, as you could saw from above.

### Maximization Criteria
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Cross-validation mean
- Throughput
- Efficiency
- Interpretability

### Minimization Criteria
- Training time
- Inference time
- Model size
- Latency
- Cross-validation standard deviation

To enable unified comparison, minimization criteria will be transformed into maximization criteria during preprocessing.

---

## Characteristics of the Experimental Results

The obtained experimental results reveal an important real-world phenomenon: several high-capacity models achieve nearly perfect predictive performance on the dataset. Consequently, predictive metrics alone cannot differentiate between alternatives.

This situation highlights the necessity of multi-criteria decision analysis, where secondary characteristics such as computational efficiency, robustness, and interpretability become decisive factors.

Therefore, this dataset provides an ideal case study demonstrating why Pareto optimality and social choice methods are required in modern machine learning model selection.

---

## Expected Outcome

The goal is not simply identifying the most accurate classifier but determining the model that achieves the best overall balance across all evaluation dimensions.

The decision process will therefore include:
- transformation and normalization of criteria
- construction of the Pareto frontier
- aggregation using weighted decision rules
- application of social choice mechanisms

This structured methodology ensures that the final decision is transparent, reproducible, and theoretically justified.

Based on preliminary empirical observations, it is hypothesized that `DecisionTree_default` will emerge as the overall preferred model.

This expectation is motivated by the following considerations:
- **Strong predictive performance** \
    The model achieves competitive accuracy and F1-score on the evaluation dataset.

- **Moderate computational cost** \
    Training and inference times remain relatively low compared to more complex ensemble or kernel-based models.

- **High interpretability** \
    Decision trees provide explicit rule-based reasoning, enabling direct inspection of classification logic, which is particularly valuable in domains analogous to medical or risk classification tasks.

- **Balanced trade-off** \
    While some models may slightly outperform it in isolated metrics, the decision tree is expected to achieve the most favorable compromise across all evaluation dimensions.
