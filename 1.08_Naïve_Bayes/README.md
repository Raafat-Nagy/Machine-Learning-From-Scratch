### Naïve Bayes Classifier Overview

**Naïve Bayes Classifier** is a probabilistic model based on Bayes' Theorem. It is used for classification tasks, particularly when dealing with categorical data or text classification.

#### 1. **Bayes' Theorem**

Bayes' Theorem is used to update the probability estimate of a hypothesis based on new evidence. It is expressed as:

$$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$$

Where:
- $P(C|X)$ is the **posterior** probability of class $C$ given feature vector $X$.
- $P(X|C)$ is the **likelihood** of feature vector $X$ given class $C$.
- $P(C)$ is the **prior** probability of class $C$.
- $P(X)$ is the **marginal** probability of feature vector $X$.

#### 2. **Naïve Assumption**

The "naïve" aspect of the model assumes that all features are independent given the class label. This simplifies the computation of $P(X|C)$:

$$P(X|C) = P(x_1, x_2, \ldots, x_n | C) = \prod_{i=1}^{n} P(x_i | C)$$
<!-- $$P(X|C) = \prod_{i=1}^{n} P(x_i | C)$$ -->

Where $x_i$ represents the $i$-th feature.

#### 3. **Classification Process**

1. **Train the Model:**
   - **Estimate Prior Probabilities:** Calculate $P(C)$ for each class based on the frequency of classes in the training dataset.
   - **Estimate Likelihoods:** Compute $P(x_i | C)$ for each feature given the class. This depends on the type of feature (e.g., Gaussian distribution for continuous features, multinomial distribution for categorical features).

2. **Predict Class for New Data:**
   - For a new feature vector $X$, compute the posterior probability for each class using the formula:
  
    $$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$$

   - Since $P(X)$ is constant for all classes, it is often omitted when determining the class with the highest probability.

3. **Apply argmax:**
   - The classification is performed by finding the class $\hat{C}$ that maximizes the posterior probability:
     $$
     \hat{C} = \arg\max_{C} \left( P(X|C) \cdot P(C) \right)
    $$
   - Here, $\arg\max$ refers to the class that gives the highest value of $P(X|C) \cdot P(C)$, which translates to the highest posterior probability for $X$.

#### 4. **Types of Naïve Bayes Classifiers**

- **Gaussian Naïve Bayes:** Assumes that the features follow a Gaussian distribution.
- **Multinomial Naïve Bayes:** Used for features that are counts or frequencies (e.g., word counts in text classification).
- **Bernoulli Naïve Bayes:** Assumes binary/boolean features.