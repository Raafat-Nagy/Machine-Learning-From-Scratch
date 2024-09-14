# One Variable Linear Regression

## Hypothesis Function
$$h_\theta(x) = \theta_0 + \theta_1 x$$

**Parameters:**
- $h_\theta(x)$: The predicted value or hypothesis given the input feature $x$.
- $\theta_0$: The y-intercept or bias term of the linear model. It represents the value of the prediction when $x = 0$.
- $\theta_1$: The slope of the line, indicating how much the predicted value changes with a unit change in $x$.
- $x$: The input feature for which the prediction is being made.

## Error Definition
The error is the difference between the predicted values and the actual target values:

$$\text{Error} = h_\theta(x) - y$$

## Cost Function (Mean Squared Error)

### Standard Form
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

### Vectorized Forms
Alternatively, the cost function can be expressed using vector notation for the error:

$$J(\theta) = \frac{\| \text{Error} \|_2^2}{2m}$$

$$J(\theta) = \frac{\text{Error} \cdot \text{Error}}{2m}$$

**Parameters:**
- $J(\theta)$: The cost function, representing the average squared error between the predicted values and the actual target values.
- $m$: The number of training examples in the dataset.
- $\text{Error}$: The vector of errors, defined as the difference between the predicted values and the actual target values.
- $h_\theta(x^{(i)})$: The predicted value for the $i$-th training example.
- $y^{(i)}$: The actual target value for the $i$-th training example.
- $x^{(i)}$: The input feature for the $i$-th training example.

## Gradient Calculation

**Gradient for $\theta_0$:**
$$\frac{\partial J(\theta)}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)$$

**Gradient for $\theta_1$:**
$$\frac{\partial J(\theta)}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}$$

**Parameters:**
- $\frac{\partial J(\theta)}{\partial \theta_0}$: The partial derivative of the cost function with respect to $\theta_0$, indicating how the cost function changes with a change in $\theta_0$.
- $\frac{\partial J(\theta)}{\partial \theta_1}$: The partial derivative of the cost function with respect to $\theta_1$, indicating how the cost function changes with a change in $\theta_1$.

## Gradient Descent Update Rules
$$\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)$$

$$\theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)} \right)$$

Alternatively, the update rules can be expressed as:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\boldsymbol{\theta})$$

**Parameters:**
- $\theta_0$: The y-intercept or bias term of the linear model.
- $\theta_1$: The slope of the line.
- $\alpha$: The learning rate, a hyperparameter that controls the step size of the parameter updates. A smaller value of $\alpha$ results in smaller updates and vice versa.
- $m$: The number of training examples in the dataset.
- $h_\theta(x^{(i)})$: The predicted value for the $i$-th training example.
- $y^{(i)}$: The actual target value for the $i$-th training example.
- $x^{(i)}$: The input feature for the $i$-th training example.

## Additional Explanation

- **Hypothesis Function**: Represents the model's prediction for a given input $x$. The linear relationship is characterized by $\theta_0$ and $\theta_1$.
- **Error**: The difference between the predicted value and the actual target value. This is the quantity we aim to minimize in our optimization process.
- **Cost Function**: Measures how well the model's predictions match the actual target values. The goal is to minimize this cost function to improve the model's accuracy.
- **Gradient Descent**: An optimization algorithm used to minimize the cost function by iteratively updating the model parameters $\theta_0$ and $\theta_1$ in the direction of the negative gradient of the cost function.
- **Gradients**: The gradients ($\frac{\partial J(\theta)}{\partial \theta_0}$ and $\frac{\partial J(\theta)}{\partial \theta_1}$) represent the direction and rate of change of the cost function with respect to the model parameters. These gradients are used to adjust the parameters during optimization.

---