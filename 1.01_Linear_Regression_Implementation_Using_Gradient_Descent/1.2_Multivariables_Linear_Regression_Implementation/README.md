# Multivariables Linear Regression

## Hypothesis Function
The hypothesis function for multiple linear regression is defined as:

$$h_\theta(\mathbf{x}) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n$$

Alternatively, in matrix notation, it can be represented as:

$$h_\theta(\mathbf{x}) = X \theta^T$$

**Parameters:**
- $h_\theta(\mathbf{x})$: The predicted value or hypothesis given the input features $\mathbf{x} = [x_1, x_2, \ldots, x_n]$.
- $\theta_0$: The y-intercept or bias term of the linear model. It represents the value of the prediction when all $x_i = 0$.
- $\theta_i$: The coefficient associated with the $i$-th feature $x_i$, indicating how much the predicted value changes with a unit change in $x_i$.
- $x_i$: The $i$-th input feature for which the prediction is being made.

## Error Definition
The error is the difference between the predicted values and the actual target values:

$$\text{Error} = h_\theta(\mathbf{x}) - y$$

## Cost Function (Mean Squared Error)
The cost function used to evaluate the accuracy of the hypothesis is the Mean Squared Error (MSE):

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(\mathbf{x}^{(i)}) - y^{(i)} \right)^2$$

Alternatively, it can be represented using vector notation:

$$J(\theta) = \frac{\| \text{Error} \|_2^2}{2m}$$

$$J(\theta) = \frac{\text{Error}^T \cdot \text{Error}}{2m}$$

**Parameters:**
- $J(\theta)$: The cost function, representing the average squared error between the predicted values and the actual target values.
- $m$: The number of training examples in the dataset.
- $h_\theta(\mathbf{x}^{(i)})$: The predicted value for the $i$-th training example.
- $y^{(i)}$: The actual target value for the $i$-th training example.
- $\mathbf{x}^{(i)}$: The input features for the $i$-th training example.

## Gradient Calculation

The gradient of the cost function with respect to each parameter $\theta_j$ is calculated as follows:

$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

**Parameters:**
- $\frac{\partial J(\theta)}{\partial \theta_j}$: The partial derivative of the cost function with respect to $\theta_j$, indicating how the cost function changes with a change in $\theta_j$.

## Gradient Descent Update Rules
To minimize the cost function, the parameters are updated using the gradient descent algorithm:

$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(\mathbf{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

Alternatively, it can be represented as:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\boldsymbol{\theta})$$

**Parameters:**
- $\theta_j$: The coefficient associated with the $j$-th feature $x_j$.
- $\alpha$: The learning rate, a hyperparameter that controls the step size of the parameter updates. A smaller value of $\alpha$ results in smaller updates and vice versa.
- $m$: The number of training examples in the dataset.
- $h_\theta(\mathbf{x}^{(i)})$: The predicted value for the $i$-th training example.
- $y^{(i)}$: The actual target value for the $i$-th training example.
- $x_j^{(i)}$: The $j$-th input feature for the $i$-th training example.

## Additional Explanation

- **Hypothesis Function**: Represents the model's prediction for a given input vector $\mathbf{x}$. The linear relationship is characterized by the coefficients $\theta_0, \theta_1, \ldots, \theta_n$.
- **Error**: The difference between the predicted value and the actual target value. This is the quantity we aim to minimize in our optimization process.
- **Cost Function**: Measures how well the model's predictions match the actual target values. The goal is to minimize this cost function to improve the model's accuracy.
- **Gradient Descent**: An optimization algorithm used to minimize the cost function by iteratively updating the model parameters $\theta_j$ in the direction of the negative gradient of the cost function.
- **Gradients**: The gradients ($\frac{\partial J(\theta)}{\partial \theta_j}$) represent the direction and rate of change of the cost function with respect to the model parameters. These gradients are used to adjust the parameters during optimization.

---