## Activation Functions 

### 1. **Sigmoid Activation Function**

**Formula:**
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Derivative:**
$$
\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))
$$

**Output Range:**
- **Formula Output Range:** (0, 1)
- **Derivative Output Range:** (0, 0.25)

**Pros:**
- **Smooth Gradient:** Helps the model to converge during training.
- **Probability Interpretation:** Useful for binary classification problems.
- **Historically Popular:** Widely used in classical neural networks and logistic regression.

**Cons:**
- **Vanishing Gradient Problem:** Can slow down or halt learning in deep networks.
- **Output Not Zero-Centered:** All outputs are positive, which may result in inefficient gradient updates.
- **Exponential Computation:** Can be computationally expensive.

---

### 2. **Tanh (Hyperbolic Tangent) Activation Function**

**Formula:**
$$
\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

**Derivative:**
$$
\text{tanh}'(x) = 1 - \text{tanh}^2(x)
$$

**Output Range:**
- **Formula Output Range:** (-1, 1)
- **Derivative Output Range:** (0, 1)

**Pros:**
- **Zero-Centered Outputs:** Helps in balancing positive and negative values, leading to better convergence.
- **Stronger Gradients:** Can result in faster learning.

**Cons:**
- **Vanishing Gradient Problem:** Similar to Sigmoid, especially for large inputs.
- **Exponential Computation:** Computationally intensive.

---

### 3. **ReLU (Rectified Linear Unit) Activation Function**

**Formula:**
$$
\text{ReLU}(x) = \max(0, x)
$$

**Derivative:**
$$
\text{ReLU}'(x) = \begin{cases} 
1 & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}
$$

**Output Range:**
- **Formula Output Range:** [0, ∞)
- **Derivative Output Range:** {0, 1}

**Pros:**
- **Efficient Computation:** Avoids exponential computations, thus computationally efficient.
- **Non-Saturating:** Allows for faster training.
- **Sparsity:** Can produce sparse activations.

**Cons:**
- **Dying ReLU Problem:** Some neurons may stop learning if they only output zero.
- **Unbounded Output:** Can lead to exploding activations.

---

### 4. **Leaky ReLU Activation Function**

**Formula:**
$$
\text{Leaky ReLU}(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}
$$

**Derivative:**
$$
\text{Leaky ReLU}'(x) = \begin{cases} 
1 & \text{if } x > 0 \\
\alpha & \text{if } x \leq 0
\end{cases}
$$

**Output Range:**
- **Formula Output Range:** (-∞, ∞)
- **Derivative Output Range:** {α, 1}, where \( α \) is a small positive number like 0.01.

**Pros:**
- **No Dying ReLU Problem:** Prevents neurons from completely dying.
- **Improved Learning:** Can improve learning by ensuring gradients do not vanish in negative regions.

**Cons:**
- **Fixed Negative Slope:** The slope for negative values is fixed and may not be optimal for all data.
- **Computational Overhead:** Slightly higher computational cost compared to standard ReLU.

---

### 5. **Softmax Activation Function**

**Formula:**
$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}
$$

**Derivative:**
$$
\frac{\partial \text{Softmax}(x_i)}{\partial x_j} = \text{Softmax}(x_i) \cdot (\delta_{ij} - \text{Softmax}(x_j))
$$

**Output Range:**
- **Formula Output Range:** (0, 1) for each class, with the sum of outputs equal to 1.
- **Derivative Output Range:** Generally in the range (-1, 1) across the vector space.

**Pros:**
- **Probability Distribution:** Converts logits into a probability distribution, essential for multi-class classification.
- **Normalization:** Outputs sum to 1, making results easy to interpret.

**Cons:**
- **Computationally Expensive:** Requires computing exponentials and normalization.
- **Not Suitable for Binary Classification:** Better suited for multi-class problems.
- **Vanishing Gradient with Cross-Entropy:** Can lead to very small gradients, especially with high confidence predictions.