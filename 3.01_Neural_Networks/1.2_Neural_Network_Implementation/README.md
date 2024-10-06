
# Neural Network Equations

## 1. **Forward Pass**

### **Layer $l$ Computations:**

1. **Linear Transformation**:
   $$
   Z^{(l)} = A^{(l-1)} W^{(l)} + b^{(l)}
   $$
   - **$Z^{(l)}$**: Pre-activation output for layer $l$.
   - **$A^{(l-1)}$**: Activations from the previous layer.
   - **$W^{(l)}$**: Weight matrix for layer $l$.
   - **$b^{(l)}$**: Bias vector for layer $l$.

2. **Activation Function**:
   $$
   A^{(l)} = \sigma(Z^{(l)})
   $$
   - **$A^{(l)}$**: Activation output of layer $l$.
   - **$\sigma(z)$**: Activation function applied element-wise to $Z^{(l)}$.

   Common activation functions include:
   - **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
   - **ReLU**: $\sigma(z) = \max(0, z)$

3. **Output Layer**:
   - The final activations $A^{(L)}$ represent the output of the network.

---

## 2. **Backward Pass**

### **Layer $l$ Computations:**

1. **Error Term for the Output Layer**:
   $$
   \delta^{(L)} = (A^{(L)} - y) \odot \sigma'(Z^{(L)})
   $$
   - **$\delta^{(L)}$**: Error term for the final layer.
   - **$A^{(L)}$**: Activation output of the final layer.
   - **$y$**: True labels.
   - **$\sigma'(Z^{(L)})$**: Derivative of the activation function for the output layer.

2. **Error Terms for Hidden Layers**:
   $$
   \delta^{(l)} = \left( \delta^{(l+1)} \cdot (W^{(l+1)})^T \right) \odot \sigma'(Z^{(l)})
   $$
   - **$\delta^{(l)}$**: Error term for layer $l$.
   - **$W^{(l+1)}$**: Weight matrix of the next layer.
   - **$\sigma'(Z^{(l)})$**: Derivative of the activation function at layer $l$.

3. **Gradients for Weights and Biases**:

   - **Weights**:
     $$
     \frac{\partial \mathcal{L}}{\partial W^{(l)}} = A^{(l-1)T} \cdot \delta^{(l)}
     $$
     - **$A^{(l-1)T}$**: Transposed activations from the previous layer.
     - **$\delta^{(l)}$**: Error term for layer $l$.

   - **Biases**:
     $$
     \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}
     $$
     - **$\delta^{(l)}$**: Error term for layer $l$.

---

## 3. **Weight and Bias Updates**

### **Weight Update**:
$$
W^{(l)} \leftarrow W^{(l)} - \alpha \cdot \frac{\partial \mathcal{L}}{\partial W^{(l)}}
$$
- **$W^{(l)}$**: Weight matrix for layer $l$.
- **$\alpha$**: Learning rate.
- **$\frac{\partial \mathcal{L}}{\partial W^{(l)}}$**: Gradient of the loss function with respect to $W^{(l)}$.

### **Bias Update**:
$$
b^{(l)} \leftarrow b^{(l)} - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b^{(l)}}
$$
- **$b^{(l)}$**: Bias vector for layer $l$.
- **$\frac{\partial \mathcal{L}}{\partial b^{(l)}}$**: Gradient of the loss function with respect to $b^{(l)}$.