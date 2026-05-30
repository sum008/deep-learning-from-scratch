# Neural Networks from Scratch: A Comprehensive Guide

This guide explains the fundamental concepts behind building and training neural networks from first principles, as implemented in `nn_from_scratch.ipynb`.

---

## Table of Contents

1. [Forward Pass](#forward-pass)
2. [Backpropagation](#backpropagation)
3. [Gradient Descent](#gradient-descent)
4. [The Neuron](#the-neuron)
5. [The Multi-Layer Perceptron (MLP)](#the-multi-layer-perceptron-mlp)
6. [Putting It All Together: Training](#putting-it-all-together-training)

---

## Forward Pass

The **forward pass** is the process of computing predictions by pushing data through the network from input to output.

### What Happens During Forward Pass?

1. **Input data** flows through each layer
2. Each neuron computes a weighted sum of inputs plus bias
3. An **activation function** (like tanh) is applied
4. The output becomes input to the next layer
5. Final layer outputs the prediction

### Mathematical View

For a single neuron with inputs $x_1, x_2, ..., x_n$ and weights $w_1, w_2, ..., w_n$:

$$z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$

Where $b$ is the bias term.

Then apply activation function:

$$output = \tanh(z) = \frac{e^{2z} - 1}{e^{2z} + 1}$$

### Example from the Notebook

```python
# inputs x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# weights w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# bias
b = Value(6.8813735870195432, label='b')

# Forward pass: z = x1*w1 + x2*w2 + b
x1w1 = x1 * w1
x2w2 = x2 * w2
z = x1w1 + x2w2 + b

# Apply activation
output = z.tanh()
```

### Key Points

- **Computation is recorded**: Each operation stores references to its inputs, building a computation graph
- **Values flow through**: Each `Value` object carries both data and tracking information
- **Graph structure**: This computation graph is crucial for backpropagation

---

## Backpropagation

**Backpropagation** is the algorithm for computing gradients (derivatives) of the loss with respect to all parameters. It uses the **chain rule** of calculus to efficiently propagate gradients backward through the network.

### Why Backpropagation?

To train a network, we need to know: *"How much does each weight affect the final loss?"*

Backpropagation answers this question by computing $\frac{\partial Loss}{\partial w}$ for every weight $w$.

### The Chain Rule Foundation

The chain rule states:

$$\frac{\partial Loss}{\partial x} = \frac{\partial Loss}{\partial y} \times \frac{\partial y}{\partial x}$$

### Step-by-Step Example

Consider: $L = (d \times f)$ where $d = (a \times b) + c$

**Forward Pass:**
- $a = 2.0, b = -3.0, c = 10.0, f = -2.0$
- $e = a \times b = -6$
- $d = e + c = 4$
- $L = d \times f = -8$

**Backward Pass (Computing Gradients):**

1. Start with $\frac{\partial L}{\partial L} = 1.0$ (the loss gradient with respect to itself)

2. For multiplication $L = d \times f$:
   - $\frac{\partial L}{\partial d} = f = -2.0$
   - $\frac{\partial L}{\partial f} = d = 4.0$

3. For addition $d = e + c$:
   - $\frac{\partial L}{\partial e} = \frac{\partial L}{\partial d} \times \frac{\partial d}{\partial e} = -2.0 \times 1.0 = -2.0$
   - $\frac{\partial L}{\partial c} = \frac{\partial L}{\partial d} \times \frac{\partial d}{\partial c} = -2.0 \times 1.0 = -2.0$

4. For multiplication $e = a \times b$:
   - $\frac{\partial L}{\partial a} = \frac{\partial L}{\partial e} \times \frac{\partial e}{\partial a} = -2.0 \times b = -2.0 \times (-3.0) = 6.0$
   - $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial e} \times \frac{\partial e}{\partial b} = -2.0 \times a = -2.0 \times 2.0 = -4.0$

### Implementation: The Value Class

The `Value` class implements automatic differentiation by storing backward functions:

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # Gradient accumulator
        self._backward = lambda: None  # Backward function
        self._prev = set(_children)  # Children in computation graph
        self._op = _op  # Operation that created this value
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # For addition, gradient passes through unchanged
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            # For multiplication, use the chain rule
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            # Derivative of tanh: 1 - tanh²(x)
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward
        return out

    def backward(self):
        # Topological sort ensures correct order
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        # Process in reverse topological order
        for node in reversed(topo):
            node._backward()
```

### Critical Insight: Topological Sort

Backpropagation processes nodes in **reverse topological order** to ensure:
- Leaf nodes are processed before their parents
- Gradients are computed in the correct dependency order
- All upstream gradients are accumulated before processing a node

### Important Addition Property

When values are added:
$$\frac{\partial (a + b)}{\partial a} = 1, \quad \frac{\partial (a + b)}{\partial b} = 1$$

This means **gradients pass through addition unchanged**—addition acts as a "gradient router" distributing the incoming gradient equally to all inputs.

---

## Gradient Descent

**Gradient descent** is the optimization algorithm that uses gradients to update weights and minimize the loss function.

### The Core Idea

1. Compute gradients: $\nabla L = \left(\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, ..., \frac{\partial L}{\partial w_n}\right)$
2. Update weights in the **opposite direction** of the gradient
3. Repeat until convergence

### Mathematical Formulation

$$w_{\text{new}} = w_{\text{old}} - \alpha \times \frac{\partial L}{\partial w}$$

Where:
- $\alpha$ is the **learning rate** (step size)
- $\frac{\partial L}{\partial w}$ is the gradient

### Intuition

- **Gradient points toward increase**: If $\frac{\partial L}{\partial w} = 0.5$, loss increases in that direction
- **Go opposite direction**: Subtract $\alpha \times 0.5$ to decrease loss
- **Learning rate controls step size**: Larger $\alpha$ means bigger steps (but risk overshooting), smaller $\alpha$ means slower but safer updates

### Example: Single Parameter Update

Suppose:
- Current weight: $w = 2.0$
- Gradient: $\frac{\partial L}{\partial w} = 0.6$
- Learning rate: $\alpha = 0.01$

Update:
$$w_{\text{new}} = 2.0 - 0.01 \times 0.6 = 2.0 - 0.006 = 1.994$$

The weight moved slightly in the negative direction because the gradient was positive.

### Practical Implementation

```python
step_size = 0.05  # Learning rate

for iteration in range(1000):
    # Forward pass: compute predictions
    ypred = [network(x) for x in xs]
    
    # Compute loss
    loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0))
    
    # Backward pass: compute gradients
    loss.backward()
    
    # Update parameters
    for param in network.parameters():
        param.data += -step_size * param.grad
    
    # Zero gradients for next iteration
    for param in network.parameters():
        param.grad = 0.0
```

### Why Zero Gradients?

The `.grad` field **accumulates** gradients. If we don't reset it, old gradients would add to new ones, causing incorrect updates in subsequent iterations.

### Learning Rate Impact

- **Too high**: Weights oscillate wildly, loss doesn't decrease
- **Too low**: Training is slow, takes many iterations
- **Goldilocks zone**: Loss steadily decreases

---

## The Neuron

A **neuron** is the fundamental building block of a neural network. It takes multiple inputs, combines them with learned weights, adds a bias, and applies an activation function.

### Neuron Architecture

```
Input x1 ──→ [w1] ──┐
Input x2 ──→ [w2] ──┤
Input x3 ──→ [w3] ──┼→ Σ (sum) ──→ activation ──→ output
                    │
                    └─→ [bias]
```

### Mathematical Definition

$$y = \tanh(w_1 x_1 + w_2 x_2 + ... + w_n x_n + b)$$

Where:
- $w_i$ are learned weights
- $b$ is the learned bias
- $\tanh()$ is the activation function

### Implementation

```python
class Neuron:
    def __init__(self, nin):
        # Initialize weights randomly
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # Initialize bias randomly
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        # Compute weighted sum
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # Apply activation function
        out = activation.tanh()
        return out
    
    def parameters(self):
        # Return all learnable parameters
        return self.w + [self.b]
```

### How It Works: Step by Step

Given inputs $x = [2.0, 0.0]$ and weights $w = [-3.0, 1.0]$, bias $b = 6.88$:

1. **Compute weighted sum**:
   - $z = (2.0)(-3.0) + (0.0)(1.0) + 6.88 = 0.88$

2. **Apply activation**:
   - $y = \tanh(0.88) ≈ 0.711$

3. **Output**: $0.711$ is passed to the next layer (or becomes final output)

### Why Tanh Activation?

- **Non-linearity**: Without it, stacking layers just creates a linear function
- **Bounded output**: $\tanh(x) \in [-1, 1]$, preventing outputs from exploding
- **Smooth derivative**: $\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$, which is smooth for backprop

### Neuron as a Classifier

A single neuron can:
- Learn **linear decision boundaries**
- Output values in $[-1, 1]$ (use as probability, threshold at 0)
- Adjust weights to fit data patterns

Multiple neurons working together (multi-layer) can learn **non-linear** boundaries.

---

## The Multi-Layer Perceptron (MLP)

A **Multi-Layer Perceptron** is a stack of neurons organized into layers. It's the most basic neural network architecture and can learn complex, non-linear relationships.

### Architecture

```
Input Layer  Hidden Layer 1  Hidden Layer 2  Output Layer
    x₁                                            ŷ
    x₂            neuron              neuron
    x₃            neuron              neuron
                  neuron
```

### Key Concept: Layered Structure

- **Each layer's output becomes the next layer's input**
- **Hidden layers** process and transform data
- **Deeper networks** can learn more complex patterns
- **Wider layers** can capture more parallel patterns

### Implementation

```python
class Layer:
    """A single layer of neurons"""
    def __init__(self, nin, nout):
        # Create nout neurons, each taking nin inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        # Apply each neuron and collect outputs
        outs = [n(x) for n in self.neurons]
        # If only one neuron, return scalar; else return list
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        # Collect parameters from all neurons
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:
    """Multi-Layer Perceptron"""
    def __init__(self, nin, nouts):
        # nin: number of inputs
        # nouts: list of neuron counts per layer
        # Example: MLP(3, [4, 4, 1]) creates:
        #   - Layer 1: 3 inputs → 4 neurons
        #   - Layer 2: 4 inputs → 4 neurons
        #   - Layer 3: 4 inputs → 1 neuron
        
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        # Push input through each layer sequentially
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        # Collect all parameters from all layers
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
```

### Example: Creating an MLP

```python
# Create a network with:
# - 3 inputs
# - First hidden layer: 4 neurons
# - Second hidden layer: 4 neurons
# - Output layer: 1 neuron
n = MLP(3, [4, 4, 1])

# Get total parameters
print(len(n.parameters()))  # 3*4 + 4 + 4*4 + 4 + 4*1 + 1 = 57 parameters

# Use the network
x = [2.0, 3.0, -1.0]
output = n(x)  # Single value (since output layer has 1 neuron)
```

### Parameter Count Breakdown

For `MLP(3, [4, 4, 1])`:

**Layer 1** (3 inputs → 4 neurons):
- Weights: 3 × 4 = 12
- Biases: 4
- Subtotal: 16

**Layer 2** (4 inputs → 4 neurons):
- Weights: 4 × 4 = 16
- Biases: 4
- Subtotal: 20

**Layer 3** (4 inputs → 1 neuron):
- Weights: 4 × 1 = 4
- Biases: 1
- Subtotal: 5

**Total: 41 parameters**

### Computational Flow

For input $[2.0, 3.0, -1.0]$:

1. **Layer 1**: 3 neurons compute 4 activations
2. **Layer 2**: 4 neurons take those 4 values as input, compute 4 activations
3. **Layer 3**: 1 neuron takes those 4 values, outputs final prediction

### Why Stack Layers?

- **Single layer**: Can only learn linear boundaries
- **Multiple layers**: Can learn arbitrary non-linear functions
- **Deeper networks**: Trade-off between expressiveness and training difficulty

---

## Putting It All Together: Training

Training a neural network combines forward pass, loss computation, backpropagation, and gradient descent.

### The Complete Training Loop

```python
# 1. Create network
n = MLP(3, [4, 4, 1])

# 2. Define dataset
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]  # Target outputs

# 3. Training loop
step_size = 0.05
iterations = 1000

for k in range(iterations):
    # Forward pass: get predictions
    ypred = [n(x) for x in xs]
    
    # Compute loss: sum of squared errors
    loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0))
    
    # Backward pass: compute gradients
    loss.backward()
    
    # Update weights: gradient descent
    for p in n.parameters():
        p.data += -step_size * p.grad
    
    # Zero gradients for next iteration
    for p in n.parameters():
        p.grad = 0.0
    
    # Print progress
    if k % 100 == 0:
        print(f"Iteration {k}: Loss = {loss.data}")
```

### What Happens in Each Iteration

**Iteration 0:**
- Loss starts high (random weights)
- Gradients computed via backprop
- Weights updated based on gradients

**Iteration 100:**
- Network has learned some patterns
- Loss is lower
- Weights have shifted toward good values

**Iteration 1000:**
- Network converges
- Loss is minimized
- Predictions close to targets

### Loss Function: Mean Squared Error

$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i^{\text{pred}} - y_i^{\text{true}})^2$$

Where:
- $y_i^{\text{pred}}$ is the network's prediction
- $y_i^{\text{true}}$ is the true label
- $n$ is the number of samples

This measures **average squared error** between predictions and targets.

### Gradient Flow Visualization

For a single sample with loss $(y_{\text{pred}} - y_{\text{true}})^2$:

1. Gradient flows from loss back through output layer
2. Each neuron's gradients accumulate from all downstream neurons
3. Layer 2 neurons receive combined gradients from output
4. Layer 1 neurons receive combined gradients from layer 2
5. All weights updated simultaneously

### Convergence Criteria

Training stops when:
- **Fixed iterations**: Run for predetermined number of epochs
- **Loss threshold**: Stop when loss is below target
- **Early stopping**: Stop when validation loss stops improving
- **Gradient norm**: Stop when gradients become very small

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Loss not decreasing | Learning rate too high | Reduce step_size |
| Training very slow | Learning rate too low | Increase step_size |
| Loss exploding | Gradients too large | Use gradient clipping or reduce learning rate |
| Overfitting | Model too complex for data | Use regularization, more data, or simpler model |
| Weights stuck | Poor initialization | Use better random initialization |

---

## Integration: How It All Works Together

### The Complete Data Flow

```
1. INPUT DATA [x₁, x₂, x₃]
   ↓
2. FORWARD PASS through MLP
   - Layer 1: 3 → 4 neurons
   - Layer 2: 4 → 4 neurons
   - Layer 3: 4 → 1 neuron
   ↓
3. PREDICTION ŷ
   ↓
4. LOSS COMPUTATION
   Loss = (ŷ - y_true)²
   ↓
5. BACKWARD PASS (Backpropagation)
   - Compute ∂Loss/∂output
   - Layer 3 gradients: ∂Loss/∂(layer 2 outputs)
   - Layer 2 gradients: ∂Loss/∂(layer 1 outputs)
   - Layer 1 gradients: ∂Loss/∂inputs
   - All weight gradients computed
   ↓
6. GRADIENT DESCENT (Update)
   w_new = w_old - learning_rate × ∂Loss/∂w
   ↓
7. REPEAT (Next iteration)
```

### Key Interactions

**Forward Pass + Computation Graph:**
- Records operations
- Enables gradient computation
- Builds dependency graph

**Backpropagation + Gradient Descent:**
- Computes how much each weight contributed to error
- Gradient points toward minimizing loss
- Update weights to reduce future errors

**Loss Function + Gradients:**
- Measures prediction quality
- Gradient of loss w.r.t. weights drives learning
- Optimization adjusts weights based on loss signal

### Why This Works

1. **Non-linear activations** allow learning complex patterns
2. **Multiple layers** compose non-linearities for expressiveness
3. **Gradient descent** efficiently searches weight space
4. **Backpropagation** computes gradients efficiently
5. **Iterative updates** gradually improve predictions

---

## Mathematical Summary

### Forward Pass Formulation

For layer $l$ with input $\mathbf{x}^{(l-1)}$:

$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{x}^{(l-1)} + \mathbf{b}^{(l)}$$

$$\mathbf{x}^{(l)} = \tanh(\mathbf{z}^{(l)})$$

Final layer output:
$$\hat{y} = \mathbf{x}^{(L)}$$

### Loss Function

$$J = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$$

### Backpropagation

For each layer, compute:
$$\frac{\partial J}{\partial \mathbf{W}^{(l)}} = \frac{\partial J}{\partial \mathbf{x}^{(l)}} \otimes \mathbf{x}^{(l-1)T}$$

$$\frac{\partial J}{\partial \mathbf{x}^{(l-1)}} = (\mathbf{W}^{(l)})^T \frac{\partial J}{\partial \mathbf{x}^{(l)}} \odot \tanh'(\mathbf{z}^{(l-1)})$$

Where $\odot$ is element-wise multiplication and $\otimes$ is outer product.

### Gradient Descent Update

$$\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \alpha \frac{\partial J}{\partial \mathbf{W}^{(l)}}$$

$$\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \alpha \frac{\partial J}{\partial \mathbf{b}^{(l)}}$$

Where $\alpha$ is the learning rate.

---

## Conclusion

Building neural networks from scratch reveals the elegant mathematics underlying deep learning:

- **Neurons** compute weighted sums with non-linear activations
- **Layers** compose neurons to learn complex functions
- **MLPs** stack layers for increased expressiveness
- **Forward passes** compute predictions and build computation graphs
- **Backpropagation** efficiently computes gradients using chain rule
- **Gradient descent** updates weights to minimize loss

Understanding these fundamentals is essential for:
- Debugging training issues
- Designing effective architectures
- Choosing appropriate hyperparameters
- Innovating new techniques

The beauty of this approach is that the same principles scale to modern deep learning frameworks like PyTorch and TensorFlow, just with more optimizations and features built on top.

---

## References from Notebook

The `nn_from_scratch.ipynb` demonstrates:

1. **Value class** - Automatic differentiation implementation
2. **Numerical derivatives** - Gradient verification technique
3. **Single neuron** - Basic tanh neuron with forward/backward
4. **Multiple neurons** - Layer combining multiple neurons
5. **MLP architecture** - Multi-layer perceptron composition
6. **Binary classification** - Simple dataset training
7. **Training loop** - Complete forward/backward/update cycle
8. **Convergence** - Network learning and loss minimization

Each concept builds upon the previous, creating a complete understanding of neural network mechanics.
