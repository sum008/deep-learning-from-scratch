# Deep Learning from Scratch: Building Neural Networks from First Principles

Welcome! This repository documents a hands-on journey learning **deep learning, neural networks, and transformers** by implementing everything from scratch.

## 🎯 Project Goals

This project aims to understand the **mathematical foundations and mechanics** of deep learning by:

1. **Building core neural network components** without relying on high-level frameworks
2. **Implementing automatic differentiation** (autodiff) to compute gradients
3. **Training simple neural networks** and understanding backpropagation
4. **Laying groundwork** for understanding transformers and modern architectures

By implementing these concepts ourselves, we gain deep intuition about how neural networks actually work.

## 📚 What's in This Repo

### Main Notebook: `1-micrograd/nn_from_scratch.ipynb`

This Jupyter notebook is the heart of the project. It progressively builds:

- **Value Class**: A custom implementation of automatic differentiation (like PyTorch's tensor with backprop)
- **Neuron Class**: A single neuron with weights and bias
- **Layer Class**: A collection of neurons forming a network layer
- **MLP (Multi-Layer Perceptron)**: A complete neural network
- **Training Loop**: Forward pass, loss calculation, backpropagation, and weight updates
- **Binary Classification**: Training on a simple dataset to classify inputs

### Supporting Documentation

- **[1-micrograd/DERIVATIVES_EXPLAINED.md](1-micrograd/DERIVATIVES_EXPLAINED.md)**: Foundational mathematics behind gradients
- **[1-micrograd/NEURAL_NETWORKS_EXPLAINED.md](1-micrograd/NEURAL_NETWORKS_EXPLAINED.md)**: Concepts and architecture overview
- **The Notebook**: [1-micrograd/nn_from_scratch.ipynb](1-micrograd/nn_from_scratch.ipynb)**: NN Notebook

---

## 🔧 Understanding the Code: From Derivatives to Neural Networks

### Part 1: Derivatives and Gradients

Before building neural networks, we need to understand **derivatives** — the core of how neural networks learn.

#### What is a Derivative?

A **derivative** measures how much a function's output changes when you make a tiny change to its input. It's the **rate of change** or **slope** of a function at a given point.

**Mathematical Definition:**
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

#### Numerical Differentiation: The Finite Difference Method

Since we can't actually compute the limit in a computer, we use a very small value for $h$ and calculate an approximate derivative:

$$f'(x) \approx \frac{f(x+h) - f(x)}{h}$$

**Example:**
```python
def f(x):
    return 3*x**2 - 4*x + 5

# Calculate derivative at x = 2/3
h = 0.000001
x = 2/3
derivative = (f(x+h) - f(x)) / h
print(derivative)  # ≈ 0 (true derivative is 6x - 4 = 0 at x=2/3)
```

### Part 2: Partial Derivatives and the Chain Rule

When a function has multiple inputs, we calculate **partial derivatives** — how much output changes when we change ONE input while keeping others fixed.

**Example:**
```python
# Function: d = a*b + c
a = 2.0
b = -3.0
c = 10.0

# Calculate ∂d/∂b (partial derivative with respect to b)
h = 0.0001

d1 = a*b + c  # d1 = 4
b += h
d2 = a*b + c  # d2 ≈ 4.0002

slope = (d2 - d1) / h  # ≈ 2.0 (equals a, which is correct!)
```

The **chain rule** allows us to compute gradients through composite functions:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \times \frac{\partial y}{\partial x}$$

### Part 3: Addition and Gradient Flow (Critical Insight!)

One key insight is how gradients flow through **addition nodes**:

When you have: `output = input1 + input2`

The derivative with respect to **any input** is always **1**:
$$\frac{\partial \text{output}}{\partial \text{input}} = 1$$

**Why?** Using the limit definition:
$$\lim_{h \to 0} \frac{(\text{input}+h + \text{other}) - (\text{input} + \text{other})}{h} = \lim_{h \to 0} \frac{h}{h} = 1$$

**Practical Implication:** In backpropagation, gradients pass through addition nodes **unchanged**. The addition operator acts as a "gradient router" — it doesn't amplify or reduce gradients, just distributes them.

```
Gradient flow through addition:
∂L/∂z ──→ (×1) ──→ ∂L/∂(input1 + input2) = ∂L/∂z
```

---

## 🧠 Building the Value Class: Automatic Differentiation

The `Value` class is our implementation of **automatic differentiation**. It tracks:
- **data**: The actual numerical value
- **grad**: The gradient (how much this value contributes to the final output)
- **_backward**: A function that computes gradients of parent nodes

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None  
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad      # gradient passes through unchanged
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad  # gradient is scaled by the other input
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
```

Notice the crucial difference:
- **Addition**: Gradient passes through as `1.0 * out.grad`
- **Multiplication**: Gradient is scaled by the other input

### Example: Computing Gradients

```python
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
f = Value(-2.0, label='f')

e = a * b
d = e + c
L = d * f  # L = (a*b + c) * f

L.backward()  # Compute all gradients

print(a.grad)  # 6.0 (gradient of L with respect to a)
print(b.grad)  # 4.0 (gradient of L with respect to b)
print(c.grad)  # -2.0 (gradient of L with respect to c)
```

---

## 🧬 Building Neural Network Components

### The Neuron Class

A single neuron implements: **output = tanh(w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b)**

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
```

### The Layer and MLP Classes

```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        return [n(x) for n in self.neurons]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

---

## 🚀 Training a Neural Network

The full training loop demonstrates supervised learning:

```python
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]  # Target outputs

n = MLP(3, [4, 4, 1])  # 3 inputs → 4 hidden → 4 hidden → 1 output

for k in range(1000):
    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0))
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    for p in n.parameters():
        p.data += -0.05 * p.grad
    
    # Zero gradients
    for p in n.parameters():
        p.grad = 0.0
    
    print(k, loss.data)
```

The network learns by:
1. Making predictions on training data
2. Computing a loss (how wrong the predictions are)
3. Computing gradients via backpropagation
4. Updating weights in the direction that **reduces loss**

---

## 📊 Key Concepts Reinforced

| Concept | Meaning | In Practice |
|---------|---------|------------|
| **Derivative** | Rate of change of output w.r.t. input | How much does loss change if we tweak a weight? |
| **Partial Derivative** | Rate of change w.r.t. one input (others fixed) | Gradient with respect to one specific weight |
| **Chain Rule** | How to compose derivatives | Backprop through multiple layers |
| **Gradient** | Vector of all partial derivatives | Direction of steepest descent in parameter space |
| **Backpropagation** | Computing gradients using chain rule | How neural networks learn |
| **Gradient Descent** | Update parameters opposite to gradient | Move weights to reduce loss |

---

## 🔮 Future Directions

This foundation prepares us to understand:

1. **Recurrent Neural Networks (RNNs)**: Handling sequences with hidden state
2. **Attention Mechanisms**: How transformers attend to different parts of input
3. **Transformers**: Modern architecture that powers LLMs
4. **Large Language Models**: Building on transformer foundations

---

## 🛠️ Running the Code

### Prerequisites
```bash
python3 -m venv nn
source nn/bin/activate
pip install numpy matplotlib torch
```

### Run the Notebook
```bash
jupyter notebook 1-micrograd/nn_from_scratch.ipynb
```

---

## 📖 References & Further Reading

- **Derivatives Deep Dive**: See [1-micrograd/DERIVATIVES_EXPLAINED.md](1-micrograd/DERIVATIVES_EXPLAINED.md)
- **Neural Network Architectures**: See [1-micrograd/NEURAL_NETWORKS_EXPLAINED.md](1-micrograd/NEURAL_NETWORKS_EXPLAINED.md)
- **The Notebook**: [1-micrograd/nn_from_scratch.ipynb](1-micrograd/nn_from_scratch.ipynb)

---

## 💡 Key Insight

**Neural networks are just composition of simple functions**: additions, multiplications, and nonlinearities (like tanh). By building a system that can compute gradients of **any composition** of these operations, we can train networks to learn patterns in data.

This is the power of **automatic differentiation** and **backpropagation** — the algorithmic foundations of deep learning.

---

*Building intuition, one derivative at a time.* 🚀
