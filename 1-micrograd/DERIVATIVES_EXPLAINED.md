# Understanding Derivatives: Numerical Differentiation Guide

## What is a Derivative?

A **derivative** measures how much a function's output changes when you make a tiny change to its input. It's the **rate of change** or **slope** of a function at a given point.

### Mathematical Definition

The derivative of a function $f(x)$ is defined as:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

In plain English: *How much does the output change when we nudge the input by a very small amount?*

---

## Numerical Differentiation: The Finite Difference Method

Since we can't actually take the limit to zero in a computer, we use **numerical differentiation** — we pick a very small value for $h$ and calculate the approximate derivative.

### The Formula

$$f'(x) \approx \frac{f(x+h) - f(x)}{h}$$

Where:
- **$f(x)$** = function value at original point
- **$f(x+h)$** = function value after a tiny bump
- **$h$** = tiny step size (e.g., 0.0001)

---

## Code Example 1: Simple Function Derivative

```python
def f(x):
    return 3*x**2 - 4*x + 5

# Calculate derivative at x = 2/3
h = 0.000001
x = 2/3
derivative = (f(x+h) - f(x)) / h
print(derivative)  # ≈ 0 (true derivative is 6x - 4 = 0 at x=2/3)
```

### Breaking It Down

1. **Define the function**: $f(x) = 3x^2 - 4x + 5$
2. **Pick a small step**: $h = 0.000001$
3. **Evaluate at two points**:
   - $f(x) = f(0.667) = 0.333...$
   - $f(x+h) = f(0.667001) ≈ 0.333...$
4. **Calculate slope**: $\frac{f(x+h) - f(x)}{h} ≈ 0$

The true derivative is $f'(x) = 6x - 4$, so $f'(2/3) = 4 - 4 = 0$ ✓

---

## Code Example 2: Partial Derivatives (Multi-variable Functions)

When a function has multiple inputs, we calculate the **partial derivative** — how much output changes when we change ONE input while keeping others fixed.

```python
# Function: d = a*b + c
a = 2.0
b = -3.0
c = 10.0

# Calculate ∂d/∂b (partial derivative with respect to b)
h = 0.0001

# Step 1: Original output
d1 = a*b + c  # d1 = 2*(-3) + 10 = 4

# Step 2: Bump b by h
b += h  # b = -2.9999

# Step 3: New output
d2 = a*b + c  # d2 = 2*(-2.9999) + 10 = 4.0002

# Step 4: Calculate slope
slope = (d2 - d1) / h  # (4.0002 - 4) / 0.0001 = 2.0
print(f'd1: {d1}')
print(f'd2: {d2}')
print(f'slope (∂d/∂b): {slope}')
```

### Output
```
d1: 4.0
d2: 4.0002
slope (∂d/∂b): 2.0
```

### What This Means

- The **partial derivative** $\frac{\partial d}{\partial b} = 2.0$
- This means: *For every 1 unit increase in $b$, $d$ increases by 2 units*
- Mathematically: Since $d = a \cdot b + c$, the derivative is $\frac{\partial d}{\partial b} = a = 2$ ✓

---

## The Same Function with Different Input

You can also calculate the partial derivative with respect to $a$:

```python
# Calculate ∂d/∂a (partial derivative with respect to a)
h = 0.0001

a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
a += h  # Bump a instead of b
d2 = a*b + c
print(f'slope (∂d/∂a): {(d2-d1)/h}')  # ≈ -3.0 (equals b)
```

This gives us $\frac{\partial d}{\partial a} = -3.0$, which equals $b$.

---

## Step-by-Step Walkthrough

### For ∂d/∂b where d = a·b + c:

| Step | Value | Explanation |
|------|-------|-------------|
| 1. Set inputs | a=2, b=-3, c=10 | Initial values |
| 2. Calculate d1 | d1 = 2×(-3)+10 = 4 | Original output |
| 3. Bump b | b = -3 + 0.0001 = -2.9999 | Tiny increase in b |
| 4. Calculate d2 | d2 = 2×(-2.9999)+10 = 4.0002 | New output |
| 5. Find slope | (4.0002 - 4) / 0.0001 = 2 | Rate of change |

---

## Why This Matters for Neural Networks

This is the **foundation of backpropagation**, the algorithm that trains neural networks:

1. **Forward Pass**: Calculate output given inputs
2. **Backward Pass**: Calculate derivatives (gradients) using the chain rule
3. **Update Weights**: Adjust weights in the direction opposite to the gradient

```
Loss decreases → Gradient tells us how much to change each weight → Update weights
```

In a neural network, we have thousands of weights, and we need to calculate how much each weight affects the final loss. Derivatives tell us the direction and magnitude of change needed.

---

## Key Takeaways

| Concept | Meaning |
|---------|---------|
| **Derivative** | Rate of change of output with respect to input |
| **Partial derivative** | Rate of change when changing ONE input (others fixed) |
| **Numerical differentiation** | Approximating derivatives using small step sizes |
| **Gradient** | Vector of all partial derivatives (multi-variable) |
| **Backpropagation** | Using derivatives to train neural networks |

---

## Common h Values

- **Too large** (h=0.1): Approximation is inaccurate
- **Good range** (h=0.0001 to 0.000001): Balances accuracy and numerical stability
- **Too small** (h=1e-10): Numerical precision errors accumulate

The sweet spot is usually $h \approx 10^{-4}$ to $10^{-6}$ depending on the scale of your function.

---

## Practice Exercise

Try calculating the partial derivatives yourself:

```python
# Function: z = x*y + y^2
x = 3.0
y = 2.0
h = 0.0001

# Calculate ∂z/∂x
z1 = x*y + y**2
x_new = x + h
z2 = x_new*y + y**2
grad_x = (z2 - z1) / h

# Calculate ∂z/∂y
x = 3.0  # Reset
z1 = x*y + y**2
y_new = y + h
z2 = x*y_new + y_new**2
grad_y = (z2 - z1) / h

print(f'∂z/∂x = {grad_x}')  # Should be ≈ 2
print(f'∂z/∂y = {grad_y}')  # Should be ≈ 7
```

Why? Because $\frac{\partial z}{\partial x} = y = 2$ and $\frac{\partial z}{\partial y} = x + 2y = 3 + 4 = 7$

---

## The Derivative of Addition: Why It's Always 1

A crucial concept in backpropagation is understanding how gradients flow through **addition operations**.

### The Simple Rule

When you have an addition operation:
$$\text{output} = \text{input1} + \text{input2} + ... + \text{inputN}$$

The derivative with respect to **any input** is always **1**:

$$\frac{\partial \text{output}}{\partial \text{input1}} = 1$$
$$\frac{\partial \text{output}}{\partial \text{input2}} = 1$$

### Why? Mathematical Derivation

Let's say `output = input1 + input2`:

$$\frac{d(\text{output})}{d(\text{input1})} = \lim_{h \to 0} \frac{(\text{input1}+h + \text{input2}) - (\text{input1} + \text{input2})}{h}$$

$$= \lim_{h \to 0} \frac{h}{h} = 1$$

The `h` terms cancel out, leaving us with **1**.

### Concrete Example from Neural Networks

Consider:
```
n = (x1*w1 + x2*w2) + b
```

Let's call `z = x1*w1 + x2*w2`, so `n = z + b`

**Question:** What is $\frac{\partial n}{\partial z}$?

**Answer:** Since $n = z + b$, the derivative is:

$$\frac{\partial n}{\partial z} = 1$$

This holds true **regardless of the values** of z or b. The addition operator is **linear** — it doesn't amplify or scale its inputs.

### Gradient Flow Through Addition

In backpropagation, when a gradient flows backward through an addition node, it passes through **unchanged**:

```
Forward:   x1*w1 ──┐
           x2*w2 ──┼→ (+) → z ──→ ... ──→ output
           b ──────┘

Backward:  ∂L/∂z  ──→ (×1) ──→ ∂L/∂(x1*w1 + x2*w2) = ∂L/∂z
```

Since we multiply by 1, the gradient at the output of the addition node equals the gradient at the input.

### Key Insight: Addition vs Multiplication

| Operation | Derivative | Gradient Flow |
|-----------|-----------|--------------|
| **Addition** `z = a + b` | `dz/da = 1`, `dz/db = 1` | **Passes through unchanged** |
| **Multiplication** `z = a * b` | `dz/da = b`, `dz/db = a` | **Scaled by the other input** |

Addition is a "gradient router" — it distributes the incoming gradient to all inputs equally.

### Chain Rule with Addition

When computing gradients through a computation graph:

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial n} \times \frac{\partial n}{\partial z}$$

If `n = z + b`:
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial n} \times 1 = \frac{\partial L}{\partial n}$$

The gradient simply passes through without modification.
