# Understanding Control Affine Dynamics Learning

This document provides a detailed explanation of the mathematical foundations, neural network architecture, and ensemble methodology used in our dynamics learning system.

## 1. Control Affine Systems: Mathematical Framework

### 1.1 System Representation

A control affine system is a specific form of nonlinear dynamical system where the control inputs appear linearly in the dynamics equation:

$$\dot{x} = f(x) + g(x)u$$

Where:
- $x \in \mathbb{R}^n$ is the state vector
- $u \in \mathbb{R}^m$ is the control input (action)
- $f(x) \in \mathbb{R}^n$ represents the autonomous dynamics (how the system evolves with no control input)
- $g(x) \in \mathbb{R}^{n \times m}$ is the control matrix determining how control inputs affect each state dimension

For a system with $n$ state dimensions and $m$ control dimensions, we can expand this equation:

$$\begin{pmatrix} \dot{x}_1 \\ \dot{x}_2 \\ \vdots \\ \dot{x}_n \end{pmatrix} = 
\begin{pmatrix} f_1(x) \\ f_2(x) \\ \vdots \\ f_n(x) \end{pmatrix} + 
\begin{pmatrix} 
g_{11}(x) & g_{12}(x) & \cdots & g_{1m}(x) \\
g_{21}(x) & g_{22}(x) & \cdots & g_{2m}(x) \\
\vdots & \vdots & \ddots & \vdots \\
g_{n1}(x) & g_{n2}(x) & \cdots & g_{nm}(x)
\end{pmatrix}
\begin{pmatrix} u_1 \\ u_2 \\ \vdots \\ u_m \end{pmatrix}$$

### 1.2 Significance of Control Affine Form

This formulation is particularly valuable because:

1. Many physical systems naturally follow control affine dynamics (e.g., robotic systems, pendulums, vehicles)
2. The separation of autonomous dynamics $f(x)$ and control influence $g(x)$ provides insights into system behavior
3. Control affine systems allow for specialized control design techniques (e.g., feedback linearization, Lyapunov-based control)
4. The structure constrains the learning problem, potentially improving sample efficiency

### 1.3 Discretization for Numerical Implementation

Since we work with discrete time steps in both simulation and learning, we use Euler integration to approximate the continuous dynamics:

$$x_{t+\Delta t} = x_t + \Delta t \cdot \dot{x}_t = x_t + \Delta t \cdot [f(x_t) + g(x_t)u_t]$$

For numerical stability, we use a small time step $\Delta t$ (typically 0.05 seconds in our implementation). More sophisticated numerical integration methods like Runge-Kutta could be used for higher accuracy, but Euler integration provides a good trade-off between simplicity and performance for our application.

## 2. Neural Network Architecture

### 2.1 Control Affine Network Design

Our neural network architecture is specifically designed to respect the control affine structure of the system dynamics:

```
                          ┌─────────┐
                          │  State  │ x ∈ ℝⁿ
                          └────┬────┘
                               │
                               ▼
                     ┌─────────────────┐
                     │ Shared Layers   │ Feature extraction
                     │ FC → Tanh → FC → Tanh │
                     └─────┬───────────┘
                      Shared features
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
     ┌─────────────────┐     ┌─────────────────┐
     │  f(x) Network   │     │  g(x) Network   │
     │  (FC Layer)     │     │  (FC Layer)     │
     └────────┬────────┘     └────────┬────────┘
              │                       │
              ▼                       ▼
      ┌───────────────┐      ┌────────────────────┐
      │ f(x) Output   │      │ g(x) Output        │
      │ ℝⁿ            │      │ ℝⁿˣᵐ (reshaped)    │
      └───────┬───────┘      └──────────┬─────────┘
              │                         │
              └──────────┬──────────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ x_dot = f(x) + g(x)u │
                 └───────────────┘
```

#### 2.1.1 Shared Feature Extraction

The network begins with shared layers that extract features from the state:

$$h(x) = \phi_2(W_2 \cdot \phi_1(W_1 x + b_1) + b_2)$$

Where:
- $h(x)$ is the shared feature representation
- $W_1, W_2$ are weight matrices of the shared layers
- $b_1, $b_2$ are bias vectors
- $\phi_1, \phi_2$ are nonlinear activation functions (tanh in our implementation)

#### 2.1.2 Separate Output Heads

From the shared features, we have two separate output heads:

**f(x) Head:**
$$f(x) = W_f \cdot h(x) + b_f$$

**g(x) Head:**
$$g_{flat}(x) = W_g \cdot h(x) + b_g$$

The g(x) output is flattened and needs to be reshaped:
$$g(x) = \text{reshape}(g_{flat}(x), [n, m])$$

Where:
- $W_f, W_g$ are weight matrices for the respective output heads
- $b_f, b_g$ are bias vectors
- $g_{flat}(x)$ is the flattened output before reshaping
- $n$ is the state dimension and $m$ is the action dimension

### 2.2 Architectural Design Considerations

1. **Shared Base Network**: By sharing the feature extraction layers, we leverage common representations between $f(x)$ and $g(x)$ since they both depend on the state. This reduces the number of parameters and improves generalization.

2. **Sparse Output Layers**: The separate output heads for $f(x)$ and $g(x)$ maintain the control affine structure while allowing each component to capture different aspects of the dynamics.

3. **Activation Functions**: We use hyperbolic tangent (tanh) activations in the hidden layers because:
   - They provide nonlinearity necessary for modeling complex dynamics
   - Output range is bounded between -1 and 1, providing numerical stability
   - The smooth gradient helps with optimization

4. **Output Layer Linearity**: No activation function is applied to the output layers, allowing $f(x)$ and $g(x)$ to take any real values, as required for general dynamical systems.

## 3. Ensemble Method for Uncertainty Quantification

### 3.1 Ensemble Architecture

Our approach uses an ensemble of multiple independent control affine networks to quantify epistemic uncertainty (uncertainty due to limited data):

```
               Input: State x, Action u
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
     ┌──────────────┐┌──────────────┐┌──────────────┐
     │ Control      ││ Control      ││ Control      │
     │ Affine       ││ Affine       ││ Affine       │  ...
     │ Network 1    ││ Network 2    ││ Network 3    │
     └──────┬───────┘└──────┬───────┘└──────┬───────┘
            │               │               │
            ▼               ▼               ▼
      ┌─────────┐     ┌─────────┐     ┌─────────┐
      │  ẋ₁     │     │  ẋ₂     │     │  ẋ₃     │  ...
      └────┬────┘     └────┬────┘     └────┬────┘
           │               │               │
           └───────────────┼───────────────┘
                           │
                           ▼
             ┌───────────────────────┐
             │ Statistical Analysis  │
             │ (Mean, Variance, CI)  │
             └───────────────────────┘
```

### 3.2 Mathematical Basis for Ensembles

Each network in the ensemble ($i = 1, 2, ..., K$) produces its own prediction:

$$\dot{x}_i = f_i(x) + g_i(x)u$$

#### 3.2.1 Training Ensembles

Each network is initialized with different random weights, creating diversity in the ensemble. During training, all networks see the same data but learn slightly different mappings due to:

1. Different initializations leading to different local optima
2. Stochastic elements in the optimization process (e.g., mini-batch selection)

The loss function for each network is:

$$\mathcal{L}_i = \frac{1}{N} \sum_{j=1}^{N} \| \hat{x}_{i,j+1} - x_{j+1} \|^2$$

Where:
- $\hat{x}_{i,j+1}$ is the prediction from model $i$ for sample $j$
- $x_{j+1}$ is the true next state for sample $j$
- $N$ is the number of samples in the batch

The overall ensemble loss is the average of individual model losses:

$$\mathcal{L}_{ensemble} = \frac{1}{K}\sum_{i=1}^{K} \mathcal{L}_i$$

#### 3.2.2 Statistical Measures from Ensembles

**Mean Prediction:**
$$\bar{\dot{x}} = \frac{1}{K} \sum_{i=1}^{K} \dot{x}_i$$

**Variance (Uncertainty Measure):**
$$\sigma^2(\dot{x}) = \frac{1}{K} \sum_{i=1}^{K} (\dot{x}_i - \bar{\dot{x}})^2$$

This variance captures disagreement between models, serving as a measure of epistemic uncertainty. High variance indicates areas where data is sparse or the dynamics are complex.

**Confidence Intervals:**
For a confidence level $\alpha$ (e.g., 95%), we compute the confidence interval for the mean prediction:

$$CI_{\alpha} = \bar{\dot{x}} \pm z_{\alpha/2} \cdot \frac{\sigma(\dot{x})}{\sqrt{K}}$$

Where $z_{\alpha/2}$ is the z-score corresponding to the desired confidence level (e.g., 1.96 for 95% CI).

### 3.3 Theoretical Justification for Ensemble Uncertainty

Ensembles approximate Bayesian inference without the computational complexity of full Bayesian methods:

1. **Bayesian Perspective**: Each model in the ensemble can be viewed as a sample from the posterior distribution over model parameters.

2. **Diversity vs. Consensus**: In data-rich regions, models tend to agree (low uncertainty), while in data-poor regions, models make different predictions (high uncertainty).

3. **Statistical Guarantees**: As the ensemble size increases, the uncertainty estimates become more reliable. Our implementation uses 5 models as a practical compromise between computational efficiency and statistical validity.

4. **Uncertainty Types**:
   - **Aleatoric Uncertainty**: Inherent randomness in the system (not explicitly modeled in our approach)
   - **Epistemic Uncertainty**: Uncertainty due to limited knowledge, which decreases as more data is collected

## 4. How Our Implementation Differs from Traditional Approaches

### 4.1 Single Network for Control Affine Structure

Traditional approaches often use:
- Separate networks for $f(x)$ and $g(x)$ with no shared parameters
- Generic neural networks that don't respect the control affine structure

Our approach:
- Uses a single network with shared representations and sparse output layers
- Explicitly models the control affine structure
- Reduces parameter count while maintaining expressive power

### 4.2 Ensemble vs. Alternative Uncertainty Methods

Alternative approaches to uncertainty quantification include:
- **Bayesian Neural Networks**: Use probability distributions over weights
- **Dropout as Bayesian Approximation**: Apply dropout during inference to sample different models
- **Gaussian Processes**: Non-parametric models with built-in uncertainty

Our ensemble approach offers several advantages:
- Simpler implementation and training compared to Bayesian methods
- More computationally efficient than Gaussian Processes for high-dimensional problems
- Can capture complex uncertainty patterns and multimodal distributions
- Allows parallel training and inference for increased performance

### 4.3 Incremental Learning with FIFO Buffer

Our implementation supports incremental learning with a configurable FIFO (First-In-First-Out) buffer:

- **FIFO Length Parameter**: Controls how many trajectories are retained for training
  - `fifo_length=1`: Only the most recent trajectory is used for training (equivalent to online learning with forgetting)
  - `fifo_length>1`: Multiple recent trajectories are retained in a sliding window approach

This approach offers several advantages:

1. **Memory Efficiency**: Limits memory usage by only storing a fixed number of trajectories
2. **Recency Bias**: Emphasizes learning from more recent experiences
3. **Forgetting Mechanism**: Automatically discards older data that may be less relevant
4. **Adaptability**: Adjusts the balance between stability and adaptability through the `fifo_length` parameter

The FIFO mechanism is implemented as:

```
Algorithm: FIFO Buffer Update
1. Collect new trajectory
2. Add trajectory to buffer
3. If buffer size > fifo_length:
   a. Remove oldest trajectory from buffer
4. Train model using all trajectories currently in buffer
```

This allows studying how learning and uncertainty evolve under different data retention strategies:
- Small buffer lengths show how quickly the model can adapt to new data
- Larger buffer lengths provide stability and help prevent catastrophic forgetting
- The optimal buffer size depends on the specific dynamics being learned and the rate of exploration

## 5. Advanced Mathematical Analysis

### 5.1 Function Approximation Theory

Neural networks serve as universal function approximators. For control affine systems, we're approximating:

$$f(x) \approx \hat{f}(x; \theta_f)$$
$$g(x) \approx \hat{g}(x; \theta_g)$$

Where $\theta_f$ and $\theta_g$ represent the network parameters. The approximation error decreases as network capacity and data increase.

### 5.2 Uncertainty Reduction with Data

The epistemic uncertainty in an ensemble model decreases as more data is collected, following a relationship similar to:

$$\sigma^2(\dot{x}) \propto \frac{1}{N_{effective}}$$

Where $N_{effective}$ is the effective sample size, which depends on the data distribution and model complexity.

### 5.3 Confidence Interval Interpretation

Our confidence intervals follow the standard statistical interpretation:
- A 95% confidence interval means that if we repeated the experiment many times, 95% of the intervals would contain the true value
- The width of the confidence interval provides a quantitative measure of prediction uncertainty
- As more data is collected, confidence intervals typically narrow

## 6. Experimental Design and Evaluation

Our implementation allows extensive experimentation to study:

1. **Learning Efficiency**: How quickly the model learns accurate dynamics
2. **Uncertainty Reduction**: How uncertainty decreases with more data
3. **Ensemble Effectiveness**: How ensemble size affects uncertainty estimates
4. **Control Affine Structure**: Benefits of the control affine formulation vs. generic models
5. **Incremental Learning**: Performance of different buffer strategies

These experiments help validate the theoretical advantages of our approach and guide practical applications of learned dynamics models.
