# Methods Summary
This serves as a collection of mathematical formulations, derivations, and justifications for all choices made in the production of this project. See the table of contents for specific inclusions.

### Table of Contents

1. [Model Choices](#model-choices)
   - [Logistic Regression](#logistic-regression)
     - [Loss](#loss)
     - [Gradient Derivation](#gradient-derivation)
     - [Solving](#solving)
     - [Expected Performance](#expected-performance)
   - [Random Forest](#random-forest)
     - [Splitting](#splitting)
     - [Variance Reduction](#variance-reduction)
     - [Expected Performance](#expected-performance-1)
   - [Support Vector Machines](#support-vector-machines)
     - [Loss](#loss-1)
     - [Dual Derivation](#dual-derivation)
     - [Platt Scaling](#platt-scaling)
     - [Loss](#loss-2)
     - [Gradient Derivation](#gradient-derivation-1)
     - [Expected Performance](#expected-performance-2)
   - [Single Hidden Layer MLP](#single-hidden-layer-mlp)
     - [Loss](#loss-3)
     - [Gradient Derivation](#gradient-derivation-2)
     - [Solving](#solving-1)
     - [Expected Performance](#expected-performance-3)
     - [Focal Loss](#focal-loss)
     - [Loss](#loss-4)
     - [Gradient Derivation](#gradient-derivation-3)
     - [Solving](#solving-2)
     - [Expected Performance](#expected-performance-4)
2. [Evaluation Metrics](#evaluation-metrics)
   - [Accuracy](#accuracy)
     - [Threshold Invariance](#threshold-invariance)
   - [Area Under the Receiver Operating Characteristic Curve](#area-under-the-receiver-operating-characteristic-curve)
     - [Rank Interpretation Derivation](#rank-interpretation-derivation)
     - [Interpretation](#interpretation)
   - [Log Loss](#log-loss)
     - [Proper Scoring Rule Derivation](#proper-scoring-rule-derivation)
     - [Interpretation](#interpretation-1)
   - [Brier Score](#brier-score)
     - [Proper Scoring Rule Derivation](#proper-scoring-rule-derivation-1)
     - [Murphy Decomposition](#murphy-decomposition)
     - [Interpretation](#interpretation-2)
   - [Reliability Diagrams](#reliability-diagrams)
     - [Expected Calibration Error](#expected-calibration-error)
     - [Interpretation](#interpretation-3)
3. [Feature Derivations](#feature-derivations)
   - [Rating Systems](#rating-systems)
     - [Massey Rating](#massey-rating)
     - [Colley Rating](#colley-rating)
   - [Momentum Features](#momentum-features)
4. [Statistical Tests](#statistical-tests)
   - [McNemar's Test](#mcnemars-test)
     - [Derivation](#derivation)
     - [Interpretation](#interpretation-4)
   - [Paired t-Test](#paired-t-test)
     - [Derivation](#derivation-1)
     - [Interpretation](#interpretation-5)
   - [Wilcoxon Signed Rank Test](#wilcoxon-signed-rank-test)
     - [Derivation](#derivation-2)
     - [Interpretation](#interpretation-6)
   - [Holm-Bonferroni Correction](#holm-bonferroni-correction)
     - [Procedure](#procedure)
     - [Derivation](#derivation-3)
     - [Interpretation](#interpretation-7)

## Model Choices

### Logistic Regression
We include logistic regression as a baseline linear model because it gives a well-calibrated, directly interpretable probability estimate against which more complex models can be measured. The coefficients of logistic regression also provide an accessible interpretation in 'log odds per unit feature change' for validating signs and magnitudes.

We define the logistic regression model as the weights $w \in \mathbb R ^d$ and bias $b \in \mathbb R$ such that for all $n$ training examples $x_i$ and labels $y_i \in {0,1}$

$$ 
P(y_i = 1 \mid x_i) = \sigma(w^T x_i + b) \\
0 < i \leq n
$$

#### Loss
Since $n>d$ we want $(w,b)$ that maximize the likelihood of the observed data, so we use MLE. Because $y_i \in {0,1}$, we model $Y$ as Bernoulli with probability

$$P(y_i = 1 \mid x_i) = \sigma(w^T x_i + b)$$

Therefore the probability mass function of $Y_i$ is

$$P(Y_i = y_i) = \sigma(w^T x_i + b)^{y_i}(1-\sigma(w^T x_i + b))^{1-y_i}$$

Assume the $n$ examples are i.i.d. Bernoulli, then the likelihood of the observed data is

$$L(w,b) = \prod_{i=1}^{n} \sigma(w^T x_i + b)^{y_i}(1-\sigma(w^T x_i + b))^{1-y_i}$$

Taking the negative log for a minimization target rather than maximum likelihood yields

$$\mathcal{L}(w,b) = -\sum_{i=1}^{n} \left(y_ilog(\sigma(w^T x_i + b)) + (1-y_i)log(\sigma(w^T x_i + b))\right)$$

Which is exactly binary cross entropy loss.

#### Gradient Derivation
Let $z_i = w^T x_i + b$ the logit of $x_i$. We first recall the derivative of the sigmoid function

$$\sigma^\prime(a) = \sigma(a)(1-\sigma(a))$$

Then the derivative of the per example loss $l_i$ with respect to $z_i$ is

$$ \begin{aligned}
l_i &= -y_ilog(\sigma(z_i)) - (1-y_i)log(\sigma(z_i)) \\ \\
\frac{\partial l_i}{\partial z_i} &= -y_i\frac{1}{\sigma(z_i)}\sigma{z_i}(1-\sigma(z_i)) + (1-y_i)\frac{1}{1 - \sigma(z_i)}\sigma{z_i}(1-\sigma(z_i)) \\ \\
&= \sigma(z_i)(1-\sigma(z_i))\left(\frac{-y_i(1-\sigma({z_i})) + (1-y_i)\sigma(z_i)}{\sigma(z_i)(1-\sigma(z_i))}\right) \\ \\
&= -y_i(1-\sigma({z_i})) + (1-y_i)\sigma(z_i) \\ \\
&= \sigma(z_i) - y_i
\end{aligned}$$

Now returning to the complete loss function, substituting $z_i = w^T x_i + b$

$$ \begin{aligned}
\mathcal{L}(w,b) &= \sum_{i=1}^{n} l_i(w,b) \\ \\
\nabla_{w} \mathcal{L} &= \sum_{i=1}^{n} \frac{\partial l_i}{\partial z_i} \frac{\partial z_i}{\partial w} \\ \\
&= \sum_{i=1}^{n} (\sigma(w^T x_i + b) - y_i)x_i \\ \\
\frac{\partial \mathcal{L}}{\partial b} &= \sum_{i=1}^{n} \frac{\partial l_i}{\partial z_i} \frac{\partial z_i}{\partial w} \\ \\
&= \sum_{i=1}^{n} (\sigma(w^T x_i + b) - y_i)
\end{aligned}$$

#### Solving
Reviewing the BCE loss function, we note that $\mathcal{L}$ as a function of $(w,b)$ is the sum of compositions of the log of sigmoids with affine maps. Since the log of a sigmoid is a convex function, by convexity preserving operations composing with an affine map is also convex, and sums of convex maps are still convex. Therefore $\mathcal{L}$ is convex in $(w,b)$. Becuase $\mathcal{L}$, if a minimizer exists it is a global minimum. Since $\mathcal{L}$ is continuous and coercive, there must exist a unique global minimizer. Therefore, gradient decent will converge to the global minimum with the correct learning rate, but the convexity of $\mathcal{L}$ and existence of the hessian also allows us to use Newton's methods for quadratic convergence or more commonly quassi-Newton methods that approximate the hessian.

#### Expected Performance
The decision boundary for for a classification threshold of $0.5$ is the hyperplane

$$w^Tx + b = 0$$

because $\sigma(0) = .5$. We therefore expect that logistic regression will under fit any interaction or nonlinear effect between features. As such it provides an effective baseline performance and can be used to infer the linearity of the true boundary.

### Random Forest
We include a random forest because our feature set mixes fundamentally different scales and structures and because we expect thresholded or interaction effects between them that a hyperplane decision boundary cannot represent. Unlike logistic regression, the random forest imposes no parametric assumption on the boundary shape at all

Let $n$ be the number of training examples and $d$ be the number of features. For binary classification, we define a single decision tree $f \colon \mathbb R^d \to [0,1]$ as a recursive partition of $\mathbb R^d$ into disjoint regions $R_1,\dots,R_M$, each with a constant prediction $\hat p_m$. The random forest is an ensemble of $T$ such trees $\{f_i\}_{i=1}^T$ where each tree is fit to a bootstrap sample of size $n$ drawn with replacement from the training data. The ensemble prediction is then computed through soft voting on all output probabilities
 
$$\hat f(x) = \frac{1}{T}\sum_{i=1}^{T} f_i(x)$$

#### Splitting
The training features are all continuous values so to grow each individual tree we determine splits by evaluating the change in impurity of a possible split on feature $j$ at threshold $tao$ as a measure of the class homogeneity of the current set of the training data. For a node with sample set $S$ and unique classes $K$, we define the Gini impurity
 
$$G(S) = 1 - \sum_{c \in K} \hat p_c^2 \qquad \hat p_c = \frac{1}{|S|}\sum_{i \in S} \mathbf{1}_{y_i = c}$$
 
which is minimized ($G=0$) when $S$ is pure and maximized when classes are equally mixed. For binary classification specifically, the Gini impurity simplifies to

$$G(S) = 2p(1-p) \qquad p = \frac{1}{|S|}\sum_{i \in S} \mathbf{1}_{y_i = 1}$$

where $p$ is the proportion of positive labels in the subset $S$. A candidate split on feature $j$ at threshold $\tau$ partitions $S$ into left and right subsets

$$S_L = \{i \in S : x_{ij} \le \tau\} \qquad S_R = S \setminus S_L$$

At each node, rather than considering all $d$ features, a random subset of features is sampled uniformly without replacement, and the best split is selected only from this subset. This additional randomization decorrelates the trees and reduces ensemble variance. The objective at each node is to choose $(j,\tau)$ maximizing the reduction in Gini impurity and producing the purest child nodes
 
$$\Delta G(j,\tau) = G(S) - \frac{|S_L|}{|S|}G(S_L) - \frac{|S_R|}{|S|}G(S_R)$$
 
This is evaluated greedily always selecting the highest decrease at each current node and recursively at every node until a stopping criterion (maximum depth, minimum leaf size, or $\Delta G = 0$) is reached. Note that unlike $\mathcal{L}(w,b)$ above, $\Delta G$ is not differentiable in $(j,\tau)$, since $\tau$ enters only through an indicator function. There is therefore no gradient to derive, and the optimization at each node is instead a discrete search over the finite set of candidate feature-threshold pairs obtained from consecutive ordered feature values in $S$.

#### Variance Reduction
A random forest is separated from single decision trees or bootstrap aggregation by the ensembling and feature subsampling steps. We seek to formally justify the use of a random forest through the framework of the bias variance trade off. 

A single tree grown to low bias has high variance because small perturbations of $\mathcal{D}$ can change the induced partition substantially, since a split near the top of the tree changes every partition beneath it. Let $D$ denote the random bootstrap dataset obtained by sampling $n$ times from the $n$ observed data. Let $J$ denote the sequence of random feature subsets sampled at each node without replacement. Both $D$ and $J$ are random elements. We formalize each fitted tree as a random variable

$$f_i: X \times \Omega \to [0,1]$$

where $X$ is the feature space and $\Omega$ is the joint sample space for all outcomes of $(D,J)$. Each tree is constructed through a deterministic algorithm on the random parameters from the same distribution, so all trees are identically distributed with marginal variance $\mathrm{Var}(f_i) = \sigma^2$. Although the bootstrap samples and feature subset selections are generated independently across trees conditional on the observed training data, the resulting predictions need not be independent. All trees are constructed from the same empirical training distribution and therefore tend to respond similarly to the same underlying structure, inducing positive correlation between their predictions. We assume every pair of trees has the same pairwise correlation $\rho = \mathrm{Corr}(f_i,f_{j})$ for $i \neq j$. The covariance between any two trees is thus

$$ \begin{aligned}
\text{Cov}(f_i,f_j) &= \text{Corr}(f_i,f_j)\sqrt{\text{Var}(f_i)\text{Var}(f_j)} \\
&= \rho \sigma^2
\end{aligned}$$

for each of the $T(T-1)$ pairs where $i \neq j$ Then for the averaged ensemble
 
$$\begin{aligned}
\mathrm{Var}(\hat f) &= \mathrm{Var}\left(\frac{1}{T}\sum_{i=1}^{T} f_i\right) \\ \\
&= \frac{1}{T^2}\left(\sum_{i=1}^{T}\mathrm{Var}(f_i) + \sum_{i \neq j}\mathrm{Cov}(f_i,f_{j})\right) \\ \\
&= \frac{1}{T^2}\left(T\sigma^2 + T(T-1)\rho\sigma^2\right) \\ \\
&= \rho\sigma^2 + \frac{1-\rho}{T}\sigma^2
\end{aligned}$$

As $T \to \infty$, the second term goes to zero leaving model variance $\rho\sigma^2$ that cannot be reduced by adding more trees. In a random forest, we sample $m<d$ candidate features at each node because forcing trees to split on different features lowers the correlation between them since they use less of the same features, lowering $\rho$. Feature subsampling can also increase the bias of individual trees because each tree is restricted from considering potentially useful features at each split. The ensemble benefits when the reduction in inter-tree covariance outweighs the resulting increase in individual-tree error.

#### Expected Performance
Each individual tree partitions feature space into axis-aligned rectangular regions (assuming ordinary axis-aligned splits). Within each leaf, $f_i(x)=\hat p_m$ is constant. The forest averages these piecewise constant functions. Therefore the forest prediction is itself piecewise constant over a collection of axis aligned regions.

This means the forest can approximate any boundary shape given enough splits but only piecewise. We therefore expect it to outperform logistic regression whenever the true boundary is nonlinear or threshold like. We expect it to underperform a model with a smooth decision boundary like an SVM or MLP only if the true boundary is smooth and if we have enough data to estimate that smoothness directly, since a piecewise-constant is an inefficient approximation of a smooth function.

### Support Vector Machines
We include a support vector machine as the maximum margin alternative to logistic regression. Rather than modeling $P(y_i=1 \mid x_i)$ directly, the SVM optimizes the geometric margin between the two classes, which statistical learning theory suggests should generalize well when the classes are close to separable and $n$ is not overwhelmingly large relative to $d$, unlike the random forest above which makes no such separability assumption at all.
 
Let $y_i \in \{-1,+1\}$ rather than $\{0,1\}$, since the margin constraints below are more naturally stated with a sign. We define the SVM as the weights $w \in \mathbb R^d$, bias $b \in \mathbb R$, and slack variables $\xi_i \geq 0$ that together solve
 
$$
\min_{w,b,\xi} \ \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i \\ \\ 
\text{s.t.} \quad y_i(w^Tx_i+b) \geq 1-\xi_i, \\ \\ 
\xi_i \geq 0, \\ \\ 
1\leq i \leq n
$$
 
where the slack $\xi_i$ allows point $i$ to fall inside or across the margin, and $C$ is a tunable hyperparameter controlling how harshly we penalize this.
 
#### Loss
Unlike logistic regression, the SVM objective is not derived from an MLE argument but rather a geometric argument. 

Under the normalization $y_i(w^Tx_i+b) \geq 1$, the two margin hyperplanes $w^Tx+b=\pm1$ are separated by distance

$$\frac{|1 - (-1)|}{||w||} = \frac{2}{||w||}$$

so maximizing the margin between classes is equivalent to minimizing $\frac{1}{2}||w||^2$. In the soft-margin formulation, this margin objective is traded off against violations through the slack penalty. At the optimum, 

$$\xi_i = \max(0, 1-y_i(w^Tx_i+b))$$

exactly, since a smaller $\xi_i$ would violate the constraint $y_i(w^Tx_i+b)\geq 1-\xi_i$ and a larger $\xi_i$ would only increase the objective unnecessarily. Substituting this back into the objective eliminates $\xi$ entirely and gives the equivalent unconstrained form
 
$$\mathcal{L}(w,b) = \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\max\big(0,\ 1-y_i(w^Tx_i+b)\big)$$
 
which is the hinge loss. The key qualitative difference from the binary cross entropy loss is that the hinge term is exactly zero once a point is classified correctly beyond the margin, so points the model already classifies confidently stop contributing to $\mathcal{L}$ altogether, whereas cross entropy continues to reward increasing confidence on orrectly classified points indefinitely.
 
#### Dual Derivation
We derive the dual problem here rather than a gradient of $\mathcal{L}(w,b)$ directly, because it is the dual, not the primal, that exposes the support vectors and permits the kernel trick used to fit nonlinear boundaries. Introducing Lagrange multipliers $\lambda_i \geq 0$ for the margin constraints and $\mu_i \geq 0$ for $\xi_i \geq 0$, the Lagrangian of the original constrained problem is
 
$$L(w,b,\xi,\lambda,\mu) = \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i - \sum_{i=1}^{n}\lambda_i\big[y_i(w^Tx_i+b) - 1+\xi_i\big] - \sum_{i=1}^{n}\mu_i\xi_i$$

The primal problem is convex since the norm is convex and all constraints are affine. We verify Slater's constraint by choosing the feasible point $w = 0, \ b=0, \ \xi_i = 2$ 

$$ 
0\geq -1 \\ \\ 
2 \geq 0
$$

Therefore KKT is necessary and sufficient for the optimal point and strong duality holds. The saddle point of $L$ give the exact solution so setting the stationarity conditions $\nabla_w L = 0$, $\partial L / \partial b = 0$, and $\partial L/\partial \xi_i = 0$,
 
$$ \begin{aligned}
\nabla_w L = w - \sum_{i=1}^{n}\lambda_iy_ix_i = 0 \qquad & \Rightarrow \ w = \sum_{i=1}^{n}\lambda_iy_ix_i \\ \\
\frac{\partial L}{\partial b} = -\sum_{i=1}^{n}\lambda_iy_i = 0 \qquad& \Rightarrow \ \sum_{i=1}^{n}\lambda_iy_i = 0 \\ \\
\frac{\partial L}{\partial \xi_i} = C - \lambda_i - \mu_i = 0 \qquad& \Rightarrow \ \lambda_i = C-\mu_i \leq C
\end{aligned}$$
 
Substituting back into $L$ removes $w,b,\xi$ from the problem entirely and leaves the dual problem
 
$$\max_{\lambda} \ \sum_{i=1}^{n}\lambda_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\lambda_i\lambda_jy_iy_j x_i^Tx_j \\ \\
\text{s.t.} \quad 0 \leq \lambda_i \leq C, \ \sum_{i=1}^{n}\lambda_iy_i=0$$
 
By complementary slackness, $\lambda_i>0$ only when the constraint is active. Since the constraint of our problem is the margin, the points sitting on or inside the margin are the so called support vectors. Further, $w$ is a sparse combination of the training data and the training points with $\lambda_i=0$ do not directly contribute to the optimal weight vector $w$. Because the training data enter the dual only through the pairwise inner products $x_i^Tx_j$, we can replace $x_i^Tx_j$ with a kernel $k(x_i,x_j) = \phi(x_i)^T\phi(x_j)$ for some feature map $\phi$ and apply the kernel trick, which lets us fit a nonlinear boundary in the space induced by $\phi$ while only ever having to evaluate $k$, never $\phi$ itself.
 
Both the primal hinge-loss form and this dual are convex programs. The primal is a quadratic objective under linear constraints, and the dual is a concave quadratic maximized over a box constraint intersected with a single linear equality, since $y_iy_jx_i^Tx_j$ forms a positive semidefinite Gram matrix. A global maximizer therefore exists by the same argument used for logistic regression, and in practice is found by quadratic programming solvers such as sequential minimal optimization, which updates pairs of $\lambda_i$ analytically rather than solving the full $n\times n$ program at once.
 
#### Platt Scaling
Because the SVM's raw score $f(x) = w^Tx+b$ is not a probability, we cannot use it directly wherever a calibrated $P(y=1\mid x)$ is required, for instance in downstream expected-value calculations. We instead fit a second, one dimensional model on top of the SVM's score to recover a probability.
 
We define the Platt scaling model as scalar parameters $c \in \mathbb R$ and $d \in \mathbb R$ such that for the SVM score $f(x_i)$
 
$$P(y_i = 1 \mid x_i) = \sigma\big(cf(x_i) + d\big)$$
 
fit on a held out calibration set $\{(f(x_i),y_i)\}_{i=1}^{m}$ that is distinct from the SVM's own training set, since calibrating on points the margin itself was fit to would bias $c,d$ toward overconfidence.
 
This is exactly the logistic regression problem above with $f(x_i)$ standing in for $x_i$ and a single scalar weight $c$ standing in for $w$, so the same MLE argument gives a negative log likelihood of
 
#### Loss
 
$$\mathcal{L}(c,d) = -\sum_{i=1}^{m}\left(t_ilog(\sigma(cf(x_i)+d)) + (1-t_i)log(1-\sigma(cf(x_i)+d))\right)$$
 
with one modification: rather than regressing on the raw labels $y_i$ directly, we use a regularized target
 
$$t_i = \frac{y_i^+ + 1}{y_i^+ + y_i^- + 2}$$
 
where $y^+,y^-$ are the total counts of positive and negative examples in the calibration set. The application of Laplace's Rule of Succession keeps $c,d$ from being fit to overconfident $0/1$ targets on the typically a small calibration set.
 
#### Gradient Derivation
Since $\mathcal{L}(c,d)$ has the identical cross entropy form as logistic regression, the gradient derivation carries over term for term and gives
 
$$\frac{\partial \mathcal{L}}{\partial c} = \sum_{i=1}^{m}\big(\sigma(cf(x_i)+d) - t_i\big)f(x_i), \qquad \frac{\partial \mathcal{L}}{\partial d} = \sum_{i=1}^{m}\big(\sigma(cf(x_i)+d) - t_i\big)$$
 
and by the same convexity argument as logistic regression, $\mathcal{L}(c,d)$ is convex in $(c,d)$, so a global minimizer exists and is found cheaply since $(c,d)$ is only two scalars.
 
#### Expected Performance
With a linear kernel we expect performance close to logistic regression, but with potentially better generalization on a small or noisy training set, since the margin objective specifically stops penalizing points once they are well classified, whereas cross entropy keeps rewarding increasing confidence on the same points indefinitely. With an RBF kernel we expect the SVM to compete with the random forest above on nonlinear structure, at the cost of $O(n^2)$ to $O(n^3)$ scaling in the Gram matrix and an additional hyperparameter search over $C$ and the kernel bandwidth.

Platt scaling only applies a monotonic reshaping to the SVM's score, so it cannot change the SVM's ranking of examples and therefore cannot change its ROC or AUC, only the mapping from score to probability. We expect it to matter when calibrated probabilities are consumed downstream, and to matter most when the SVM's margin scores are visibly non sigmoidal in shape, which is exactly the case when classes are close to separable and scores run to extremes.

### Single Hidden Layer MLP
We include a single hidden layer multilayer perceptron because, by the universal approximation theorem, an MLP with one sufficiently wide hidden layer and a nonlinear activation can approximate any continuous function on a compact domain to arbitrary accuracy. It therefore serves as a check on whether the linear and tree based models above are leaving nonlinear signal on the table.
 
Let $x \in \mathbb R^d$ be the input, $h$ the hidden width, $W^{(1)} \in \mathbb R^{h \times d}$ and $b^{(1)} \in \mathbb R^h$ the hidden layer parameters, $W^{(2)} \in \mathbb R^{1 \times h}$ and $b^{(2)} \in \mathbb R$ the output layer parameters, and $\phi$ an elementwise nonlinear activation. We define the model's forward pass as
 
$$
z^{(1)} = W^{(1)}x + b^{(1)}, \quad a^{(1)} = \phi(z^{(1)}), \quad z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}, \quad \hat p = \sigma(z^{(2)})
$$
 
Without the activation function $\phi$, the composition of two affine maps $z^{(2)} = W^{(2)}(W^{(1)}x+b^{(1)})+b^{(2)}$ is itself affine in $x$, so the model would collapse back to logistic regression regardless of how large $h$ is.
 
#### Loss
We again take $y_i \in \{0,1\}$ and model $Y_i$ as Bernoulli with probability $\hat p_i$, so by the identical MLE argument used for logistic regression,
 
$$\mathcal{L}\big(W^{(1)},b^{(1)},W^{(2)},b^{(2)}\big) = -\sum_{i=1}^{n}\left(y_ilog(\hat p_i) + (1-y_i)log(1-\hat p_i)\right)$$
 
which is again exactly binary cross entropy, now composed with the two layer forward pass above rather than a single affine map.
 
#### Gradient Derivation
Let $\delta_i^{(2)} = \partial l_i / \partial z_i^{(2)}$. The output layer of the MLP is structurally identical to logistic regression applied to $a_i^{(1)}$ in place of $x_i$, so the same derivative computed above applies directly without rederivation,
 
$$\delta_i^{(2)} = \hat p_i - y_i$$
 
Backpropagating this error through the output weights by the chain rule,
 
$$\frac{\partial l_i}{\partial W^{(2)}} = \delta_i^{(2)}\big(a_i^{(1)}\big)^T, \qquad \frac{\partial l_i}{\partial b^{(2)}} = \delta_i^{(2)}$$
 
To propagate the error back to the hidden layer we apply the chain rule once more, through $z_i^{(2)} = W^{(2)}a_i^{(1)}+b^{(2)}$ and then $a_i^{(1)} = \phi(z_i^{(1)})$,
 
$$ \begin{aligned}
\delta_i^{(1)} &= \frac{\partial l_i}{\partial z_i^{(1)}} \\ \\
&= \frac{\partial l_i}{\partial z_i^{(2)}}\frac{\partial z_i^{(2)}}{\partial a_i^{(1)}}\frac{\partial a_i^{(1)}}{\partial z_i^{(1)}} \\ \\
&= \big(W^{(2)}\big)^T \delta_i^{(2)} \odot \phi^\prime(z_i^{(1)})
\end{aligned}$$
 
where $\odot$ denotes the elementwise Hadamard product, since $a_i^{(1)}=\phi(z_i^{(1)})$ is applied elementwise, and $\phi^\prime$ is the elementwise activation derivative, for example $\phi^\prime(z) = \mathbf{1}_{z>0}$ for ReLU. Applying the chain rule a final time through $z_i^{(1)} = W^{(1)}x_i+b^{(1)}$ gives the remaining two gradients,
 
$$\frac{\partial l_i}{\partial W^{(1)}} = \delta_i^{(1)}x_i^T, \qquad \frac{\partial l_i}{\partial b^{(1)}} = \delta_i^{(1)}$$
 
Computing these four gradients in this order, output layer first and hidden layer second, rather than differentiating $\mathcal{L}$ with respect to $W^{(1)}$ directly, is precisely the backpropagation algorithm, and is what a gradient based optimizer such as Adam requires at each step.
 
#### Solving
Unlike $\mathcal{L}(w,b)$ for logistic regression, $\mathcal{L}$ here is a function of $W^{(1)},b^{(1)},W^{(2)},b^{(2)}$ jointly, and $z_i^{(2)} = W^{(2)}\phi(W^{(1)}x_i+b^{(1)})+b^{(2)}$ contains a product of $W^{(2)}$ with a nonlinear function of $W^{(1)}$. This product structure breaks the convexity preserving composition argument used for logistic regression, since a product of two non-constant functions of the same variables is not in general convex, so $\mathcal{L}$ is non-convex in the full parameter set here. Gradient descent is therefore only guaranteed to converge to a stationary point rather than a global minimum, and which stationary point is reached depends on the random initialization of $W^{(1)},W^{(2)}$. In practice we address this with multiple random restarts and with regularization, such as weight decay, dropout, or early stopping, to reduce sensitivity to the particular local minimum reached rather than attempting to certify global optimality the way we could for logistic regression or the SVM.
 
#### Expected Performance
Because $\phi$ is applied continuously rather than through the indicator function splits of the random forest above or the fixed kernel form of the SVM, we expect the MLP to represent smooth decision boundaries more efficiently than either, and to match or exceed both whenever the true boundary is smooth rather than piecewise axis-aligned. Given the comparatively small size of a single sports season dataset relative to typical deep learning training sets, the model's flexibility is also its chief risk: with a wide hidden layer and without adequate regularization we would expect it to overfit and underperform the random forest despite having greater representational capacity, which is the same bias-variance trade-off invoked to justify feature subsampling in the random forest, just working against us here instead of for us.
 
#### Focal Loss
Sports outcome data is frequently imbalanced in a soft sense, since heavy favorites win the large majority of their games, so under plain cross entropy the large mass of easy, confidently and correctly classified examples dominates the sum in $\mathcal{L}$ and contributes comparatively little useful gradient, while the harder, closer games that we most want the model to get right contribute a proportionally smaller share. Focal loss modifies the MLP's output loss to reweight the sum toward these harder examples.
 
Let $p_{t,i}$ denote the predicted probability assigned to the true class of example $i$,
 
$$
p_{t,i} = \begin{cases} \hat p_i & y_i = 1 \\ 1-\hat p_i & y_i = 0 \end{cases}
$$
 
so that ordinary cross entropy can be written $l_i = -log(p_{t,i})$. We define the focal loss with focusing parameter $\gamma \geq 0$ and class weight $\alpha_{t,i} \in [0,1]$ as
 
$$l_i = -\alpha_{t,i}(1-p_{t,i})^\gamma log(p_{t,i})$$
 
#### Loss
Setting $\gamma=0$ and $\alpha_{t,i}=1$ recovers $l_i=-log(p_{t,i})$ exactly, so focal loss is a direct generalization of cross entropy loss rather than a different objective entirely. As $p_{t,i}\to 1$, meaning example $i$ is confidently and correctly classified, the factor $(1-p_{t,i})^\gamma \to 0$, so $l_i \to 0$ faster than ordinary cross entropy's $l_i=-log(p_{t,i})$ does, meaning easy examples contribute less to the overall loss sum and forcing the model to learn the harder examples that have more significant loss contributions.
 
#### Gradient Derivation
Consider first $y_i=1$, so $p_{t,i}=\hat p_i = \sigma(z_i)$. The $y_i=0$ case follows symmetrically. Differentiating with respect to $\hat p_i$ by the product rule, treating $\alpha$ as constant,
 
$$ \begin{aligned}
\frac{\partial l_i}{\partial \hat p_i} &= -\alpha\left[\gamma(1-\hat p_i)^{\gamma-1}(-1)log(\hat p_i) + (1-\hat p_i)^\gamma \frac{1}{\hat p_i}\right] \\ \\
&= \alpha\left[\gamma(1-\hat p_i)^{\gamma-1}log(\hat p_i) - \frac{(1-\hat p_i)^\gamma}{\hat p_i}\right]
\end{aligned}$$
 
Applying the chain rule through $\partial \hat p_i/\partial z_i = \hat p_i(1-\hat p_i)$, exactly as in the BCE case
 
$$ \begin{aligned}
\frac{\partial l_i}{\partial z_i} &= \frac{\partial l_i}{\partial \hat p_i}\cdot\hat p_i(1-\hat p_i) \\ \\
&= \alpha\left[\gamma \hat p_i(1-\hat p_i)^{\gamma}log(\hat p_i) - (1-\hat p_i)^{\gamma+1}\right] \\ \\
&= \alpha(1-\hat p_i)^{\gamma}\Big[\gamma \hat p_i \ log(\hat p_i) - (1-\hat p_i)\Big]
\end{aligned}$$
 
Comparing this to the plain cross entropy gradient derived for logistic regression, $\partial l_i/\partial z_i = \hat p_i - y_i = -(1-\hat p_i)$ for $y_i=1$, the focal loss gradient is this same $-(1-\hat p_i)$ term scaled by $(1-\hat p_i)^\gamma$, plus an additional $\gamma\hat p_i \ log(\hat p_i)$ correction term that also vanishes as $\hat p_i \to 1$. This confirms algebraically, rather than only by appeal to the shape of $(1-p_{t,i})^\gamma$, that the gradient contributed by a well classified example shrinks polynomially in $(1-\hat p_i)$ at rate $\gamma$, whereas the plain cross entropy gradient shrinks only linearly.
 
#### Solving
Substituting this $\partial l_i/\partial z_i$ in place of $\delta_i^{(2)} = \hat p_i - y_i$ in the MLP's gradient derivation above leaves the rest of backpropagation unchanged, since $\delta_i^{(1)}$ was only ever computed from $\delta_i^{(2)}$ as an opaque quantity. The joint objective therefore inherits the same non-convexity in $(W^{(1)},b^{(1)},W^{(2)},b^{(2)})$ discussed above, plus two additional hyperparameters $\gamma,\alpha$ that have to be selected by cross validation rather than fit by gradient descent.
 
#### Expected Performance
Because focal loss is no longer the exact negative log likelihood that justified using cross entropy by MLE in the first place, we lose the probabilistic interpretation that motivated the loss for logistic regression and the plain MLP above. We would want to evaluate calibration explicitly, for instance with a reliability diagram or Brier score, rather than assume it holds. In exchange, we expect training the MLP with focal loss in place of plain cross entropy to improve performance specifically on the harder, closer games in our dataset, at the cost of an additional $(\alpha,\gamma)$ hyperparameter search.

## Evaluation Metrics

### Accuracy
We include accuracy as the simplest possible summary of model performance, since it is the quantity most directly tied to the decision a threshold classifier actually makes, but we use it primarily as an intuitive evaluation of model performance rather than a primary metric.

At a classification threshold of $0.5$, define the predicted label $\hat y_i = \mathbf{1}_{\hat p_i > 0.5}$, and accuracy over $n$ examples as

$$\mathrm{Acc} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}_{\hat y_i = y_i}$$

#### Threshold Invariance
Accuracy depends on $\hat p_i$ only through $\hat y_i = \mathbf{1}_{\hat p_i > 0.5}$, so for any strictly increasing function $g:[0,1]\to[0,1]$ with $g(0.5)=0.5$, replacing $\hat p_i$ with $g(\hat p_i)$ leaves $\mathrm{Acc}$ unchanged, since $\mathbf{1}_{g(\hat p_i)>0.5} = \mathbf{1}_{\hat p_i>0.5}$ for such $g$. Two models can therefore achieve identical accuracy while disagreeing arbitrarily on the magnitude of $\hat p_i$ away from the boundary, for instance one assigning $\hat p_i=0.51$ and the other $\hat p_i=0.99$ to the same correctly classified example. This is precisely the information accuracy discards and that other metrics like log loss and the Brier score are designed to capture, which is why we do not treat accuracy alone as sufficient for a use case where the magnitude of $\hat p_i$, not just its side of the threshold, is what downstream decisions depend on.

### Area Under the Receiver Operating Characteristic Curve
We include AUC ROC because it evaluates the model's ranking of examples independent of any specific threshold since the $0.5$ threshold implicit in accuracy is arbitrary. For a chosen threshold $\tau \in [0,1]$, define the true and false positive rates

$$\mathrm{TPR}(\tau) = P(\hat p_i > \tau \mid y_i = 1), \qquad \mathrm{FPR}(\tau) = P(\hat p_i > \tau \mid y_i = 0)$$

The ROC curve is the parametric curve $\big(\mathrm{FPR}(\tau), \mathrm{TPR}(\tau)\big)$ traced as $\tau$ sweeps from $1$ to $0$, and AUC-ROC is the area beneath it,

$$\mathrm{AUC} = \int_0^1 \mathrm{TPR}\big(\mathrm{FPR}^{-1}(u)\big) \, du$$

#### Rank Interpretation Derivation
We seek a computationally feasible and more intuative formulation for AUC ROC. Let $\{s_1^+,\dots,s_{n_+}^+\}$ be the predicted probabilities $\hat p_i$ for the $n_+$ positive examples and $\{s_1^-,\dots,s_{n_-}^-\}$ the predicted probabilities for the $n_-$ negative examples. Discretizing the integral over the $n_++n_-$ observed score values and simplifying gives

$$\mathrm{AUC} = \frac{1}{n_+n_-}\sum_{i=1}^{n_+}\sum_{j=1}^{n_-}\left(\mathbf{1}_{s_i^+ > s_j^-} + \tfrac{1}{2}\mathbf{1}_{s_i^+ = s_j^-}\right)$$

which is exactly the Mann-Whitney U statistic used for non-parametric independent group comparisons normalized by $n_+n_-$, and is an unbiased estimator of

$$\mathrm{AUC} = P(\hat p_i > \hat p_j \mid y_i=1, y_j=0)$$

the probability that a randomly drawn positive example is scored higher than a randomly drawn negative example. Unlike accuracy, AUC ROC never considers whether $\hat p_i$ crosses $0.5$, only whether positives are scored above negatives on average across all possible thresholds simultaneously.

#### Interpretation
$\mathrm{AUC}=0.5$ corresponds to a model no better than randomly ranking examples, and $\mathrm{AUC}=1$ corresponds to perfect separation at all possible thresholds, so unlike log loss or the Brier score there is a fixed, model independent baseline to read the number against rather than only a relative comparison between models. An intuative way to read a given value is via the rank interpretation where an AUC of $0.8$ can be interpreted to mean that if we draw one randomly won game and one randomly lost game, the model assigns the winner a higher score about $80\%$ of the time. A high AUC alongside poor log loss or Brier score is the signature of a model like the raw SVM margin that ranks examples well but is not yet expressed on a probability scale, which is why we often read AUC together with calibration sensitive metrics.

### Log Loss
We include log loss because it is the same negative log-likelihood objective derived for BCE, evaluated out of sample. Using it as an evaluation metric therefore directly measures whether the quantity each of those models was trained to optimize actually generalizes. Over $n$ held out examples,

$$\mathrm{LogLoss} = -\frac{1}{n}\sum_{i=1}^{n}\Big(y_ilog(\hat p_i) + (1-y_i)log(1-\hat p_i)\Big)$$

#### Proper Scoring Rule Derivation
A scoring system for a probabilistic model is considered proper when a forecaster achieves the best expected score by stating their honest, true beliefs. Log loss is minimized in expectation exactly when $\hat p_i$ equals the true probability, which justifies treating it as an evaluation metric for calibration rather than just discrimination. Fix an example with true probability $p^* = P(y_i=1)$, and consider the expected per-example loss as a function of a reported probability $q$,

$$
\mathbb{E}\big[l(q)\big] = -p^*log(q) - (1-p^*)log(1-q)
$$

Differentiating with respect to $q$ and setting the result to zero,

$$
\begin{aligned}
\frac{d}{dq}\mathbb{E}[l(q)] &= -\frac{p^*}{q} + \frac{1-p^*}{1-q} = 0 \\ \\
p^*(1-q) &= q(1-p^*) \\ \\
q &= p^*
\end{aligned}
$$

and since $\frac{d^2}{dq^2}\mathbb{E}[l(q)] = \frac{p^*}{q^2} + \frac{1-p^*}{(1-q)^2} > 0$ everywhere on $(0,1)$, this stationary point is a global minimum. Reporting $q=p^*$ is therefore the unique optimal strategy in expectation. Thus log loss is a strictly proper scoring rule, and a model cannot achieve a lower expected log loss by reporting anything other than its true belief about $P(y_i=1)$.

#### Interpretation
Unlike accuracy, log loss has no fixed upper bound to read against, and its scale is not directly intuitive in units of 'probability,' so it is best read as a relative comparison between models on the same dataset rather than as a standalone number. For a useful reference point we consider the log loss of the constant model that always predicts the base rate $\bar y$. Any model scoring worse than this baseline is worse than simply reporting the historical win rate and predicting nothing else. Because $log(q) \to -\infty$ as $q \to 0$, a single confident and wrong prediction can move the metric substantially more than several mildly wrong ones, so when reading a log loss value it is worth checking whether a small number of extreme misses are driving it before concluding a model is broadly miscalibrated. This also means that when comparing across our models, log loss will be the metric most sensitive to the MLP or SVM producing an overconfident probability near $0$ or $1$ on an example it gets wrong, an effect that would be invisible in accuracy and only partially visible in the bounded Brier score.

### Brier Score
We include the Brier score as a second proper scoring rule alongside log loss, but one that is bounded rather than divergent, so that it is not dominated by a small number of extreme misses the way log loss can be. Over $n$ held out examples,

$$\mathrm{BS} = \frac{1}{n}\sum_{i=1}^{n}(\hat p_i - y_i)^2$$

#### Proper Scoring Rule Derivation
We seek to show that a model minimizes Brier in expectation when it reports the true belief. Fix $p^*=P(y_i=1)$ and consider the expected squared error of reporting $q$. Using $\mathbb{E}[Y^2]=\mathbb{E}[Y]=p^*$ since $Y$ is Bernoilli with probability $p*$,

$$
\begin{aligned}
\mathbb{E}\big[(q-Y)^2\big] &= q^2 - 2q\,\mathbb{E}[Y] + \mathbb{E}[Y^2] \\ \\
&= q^2 - 2qp^* + p^*
\end{aligned}
$$

Differentiating with respect to $q$ and setting the result to zero, $\frac{d}{dq}\mathbb{E}[(q-Y)^2] = 2q-2p^*=0 \Rightarrow q=p^*$, and since the second derivative $2>0$ is constant, this is again a global minimum. The Brier score is therefore also a strictly proper scoring rule, with the same practical consequence as for log loss. A model minimizes its expected Brier score exactly by reporting its true belief.

#### Murphy Decomposition
We seek a decomposition of the Brier score that separates calibration error from the model's ability to discriminate between classes as a diagnostic rather than only a scalar summary. Partition the $n$ examples into $B$ bins by predicted probability, with bin $b$ containing $n_b$ examples, mean predicted probability $\bar p_b = \frac{1}{n_b}\sum_{i \in b}\hat p_i$, and observed positive frequency $\bar y_b = \frac{1}{n_b}\sum_{i \in b}y_i$. Let $\bar y = \frac{1}{n}\sum_{i=1}^n y_i$ be the overall base rate. Adding and subtracting $\bar y_b$ inside each squared term of $\mathrm{BS}$ and expanding,

$$
\begin{aligned}
\mathrm{BS} &= \frac{1}{n}\sum_{b=1}^{B}\sum_{i \in b}\big(\hat p_i - y_i\big)^2 \\ \\
&= \frac{1}{n}\sum_{b=1}^{B}\sum_{i \in b}\Big[\big(\hat p_i - \bar y_b\big)^2 + 2\big(\hat p_i - \bar y_b\big)\big(\bar y_b - y_i\big) + \big(\bar y_b - y_i\big)^2\Big]
\end{aligned}
$$

Since $\hat p_i$ is well approximated within a bin by the constant $\bar p_b$, and the cross term vanishes on average within each bin because $\frac{1}{n_b}\sum_{i\in b}(\bar y_b - y_i) = 0$ by definition of $\bar y_b$, this reduces to the standard three-term decomposition

$$
\mathrm{BS} = \underbrace{\frac{1}{n}\sum_{b=1}^{B}n_b(\bar p_b - \bar y_b)^2}_{\text{reliability}} \; - \; \underbrace{\frac{1}{n}\sum_{b=1}^{B}n_b(\bar y_b - \bar y)^2}_{\text{resolution}} \; + \; \underbrace{\bar y(1-\bar y)}_{\text{uncertainty}}
$$

The reliability term is exactly zero when the model is perfectly calibrated within every bin ($\bar p_b = \bar y_b$ for all $b$), the resolution term rewards bins whose observed frequency differs from the base rate, and the uncertainty term depends only on the base rate $\bar y$ and not on the model at all, so it is identical for every model evaluated on the same data.

#### Interpretation
Because the Brier score is bounded in $[0,1]$, it is more directly readable on its own than log loss. A constant model predicting the base rate $\bar y$ for every example scores $\bar y(1-\bar y)$, which by the decomposition above is exactly the uncertainty term with zero reliability and resolution, so this value is a natural floor to compare against rather than an arbitrary reference point. Reading the Brier score alongside its decomposition is more informative than reading the scalar alone: a low overall score with a large resolution term indicates a model that is discriminating well between likely winners and toss-ups, while a low score driven mainly by a small uncertainty term (a base rate close to $0$ or $1$) indicates a dataset that was easy to score well on regardless of model quality. Because the Brier score is bounded while log loss is not, the two metrics should mostly agree across our models but can diverge specifically when a small number of extremely confident wrong predictions are present; when they do diverge, we would read that as evidence that model comparison is being driven by tail behavior on a handful of examples rather than by broad performance differences.

### Reliability Diagrams
We include reliability diagrams because the reliability term of the Murphy decomposition above is a single scalar that can hide where in the probability range a model's calibration actually breaks down, for instance a model could be well calibrated among close games but systematically overconfident on heavy favorites, and the scalar reliability term alone would not distinguish this from a model that is uniformly slightly miscalibrated everywhere.

Using the same binning, a reliability diagram plots the empirical frequency $\bar y_b$ against the mean predicted probability $\bar p_b$ for each bin $b$, typically with bins of equal width in $[0,1]$ or equal count. A perfectly calibrated model has $\bar p_b = \bar y_b$ for every bin, so its points lie exactly on the line $y=x$. A model that is systematically overconfident on favorites will show points below this line at high $\bar p_b$, and one that is systematically underconfident will show points above it.

#### Expected Calibration Error
From the same bins we define a scalar summary of the diagram itself, the expected calibration error,

$$
\mathrm{ECE} = \frac{1}{n}\sum_{b=1}^{B}n_b\,\big|\bar p_b - \bar y_b\big|
$$

which is directly comparable in form to the reliability term of the Murphy decomposition, $\frac{1}{n}\sum_b n_b(\bar p_b-\bar y_b)^2$, but uses an absolute rather than squared deviation. As a consequence ECE is expressed in the same units as probability itself and is not dominated by any single badly miscalibrated bin the way the squared reliability term can be, which makes it a more directly interpretable summary of the diagram, at the cost of losing the exact additive decomposition into reliability, resolution, and uncertainty that motivated using squared error.

#### Interpretation
The diagram should be read bin by bin rather than as a single trend: points above the diagonal indicate bins where the model is underconfident (the outcome happens more often than predicted), and points below the diagonal indicate overconfidence (the outcome happens less often than predicted), and the location of these deviations along the x-axis tells us which part of the probability range is affected, for instance a model could be well calibrated on close games and specifically overconfident on heavy favorites, which a single scalar metric would not localize. Bin size matters when reading the diagram as too many bins and individual points become noisy estimates of $\bar y_b$ from too few examples, and too few bins and genuine local miscalibration gets averaged away, so we would read a diagram together with the per-bin counts $n_b$ rather than the plotted points alone. We would expect the SVM's raw margin score, reinterpreted directly as a $[0,1]$ quantity, to show visible deviation from the diagonal particularly near $\bar p_b$ close to $0$ or $1$, since the margin is not constructed to be probability-like in the first place, and we would read a flattened, closer-to-diagonal diagram after Platt scaling as a direct visual check on the calibration argument made in that section, rather than assuming the MLE fit of $(c,d)$ necessarily produces good calibration on new data without looking.

## Feature Derivations

### Rating Systems

Mathematical rating systems for sports teams are a well researched topic and play a significant role in betting odds specifically. In an attempt to capture industry standard information beyond raw stats, we consider adding such a feature. There are generally two common automated rating algorithms based on interconnected schedules, the Colley system and the Massey system. 

#### Massey Rating
The Massey system is based on the assumption that the difference in ratings between two teams should equal the expected margin of victory and uses least squares to minimize errors. We formulate the Massey problem as

$$Xr = y$$

where $r$ is the unknown vector of team ratings, $y$ is a vector of point differentials for each game k, and $X$ is an $m \times n$ matrix of $m$ games and $n$ teams. For each game $k$ with winning team $i$ and losing team $j$,

$$
X_{ki} = +1 \\
X_{kj} = -1 \\
X_{kc} = 0 \text{ for all other teams not involved in game } k
$$

Interpreting the construction of the system, we see that the ratings $r$ should exactly equal the point differential between two teams should they play in a game. Since $m > n$ for any given season, $X$ is a tall matrix and the above system most likely cannot be solved explicitly. We now formulate solving for the ratings $r$ as a least squares problem and construct the normal equations

$$X^TXr=X^Ty$$

and reinterpreting the normal equations gives the traditional Massey formulation

$$Mr = p$$

where $p$ becomes a vector where each element $p_i$ is the net point differential for team $i$ calculated as total points scored minus total points allowed across all games and $M$ is the $n \times n$ Massey matrix where n is the number of teams. The matrix is constructed as 
$$
M_{ii} = \text{total number of games team i has played} \\
M_{ij} = \text{negative number of times team i played team j} \\
\text{for } 0<i,j\leq n
$$
The system attempts to find the ratings $r$ such that the season per game sum a teams rating minus all of their opponents ratings is equal to that teams season point differential. By construction, all rows of the Massey matrix add to 0. As a result, the matrix is singular since multiplying by a vector of all ones will give the zero vector, meaning the kernel is non-zero and the matrix is not invertible. The canonical solution to this problem thus involves replacing the final row of $M$ with a row of ones and setting $p_n = 0$, forcing the sum of all rating to be zero and thus standardizing the Massey rating such that $r = 0$ represents the average performance, and the difference between two Massey ratings predicts the point spread of their matchup. Solving for the rating system then simply becomes

$$r = M^{-1}p$$

though in practice the system is often solved using factorization algorithms like LU-decomposition or Cholesky decomposition, or iterative algorithms like GMRES because of the numerical instability of inverse computations. This post hoc approach does not sacrifice information since the ratings are inherently relative and the rating for the final team lost to the replacement is still fully constrained by the information of all other teams. 

#### Colley Rating
Due to the necessity of post hoc modification for the stability of the Massey rating, the inclusion of point differential as a current feature in the dataset, and the potential for blowouts to distort the predictive power of the Massey rating, we instead consider the Colley rating. The Colley system ignores the margin of victory and relies on Laplace's Rule of Succession to calculate a bias-free, self-consistent win percentage. The Colley rating fundamentally builds on Laplace's rule. Assuming every game is independent and has an equal chance of winning

$$r_i = \frac{w_i + 1}{t_i + 2}$$

where for team $i$, $r_i$ is the rating or win probability, $w_i$ is the total number of observed wins, and $t_i$ is the total number of observed games. However, not all wins in sports are equally likely as beating a good team should be much less likely than beating a bad team. To address this, we first recognize that we can rewrite the number of wins as

$$w_i = \frac{w_i - l_i}{2} + \frac{w_i + l_i}{2} = \frac{w_i - l_i}{2} + \frac{t_i}{2}$$

Then since $\frac{t_i}{2}$ inherently follows Laplace's assumption that all teams are average and thus give an equal win probability, we replace it with $\sum_{j \in Opp(i)} r_j$ the sum of the actual opponents ratings, since each represents the win probability against that opponent. Substituting back into the original formula gives

$$r_i = \frac{1 + \frac{w_i - l_i}{2} + \sum_{j \in Opp(i)} r_j}{t_i + 2}$$

rearranging the equation so that we can solve for all ratings simultaneously and defining $n_{ij}$ as the number of times team $i$ plays team $j$ to simplify the sum index we obtain

$$(t_i + 2)r_i - \sum_{j \neq i} n_{ij}r_j = 1 + \frac{w_i - l_i}{2}$$

which is a system of linear equations we can write in matrix form. We formulate the Colley problem as

$$Cr = b$$

where $r$ is the vector of unknown ratings, $b$ is a vector with each element $b_i$ is constructed as

$$b_i = 1 + \frac{w_i - l_i}{2} \text{ where $w_i$ and $l_i$ are the wins and losses for team i}$$

and C is the Colley matrix, constructed as

$$
C_{ii} = 2 + \text{ total number of games team i has played} \\
C_{ij} = \text{negative number of times team i played team j} \\
\text{for } 0<i,j\leq n
$$

Because of the $+2$ on the diagonal of the Colley matrix, $C$ is strictly diagonally dominant, so by the Levy–Desplanques Theorem it must be invertible and we can solve the system cleanly for $r$. Since $r_i$ is constructed based on Laplace's rule, it serves as an adjusted win probability, with the average rating across all teams being $0.5$.

### Momentum Features
Heuristically, momentum based features follow conventional logic that successful teams are the teams that 'get hot at the right time' or are having outlier years. To rigorously develop a feature that measures current performance vs previous years results, we start by defining a baseline. For each year, we exclude the current years performance to prevent target leakage and then calculate the trailing average of a given statistic over the past $w$ seasons. For a given team's output in a stat $x$ at season $t$, we calculate this average as

$$\bar{x}_{t-1}^{(w)} = \frac{1}{k} \sum_{i=1}^{k} x_{t-i}$$

where $k = \min(w, t-1)$ to account for limited history (ie start of the observation period) and $x_{t-i}$ is the value of the statistic from $i$ seasons ago. We then define 'momentum' as the arithmetic difference between a team's current season performance vs their historical baseline to determine when a program is over or under performing relative to their recent years. The momentum $m_t$ for the current season $t$ is then

$$m_t = x_t - \bar{x}_{t-1}^{(w)}$$

allowing our feature to track the current stats divergence from the historical 'center of gravity' and detect outlier years.


## Statistical Tests
We evaluate four classifiers (logistic regression, random forest, SVM, and MLP) on the same held out set of tournament games, so any comparison between two models is a comparison between paired observations rather than independent samples. The tests below are included specifically because they exploit that pairing, and because the metrics being compared (0/1 correctness, and continuous scores like log loss or Brier score) call for different tests depending on whether the underlying quantity is discrete or continuous and whether a normality assumption is defensible.
 
### McNemar's Test
We include McNemar's test to determine whether a difference in raw accuracy between two classifiers is real signal or an artifact of the particular games in our test set. McNemar's test is built on the foundation that a naive comparison of two accuracy numbers, or a two sample test that treats each model's correct/incorrect calls as independent samples, throws away the fact that both models are scored on the exact same games and ignores every game both models get right or both get wrong, which carry no information about which model is better.
 
For two classifiers $A,B$ evaluated on the same $n$ held out games, we only care about the games on which they disagree. We construct the $2\times2$ contingency table of paired outcomes
 
$$
\begin{aligned}
n_{01} &= |\{i : A \text{ wrong}, \ B \text{ right}\}| \\
n_{10} &= |\{i : A \text{ right}, \ B \text{ wrong}\}|
\end{aligned}
$$
 
and discard the $n_{00}$ and $n_{11}$ counts where both models agree, since those games are uninformative about which model is better. Under the null hypothesis that the two classifiers have equal error rate, a disagreement is equally likely to favor $A$ or $B$, so conditional on the total number of disagreements $m = n_{01}+n_{10}$, $n_{01}$ is Binomial $\sim (m,\tfrac12)$.
 
#### Derivation
We want a test statistic on $n_{01}$ that we can evaluate without needing the exact Binomial CDF, since $m$ can be large. Since $n_{01}\sim\text{Binomial}(m,\tfrac12)$ under $H_0$, it has mean $\tfrac{m}{2}$ and variance $\tfrac{m}{4}$, so by the central limit theorem
 
$$\frac{n_{01} - \tfrac{m}{2}}{\sqrt{m/4}} \;\xrightarrow{d}\; \mathcal N(0,1)$$
 
Squaring a standard normal gives a $\chi^2_1$ random variable, and noting $n_{01}-\tfrac m2 = \tfrac12(n_{01}-n_{10})$ since $n_{10}=m-n_{01}$, we obtain
 
$$\chi^2 = \frac{(n_{01}-n_{10})^2}{n_{01}+n_{10}} \;\sim\; \chi^2_1$$
 
Because $n_{01}$ is discrete while the $\chi^2_1$ approximation it is compared against is continuous, we apply Yates' continuity correction, which shrinks $|n_{01}-n_{10}|$ toward $0$ by one before squaring, to avoid systematically overstating significance when $m$ is small
 
$$\chi^2_{\text{corr}} = \frac{(|n_{01}-n_{10}|-1)^2}{n_{01}+n_{10}}$$
 
#### Interpretation
McNemar's test says nothing about how many predictions either model got right in total, only about which model wins on the disputed predictions, so it should be read alongside raw accuracy rather than as a replacement for it. Because the test conditions on $m$ and discards agreements entirely, its power depends on the number of disagreements between the two specific models being compared rather than on $n$ directly, so two models that make very similar calls will have low power to reject $H_0$ even with a large test set, which is itself useful information about how differently the two models are behaving.
 
### Paired t-Test
We include the paired t-test to compare two models on a continuous per season metric, such as per season bracket score, again exploiting that both models are scored on the same games in the same held out season rather than treating the two sets of scores as independent samples.

For $n$ paired seasons, define the per season average difference $d_i = m_{A,i} - m_{B,i}$ between the two models' per season bracket score values. The null hypothesis is that the true mean difference across all seasons $\mu_d = \mathbb E[d_i]$ is zero. With sample mean $\bar d$ and sample standard deviation $s_d$ of the $d_i$, the test statistic is
 
$$t = \frac{\bar d}{s_d/\sqrt n}$$
 
which, under the assumption that the $d_i$ are i.i.d. normal, follows a Student's $t$ distribution with $n-1$ degrees of freedom under $H_0$.
 
#### Derivation
Assuming $d_i {\sim}\mathcal N(\mu_d,\sigma_d^2)$ i.i.d., the sample mean satisfies $\bar d \sim \mathcal N(\mu_d, \sigma_d^2/n)$, so under $H_0:\mu_d=0$,
 
$$Z = \frac{\bar d}{\sigma_d/\sqrt n} \sim \mathcal N(0,1)$$
 
but $\sigma_d$ is unknown and must be estimated by $s_d$, introducing additional uncertainty from a finite sample. By Cochran's theorem, $(n-1)s_d^2/\sigma_d^2 \sim \chi^2_{n-1}$ independently of $\bar d$, and the ratio of a standard normal to the square root of an independent $\chi^2_{n-1}$ scaled by its degrees of freedom is, by definition of the $t$ distribution,
 
$$t = \frac{Z}{\sqrt{\chi^2_{n-1}/(n-1)}} = \frac{\bar d/(\sigma_d/\sqrt n)}{s_d/\sigma_d} = \frac{\bar d}{s_d/\sqrt n} \sim t_{n-1}$$
 
which recovers the statistic above without ever needing to know $\sigma_d$. The reason to pair rather than run an unpaired two sample test on $\{m_{A,i}\}$ and $\{m_{B,i}\}$ directly is a variance argument: writing $\mathrm{Var}(d_i) = \mathrm{Var}(m_{A,i})+\mathrm{Var}(m_{B,i})-2\mathrm{Cov}(m_{A,i},m_{B,i})$, since both models are scored on the same season and therefore tend to do well or badly together (an upset is hard for every model, a blowout is easy for every model), $\mathrm{Cov}(m_{A,i},m_{B,i})>0$, which makes $\mathrm{Var}(d_i)$ smaller than the variance an unpaired test would implicitly assume, giving the paired test more power to detect the same true difference in means.
 
#### Interpretation
The validity of $t$ here rests on the $d_i$ being approximately normal, which is a stronger assumption than McNemar's test requires and one that is more plausible for a bounded, averaged metric like Brier score than for log loss, which we noted above can be dominated by a small number of extreme misses and is therefore prone to heavy tailed or skewed differences. Where that assumption is doubtful we instead use the Wilcoxon signed rank test below on the same $d_i$.
 
### Wilcoxon Signed Rank Test
We include the Wilcoxon signed rank test as a non-parametric alternative to the paired t-test where the per season average differences $d_i$ are not plausibly normal, while still using more information than a simple sign test that only asks which model won on each season.
 
Given the same paired differences $d_i$, discard any $d_i=0$, leaving $m$ nonzero differences. Rank $|d_i|$ from $1$ to $m$, and let $R_i$ denote the rank assigned to $|d_i|$. Define
 
$$W^+ = \sum_{i:\,d_i>0} R_i, \qquad W^- = \sum_{i:\,d_i<0} R_i$$
 
so $W^++W^- = \tfrac{m(m+1)}{2}$, the sum of the first $m$ integers, and the test statistic is $W=\min(W^+,W^-)$. The null hypothesis is that the $d_i$ are drawn from a distribution symmetric about $0$, under which each rank is equally likely to carry a positive or negative sign.
 
#### Derivation
Under $H_0$, symmetry about $0$ means that, independent of the magnitude ranks, each sign is an independent fair coin flip. Write $W^+ = \sum_{k=1}^m k\cdot I_k$ where $I_k\sim\text{Bernoulli}(\tfrac12)$ i.i.d. indicates whether the difference with rank $k$ is positive. Then
 
$$\mathbb E[W^+] = \frac12\sum_{k=1}^m k = \frac{m(m+1)}{4}$$
 
and, since the $I_k$ are independent with $\mathrm{Var}(I_k)=\tfrac14$,
 
$$\mathrm{Var}(W^+) = \sum_{k=1}^m k^2\,\mathrm{Var}(I_k) = \frac14\sum_{k=1}^m k^2 = \frac{m(m+1)(2m+1)}{24}$$
 
using $\sum_{k=1}^m k^2 = \tfrac{m(m+1)(2m+1)}{6}$. For $m$ not too small, $W^+$ is a sum of $m$ bounded independent terms, so by the Lyapunov central limit theorem
 
$$Z = \frac{W^+ - \tfrac{m(m+1)}{4}}{\sqrt{m(m+1)(2m+1)/24}} \;\xrightarrow{d}\; \mathcal N(0,1)$$
 
which we compare against the standard normal rather than the exact, tabulated distribution of $W$.
 
#### Interpretation
Because the test uses only the ranks of $|d_i|$ rather than their actual magnitudes, it is more robust to a few extreme differences than the paired t-test, at the cost of discarding some information the t-test would use and therefore somewhat less powerful when the normality assumption genuinely does hold. We use it as a check on the paired t-test result on the same metric: broad agreement between the two increases our confidence that a detected difference is not an artifact of non-normal $d_i$, while a disagreement is itself a sign that the t-test's normality assumption is doing meaningful, and potentially misleading, work.
 
### Holm-Bonferroni Correction
We include the Holm-Bonferroni correction because comparing four models pairwise, or comparing several metrics between the same two models, means running multiple hypothesis tests rather than one, and each individual test above controls its own Type I error rate $\alpha$ only in isolation.
 
Running $m$ independent tests each at level $\alpha$, the probability of at least one false rejection across the family is
 
$$P(\text{at least one false positive}) = 1-(1-\alpha)^m$$
 
which grows toward $1$ as $m$ increases even though each individual test is well calibrated, so with $6$ pairwise model comparisons a nominal $\alpha=0.05$ per test can correspond to a substantially higher family-wise error rate. The classical Bonferroni correction controls this by testing each hypothesis at level $\alpha/m$, which is simple but conservative because it applies the same threshold to every test regardless of the other $p$-values observed. The Holm-Bonferroni procedure controls the same family-wise error rate with a step-down threshold.
 
#### Procedure
Given $m$ hypotheses with $p$-values sorted ascending $p_{(1)} \leq p_{(2)} \leq \dots \leq p_{(m)}$, find the smallest index $k$ such that
 
$$p_{(k)} > \frac{\alpha}{m-k+1}$$
 
reject hypotheses $H_{(1)},\dots,H_{(k-1)}$ and fail to reject $H_{(k)},\dots,H_{(m)}$. If no such $k$ exists, reject all $m$.

#### Derivation
We show the family-wise error rate under this procedure is at most $\alpha$. Let $I_0$ be the set of hypotheses that are truly null, with $|I_0|=m_0\leq m$. A false rejection occurs only if some $p$-value in $I_0$ is rejected, which under the step-down rule requires $p_{(j)}\leq \alpha/(m-j+1)$ for the rank $j$ at which that null hypothesis is examined. Since at most $m-m_0$ of the smaller ranks can be occupied by non-null hypotheses, any true null is examined at a rank $j$ no smaller than $m-m_0+1$, so it is rejected only if its $p$-value is at most $\alpha/(m-j+1) \leq \alpha/m_0$. Applying Bonferroni's inequality (a union bound) to just the $m_0$ true nulls at this common threshold,
 
$$P(\text{any true null has } p\text{-value} \leq \alpha/m_0) \leq \sum_{i\in I_0} P(p_i \leq \alpha/m_0) = m_0\cdot\frac{\alpha}{m_0} = \alpha$$
 
so the family-wise error rate is bounded by $\alpha$ regardless of how many of the $m$ hypotheses are actually non-null, and regardless of any dependence between the tests, since the union bound step requires no independence assumption. Because Holm-Bonferroni only requires $p_{(k)}$ to clear a threshold that grows as $k$ increases (correcting by $m-k+1$ rather than the fixed $m$ used by classical Bonferroni), every hypothesis that Bonferroni would reject is also rejected by Holm-Bonferroni, making it uniformly more powerful for the same guaranteed error control.
 
#### Interpretation
We apply this correction across the full family of pairwise model comparisons and metric comparisons run in a single round of evaluation, rather than to each test in isolation, since it is the act of running many tests and reporting whichever come back significant that inflates the family-wise error rate. A comparison that is significant before correction but not after should be read as a difference we cannot distinguish from noise once we account for how many comparisons were run to find it, not as a failed or discarded result.