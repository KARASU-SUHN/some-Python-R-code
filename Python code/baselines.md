## SCG

baseline:<br>
·KOCG(matlab)<br> 
·BNC/SPONGE(from SigNet)


```python
from signet.cluster import Cluster 
from signet.block_models import SSBM
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score

# simple test on the signed stochastic block model 

n = 50000  # number of nodes
k = 2      # number of clusters
eta = 0.1  # sign flipping probability
p = 0.0002 # edge probability

(Ap, An), true_assignment = SSBM(n = n, k = k, pin = p, etain = eta) # construct a graph
print(true_assignment)

c = Cluster((Ap, An))

predictions1 = c.spectral_cluster_laplacian(k = k, normalisation='sym') # cluster with the signed laplacian
score1 = f1_score(predictions1, true_assignment)
print(score1)

#SDP_cluster(k, solver='BM_proj_grad', normalisation='sym_sep')
```

    [1 1 0 ... 1 0 1]
    0.9970814009275548


### This module contains a series of function that can generate random graphs with a signed community structure.

#### 1. block_models.SBAM(n, k, p, eta)
A signed Barabási–Albert model graph generator.<br>
n – (int) Number of nodes.<br>
k – (int) Number of communities.<br>
p – (float) Sparsity value.<br>
eta – (float) Noise value.<br>

Returns:	
(a,b),c where a is a sparse n by n matrix of positive edges, b is a sparse n by n matrix of negative edges c is an array of cluster membership. 


```python
#SBAM- Barabási–Albert model
from signet.block_models import SBAM
n = 50000  # number of nodes
k = 2 # number of clusters
eta = 0.1  # Noise value.
p = 0.002  # Sparsity value.

(Ap, An), true_assignment1 = SBAM(n = n, k = k, p = p, eta = eta) # construct a graph
print(true_assignment1)
```

    [0. 0. 0. ... 1. 1. 1.]


#### 2. block_models.SRBM(n, k, p, eta)
A signed regular graph model generator.<br>
n – (int) Number of nodes.<br>
k – (int) Number of communities.<br>
p – (float) Sparsity value.<br>
eta – (float) Noise value.<br>

Returns:	
(a,b),c where a is a sparse n by n matrix of positive edges, b is a sparse n by n matrix of negative edges c is an array of cluster membership.


```python
#SRBM- regular graph model
from signet.block_models import SRBM
n = 50000  # number of nodes
k = 2 # number of clusters
eta = 0.1  # Noise value.
p = 0.002  # Sparsity value.

(Ap, An),true_assignment2 = SRBM(n, k , p , eta) # construct a graph
print(true_assignment2)
```

    ((<5000x5000 sparse matrix of type '<class 'numpy.float64'>'
    	with 24948 stored elements in Compressed Sparse Column format>, <5000x5000 sparse matrix of type '<class 'numpy.float64'>'
    	with 25052 stored elements in Compressed Sparse Column format>), array([0., 1., 0., ..., 1., 1., 1.]))


#### ***3. block_models.SSBM(n, k, pin, etain, pout=None, etaout=None, values='ones', sizes='uniform')
A signed stochastic block model graph generator.<br>
n – (int) Number of nodes.<br>
k – (int) Number of communities.<br>
pin – (float) Sparsity value within communities.<br>
etain – (float) Noise value within communities.<br>
pout – (float) Sparsity value between communities.<br>
etaout – (float) Noise value between communities.<br>
values – (string) Edge weight distribution (within community and without sign flip; otherwise weight is negated): ‘ones’: Weights are 1. ‘gaussian’: Weights are Gaussian, with variance 1 and expectation of 1.# ‘exp’: Weights are exponentially distributed, with parameter 1. ‘uniform: Weights are uniformly distributed between 0 and 1.<br>
sizes – (string) How to generate community sizes: ‘uniform’: All communities are the same size (up to rounding). ‘random’: Nodes are assigned to communities at random. ‘uneven’: Communities are given affinities uniformly at random, and nodes are randomly assigned to communities weighted by their affinity.<br>

Returns:	
(a,b),c where a is a sparse n by n matrix of positive edges, b is a sparse n by n matrix of negative edges c is an array of cluster membership.


#### < Discovering conflicting groups in signed networks >
block model (m-SSBM), which has 4 parameters; <br>
n: the graph size;<br>
k: the number of conflicting groups; <br>
l: the size of each of the conflicting groups (all have the same size); and η ∈ [0, 1]: a parameter that controls the edge probabilities. Edges in the same group are positive with probability 1 − η and negative or absent with probability η/2. Edges between distinct groups are negative with probability 1 − η and positive or absent with probability η/2. All other edges have equal probability of min(η, 1/2) of being positive or negative. Hence, the smaller the value of η, the denser the conflicting groups and the lower the noise level. Note that the conflicting groups only emerge when η ≤ 2/3, since m-SSBM is expect to have more negative edges in the groups and more positive edge between groups if η>2/3


```python
from signet.block_models import SSBM
n = 50000  # number of nodes
k = 2      # number of clusters
eta = 0.01  # sign flipping probability
p = 0.0002 # edge probability

(Ap, An), true_assignment3 = SSBM(n, k, 
                                  pin= p,etain = eta,pout=None, etaout=None, values='ones', sizes='uniform') # construct a graph
print(true_assignment3)
```

    [0 1 1 ... 0 0 0]


## Comparing signed clustering algorithms using SigNet


```python
# import the relevant classes and functions from signet
from signet.cluster import Cluster
from signet.block_models import SSBM
from sklearn.metrics import adjusted_rand_score
```


```python
# generate a random graph with community structure by the signed stochastic block model 

n = 5000    # number of nodes
k = 15      # number of communities
eta = 0.05  # sign flipping probability
p = 0.01    # edge probability

(A_p, A_n), true_assign = SSBM(n = n, k = k, pin=p, etain=eta) 
```


```python
# initialise the Cluster object with the data (adjacency matrix of positive and negative graph)

c = Cluster((A_p, A_n))
```

#### Clusters the graph using eigenvectors of the adjacency matrix.
spectral_cluster_adjacency(k=2, normalisation='sym_sep', eigens=None, mi=None)

k (int, or list of int) – The number of clusters to identify. If a list is given, the output is a corresponding list.<br>
normalisation (string) – How to normalise for cluster size: ‘none’ - do not normalise. ‘sym’ - symmetric normalisation. ‘rw’ - random walk normalisation. ‘sym_sep’ - separate symmetric normalisation of positive and negative parts. ‘rw_sep’ - separate random walk normalisation of positive and negative parts.<br>
eigens (int) – The number of eigenvectors to take. Defaults to k.<br>
mi (int) – The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.


```python
# calculate the assignments provided by the algorithms you want to analyse

A_assign = c.spectral_cluster_adjacency(k = k,normalisation='sym_sep', eigens=None, mi=None)

# compute the recovery score of the algorithms against the SSBM ground truth
score_A = adjusted_rand_score(A_assign, true_assign)

print('score_A: ', score_A)
```

    score_A:  6.460311072999736e-05


## sklearn.metrics.adjusted_rand_score

Rand index adjusted for chance.

The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.

The raw RI score is then “adjusted for chance” into the ARI score using the following scheme:
ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

notes: 
$$
RI(rand index)=\frac{TP+TN}{TP+FP+FN+TN}
$$ 
where TP is the number of true positives, TN is the number of true negatives, FP is the number of false positives, and FN is the number of false negatives.

The adjusted Rand index is thus ensured to have a value close to 0.0 for random labeling independently of the number of clusters and samples and exactly 1.0 when the clusterings are identical (up to a permutation).

ARI is a symmetric measure:
adjusted_rand_score(a, b) == adjusted_rand_score(b, a)

labels_true:int array, shape = $[n$_samples,]
Ground truth class labels to be used as a reference

labels_pred:array-like of shape (n_samples,)
Cluster labels to evaluate

#### Similarity score between -1.0 and 1.0. Random labelings have an ARI close to 0.0. <br>1.0 stands for perfect match.

#### Clusters the graph using the eigenvectors of the graph signed Laplacian.
spectral_cluster_laplacian(k=2, normalisation='sym_sep', eigens=None, mi=None)

k (int, or list of int) – The number of clusters to identify. If a list is given, the output is a corresponding list.<br>
normalisation (string) – How to normalise for cluster size: ‘none’ - do not normalise. ‘sym’ - symmetric normalisation. ‘rw’ - random walk normalisation. ‘sym_sep’ - separate symmetric normalisation of positive and negative parts. ‘rw_sep’ - separate random walk normalisation of positive and negative parts.<br>
eigens (int) – The number of eigenvectors to take. Defaults to k.<br>
mi (int) – The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.


```python
L_assign = c.spectral_cluster_laplacian(k = k, normalisation='sym')

score_L = adjusted_rand_score(L_assign, true_assign)

print('score_L: ', score_L)
```

    score_L:  -0.00015935867998148105


#### Clusters the graph by solving a Laplacian-based generalised eigenvalue problem.
geproblem_laplacian(k=4, normalisation='multiplicative', eigens=None, mi=None, tau=1.0)

k (int, or list of int) – The number of clusters to identify. If a list is given, the output is a corresponding list.<br>
normalisation (string) – How to normalise for cluster size: ‘none’ - do not normalise. ‘additive’ - add degree matrices appropriately. ‘multiplicative’ - multiply by degree matrices appropriately.<br>
eigens (int) – The number of eigenvectors to take. Defaults to k.<br>
mi (int) – The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.<br>
nudge (int) – Amount added to diagonal to bound eigenvalues away from 0.


```python
gela_assign = c.geproblem_laplacian(k = k, normalisation='additive')

score_gela = adjusted_rand_score(gela_assign, true_assign)

print('score_gela: ', score_gela)
```

    score_gela:  -0.00013972147651280587



```python
#change normalisation

gela_assign_mup = c.geproblem_laplacian(k = k, normalisation='multiplicative')

score_gela_mup = adjusted_rand_score(gela_assign_mup, true_assign)

print('score_gela_mup: ', score_gela_mup)
```

    score_gela_mup:  7.461492884601056e-05


#### Clustering based on a SDP relaxation of the clustering problem.
(A low dimensional embedding is obtained via the lowest eigenvectors of positive-semidefinite matrix Z which maximises its Frobenious product with the adjacency matrix and k-means is performed in this space.)

SDP_cluster(k, solver='BM_proj_grad', normalisation='sym_sep')<br>
k (int, or list of int) – The number of clusters to identify. If a list is given, the output is a corresponding list.<br>
solver (str) – Type of solver for the SDP formulation. ‘interior_point_method’ - Interior point method. ‘BM_proj_grad’ - Burer Monteiro method using projected gradient updates. ‘BM_aug_lag’ - Burer Monteiro method using augmented Lagrangian updates.


```python
SDP_assign = c.SDP_cluster(k = k, solver='BM_proj_grad', normalisation='sym_sep')

score_SDP = adjusted_rand_score(SDP_assign, true_assign)

print('score_SDP: ', score_SDP)
```

    score_SDP:  -0.0003393645427275254


#### Clusters the graph using the Signed Positive Over Negative Generalised Eigenproblem (SPONGE) clustering.
SPONGE(k=4, tau_p=1, tau_n=1, eigens=None, mi=None)

The algorithm tries to minimises the following ratio (Lbar^+ + tau_n D^-)/(Lbar^- + tau_p D^+). The parameters tau_p and tau_n can be typically set to one.

https://arxiv.org/pdf/1904.08575.pdf (3.5) <br>
ref: 「SPONGE: A generalized eigenproblem for clustering signed networks」
<br>
minimize
$(\bar{L}^++\tau^-D^-)/(\bar{L}^-+\tau^+D^+)$ <br>
$L^+$ denotes the Laplacian of $G^+$ <br>
$D^+$ denotes a diagonal matrix with the degrees of $G^+$

k (int, or list of int) – The number of clusters to identify. If a list is given, the output is a corresponding list.<br>
tau_n (float) – regularisation of the numerator <br>
tau_p (float) – regularisation of the denominator <br>
eigens (int) – The number of eigenvectors to take. Defaults to k.<br>
mi (int) – The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.<br>
nudge (int) – Amount added to diagonal to bound eigenvalues away from 0.


```python
SPONGE_assign = c.SPONGE(k = k, tau_p=1, tau_n=1, eigens=None, mi=None)

score_SPONGE = adjusted_rand_score(SPONGE_assign, true_assign)
print('score_SPONGE: ', score_SPONGE)

f1_score_SPONGE = f1_score(SPONGE_assign, true_assign, labels=None, pos_label=1, average='weighted', sample_weight=None, zero_division='warn')
print('f1_score_SPONGE: ', f1_score_SPONGE)
```

    score_SPONGE:  0.03417362533701187
    f1_score_SPONGE:  0.061263067901204775


## sklearn.metrics.f1_score
https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')<br>

The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:<br>

F1 = 2 * (precision * recall) / (precision + recall) <br>
In the multi-class and multi-label case, this is the average of the F1 score of each class with weighting depending on the average parameter.


y_true:1d array-like, or label indicator array / sparse matrix
Ground truth (correct) target values.<br>
y_pred:1d array-like, or label indicator array / sparse matrix
Estimated targets as returned by a classifier.<br>
labels:array-like, default=None<br>
pos_label: str or int, default=1
The class to report if average='binary' and the data is binary. If the data are multiclass or multilabel, this will be ignored; setting labels=[pos_label] and average != 'binary' will report scores for that label only.<br>
average:{‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’ This parameter is required for multiclass/multilabel targets. If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data <br>
zero_division:“warn”, 0 or 1, default=”warn”
Sets the value to return when there is a zero division, i.e. when all predictions and labels are negative. If set to “warn”, this acts as 0, but warnings are also raised.

Returns:
F1 score of the positive class in binary classification or weighted average of the F1 scores of each class for the multiclass task.

In the most simple terms, higher F1 scores are generally better.


```python

```

#### Clusters the graph by using the Balance Normalised Cut or Balance Ratio Cut objective matrix.
spectral_cluster_bnc(k=2, normalisation='sym', eigens=None, mi=None)

k (int, or list of int) – The number of clusters to identify. If a list is given, the output is a corresponding list.<br>
normalisation (string) – How to normalise for cluster size: ‘none’ - do not normalise. ‘sym’ - symmetric normalisation. ‘rw’ - random walk normalisation.<br>
eigens (int) – The number of eigenvectors to take. Defaults to k.<br>
mi (int) – The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.


```python
BNC_assign = c.spectral_cluster_bnc(k = k, normalisation='sym', eigens=None, mi=None)

score_BNC = adjusted_rand_score(BNC_assign, true_assign)
print('score_BNC: ', score_BNC)

f1_score_BNC = f1_score(BNC_assign, true_assign, labels=None, pos_label=1, average='weighted', sample_weight=None, zero_division='warn')
print('f1_score_BNC: ', f1_score_BNC)
```

    score_BNC:  0.005871912857370246
    f1_score_BNC:  0.06485404177933564



```python

```


```python
#Summarize

# calculate the assignments provided by the algorithms you want to analyse

A_assign = c.spectral_cluster_adjacency(k = k,normalisation='sym_sep', eigens=None, mi=None)

L_assign = c.spectral_cluster_laplacian(k = k, normalisation='sym')

gela_assign = c.geproblem_laplacian(k = k, normalisation='additive')

gela_assign_mup = c.geproblem_laplacian(k = k, normalisation='multiplicative')

SDP_assign = c.SDP_cluster(k = k, solver='BM_proj_grad', normalisation='sym_sep')

SPONGE_assign = c.SPONGE(k = k, tau_p=1, tau_n=1, eigens=None, mi=None)


# compute the recovery score of the algorithms against the SSBM ground truth

score_A = adjusted_rand_score(A_assign, true_assign)

score_L = adjusted_rand_score(L_assign, true_assign)

score_gela = adjusted_rand_score(gela_assign, true_assign)

score_gela_mup = adjusted_rand_score(gela_assign_mup, true_assign)

score_SDP = adjusted_rand_score(SDP_assign, true_assign)

score_SPONGE = adjusted_rand_score(SPONGE_assign, true_assign)



print('score_A: ', score_A)
print('score_L: ', score_L)
print('score_gela: ', score_gela)
print('score_gela_mup: ', score_gela_mup)
print('score_SDP: ', score_SDP)
print('score_SPONGE: ', score_SPONGE)
```

    score_A:  0.00014565193416984847
    score_L:  -0.00015945810069998992
    score_gela:  0.00015105756555625264
    score_gela_mup:  9.989619547918406e-05
    score_SDP:  -0.00032450966879740884
    score_SPONGE:  0.00017483049606168446



```python

```
