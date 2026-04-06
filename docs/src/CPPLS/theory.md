# Theory

Canonical Powered Partial Least Squares (CPPLS) is a supervised projection
method for regression and classification. Its purpose is to extract latent
components that summarize the predictor matrix $X \in \mathbb{R}^{n \times p}$,
where $n$ is the number of samples and $p$ the number of predictors, in a way
that best reflects the structure in a multivariate response matrix
$Y \in \mathbb{R}^{n \times q}$, where $q$ is the number of response variables.
The central idea is to construct a response-guided supervised space using a power 
parameter $\gamma$ that balances predictor variance and predictor-response association, 
and then to use canonical correlation to combine that supervised space into latent 
components aligned with the response block. The framework is naturally multivariate in
$Y$, supporting both regression and classification.

This package follows that core structure. In addition, it allows optional auxiliary 
responses, observation weights, as well as two response-column weighting mechanisms, 
response weights and target weights.

Each CPPLS component is extracted in two conceptual stages. First, the predictors are 
projected onto supervised directions, one for each column of $Y$, using the 
$\gamma$-controlled mixture of weighted predictor variance and weighted predictor–response 
correlation. Second, a canonical correlation analysis (CCA) determines how these 
supervised directions should be linearly combined into a single latent variable that is 
optimally aligned with the primary responses. Auxiliary responses, observation weights, and 
response weights enter the computation of supervised directions in the first stage, shaping 
the latent space that is subsequently analyzed by CCA. The second stage performs CCA only 
against the primary response block; that block may additionally be modified by target 
weights, and the same observation weighting is used there as well.

## Preprocessing and Observation Weights

Let $v_i \ge 0$ denote the observation weight of sample $i$. CPPLS uses these weights in 
centering, optional standard-deviation scaling, predictor-response correlations, and the 
later CCA step. The weighted mean of a variable $x$ is

```math
\bar{x}_w = \frac{\sum_i v_i x_i}{\sum_i v_i}.
```

For a centered variable, the weighted variance and weighted standard deviation
are

```math
\operatorname{Var}_w(x) = \frac{\sum_i v_i x_i^2}{\sum_i v_i},
\qquad
\operatorname{std}_w(x) = \sqrt{\operatorname{Var}_w(x)}.
```

For two centered variables $x$ and $y$, the weighted covariance and weighted
correlation are

```math
\operatorname{Cov}_w(x,y) = \frac{\sum_i v_i x_i y_i}{\sum_i v_i},
```

```math
\operatorname{corr}_w(x,y) =
\frac{\operatorname{Cov}_w(x,y)}
{\sqrt{\operatorname{Var}_w(x)\operatorname{Var}_w(y)}} .
```

The implementation preprocesses predictors, primary responses, and auxiliary
responses asymmetrically because they enter the algorithm differently.

For the predictor block $X$, CPPLS optionally centers and optionally scales
each column. If both are enabled, predictor $x_j$ becomes

```math
\tilde x_j = \frac{x_j - \bar{x}_{j,w}}{\operatorname{std}_w(x_j)}.
```

If only centering is enabled, only the weighted mean is removed. If only scaling is 
enabled, the weighted standard deviation is computed from the uncentered column. 
Columns with zero or non-finite standard deviation are left unscaled by replacing the 
divisor with `1`.

For the primary response block $Y_{\mathrm{prim}}$, CPPLS may scale columns but does not 
apply a separate preprocessing centering step. Thus

```math
\tilde y_k =
\frac{y_k}{\operatorname{std}_w(y_k)}
```

when response scaling is enabled, and $\tilde y_k = y_k$ otherwise. This is not in 
conflict with the correlation-based theory, because the response columns are centered 
internally later when predictor-response correlations are computed.

Auxiliary responses are concatenated after this preprocessing step,

```math
Y = [\,\tilde Y_{\mathrm{prim}} \;\; Y_{\mathrm{aux}}\,],
```

and they are not given separate centering or scaling options. Their influence
is instead controlled through response-column weights in the supervised compression step.

## Supervised Compression

CPPLS constructs a supervised transformation matrix by combining predictor
scale and predictor–response correlation, with the balance controlled by the
power parameter $\gamma \in [0,1]$. The expressions below are written for the
interior case $0 < \gamma < 1$; the endpoint values $\gamma = 0$ and
$\gamma = 1$ are handled in CPPLS as limiting cases. For each predictor $x_j$
(a column of $X$) and each response $y_k$ (a column of the combined response
matrix $Y$), CPPLS computes the weighted standard deviation
$\operatorname{std}_w(x_j)$ and the weighted correlation
$\operatorname{corr}_w(x_j,y_k)$. In the implementation, both the current
predictor matrix and the combined response matrix are centered internally
before these correlations are computed. These quantities are not combined
additively, but multiplicatively through $\gamma$-dependent powers.

The resulting supervised weight matrix is

```math
W_0(\gamma) \in \mathbb{R}^{p \times q},
```

where $p$ is the number of predictors and $q$ is the number of response
columns used to construct the supervised space, including auxiliary ones. It
can be written as a product of a diagonal scale matrix and a correlation
matrix,

```math
W_0(\gamma) = S_x(\gamma)\,C(\gamma),
```

with diagonal entries

```math
S_x(\gamma)_{jj} = \operatorname{std}_w(x_j)^{\frac{1-\gamma}{\gamma}},
```

and correlation entries

```math
C(\gamma)_{jk} = \operatorname{sign}\!\big(\operatorname{corr}_w(x_j,y_k)\big)\, \left|\operatorname{corr}_w(x_j,y_k)\right|^{\frac{\gamma}{1-\gamma}} .
```

Thus, each entry of $W_0(\gamma)$ is proportional to a product of a predictor
standard-deviation term, which serves as a measure of predictor scale, and a
predictor–response correlation term, each raised to a power determined by
$\gamma$. When $\gamma$ is small, predictors with large weighted standard
deviation are emphasized; when $\gamma$ approaches one, predictors that are
strongly correlated with the responses dominate. In traditional PLS workflows,
predictor scaling is often used to control whether high-variance variables
dominate the model. In CPPLS, the relative influence of predictor scale and
predictor–response correlation is instead controlled explicitly through the
power parameter $\gamma$ within the weight construction, and the model
evaluates candidate values of $\gamma$ to identify the supervised
representation of $X$ that is best aligned with $Y$. This reduces the need for
separate scaling decisions as a preprocessing step while making the
variance-correlation trade-off part of the fitted model itself.

## Response Weights and Target Weights

Besides observation weights, this implementation supports two additional
response-column weighting schemes.

Let $r_k \ge 0$ denote the `response_weights` for the columns of the combined
response block $Y$, including auxiliary responses when present. These weights
are injected into the supervised compression step by weighting each response
column before the supervised directions are formed. In matrix form,

```math
Y^{(r)} = Y D_r,
\qquad
D_r = \operatorname{diag}(r_1,\dots,r_q).
```

Equivalently, one may view this as weighting the predictor-response
correlation matrix columnwise:

```math
C^{(r)}_{jk} = r_k\, \operatorname{corr}_w(x_j,y_k).
```

The supervised weight matrix $W_0(\gamma)$ is then built from predictor scale
and these weighted correlations. This is the stage at which both primary and
auxiliary response columns are allowed to have more or less influence on the
construction of the supervised space.

Now let $t_\ell \ge 0$ denote the `target_weights` for the primary-response
columns only. These weights do not alter the construction of $W_0(\gamma)$.
Instead, they are injected into the CCA alignment step through

```math
Y_{\mathrm{prim}}^{(t)} = \tilde Y_{\mathrm{prim}} D_t,
\qquad
D_t = \operatorname{diag}(t_1,\dots,t_{q_{\mathrm{prim}}}).
```

CCA is therefore performed between $Z(\gamma)$ and
$Y_{\mathrm{prim}}^{(t)}$, not between $Z(\gamma)$ and the unweighted primary
response block. Consequently, `target_weights` control how strongly each
primary response contributes when candidate values of $\gamma$ are scored and
when the final canonical direction is chosen.

In short, `response_weights` influence the supervised compression stage,
whereas `target_weights` influence the CCA alignment stage.

Each column of $W_0(\gamma)$ admits a direct geometric interpretation. For
each response variable $y_k$, CPPLS constructs a direction in the original
predictor space that emphasizes predictors with large weighted variance and
predictors that are strongly correlated with that response, with the balance
controlled by the power parameter $\gamma$. These directions represent
response-specific supervised views of the predictor space.

Projecting the predictor matrix onto these directions,

```math
Z(\gamma) = X W_0(\gamma), 
```

maps each sample from the original $p$-dimensional predictor space into a
$q$-dimensional supervised space, where each coordinate summarizes the sample
along a direction tailored to one column of $Y$. Auxiliary response variables
contribute additional supervised directions and thereby enrich this
intermediate representation. In this sense, $W_0(\gamma)$ defines a
supervised low-dimensional coordinate system for the predictors. Every entry
in $Z(\gamma) = X W_0(\gamma)$ is a single scalar summary value for a sample
in the supervised direction associated with a response column. This
representation is an intermediate supervised projection of $X$ and does not
yet define a CPPLS component; rather, it provides the response-guided axes
that are subsequently combined by canonical correlation analysis to form the
final latent component.

## Choosing the Power Parameter

The power parameter $\gamma$ can either be supplied directly by the user or
selected by CPPLS from a user-defined interval or grid of candidate values.
When $\gamma$ is selected rather than fixed, CPPLS evaluates each candidate
value by constructing the supervised compression

```math
Z(\gamma) = X W_0(\gamma), 
```

where the matrix $W_0(\gamma)$ depends on $\gamma$ through a power-based
trade-off between predictor scale, represented by weighted standard
deviations, and predictor–response association, represented by weighted
correlations after response-column weighting. CPPLS then performs a weighted
canonical correlation analysis between $Z(\gamma)$ and the target-weighted
primary response block $Y_{\mathrm{prim}}^{(t)}$, and records the first
(largest) canonical correlation

```math
\rho_1(\gamma) = \operatorname{ccorr}_w\!\big(Z(\gamma),\, Y_{\mathrm{prim}}^{(t)}\big) 
```

as a score for that value of $\gamma$. The optimal value

```math
\gamma_{\mathrm{best}} = \arg\max_{\gamma \in \mathcal{G}} \rho_1(\gamma)
```

is therefore the $\gamma$ whose variance–correlation–weighted construction of
$Z(\gamma)$ yields a representation of $X$ that is maximally aligned, in the
canonical correlation sense, with the primary responses. This step does not
yet extract latent components or deflate the data; it only compares candidate
supervised representations under identical conditions in order to select
$\gamma$. If the user provides a fixed $\gamma$, this selection step is
skipped and CPPLS proceeds directly with that value.

## Component Extraction

Once the optimal $\gamma$ has been determined, CPPLS recomputes
```math
Z = Z(\gamma_{\mathrm{best}})
```
and performs a full weighted CCA between this matrix and the target-weighted
primary response block $Y_{\mathrm{prim}}^{(t)}$. The result is a canonical
direction $a$ in the $Z$-space and a corresponding direction $b$ in the
primary response space. The direction $a$ specifies how to combine the
supervised directions in $Z$ into one axis that maximally correlates with the
weighted primary responses. Providing
$Y_{\mathrm{aux}}$ changes the supervised directions $W_0$, so the
intermediate representation $Z = X W_0$ reflects both the primary responses
and auxiliary structure. The CCA direction $a$ is still chosen only to align
$Z$ with $Y_{\mathrm{prim}}^{(t)}$, but it is chosen inside a supervised space
that already accounts for systematic variation captured by
$Y_{\mathrm{aux}}$. In practice, this means auxiliary variables can steer the
construction of the latent space without becoming prediction targets
themselves.

The canonical direction is then mapped back into the predictor space through
```math
w = W_0(\gamma_{\mathrm{best}})\, a ,
```
producing the final CPPLS weight vector. This vector lies in the original
predictor space and defines the direction used to compute the component score
```math
t = X w .
```

The component score $t$ acts as a latent one-dimensional slider: each sample
receives a coordinate $t_i$, and moving along this latent axis corresponds to
sliding along the component in predictor space. The relationship between the
component and the original variables is captured by the loadings. If
$W = \operatorname{diag}(w_1,\dots,w_n)$ denotes the diagonal matrix of
normalized sample weights, the weighted X-loading is given by
```math
p = \frac{X^\top W t}{t^\top W t},
```
which is the weighted regression of the predictors on the component. Each
entry of $p$ is therefore a linear coefficient describing how the corresponding
predictor changes, on average, as one moves along $t$. The weighted Y-loading
```math
c = \frac{Y^\top W t}{t^\top W t}
```
likewise contains linear coefficients describing how each response variable,
including auxiliary responses when present, varies with the component under
the weighting structure.

Deflation removes the part of $X$ explained by this component:
```math
X \leftarrow X - t p^\top .
```

After this deflation, the dominant predictor structure captured by the current
component has been removed from $X$. Subsequent components are therefore
extracted from a predictor matrix that no longer contains the previously
explained direction, which helps later components describe remaining
structured variation rather than repeating the same dominant signal. In
discriminant analysis, for example, the first component may capture most of
the class separation, while later components may describe residual within-class
structure or additional sources of systematic variation.

## Sample Weights and Auxiliary Responses

Sample weighting becomes particularly important in discriminant analysis
(CPPLS-DA) when classes are unbalanced, when some samples are more reliable
than others, or when the sampling design itself introduces unequal
representation. For example, if one class has 20 samples and another 60, an
unweighted analysis gives the larger class three times the influence on
weighted variances, covariances, and correlations. The supervised compression
can then be driven more by variation inside the majority class than by the
separation between classes. By assigning larger weights to the minority class
and smaller weights to the majority class, the effective contribution of each
class can be balanced. The resulting components better reflect between-class
structure instead of mostly tracking the most common samples.

Weights can also compensate for auxiliary structure that is unevenly
distributed across classes. If collection time, instrument batch, or another
nuisance factor is associated more strongly with one class than another, that
imbalance can distort the estimated correlations used to build $W_0(\gamma)$.
Reweighting samples can reduce this distortion by making the weighted
covariance structure better reflect the comparison of interest. Similarly,
samples known to be noisy or unreliable can be down-weighted, while
representative or carefully controlled samples can be up-weighted, improving
the robustness of the extracted components.

Auxiliary responses address the problem of structured variation in $X$ that is
not itself the primary target but can still influence the extracted
components. Instead of changing how much influence each sample has, they tell
CPPLS which additional structured variation should be represented when
constructing the supervised space. If a nuisance factor such as collection
date, instrument batch, or processing condition explains part of the variation
in $X$, adding it as a column in $Y_{\mathrm{aux}}$ gives CPPLS a dedicated
supervised direction for that factor. The primary responses are still the only
targets used when selecting the final canonical direction and when building the
prediction model, but that direction is now chosen in a latent space that has
already organized part of the nuisance variation explicitly. This reduces the
risk that nuisance structure is absorbed into the primary component simply
because it happens to correlate with the target labels in the observed sample.

A concrete example illustrates the benefit of combining auxiliary responses and
sample weighting. Suppose two insect species are analyzed by GC-MS to
characterize their cuticular hydrocarbons. The primary task is to discriminate
species, but chemical profiles also change with season, and the two species
may not be collected uniformly throughout the year. In this situation, the
largest variation in $X$ may reflect seasonal drift rather than species. If
season is omitted from the response structure, peaks that vary with collection
date may appear spuriously associated with species because the sampling times
differ. Including collection date or season as an auxiliary response gives
CPPLS a supervised direction that represents this temporal effect explicitly.
If the sampling is also imbalanced, class-balanced weights can prevent one
species or one part of the season from dominating the covariance estimates used
to build the components. Together, auxiliary responses and sample weights help
separate the biological signal of interest from structured sampling effects,
producing a more stable and interpretable model.

## Final Regression Model

After all components are extracted, regression coefficients for predicting the
primary responses are assembled using
```math
B =
W_{\mathrm{comp}}
\left( P^\top W_{\mathrm{comp}} \right)^{-1}
C_{\mathrm{primary}}^\top ,
```
where $W_{\mathrm{comp}}$ contains the component weight vectors, $P$ the
corresponding X-loadings, and $C_{\mathrm{primary}}$ the primary Y-loadings.

Only the primary response block contributes to $C_{\mathrm{primary}}$ and thus
to the final coefficient matrix $B$. Auxiliary responses affect the fitted
model indirectly, by shaping the supervised compression and therefore the
extracted components, but they are not themselves predicted.

In summary, the core CPPLS mechanism in this package is the
$\gamma$-controlled supervised compression followed by CCA. The implementation
further allows auxiliary responses, sample weights, and package-specific
response-column weighting through `response_weights` and `target_weights`.
Together, these controls make it possible to tailor the fitted model to
complex, high-dimensional, and potentially confounded data settings without
blurring the distinction between the CPPLS core algorithm and implementation
extensions.
