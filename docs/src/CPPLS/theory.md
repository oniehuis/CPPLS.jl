# Theory

Canonical Powered Partial Least Squares (CPPLS) is a supervised projection method for regression and classification. Its purpose is to extract latent components that summarize the predictor matrix $X \in \mathbb{R}^{n \times p}$, where $n$ is the number of samples and $p$ the number of predictors, in a way that best reflects the structure in a multivariate response matrix $Y \in \mathbb{R}^{n \times q}$, where $q$ is the number of response variables. The method extends standard PLS in three important ways. First, it allows multiple response variables, including both primary responses that one wishes to predict and optional auxiliary responses that guide the extraction of components. Second, it incorporates a power parameter $\gamma$ that controls the balance between predictor variance and predictor–response correlation, giving the user explicit control over how strongly the model should emphasize correlation structure. Third, CPPLS can operate with a vector of non-negative sample weights, allowing some observations to contribute more or less to the fitted model. This is useful when classes are unbalanced, when some samples are more reliable, or when experimental design considerations suggest that certain samples should carry greater influence.

Each CPPLS component is extracted in two conceptual stages. First, the predictors are projected onto supervised directions, one for each column of $Y$, using the $\gamma$-controlled mixture of weighted predictor variance and weighted predictor–response correlation. Second, a canonical correlation analysis (CCA) determines how these supervised directions should be linearly combined into a single latent variable that is optimally aligned with the primary responses. Auxiliary responses and observation weights enter the computation of supervised directions in the first stage, shaping the latent space that is subsequently analyzed by CCA, while the CCA itself is guided solely by the primary responses under the same weighting structure.

## Weighted Preprocessing

To begin, $X$ and $Y$ are centered using the supplied sample weights. If $w_i$ is the weight of sample $i$ and the weights are normalized to sum to one, the weighted mean of a variable $x$ becomes
```math
\bar{x} = \sum_i w_i x_i .
```
All variances, covariances, and correlations are computed in a weighted sense. For a centered variable $x$, the weighted variance is
```math
\operatorname{Var}_w(x) = \sum_i w_i x_i^2 ,
```
and for two centered variables $x$ and $y$, the weighted covariance is
```math
\operatorname{Cov}_w(x,y) = \sum_i w_i x_i y_i .
```
Weighted correlations are obtained by normalizing the weighted covariance by the corresponding weighted standard deviations.
For two centered variables $x$ and $y$, the weighted correlation is
```math
\operatorname{corr}_w(x,y) =
\frac{\operatorname{Cov}_w(x,y)}
{\sqrt{\operatorname{Var}_w(x)\operatorname{Var}_w(y)}} .
```

## Supervised Compression

CPPLS constructs a supervised transformation matrix by combining predictor scale and predictor–response correlation, with the balance controlled by the power parameter $\gamma \in [0,1]$. The expressions below are written for the interior case $0 < \gamma < 1$; the endpoint values $\gamma = 0$ and $\gamma = 1$ are handled in CPPLS as limiting cases. For each predictor $x_j$ (a column of $X$) and each response $y_k$ (a column of $Y$), CPPLS computes the weighted standard deviation $\operatorname{std}_w(x_j)$ and the weighted correlation $\operatorname{corr}_w(x_j,y_k)$. These quantities are not combined additively, but multiplicatively through $\gamma$-dependent powers.

The resulting supervised weight matrix is

```math
W_0(\gamma) \in \mathbb{R}^{p \times q},
```

where $p$ is the number of predictors and $q$ is the number of response columns, including auxiliary ones. It can be written as a product of a diagonal scale matrix and a correlation matrix,

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

Thus, each entry of $W_0(\gamma)$ is proportional to a product of a predictor standard-deviation term, which serves as a measure of predictor scale, and a predictor–response correlation term, each raised to a power determined by $\gamma$. When $\gamma$ is small, predictors with large weighted standard deviation are emphasized; when $\gamma$ approaches one, predictors that are strongly correlated with the responses dominate. In traditional PLS workflows, predictor scaling is often used to control whether high-variance variables dominate the model. In CPPLS, the relative influence of predictor scale and predictor–response correlation is instead controlled explicitly through the power parameter $\gamma$ within the weight construction, and the model evaluates candidate values of $\gamma$ to identify the supervised representation of $X$ that is best aligned with $Y$. This reduces the need for separate scaling decisions as a preprocessing step while making the variance-correlation trade-off part of the fitted model itself.

Each column of $W_0(\gamma)$ admits a direct geometric interpretation. For each response variable $y_k$, CPPLS constructs a direction in the original predictor space that emphasizes predictors with large weighted variance and predictors that are strongly correlated with that response, with the balance controlled by the power parameter $\gamma$. These directions represent response-specific supervised views of the predictor space.

Projecting the predictor matrix onto these directions,

```math
Z(\gamma) = X W_0(\gamma), 
```

maps each sample from the original $p$-dimensional predictor space into a $q$-dimensional supervised space, where each coordinate summarizes the sample along a direction tailored to one column of $Y$. Auxiliary response variables contribute additional supervised directions and thereby enrich this intermediate representation. In this sense, $W_0(\gamma)$ defines a supervised low-dimensional coordinate system for the predictors. Every entry in $Z(\gamma) = X W_0(\gamma)$ is a single scalar summary value for a sample in the supervised direction associated with a response column. This representation is an intermediate supervised projection of $X$ and does not yet define a CPPLS component; rather, it provides the response-guided axes that are subsequently combined by canonical correlation analysis to form the final latent component.

## Choosing the Power Parameter

The power parameter $\gamma$ can either be supplied directly by the user or selected by CPPLS from a user-defined interval or grid of candidate values. When $\gamma$ is selected rather than fixed, CPPLS evaluates each candidate value by constructing the supervised compression

```math
Z(\gamma) = X W_0(\gamma), 
```

where the matrix $W_0(\gamma)$ depends on $\gamma$ through a power-based trade-off between predictor scale, represented by weighted standard deviations, and predictor–response association, represented by weighted correlations. CPPLS then performs a weighted canonical correlation analysis between $Z(\gamma)$ and the primary response block $Y_{\mathrm{prim}}$, and records the first (largest) canonical correlation

```math
\rho_1(\gamma) = \operatorname{ccorr}_w\!\big(Z(\gamma),\, Y_{\mathrm{prim}}\big) 
```

as a score for that value of $\gamma$. The optimal value

```math
\gamma_{\mathrm{best}} = \arg\max_{\gamma \in \mathcal{G}} \rho_1(\gamma)
```

is therefore the $\gamma$ whose variance–correlation–weighted construction of $Z(\gamma)$ yields a representation of $X$ that is maximally aligned, in the canonical correlation sense, with the primary responses. This step does not yet extract latent components or deflate the data; it only compares candidate supervised representations under identical conditions in order to select $\gamma$. If the user provides a fixed $\gamma$, this selection step is skipped and CPPLS proceeds directly with that value.

## Component Extraction

Once the optimal $\gamma$ has been determined, CPPLS recomputes
```math
Z = Z(\gamma_{\mathrm{best}})
```
and performs a full weighted CCA between this matrix and the primary response columns of $Y$. The result is a canonical direction $a$ in the $Z$-space and a corresponding direction $b$ in the primary response space. The direction $a$ specifies how to combine the supervised directions in $Z$ into one axis that maximally correlates with the primary responses. 
Providing $Y_{\mathrm{aux}}$ changes the supervised directions $W_0$, so the intermediate representation $Z = X W_0$ reflects both the primary responses and auxiliary structure. The CCA direction $a$ is still chosen only to align $Z$ with $Y_{\mathrm{prim}}$, but it is chosen inside a supervised space that already accounts for systematic variation captured by $Y_{\mathrm{aux}}$. In practice, this means auxiliary variables can steer the construction of the latent space without becoming prediction targets themselves.

The canonical direction is then mapped back into the predictor space through
```math
w = W_0(\gamma_{\mathrm{best}})\, a ,
```
producing the final CPPLS weight vector. This vector lies in the original predictor space and defines the direction used to compute the component score
```math
t = X w .
```

The component score $t$ acts as a latent one-dimensional slider: each sample receives a coordinate $t_i$, and moving along this latent axis corresponds to sliding along the component in predictor space. The relationship between the component and the original variables is captured by the loadings. The weighted X-loading is given by
```math
p = \frac{X^\top W t}{t^\top W t},
```
which is the weighted regression of the predictors on the component. Each entry of $p$ is therefore a linear coefficient describing how the corresponding predictor changes, on average, as one moves along $t$. The weighted Y-loading
```math
c = \frac{Y^\top W t}{t^\top W t}
```
likewise contains linear coefficients describing how each response variable, including auxiliary responses when present, varies with the component under the weighting structure.

Deflation removes the part of $X$ and $Y$ that can be explained by this component:
```math
X \leftarrow X - t p^\top,\qquad
Y \leftarrow Y - t c^\top .
```

After this deflation, the dominant structure captured by the current component has been removed from both $X$ and $Y$. Because subsequent components are extracted from the deflated matrices, they often describe remaining structured variation rather than repeating the same dominant signal. In discriminant analysis, for example, the first component may capture most of the class separation, while later components may describe residual within-class structure or additional sources of systematic variation.

## Sample Weights and Auxiliary Responses

Sample weighting becomes particularly important in discriminant analysis (CPPLS-DA) when classes are unbalanced, when some samples are more reliable than others, or when the sampling design itself introduces unequal representation. For example, if one class has 20 samples and another 60, an unweighted analysis gives the larger class three times the influence on weighted variances, covariances, and correlations. The supervised compression can then be driven more by variation inside the majority class than by the separation between classes. By assigning larger weights to the minority class and smaller weights to the majority class, the effective contribution of each class can be balanced. The resulting components better reflect between-class structure instead of mostly tracking the most common samples.

Weights can also compensate for auxiliary structure that is unevenly distributed across classes. If collection time, instrument batch, or another nuisance factor is associated more strongly with one class than another, that imbalance can distort the estimated correlations used to build $W_0(\gamma)$. Reweighting samples can reduce this distortion by making the weighted covariance structure better reflect the comparison of interest. Similarly, samples known to be noisy or unreliable can be down-weighted, while representative or carefully controlled samples can be up-weighted, improving the robustness of the extracted components.

Auxiliary responses address the problem of structured variation in $X$ that is not itself the primary target but can still influence the extracted components. Instead of changing how much influence each sample has, they tell CPPLS which additional structured variation should be represented when constructing the supervised space. If a nuisance factor such as collection date, instrument batch, or processing condition explains part of the variation in $X$, adding it as a column in $Y_{\mathrm{aux}}$ gives CPPLS a dedicated supervised direction for that factor. The primary responses are still the only targets used when selecting the final canonical direction and when building the prediction model, but that direction is now chosen in a latent space that has already organized part of the nuisance variation explicitly. This reduces the risk that nuisance structure is absorbed into the primary component simply because it happens to correlate with the target labels in the observed sample.

A concrete example illustrates the benefit of combining auxiliary responses and sample weighting. Suppose two insect species are analyzed by GC-MS to characterize their cuticular hydrocarbons. The primary task is to discriminate species, but chemical profiles also change with season, and the two species may not be collected uniformly throughout the year. In this situation, the largest variation in $X$ may reflect seasonal drift rather than species. If season is omitted from the response structure, peaks that vary with collection date may appear spuriously associated with species because the sampling times differ. Including collection date or season as an auxiliary response gives CPPLS a supervised direction that represents this temporal effect explicitly. If the sampling is also imbalanced, class-balanced weights can prevent one species or one part of the season from dominating the covariance estimates used to build the components. Together, auxiliary responses and sample weights help separate the biological signal of interest from structured sampling effects, producing a more stable and interpretable model.

## Final Regression Model

After all components are extracted, regression coefficients for predicting the primary responses are assembled using
```math
B =
W_{\mathrm{comp}}
\left( P^\top W_{\mathrm{comp}} \right)^{-1}
C_{\mathrm{primary}}^\top ,
```
where $W_{\mathrm{comp}}$ contains the component weight vectors, $P$ the corresponding X-loadings, and $C_{\mathrm{primary}}$ the primary Y-loadings.

Only the primary response block contributes to $C_{\mathrm{primary}}$ and thus to the final coefficient matrix $B$. Auxiliary responses affect the fitted model indirectly, by shaping the supervised compression and therefore the extracted components, but they are not themselves predicted.

## Summary

In summary, CPPLS combines three complementary forms of supervision: the power parameter $\gamma$ that controls the balance between variance and correlation, auxiliary responses that provide additional structured guidance for the supervised compression, and sample weights that ensure appropriate influence of different samples or classes. Together, these features allow CPPLS to build stable, interpretable, and discriminative models even in complex, high-dimensional, and confounded data settings.
