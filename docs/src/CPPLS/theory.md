# Canonical Powered Partial Least Squares (CPPLS)

Canonical Powered Partial Least Squares (CPPLS) is a supervised projection method for regression and classification. Its purpose is to extract latent components that summarize the predictor matrix $X \in \mathbb{R}^{n \times p}$ in a way that best reflects the structure in a multivariate response matrix $Y \in \mathbb{R}^{n \times q}$. The method extends standard PLS in three important ways. First, it allows multiple response variables, including both primary responses that one wishes to predict and optional auxiliary responses that guide the extraction of components. Second, it incorporates a power parameter $\gamma$ that controls the balance between predictor variance and predictor–response correlation, giving the user explicit control over how strongly the model should emphasize correlation structure. Third, CPPLS can operate with a vector of non-negative sample weights, allowing some observations to contribute more or less to the fitted model. This is useful when classes are unbalanced, when some samples are more reliable, or when experimental design considerations suggest that certain samples should carry greater influence.

Each CPPLS component is extracted in two conceptual stages. First, the predictors are projected onto supervised directions, one for each column of $Y$, using the $\gamma$-controlled mixture of weighted predictor variance and weighted predictor–response correlation. Second, a canonical correlation analysis (CCA) determines how these supervised directions should be linearly combined into a single latent variable that is optimally aligned with the primary responses. Auxiliary responses and observation weights enter the computation of supervised directions in the first stage, shaping the latent space that is subsequently analyzed by CCA, while the CCA itself is guided solely by the primary responses under the same weighting structure.

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


CPPLS constructs a supervised transformation matrix by combining predictor scale and predictor–response correlation, with the balance controlled by the power parameter $\gamma \in (0,1)$. For each predictor $x_j$ (a column of $X$) and each response $y_k$ (a column of $Y$), CPPLS computes the weighted standard deviation $\operatorname{std}_w(x_j)$ and the weighted correlation $\operatorname{corr}_w(x_j,y_k)$. These quantities are not combined additively, but multiplicatively through $\gamma$-dependent powers.

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

Thus, each entry of $W_0(\gamma)$ is proportional to a product of a predictor-scale term and a predictor–response correlation term, each raised to a power determined by $\gamma$. When $\gamma$ is small, predictors with large weighted standard deviation are emphasized; when $\gamma$ approaches one, predictors that are strongly correlated with the responses dominate. In traditional PLS workflows, predictor scaling is often used to control whether high-variance variables dominate the model. In CPPLS, the relative influence of predictor scale and predictor–response correlation is instead controlled explicitly through the power parameter $\gamma$ within the weight construction. This reduces the need for separate scaling decisions as a preprocessing step.

Each column of $W_0(\gamma)$ admits a direct geometric interpretation. For each response variable $y_k$, CPPLS constructs a direction in the original predictor space that emphasizes predictors with large weighted variance and predictors that are strongly correlated with that response, with the balance controlled by the power parameter $\gamma$. These directions represent response-specific supervised views of the predictor space.

Projecting the predictor matrix onto these directions,

```math
Z(\gamma) = X W_0(\gamma), 
```

maps each sample from the original $p$-dimensional predictor space into a $q$-dimensional supervised space, where each coordinate summarizes the sample along a direction tailored to one column of $Y$. Auxiliary response variables contribute additional supervised directions and thereby enrich this intermediate representation. In this sense, $W_0(\gamma)$ defines a supervised low-dimensional coordinate system for the predictors. Every entry in $Z(\gamma) = X W_0(\gamma)$ is a single scalar summary value for a sample in the supervised direction associated with a response column. This representation is an intermediate supervised projection of $X$ and does not yet define a CPPLS component; rather, it provides the response-guided axes that are subsequently combined by canonical correlation analysis to form the final latent component.

To choose the power parameter $\gamma$, CPPLS evaluates a grid of candidate values. For each candidate $\gamma$, it constructs the supervised compression

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

is therefore the $\gamma$ whose variance–correlation–weighted construction of $Z(\gamma)$ yields a representation of $X$ that is maximally aligned, in the canonical correlation sense, with the primary responses. This step does not yet extract latent components or deflate the data; it only compares candidate supervised representations under identical conditions in order to select $\gamma$.

Once the optimal $\gamma$ has been determined, CPPLS recomputes
```math
Z = Z(\gamma_{\mathrm{best}})
```
and performs a full weighted CCA between this matrix and the primary response columns of $Y$. The result is a canonical direction $a$ in the $Z$-space and a corresponding direction $b$ in the primary response space. The direction $a$ specifies how to combine the supervised directions in $Z$ into one axis that maximally correlates with the primary responses. 
Providing $Y_{\mathrm{aux}}$ changes the supervised directions $W_0$, so the intermediate representation $Z = X W_0$ reflects both the primary responses and auxiliary structure. The CCA direction $a$ is then chosen in this augmented space, which can rotate it toward directions that are jointly aligned with $Y_{\mathrm{prim}}$ and $Y_{\mathrm{aux}}$. This helps prevent variance that is mainly explained by $Y_{\mathrm{aux}}$ from being misattributed to $Y_{\mathrm{prim}}$.

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
which is the weighted regression of the predictors on the component. It describes how each predictor changes as one moves along $t$. The weighted Y-loading
```math
c = \frac{Y^\top W t}{t^\top W t}
```
describes how each response variable—including auxiliary responses when present—varies with the component under the weighting structure.

Deflation removes the part of $X$ and $Y$ that can be explained by this component:
```math
X \leftarrow X - t p^\top,\qquad
Y \leftarrow Y - t c^\top .
```

After this deflation, the dominant species structure has been removed from both $X$ and $Y$. Because subsequent components are extracted from these residuals, they often describe remaining structured variation—such as seasonal drift or batch effects—rather than additional species separation. This does not impair discrimination: the first component captures the primary class separation, and later components model the remaining confounding structure, improving stability and interpretability.

Sample weighting becomes particularly important in discriminant analysis (CPPLS-DA) when classes are unbalanced or when certain samples are more reliable. For example, if one species has 20 samples and another 60, the unweighted analysis gives the larger class three times the influence on variances and correlations, and thus on the supervised compression. This often causes the dominant component to reflect within-class variation of the majority class rather than the between-class difference one intends to model. By assigning larger weights to minority-class samples and smaller weights to majority-class samples, the total effective weight of each class becomes balanced. All weighted variances, covariances, and correlations then reflect this adjusted influence, and the extracted components focus on genuine between-class separation rather than being overwhelmed by the larger class. Similarly, samples known to be noisy or unreliable can be down-weighted, while representative or carefully controlled samples can be up-weighted, improving the robustness of the extracted components.

A concrete example illustrates the benefit of combining auxiliary responses and sample weighting. Suppose two insect species are analyzed by GC–MS to characterize their cuticular hydrocarbons. The primary task is to discriminate species, but chemical profiles change with season, and species may not be collected uniformly throughout the year. In this situation, the largest variation in $X$ may reflect seasonal drift rather than species. Even if the response $Y$ encodes only species, the supervised compression may inadvertently emphasize seasonal structure, because GC–MS peaks that vary with season often correlate indirectly with species when sampling times differ. Including sampling date or season as an auxiliary Y-column provides CPPLS with a supervised direction dedicated to seasonal variation, preventing this structure from leaking into the species component. If sampling imbalance is also present—say one species is collected mostly early in the season and the other mostly later—assigning balanced sample weights prevents the majority class or sampling pattern from dominating the supervised compression. Together, auxiliary $Y$ and sample weighting produce a far more stable and interpretable latent structure, with the X-loadings highlight species-specific signal.

After all components are extracted, regression coefficients for predicting the primary responses are assembled using
```math
B =
W_{\mathrm{comp}}
\left( P^\top W_{\mathrm{comp}} \right)^{-1}
C_{\mathrm{primary}}^\top ,
```
where $W_{\mathrm{comp}}$ contains the component weight vectors, $P$ the corresponding X-loadings, and $C_{\mathrm{primary}}$ the primary Y-loadings. Auxiliary responses influence the latent components but do not appear in the final regression model unless explicitly designated as prediction targets.

In summary, CPPLS combines three complementary forms of supervision: the power parameter $\gamma$ that controls the balance between variance and correlation, auxiliary responses that provide additional structured guidance for the supervised compression, and sample weights that ensure appropriate influence of different samples or classes. Together, these features allow CPPLS to build stable, interpretable, and discriminative models even in complex, high-dimensional, and confounded data settings.
