# GTA: a unified method and the AVT-v1 instance

This document specifies the implemented method. It reports no new experimental
performance. The release implements the AVT instance; the historical airway
instance retains its own architecture, training protocol and experimental identity.

## Anatomical attribute-conditioned feature adaptation

Let $X$ be a CT volume and $Y$ its training segmentation. For anatomical task
$t$, a fixed geometry procedure constructs attribute targets
$a_t^*=\Phi_t(Y)$ and their supervision domains. An image-only estimator
$P_t$ predicts $\hat a_t=P_t(X)$ and multi-scale features $H_t$.
At selected segmentation layers, attribute features are fused with segmentation
features and attributes modulate the subsequent transformation:

$$B_l=\operatorname{Fuse}_l(F_l,H_t),\qquad
  F'_l=B_l+\Delta_{t,l}(B_l,\hat a_t).$$

This formulation includes multiplicative enhancement by expressing it as an
increment relative to $B_l$. It defines an implementation principle rather than
a theorem: geometrically motivated attributes should control feature processing
where the corresponding local heterogeneity occurs. Attribute definitions,
prediction granularity and conditional operators may differ between tasks.
Predictions from the same CT introduce a supervised inductive bias, not additional
test-time observations.

The airway instance uses the original discrete generation, morphology and location
attributes and its original adaptive modules. The AVT instance uses continuous
physical scale with explicit applicability. Aortic radius is not renamed airway
generation, and no depth-to-calibre equivalence is assumed. The exact airway
category counts, integration sites and training schedule must follow the version
that produced the original results.

## AVT attributes and estimator

The native-spacing Euclidean distance transform of the annotated lumen is sampled
at centreline nodes obtained by Lee thinning on the 1 mm grid. The resulting
$r$ is a geometric local inscribed-scale proxy. We use
$R=\log(r/1\mathrm{mm})$. A fixed partial measurement procedure associates
queries with reliable nonjunction nodes whose radius balls remain inside the
observed domain. Junction, censored and conflicting support is excluded from
radius measurement. Exact rules and reason codes are in `gta_avt/geometry.py`.

Operational applicability is $A(q)=1$ iff this procedure emits a radius at query
$q$; otherwise $A(q)=0$ on the real image grid. Padding is excluded. Thus $A$
is neither the vessel mask nor calibrated prediction confidence. An unavailable
radius is not supervised as radius zero. TRAIN-only, case-equal reference moments
give $z_R=(R-\mu)/\sigma$ and an emission prior $\pi$.

The attribute estimator has the historical eight Conv3D--BN--ReLU blocks with
stage widths 32, 64, 128 and 256; three SE blocks; and a 448-to-256 multiscale
fusion. Its local 64-channel feature and interpolated fused 256-channel feature
feed a 1x1x1, 320-to-2 head. Outputs are interpolated to half the input resolution.
The head is initialized to $p_A=\pi$ and $z_R=0$. Independent pretraining minimizes
applicability BCE over real cells and standardized-radius MSE only on $A=1$.
Each loss first averages its supported cells within a case, then averages cases.

## Segmentation and bounded local routing

The segmentation encoder has widths 32/64/128/256 and the three decoder widths
are 128/64/32. It uses the active historical convolution, dilated residual and
attention primitives. One frozen attribute estimator provides both conditions and
features. The deep decoder retains the historical feature-fusion/enhancement
operator with fixed conditions; dynamic AVT conditioning is isolated at the
64-channel middle decoder. A segmentation-owned, trainable 256-to-128-to-16
guidance adapter also consumes frozen fused attribute features. Decoder projections
provide six channels, concatenated with this guidance before a 22-to-32 projection
and two-class output. The binary foreground logit is foreground minus background.

At the middle decoder, the original four 64-to-16 convolution branches have
dilations $(1,1,2,3)$. Let their concatenation be $h=[h_0,h_1,h_2,h_3]$, the shared
1x1 projection be $W$, its bias be $b_W$, and
$k=(1-1/3)(1+0.1/3)$. With $B$ denoting fused segmentation/attribute features,

$$F_0=B+k(Wh+b_W),$$
$$D_1=kW[-h_0,-h_1,h_2,h_3],\quad
D_2=kW[h_0,-h_1,0,0],\quad D_3=kW[0,0,h_2,-h_3].$$

Using $\hat A=\mathbf1[p_A\ge0.5]$ and $c_R=\hat A\hat z_R$, three
channelwise, spatially varying gains are

$$g_j=\tanh(b_j+u_j\hat A+w_jc_R),\quad j=1,2,3,$$
$$F=F_0+\alpha g_1D_1+
\alpha(1-|g_1|)(g_2D_2+g_3D_3),\qquad0\le\alpha\le1.$$

The 576 parameters $b_j,u_j,w_j$ are initialized to zero. The equivalent four
branch coefficients are
$1+\alpha[-g_1\pm(1-|g_1|)g_2]$ and
$1+\alpha[g_1\pm(1-|g_1|)g_3]$. They lie in $[0,2]$ and sum to four in
real/same-precision arithmetic. This is not feature-energy conservation or a
topology guarantee. The branches are convolutional alternatives, not proven
anatomical experts. $\hat A=0$ neutralizes the attribute path without vetoing
foreground prediction.

## Staged training and optional shape auxiliary

After independent attribute pretraining and validation against constant and
case-centred spatial references, the estimator is frozen, including BN state.
The segmentation network is randomly initialized and trained with $\alpha=0$.
Matched C0, A-only and AR continuations load the same warmup checkpoint within
each seed. C0 sets both conditional inputs to zero but retains learned routing
biases; A-only consumes $\hat A$; AR also consumes $c_R$. All segmentation
parameters remain trainable. Routing strength is ramped according to a declared
schedule, with common update budgets and validation selection. Training and normal
inference both use predicted conditions, never an oracle-to-prediction substitution.
Independent full repeats retrain both networks. Same-weight radius re-pairing is
an additional correspondence diagnostic, not another trained baseline.

The optional Gaussian auxiliary predicts

$$G(q)=\frac{Y(q)}{s_{\mathrm{TRAIN}}}
\log\left(1+\sum_{c\in C}
\exp[-\|q-c\|_{\mathrm{mm}}^2/(2r_c^2)]\right).$$

All full-case centreline seeds contribute. A fixed TRAIN, case-equal RMS scale
normalizes the log field. A 32-to-1 projection and Softplus regress this target
with valid-domain MSE. Known background has target zero; padding and foreground
with unavailable whole-case geometry are excluded. This is a shape auxiliary,
not a geodesic distance or general topology-preserving constraint. Its loss weight
is independently configurable and its benefit requires a matched on/off test.
The segmentation objective combines Dice, BCE and optional Gaussian/deep
supervision terms, with all weights recorded in the configuration.

Inference uses CT-only intensity correction, orientation/resampling, an image-derived
crop, overlapping tiles and probability blending, followed by native-space
restoration. Neither annotations nor geometric targets enter inference.

## Paper claims and evidence scope

Attribute-feature learning, dynamic consumption and Gaussian supervision are
distinct contributions. Their effects require separate matched comparisons;
mean accuracy, case dispersion, poor-case behaviour and training-seed variability
are separate outcomes. Original airway results support only that original
instance. Prior AVT local adaptation studies do not constitute from-initialization
evaluation of this complete v1 pipeline. This code release does not claim such
evaluation has occurred.

Feature conditioning and adaptive multiscale selection already have established
precedents, including [FiLM](https://arxiv.org/abs/1709.07871) and
[Selective Kernel Networks](https://arxiv.org/abs/1903.06586). GTA should position
its contribution around explicit task-grounded attributes, their controlled
consumption and cross-anatomy instantiation, rather than claiming to introduce
conditional computation itself. The supplied earlier draft uses HARL (and HRAL
in some tables), whereas the project uses GTA; final naming and original-version
details require consistent editorial alignment.
