# Cover Letter — *The European Physical Journal C* (Computing, Software and Data Science)

**To:** Editorial Board, *The European Physical Journal C (EPJC)* — Section: *Computing, Software and Data Science*  
**Manuscript Title:** *Non-monotonic benefits of physical target centering in conditional models of pion momentum loss*  
**Authors:** Xuansong Chang, Yiran Xiao, and Chengtao Song  

Dear Editors,

We submit our original research manuscript, *"Non-monotonic benefits of physical target centering in conditional models of pion momentum loss,"* for consideration as a Regular Article in the **Computing, Software and Data Science** section of *The European Physical Journal C*.

### Motivation and Fit for EPJC (*Computing, Software and Data Science*)

Deep generative models—particularly conditional normalizing flows and physics-constrained generative frameworks—are increasingly deployed in HEP both for fast Monte Carlo detector transport surrogates (e.g., *ParaFlow*, *Eur. Phys. J. C* **85**, 857, 2025) and for recovering sub-granularity physical structure from coarse detector readouts (e.g., Liu & Tung, *Eur. Phys. J. C* **86**, 1351, 2026). In forward hybrid surrogates, analytic physics baselines are often combined with learned residual models, yet while an invertible target translation leaves an ideal conditional density family invariant under complete conditioning, finite-capacity surrogates trained under finite optimization budgets or reduced/tabulated neural conditioning inputs do not share that invariance.

Rather than proposing yet another heuristic surrogate architecture, this work contributes a **pre-registered, strictly controlled methodological study** of how prescribed Bethe–Bloch physical target centering interacts with the momentum information available to a conditional rational-quadratic spline flow on simulated charged-pion ($0.3\text{--}1.0\text{ GeV}/c$) transport in thin silicon slabs ($0.1, 0.3, 1.0\text{ mm}$, Geant4 11.4.1 `FTFP_BERT`). All 80 primary models, 160 secondary/control models, preprocessing parameters, and contrast families were cryptographically frozen prior to generating two independent $500{,}000$-event confirmation catalogs (`C` and `D`).

Our principal findings provide concrete methodological guidance for hybrid fast-simulation design in HEP:
1. **Non-monotonic representation–information interaction:** Relative centering gains over matched raw-target flows peak at intermediate two-bin momentum conditioning ($0.0104\text{--}0.0112\text{ nat/event}$) rather than under no direct momentum or exact momentum, consistently across all five paired optimizer seeds and both confirmation catalogs.
2. **Relative centering gain vs. absolute likelihood:** Exact incident momentum in the neural conditioner still yields the lowest absolute negative log-likelihood for every representation, demonstrating that relative centering benefit and absolute predictor quality select different operating points.
3. **Regional trade-offs and physical mechanism:** Positive catalog-mean gains coexist with statistically significant regional likelihood losses and degraded high-momentum quantile calibration (20–23 of 42 strata simultaneously negative under no direct momentum), consistent with the divergence between the nearly flat Landau–Vavilov–Bichsel bulk median and the relativistic rise of unscreened mean Bethe–Bloch stopping power above minimum ionization.

### Open Science, Reproducibility, and Declarations

In full alignment with EPJC's open-access (SCOAP³) and reproducibility standards, all source code, frozen protocol contracts, 240 run manifests, cryptographic SHA-256 checkpoint digests, saved 2,000-draw paired bootstrap replicates, and a single-command isolated verification script (`python reproduce.py --output reproduced`) that reconstructs all 21 figures, all main and supplementary tables, and all diagnostic summaries in ~30 seconds are openly archived under a dual MIT / CC-BY-4.0 license at:

> https://github.com/xschang1002/EPJC (release tag `v1.0.0-epjc`)

The manuscript has not been published previously and is not under consideration elsewhere. All authors have read and approved the submission and declare no competing interests.

Thank you for your time and consideration.

Sincerely,  
**Xuansong Chang** (on behalf of all co-authors)  
Anhui University, Hefei, China  
E-mail: alec.xs.chang@gmail.com
