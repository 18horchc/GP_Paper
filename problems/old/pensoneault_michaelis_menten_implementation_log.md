# Implementation log: Pensoneault nonnegative GP on Michaelis–Menten assay data

This document records how we adapted **Pensoneault et al. (2020), “Nonnegativity-Enforced Gaussian Process Regression”** (arXiv:2004.04632 / *Theoretical and Applied Mechanics Letters*) for the **`michaelis_menten.m`** script, what we changed relative to the paper’s stated numerical recipe, and why. It also contrasts the **constrained** procedure with the **unconstrained** GPML baseline.

**Primary code file:** [`michaelis_menten.m`](./michaelis_menten.m)

---

## 1. What the paper prescribes (baseline recipe)

The method keeps the **standard GP regression model** (Gaussian likelihood, SE kernel, zero mean in their examples). It does **not** change the posterior formulas; it changes **how hyperparameters** \(\theta = (\ell, \sigma_f, \sigma_n)\) are chosen.

| Ingredient | Paper (as written) |
|------------|---------------------|
| **Objective** | Minimize **negative log marginal likelihood (NLML)**, Eq. (7). |
| **Probabilistic nonnegativity** | Impose \(P(Y(x) < 0) \le \eta\) with \(\eta = 2.2\%\), which for a Gaussian predictive gives a **lower tail bound** \(m(x) - k\,s(x) \ge 0\) with \(k \approx 2\) (their Eqs. 8–11, 13). |
| **Finite grid** | Enforce that inequality on **\(m\) constraint points** \(X_c\) (e.g. **30** equispaced points on \([0,1]\) in Example 1), not on the whole continuum. |
| **Data fidelity** | \(|y_j - m(x_j)| \le \varepsilon\) for each training index \(j\), with **\(\varepsilon = 0.03\)** in their experiments (Eq. 14). |
| **Unconstrained fit** | MATLAB **GPML** + **`minimize`** on NLML. |
| **Constrained fit** | MATLAB **`fmincon`**, **interior-point** algorithm. |
| **Bad initialization** | Retry with \(\log(\ell), \log(\sigma), \log(\sigma_n) = (-3,-3,-10)\) plus small Gaussian noise (Remark 4.1). |

The predictive **\(m\)** and **\(s\)** in the paper are the usual GP **posterior predictive mean and standard deviation** at \(x\). In GPML’s prediction call, the first outputs **`ymu`**, with variance **`ys2`**, correspond to the **noisy observation** predictive distribution; latent function moments are **`fmu`**, **`fs2`**.

---

## 2. Application context in our script (Michaelis–Menten assay)

- **Inputs:** substrate concentration **[S]** in **mM** (7 levels, including 0).
- **Outputs:** initial rate **\(v_0\)** in **μM/s**, with **3 technical replicates** per **[S]** → **21** training pairs \((x_j, y_j)\).
- **Ground truth curve** `y_true`: analytic Michaelis–Menten with parameters **`Vmax`**, **`Km`** for comparison plots only; hyperparameters are **not** forced to match that curve.
- **Software:** **GPML** for NLML and predictions; **Optimization Toolbox** **`fmincon`** for the constrained problem.

---

## 3. What we changed from the paper’s recipe, and why

The paper illustrates the method on synthetic curves on a unit interval. We instead use **real assay data**: seven substrate concentrations in **mM**, each with **three measured \(v_0\)** values in **μM/s**, so the training set has **21** rows. We build **`x_train`** and **`y_train`** so every replicate is kept—GPML is built for this: repeated **\(x\)** with i.i.d. Gaussian noise is a standard regression setup, and the marginal likelihood and predictive means remain well defined. The **evaluation grid** **`x_grid`** spans a sensible mM range (up to about **1.2×** the largest observed **[S]**, at least **2 mM**) so the plotted surrogate is easy to read; that is the same idea as the paper’s equispaced test grid, only rescaled to your domain.

We keep **\(m = 30\)** constraint locations, as in their Example 1, but place them along **[S]** from **0** to the upper end of **`x_grid`**. Rather than relying on **`linspace(0, …)`** alone, we **force the first constraint point to be exactly **[S] = 0**. You have observations at zero substrate; enforcing the nonnegative tail there should not be left to floating-point chance at an endpoint.

For the **tail probability** \(\eta = 2.2\%\), the paper uses \(k \approx 2\) in the bound \(m(x) - k\,s(x) \ge 0\). We compute the same **\(k\)** from \(\eta\) using **`erfinv`** in base MATLAB instead of **`norminv`**, so the script does not depend on the Statistics Toolbox. The probability content is unchanged; only the implementation of \(\Phi^{-1}(\eta)\) differs.

The **data-fidelity** parameter \(\varepsilon\) is where our problem scale really departs from the paper’s demo. They fix **\(\varepsilon = 0.03\)** on a toy function whose values are **\(O(1)\)**. Your rates live on the order of **0–5 μM/s**, and the reported **within-concentration standard deviations** are typically **0.05–0.10 μM/s**. A tube of width **0.03 μM/s** applied to **every** replicate is often **too tight** to be feasible together with the tail constraints, or it forces the optimizer into badly conditioned regions. We therefore set **\(\varepsilon \approx k_\varepsilon\)** times the **mean of the nonzero \(v_0\) standard deviations** (default **\(k_\varepsilon = 1\)**), so the “how close must the posterior mean track each observation?” rule is expressed in **your units** and at a scale comparable to replicate noise. If **`fmincon`** still fails to finish satisfactorily, we **automatically relax** \(\varepsilon\) once using a larger multiple of that same spread—a pragmatic analogue of “increase \(\varepsilon\) until the problem is feasible,” without abandoning the Pensoneault structure.

The paper writes the nonnegative constraint for the **predictive output** \(Y(x)\). In GPML, the **noisy** predictive mean and variance (**`ymu`**, **`ys2`**) can behave slightly differently from the **latent** (**noise-free**) function moments (**`fmu`**, **`fs2`**). For a **reaction rate**, the quantity that “should not go negative” in a physical reading is usually the **underlying** curve, not the measured draw with observation noise. By default we therefore enforce the **\(m - ks \ge 0\)** inequality on **\(f\)** at the constraint grid (**`use_latent_nonneg = true`**), while keeping the **data** constraints as **\(|y_j - m(x_j)| \le \varepsilon\)** using the **noisy** predictive mean **`ymu`** at training inputs—so the model is still required to stay close to what you actually measured. If you want to hew as closely as possible to the paper’s literal use of the **observation** predictive for both pieces, you can set **`use_latent_nonneg = false`**; in our experience that version is **harder** for the optimizer.

On the **numerical** side, the paper recommends **`fmincon`** with the **interior-point** method. When we tried that first on your problem, the solver often reported **near-feasible** points with small remaining violations while the **KKT** linear algebra became **singular**—a sign that the inequality set is stiff or ill-conditioned for interior-point at very tight tolerances. We now run **`sqp`** first (often more dependable for **small, dense** problems with many inequalities), then fall back to **interior-point** with **feasibility mode** if needed. We also **relax** constraint and optimality tolerances slightly (order **\(10^{-4}\)** instead of **\(10^{-6}\)**) and allow **many more** function evaluations and iterations, so the run is less likely to stop early on numerical noise or a **500-evaluation** cap.

Another practical issue was **\(\sigma_n\)** collapsing toward **zero** under constraints, which balloons the **NLML** and makes the problem poorly scaled. The paper does not impose a minimum noise level; we add a **lower bound** on **\(\sigma_n\)** (via **`log(sn_floor)`** in **`lb`**) tied to a fraction of your **empirical noise scale**. That encodes the belief that measurement error does not vanish and mirrors the stabilizing role of \(\varepsilon\): both stop the optimizer from “cheating” with extreme hyperparameters. We still start **`fmincon`** from the **unconstrained** GPML solution (**`hyp_unc`**), clipped to the bounds, and only use the paper’s **\((-3,-3,-10)\)**-style restart if the exits remain poor—because in practice, warm-starting from `minimize` is far more informative than a cold random point.

The **mathematical** link between the paper and GPML is unchanged: three hyperparameters in **log** space, **NLML** from **`gp`** with **no** test points, and predictions from **`gp`** with test inputs. Constraint evaluation stacks **\(X_c\)** and training **\(x\)** into one predictive call for efficiency.

Separately from Pensoneault, the script was stripped of earlier **debug instrumentation** (logging, timing, memory traces, commented rejection-sampling experiments). That cleanup does not alter the method; it only makes the file easier to maintain.

---

## 4. How the unconstrained and constrained paths differ in code

**Same model, different optimization problem.** Both branches use the **same** GPML setup: zero mean, squared-exponential covariance, Gaussian likelihood, exact inference **`infGaussLik`**. The **posterior** at any **\(\theta\)** is still the usual GP regression posterior; nothing in the Pensoneault paper replaces the Kalman-style algebra inside **`gp`**.

The **unconstrained** path is the textbook procedure: GPML’s **`minimize`** adjusts **`hyp`** to reduce the **negative log marginal likelihood** and stops when its internal stopping rule is satisfied. There are **no** inequality constraints; the only pressure on **length scale**, **signal scale**, and **noise** is “explain the data” plus the implicit Occam penalty in the NLML.

The **constrained** path **reuses that same NLML** as the **objective** passed to **`fmincon`**, but every candidate **\(\theta\)** must also satisfy **`pens_nonlcon(\theta) \le 0`**. Those inequalities encode two ideas from the paper: at each point on **\(X_c\)**, the **lower tail** of the (latent or predictive) Gaussian must stay **nonnegative** with probability at least **\(1-\eta\)** in the Gaussian sense; and at **each training row**, the **predictive mean** must lie within **\(\varepsilon\)** of the observed **\(y_j\)**. So the constrained fit is **not** “a different GP”—it is **hyperparameter selection under physics- and data-shaped side conditions**.

Because of those side conditions, the **best** **\(\theta\)** for the constrained problem will generally **not** minimize NLML over the full unconstrained space. You should expect a **larger** NLML when constraints bite; that is not a bug, it is the **cost** of feasibility. The meaningful comparison between the two fits is **visual and scientific**: whether the **mean** and **uncertainty** look reasonable, whether **negative** **\(v_0\)** in regions with little data is suppressed, and whether the **data tube** is respected—not which run has the smaller scalar NLML.

In the figures we plot the **unconstrained** posterior using **`hyp_unc`** and the **constrained** posterior using **`hyp_con`**, side by side, with the **same** **ylim** so the eye can compare shape and band width without rescaling artifacts.

---

## 5. Quick reference: flags you can tune

| Variable | Role |
|----------|------|
| **`k_epsilon`** | Scales **`epsilon_data`** relative to **`mean(v0_std)`**. |
| **`use_latent_nonneg`** | **`true`**: nonneg on **latent** **\(f\)** at **`X_c`**; **`false`**: on **noisy** predictive **\(y\)** (closer to a literal reading of the paper’s \(Y(x)\) notation). |
| **`m_constraint`** | Number of nonnegative tail constraints (grid on **[S]**). |
| **`eta_quantile`** | Sets **`k_nonneg`** via the Gaussian tail (default **2.2%**). |
| **`sn_floor`** | Minimum **\(\sigma_n\)** (via **`lb(3)`**) for numerical stability. |

---

## 6. References

- Pensoneault, Yang, Zhu, *Nonnegativity-Enforced Gaussian Process Regression*, 2020. [arXiv:2004.04632](https://arxiv.org/abs/2004.04632)
- Rasmussen & Nickisch, **GPML** toolbox (`gp.m`, `minimize.m`).

---

*This log reflects the implementation in the repository as of the last edit to `michaelis_menten.m` alongside this file.*
