





























































# Run Command Options Reference

This file is the running reference for what you can pass into:

- `run_experiment(...)`
- `run_lotka_compare_naive_vs_lmc(...)`

## When to use `struct(...)`

- Use **no struct** when you want all defaults:
  - `out = run_experiment();`
  - `out = run_lotka_compare_naive_vs_lmc();`
- Use `struct(...)` when you want to override any option:
  - `out = run_experiment(struct('problem_name','lotka_volterra','method_name','naive_independent'));`

Think of `struct(...)` as a named-options container.

---

## `run_experiment` options

All options are passed as fields in one struct, e.g.:

```matlab
out = run_experiment(struct('problem_name','lotka_volterra','seed',2));
```

| Field | Type | Default | Valid / Examples | What it controls |
|---|---|---|---|---|
| `gpml_path` | char/string | `''` | `'C:\path\to\gpml-matlab'` | Explicit GPML root path override. |
| `seed` | scalar int | `1` | `1`, `42`, `123` | Random seed for reproducibility. |
| `n_train` | scalar int | `20` | `10`, `14`, `30` | Number of sampled training time points (unless `x_train` is provided). |
| `noise_model` | char/string | `'additive'` | `'additive'`, `'proportional'` | Observation noise model used in data generation. |
| `noise_std` | scalar double | `0.05` | `0.01`, `0.05`, `0.1` | Noise level used by problem generator. |
| `xmin` | scalar double | `0` | `0` | Left edge of time domain. |
| `xmax` | scalar double | `20` | `15`, `20`, `50` | Right edge of time domain. |
| `n_grid` | scalar int | `300` | `200`, `300`, `600` | Number of dense prediction/truth points. |
| `max_iters` | scalar int | `-200` | `-100`, `-200`, `-300` | GPML optimization budget (`<0` = function eval budget). |
| `problem_name` | char/string | `'acute_transient'` | `'acute_transient'`, `'lotka_volterra'` | Which toy system to simulate. |
| `method_name` | char/string | `'naive_independent'` | `'naive_independent'`, `'lmc'` | GP method (used for Lotka-Volterra). |
| `sample_mode` | char/string | `'uniform'` | `'uniform'`, `'random'` | How training time points are chosen when `x_train` not provided. |
| `x_train` | vector | (not set) | `[0 1 2 3 5 7 14]` | Explicit sampling times (overrides `n_train` + `sample_mode`). |
| `n_train_prey` | scalar int | (inherits `n_train`) | `30` | Lotka only: prey sample count when `x_train_prey` is not provided. |
| `n_train_pred` | scalar int | (inherits `n_train`) | `6` | Lotka only: predator sample count when `x_train_pred` is not provided. |
| `sample_mode_prey` | char/string | (inherits `sample_mode`) | `'uniform'`, `'random'` | Lotka only: prey sampling mode. |
| `sample_mode_pred` | char/string | (inherits `sample_mode`) | `'uniform'`, `'random'` | Lotka only: predator sampling mode. |
| `x_train_prey` | vector | (not set) | `[0 1 2 3 4 5 7 9]` | Lotka only: explicit prey sample times. |
| `x_train_pred` | vector | (not set) | `[0 3 6 10 15]` | Lotka only: explicit predator sample times. |
| `lv_params` | struct | (internal defaults) | `struct('alpha',1.1,'beta',0.09,...)` | Lotka-Volterra ODE parameters/initial conditions. |
| `lmc` | struct | see below | `struct('q_latent',2,...)` | LMC-specific options (when `method_name='lmc'`). |

### `lmc` sub-options

| Field | Type | Default | Meaning |
|---|---|---|---|
| `q_latent` | scalar int | `1` | Number of latent components (`1` behaves as ICM). |
| `separate_noise` | logical | `true` | Whether prey/predator use separate noise scales in covariance. |
| `jitter` | scalar double | `1e-6` | Numerical stabilization noise floor. |
| `n_restarts` | scalar int | `3` | Number of random-restart fits; best NLML is kept. |

### Noise model semantics

- `noise_model='additive'`:
  - `noise_std` is absolute Gaussian standard deviation.
  - Formula: `y = y_true + noise_std * randn(...)`
- `noise_model='proportional'`:
  - `noise_std` is relative scale (fraction of signal magnitude).
  - Formula: `y = y_true + (noise_std * |y_true|) .* randn(...)`

### Asymmetric Lotka sampling precedence

For each state separately:

1. if `x_train_prey` / `x_train_pred` is provided, that explicit vector is used;
2. else `sample_mode_prey` / `sample_mode_pred` with `n_train_prey` / `n_train_pred` is used;
3. else fall back to shared `sample_mode` and `n_train`.

---

## `run_lotka_compare_naive_vs_lmc` options

This is a convenience wrapper that:

1. builds one shared Lotka dataset config,
2. runs `naive_independent`,
3. runs `lmc`,
4. returns combined metrics.

It accepts the same override style:

```matlab
out = run_lotka_compare_naive_vs_lmc(struct('seed',1,'sample_mode','random'));
```

You can pass top-level options (`seed`, `n_train`, `xmax`, etc.) and nested `lmc` options.

---

## Common command examples

### 1) Acute transient defaults

```matlab
out = run_experiment();
```

### 2) Lotka + naive independent

```matlab
out = run_experiment(struct( ...
    'problem_name','lotka_volterra', ...
    'method_name','naive_independent'));
```

### 2b) Acute with proportional noise (explicit)

```matlab
out = run_experiment(struct( ...
    'problem_name','acute_transient', ...
    'noise_model','proportional', ...
    'noise_std',0.05));
```

### 3) Lotka + LMC (ICM behavior with `q_latent=1`)

```matlab
out = run_experiment(struct( ...
    'problem_name','lotka_volterra', ...
    'method_name','lmc', ...
    'lmc',struct('q_latent',1,'n_restarts',3)));
```

### 4) Lotka + full LMC (`q_latent=2`)

```matlab
out = run_experiment(struct( ...
    'problem_name','lotka_volterra', ...
    'method_name','lmc', ...
    'lmc',struct('q_latent',2,'n_restarts',3)));
```

### 5) Naive vs LMC comparison wrapper

```matlab
out = run_lotka_compare_naive_vs_lmc(struct( ...
    'sample_mode','random', ...
    'n_train',14, ...
    'seed',1));
```

### 5b) Compare wrapper with additive vs proportional noise

```matlab
out_add = run_lotka_compare_naive_vs_lmc(struct( ...
    'noise_model','additive', ...
    'noise_std',0.05, ...
    'seed',1));

out_prop = run_lotka_compare_naive_vs_lmc(struct( ...
    'noise_model','proportional', ...
    'noise_std',0.05, ...
    'seed',1));
```

### 6) Asymmetric counts by state (many prey, few predator)

```matlab
out = run_lotka_compare_naive_vs_lmc(struct( ...
    'n_train_prey',30, ...
    'n_train_pred',6, ...
    'sample_mode_prey','random', ...
    'sample_mode_pred','uniform', ...
    'seed',1));
```

### 7) Explicit asymmetric time vectors by state

```matlab
out = run_lotka_compare_naive_vs_lmc(struct( ...
    'x_train_prey',[0 1 2 3 4 5 7 9 11 13 15], ...
    'x_train_pred',[0 3 6 10 15], ...
    'seed',1));
```

---

## Output fields to expect

For most runs:

- `out.cfg` : resolved config used
- `out.problem` : generated truth + sampled data
- `out.pred` : predictions/hyperparameters
- `out.metrics_table` : RMSE/MAE/Coverage95 table

For compare wrapper:

- `out.naive` : full output of naive run
- `out.lmc` : full output of LMC run
- `out.metrics_combined` : stacked method/state comparison table
