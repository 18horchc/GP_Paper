% GP Research Project: Acute Transient Equation Sampling
clear; clc; close all;

%% 1. Define Realistic Parameters 
% Parameters 
Vmax = 6;       % Maximum velocity (units/s) 
Km   = 0.15;        % Michaelis constant (units of [S]) 

% Define function
% v = (Vmax * [S]) / (Km + [S])
mm_static = @(S) (Vmax .* S) ./ (Km + S); 


%% 2. Training data ([S] in mM, v_0 in μM/s)
% Baseline: n_train uniform [S] on [0, x_max]; additive noise N(0, (noise_frac*Vmax)^2) on every point.
n_train = 10;
x_max = 2;              % upper [S] (mM): sampling, ground truth, and plot extent
noise_frac = 0.05;      % sigma = noise_frac * Vmax

% Baseline constraint grid: X_c = {0, 0.02, ..., x_max} (~101 points).
constraint_step = 0.02;
X_c = (0:constraint_step:x_max)';
X_c_mono = X_c;

rng(42);
x_train = linspace(0, x_max, n_train);
v_true = mm_static(x_train);
noise_sd_true = noise_frac * Vmax;
y_train = v_true + noise_sd_true * randn(size(v_true));

%% 3. Ground truth curve on an [S] grid (mM)
x_grid = linspace(0, x_max, 500);
y_true = mm_static(x_grid);

%% 4. Visualization
% figure('Color', 'w', 'Position', [100, 100, 800, 500]);
% hold on; grid on;
% 
% % Plot continuous solution curve 
% plot(x_grid, y_true, 'b-', 'LineWidth', 2, 'DisplayName', 'Ground Truth');
% 
% % Plot the horizontal asymptote (Vmax) [cite: 12, 145]
% yline(Vmax, 'r--', 'V_{max} (Asymptote)', 'LabelVerticalAlignment', 'bottom');
% 
% % Plot sampled noisy points
% plot(x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
%     'DisplayName', sprintf('Sampled Data (n=%d, %g%%  Noise)', n_samples, noise_level*100));
% 
% % Labels and Formatting
% xlabel('Substrate Concentration [S]');
% ylabel('Initial Reaction Velocity (v_0)');
% title('Michaelis-Menten Kinetics');
% legend('Location', 'southeast');
% set(gca, 'FontSize', 12);
% 
% %fprintf('Simulation complete. n=%d samples generated.\n', n_samples);




%% Fit the Naive GP
% We check for a core GPML function like 'gp.m'
gpml_folder_name = "C:\Users\chorc\OneDrive\Documents\Stroke Research\Gaussian Processes\Old\gpml-matlab-master\gpml-matlab-master";

% Check if the core 'gp' function is available in the current path 
if ~exist('gp', 'file')
    fprintf('GPML not found. Searching in local directory: %s...\n', gpml_folder_name);
    
    % Check if the folder exists on your drive
    if exist(gpml_folder_name, 'dir')
        addpath(genpath(gpml_folder_name)); % Adds the folder and all its math subfolders
        fprintf('GPML successfully added to path.\n');
    else
        % If it's still missing, we throw an error with instructions
        error(['GPML toolbox missing! Please download it and ensure ' ...
               'the folder "%s" is in your project directory.'], gpml_folder_name);
    end
else
    fprintf('GPML toolbox is already loaded and ready.\n');
end

% Run GPML's internal startup to ensure MEX files/utilities are ready
try
    startup; 
catch
    % If GPML doesn't have a startup.m in the version you got, this just skips
end

% Configuration & Initial Hyperparameters
% Unconstrained: GPML minimize (no box). Constrained: fmincon with hyp_lb/hyp_ub below.
ell0 = std(x_train);
sf0  = std(y_train);
sn0  = max(1e-3, noise_sd_true);

hyp = struct();
hyp.mean = [];              % @meanZero: no mean hyperparameters
hyp.cov  = log([ell0; sf0]);
hyp.lik  = log(sn0);

% Hyperparameter box (constrained fmincon only; ell upper swept in sensitivity loop)
ell_bounds_lo = 0.02;
ell_ub_sweep = [10, 15, 20, 30];
sf_bounds  = [0.1, 12];
sn_bounds  = [0.05, 2.5];

% Define GP components
meanfunc = @meanZero;
%covfunc  = {@covMaterniso,5}; 
covfunc = @covSEiso;
likfunc = @likGauss;
inffunc = @infGaussLik;

x_col = x_train(:);
y_col = y_train(:);

% --- Cholesky / PD diagnostics (set false to skip; see plan: debug Cholesky) ---
debug_chol = true;
% Optional Ky jitter in gp_seiso_deriv_pred is off for now (see commented line on Ky there).

if debug_chol
    ndup = numel(x_col) - numel(unique(x_col));
    xs = sort(x_col(:));
    if numel(xs) > 1
        dxmin = min(diff(xs));
    else
        dxmin = NaN;
    end
    fprintf(['[CHOL debug inputs] n=%d duplicate_rows=%d min_positive_diff(x)=%.6g\n'], ...
        numel(x_col), ndup, dxmin);
    if ndup > 0
        warning('michaelis_menten:duplicateX', ...
            'Duplicate training x values make K_ff rank-deficient; tiny sigma_n breaks chol.');
    end
end

%% 1. Fit unconstrained GP
fprintf('Optimizing hyperparameters (unconstrained NLML, GPML minimize)...\n');
hyp_unc = minimize(hyp, @gp, -100, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);

if debug_chol
    fprintf('[CHOL debug hyp_unc] ell=%.6g sf=%.6g sn=%.6g (exp of log hyp)\n', ...
        exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik(1)));
    try
        nlml_dbg = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
        fprintf('[CHOL debug hyp_unc] gp(NLML only) OK, NLML=%.6g\n', nlml_dbg);
    catch ME
        fprintf('[CHOL debug hyp_unc] gp(NLML) FAILED: %s\n', ME.message);
        rethrow(ME);
    end
    mnKy = mm_min_eig_Ky_seiso(hyp_unc, x_col);
    fprintf('[CHOL debug hyp_unc] min real eigenvalue of sym(Ky)=%.6g (<=0 => chol at risk)\n', mnKy);
end

[m_unc, s2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
f_upper_unc = m_unc + 2 * sqrt(max(s2_unc, 0));
f_lower_unc = m_unc - 2 * sqrt(max(s2_unc, 0));

%% 2. Build constraints
% Pensoneault tails: mu* - k*s* >= 0, mu* + k*s* <= Vmax, mu*' - k*s*' >= 0 on X_c.
hyp_tpl = hyp_unc;
eta = 0.022;
k   = -sqrt(2) * erfinv(2*eta - 1);   % Phi^{-1}(eta) ~ -2
y_max = Vmax;
enforce_upper_bound = true;
enforce_data_fidelity = false;
enforce_monotonicity = true;
k_mono = [];
epsilon = 0.165;   % reserved for data-fidelity tube experiments

if debug_chol && enforce_monotonicity
    try
        gp_seiso_deriv_pred(hyp_unc, x_col, y_col, X_c_mono);
        fprintf('[CHOL debug hyp_unc] gp_seiso_deriv_pred OK at unconstrained hyp.\n');
    catch ME
        fprintf('[CHOL debug hyp_unc] gp_seiso_deriv_pred FAILED: %s\n', ME.message);
        rethrow(ME);
    end
end

objfun_inner = @(theta) gp(theta_to_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
obj_con = objfun_inner;   % NLML for ranking feasible random starts
if debug_chol
    objfun = @(theta) mm_gp_wrap_chol(objfun_inner, theta, hyp_tpl, x_col, 'objfun');
else
    objfun = objfun_inner;
end
nonlcon_inner = @(theta) pens_constraints(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, X_c, X_c_mono, k, epsilon, y_max, ...
    enforce_upper_bound, enforce_data_fidelity, enforce_monotonicity, k_mono);
if debug_chol
    nonlcon = @(theta) mm_nonlcon_wrap_chol(nonlcon_inner, theta, hyp_tpl, x_col, 'nonlcon');
else
    nonlcon = nonlcon_inner;
end
opts = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, ...
    'Display', 'off', ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);
nTry = 5000;
nMultistart = 10;
active_tol = 1e-5;

nlml_unc = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
cfg = struct('hyp_unc', hyp_unc, 'hyp_tpl', hyp_tpl, 'inffunc', inffunc, 'meanfunc', meanfunc, ...
    'covfunc', covfunc, 'likfunc', likfunc, 'objfun', objfun, 'obj_con', obj_con, 'nonlcon', nonlcon, ...
    'x_col', x_col, 'y_col', y_col, 'X_c', X_c, 'X_c_mono', X_c_mono, 'k', k, 'epsilon', epsilon, ...
    'y_max', y_max, 'enforce_upper_bound', enforce_upper_bound, ...
    'enforce_data_fidelity', enforce_data_fidelity, 'enforce_monotonicity', enforce_monotonicity, ...
    'k_mono', k_mono, 'sf_bounds', sf_bounds, 'sn_bounds', sn_bounds, 'nTry', nTry, ...
    'nMultistart', nMultistart, 'opts', opts, 'active_tol', active_tol, ...
    'nlml_unc', nlml_unc, 'x_grid', x_grid, 'y_true', y_true, 'Vmax', Vmax);

%% 3–6. Length-scale upper-bound sensitivity (constrained pipeline per ell_ub)
fprintf('\n=== Length-scale upper-bound sensitivity ===\n');
fprintf('Random design: rng(42) per ell_ub case (common unit-cube draws, mapped through each box).\n');
sens = [];
theta_prev = [];
for i = 1:numel(ell_ub_sweep)
    fprintf('\n--- ell upper = %g (%d/%d) ---\n', ell_ub_sweep(i), i, numel(ell_ub_sweep));
    out_i = mm_constrained_ell_ub_run(ell_ub_sweep(i), ell_bounds_lo, cfg, theta_prev);
    if isempty(sens)
        sens = out_i;
    else
        sens(i) = out_i;
    end
    theta_prev = sens(i).theta_opt;
end
mm_print_ell_ub_sensitivity_table(sens, ell_ub_sweep);

% Use ell_ub = 4 case for plots and detailed binding diagnostics
idx_plot = 1;
theta_opt = sens(idx_plot).theta_opt;
nlml_opt = sens(idx_plot).nlml_opt;
exitflag = sens(idx_plot).exitflag;
hyp_con = theta_to_hyp(theta_opt, hyp_tpl);
[c_final, ~] = pens_constraints(theta_opt, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x_col, y_col, X_c, X_c_mono, k, epsilon, y_max, ...
    enforce_upper_bound, enforce_data_fidelity, enforce_monotonicity, k_mono);
v_final = sens(idx_plot).max_c_final;
hyp_lb = sens(idx_plot).hyp_lb;
hyp_ub = sens(idx_plot).hyp_ub;
ell_bounds = sens(idx_plot).ell_bounds;
in_box = sens(idx_plot).in_box;

fprintf('\n=== Baseline plot case (ell upper = %g) ===\n', ell_ub_sweep(idx_plot));
fprintf('Final max(c) = %.6g (feasible if <= 0)\n', v_final);
fprintf('Final in hyp box: %d\n', in_box);
fprintf('fmincon exitflag = %d (%s)\n', exitflag, sens(idx_plot).fmincon_message);
mm_report_binding_diagnostics(theta_opt, c_final, hyp_lb, hyp_ub, numel(X_c), numel(X_c_mono), ...
    enforce_upper_bound, enforce_data_fidelity, enforce_monotonicity, numel(x_col));

x_lim_lo = 0;
ylim_unc = mm_plot_ell_ub_sensitivity_grid(sens, ell_ub_sweep, x_grid, y_true, x_train, y_train, ...
    m_unc, s2_unc, f_upper_unc, f_lower_unc, x_lim_lo, x_max, Vmax, n_train, noise_sd_true);

fprintf('GP optimization complete.\n');
fprintf('Unconstrained: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);
fprintf('Constrained:   ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | fmincon exitflag=%d\n', ...
    exp(hyp_con.cov(1)), exp(hyp_con.cov(2)), exp(hyp_con.lik), ...
    nlml_opt, exitflag);

%% Encodings to try:
% Zero-mean prior: MM curves are positive; sparse data can pull the posterior toward
% zero between points. @meanConst is an alternative if that behavior is unwanted.

% Hyperparameter initialization techniques:  $std(x)$ for the lengthscale 
% ($\ell$) and $std(y)$ for the signal variance ($\sigma_f$) is a classic, 
% robust heuristic for stationary data. Just remember that GPML's minimize 
% function can sometimes get stuck in local minima if the initial noise 
% guess ($sn_0$) is too far from the actual 5% relative noise you've 
% injected. 
% Log-Space Optimization: Remember that GPML optimizes in 
% log-space. If your initial noise guess ($sn_0$) is too high, the 
% optimizer might converge to a "noise-only" solution where the GP treats 
% the entire biological signal as random fluctuation

% Kernel Choice: You are using covSEiso (Squared Exponential). This assumes 
% the function is infinitely smooth. While fine for the saturation curve, 
% if you eventually switch to time-series data with sharp transitions (like 
% the action potential mentioned in your research framing), this kernel 
% might "over-smooth" the peak.

%% Boundedness
% Transform bounded observations into an unbounded latent space, fit a
% standard GP, and then warp the predictions back.

% In literature used probit function for transformation
%Consider also Lineweaver-Burk space

% Note that warping the output distorts the uncertainty bands. Uncertainty
% will become asymmetric.

function hyp = theta_to_hyp(theta, hyp_tpl)
% Pack theta = [log(ell); log(sf); log(sn)] into a GPML hyp struct (@meanZero).
hyp = hyp_tpl;
hyp.cov  = theta(1:2);
hyp.lik  = theta(3);
hyp.mean = [];
end

function [m_deriv, s2_deriv] = gp_seiso_deriv_pred(hyp, x, y, X_c)
% Posterior predictive mean and variance of df/dx at each X_c under covSEiso + likGauss + meanZero.
% Matches GPML covSEiso: k(x,z) = sf^2 * exp(-0.5*(x-z)^2/ell^2); hyp.cov = [log(ell); log(sf)],
% hyp.lik = log(sn); prior mean has zero derivative (flat m(x) => contributes 0 to f').
%
% Cross-covariance Cov[f'(x*), f(z)] = d/dx* k(x*,z) = -(x*-z)/ell^2 * k(x*,z).
% Prior variance of derivative: Cov[f'(x*), f'(x*)] = sf^2/ell^2 (limit of second derivative form).

ell = exp(hyp.cov(1));
sf2 = exp(2 * hyp.cov(2));
sn2 = exp(2 * hyp.lik(1));

x = x(:);
y = y(:);
X_c = X_c(:);
n = numel(x);
m = numel(X_c);

if isempty(hyp.mean)
    mu = 0;
else
    mu = hyp.mean;
end
ytil = y - mu;

% Training Gram Ky = K_ff + sn^2 I
dxx = (x - x.') ./ ell;
K = sf2 * exp(-0.5 * dxx.^2);
Ky = K + sn2 * eye(n);
% Ky = Ky + 1e-8 * eye(n);   % optional stabilizing jitter for chol(Ky) (disabled)

% Cross m x n: R(j,i) = X_c(j) - x(i)
R = X_c - x.';
Kxc = sf2 * exp(-0.5 * (R ./ ell).^2);
K_df = -Kxc .* (R ./ (ell^2));   % Cov[f'(X_c(j)), f(x(i))]

% Prior (noise-free) marginal variance of derivative at each X_c (diagonal of K_dd)
k_dd_diag = (sf2 / ell^2) * ones(m, 1);

alpha = Ky \ ytil;
m_deriv = K_df * alpha;

% Posterior marginal variances: k_dd - K_df Ky^{-1} K_df'
L = chol(Ky, 'lower');
B = L \ K_df';                  % n x m
q = sum(B.^2, 1).';             % diag(K_df * inv(Ky) * K_df')
s2_deriv = max(k_dd_diag - q, 0);
end

function [c, ceq] = pens_constraints(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, x, y, X_c, X_c_mono, k, epsilon, y_max, ...
    enforce_upper_bound, enforce_data_fidelity, enforce_monotonicity, k_mono)
% Pensoneault-style inequalities c(theta) <= 0: nonnegative tail at X_c (Eq. 13 lower);
% optional upper tail y*(x_c)+k*s<=y_max; optional data tube |y-y*|<=epsilon (Eq. 14);
% optional monotonicity on f' at X_c_mono (or X_c if X_c_mono is empty). fmincon expects c <= 0.

hyp = theta_to_hyp(theta, hyp_tpl);

if nargin < 14 || isempty(enforce_upper_bound), enforce_upper_bound = true; end
if nargin < 15 || isempty(enforce_data_fidelity), enforce_data_fidelity = true; end
if nargin < 16 || isempty(enforce_monotonicity), enforce_monotonicity = true; end
if nargin < 17 || isempty(k_mono), k_mono = k; end

% GPML predictive: need training x in xstar only when the data-fidelity block is on.
if enforce_data_fidelity
    xstar = [X_c(:); x(:)];
else
    xstar = X_c(:);
end
[ymu, ys2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xstar);

nC   = numel(X_c);
y_star_xc = ymu(1:nC);                       % y*(x_c^(i))
s_xc      = sqrt(max(ys2(1:nC), 0));         % s(x_c^(i))

% Eq. (13) lower: 0 <= y*(x_c) - k * s(x_c)   =>   c_nonneg = k * s(x_c) - y*(x_c) <= 0
c_nonneg = k * s_xc - y_star_xc;

c = c_nonneg(:);
if enforce_upper_bound
    % Upper-tail: y*(x_c) + k * s(x_c) <= y_max  =>  c_upper <= 0
    c_upper = y_star_xc + k * s_xc - y_max;
    c = [c; c_upper(:)];
end
if enforce_data_fidelity
    y_star_xj = ymu(nC+1:end);               % y*(x^(j))
    % Eq. (14): 0 <= epsilon - |y - y*(x)|  =>  c_data <= 0
    c_data = abs(y - y_star_xj) - epsilon;
    c = [c; c_data(:)];
end
if enforce_monotonicity
    if isempty(X_c_mono)
        Xg = X_c(:);
    else
        Xg = X_c_mono(:);
    end
    % Monotonicity (increasing): posterior on f'(x) is Gaussian; same tail form as
    % Eq. (13) with optional k_mono (defaults to k): m_deriv - k_mono * s_deriv >= 0.
    [m_deriv, s2_deriv] = gp_seiso_deriv_pred(hyp, x, y, Xg);
    s_deriv = sqrt(max(s2_deriv, 0));
    c_mono = k_mono .* s_deriv - m_deriv;
    c = [c; c_mono];
end
ceq = [];
end

function mn = mm_min_eig_Ky_seiso(hyp, x)
% Smallest eigenvalue of sym(Ky) with same Ky as gp_seiso_deriv_pred (no extra jitter).
x = x(:);
ell = exp(hyp.cov(1));
sf2 = exp(2 * hyp.cov(2));
sn2 = exp(2 * hyp.lik(1));
n = numel(x);
dxx = (x - x.') ./ ell;
K = sf2 * exp(-0.5 * dxx.^2);
Ky = K + sn2 * eye(n);
mn = min(real(eig((Ky + Ky') / 2)));
end

function v = mm_gp_wrap_chol(fun, theta, hyp_tpl, x, tag)
try
    v = fun(theta);
catch ME
    msgl = lower(ME.message);
    if contains(msgl, 'chol') || contains(msgl, 'posdef') || contains(msgl, 'positive definite')
        hyp_bad = theta_to_hyp(theta, hyp_tpl);
        mnKy = mm_min_eig_Ky_seiso(hyp_bad, x);
        fprintf(['[CHOL debug %s] %s\n  log(theta)=[%.6f; %.6f; %.6f]  ell,sf,sn=[%.6g %.6g %.6g]  min_eig(Ky)=%.6g\n'], ...
            tag, ME.message, theta(1), theta(2), theta(3), exp(theta(1)), exp(theta(2)), exp(theta(3)), mnKy);
    end
    rethrow(ME);
end
end

function [c, ceq] = mm_nonlcon_wrap_chol(fh, theta, hyp_tpl, x, tag)
try
    [c, ceq] = fh(theta);
catch ME
    msgl = lower(ME.message);
    if contains(msgl, 'chol') || contains(msgl, 'posdef') || contains(msgl, 'positive definite')
        hyp_bad = theta_to_hyp(theta, hyp_tpl);
        mnKy = mm_min_eig_Ky_seiso(hyp_bad, x);
        fprintf(['[CHOL debug %s] %s\n  log(theta)=[%.6f; %.6f; %.6f]  ell,sf,sn=[%.6g %.6g %.6g]  min_eig(Ky)=%.6g\n'], ...
            tag, ME.message, theta(1), theta(2), theta(3), exp(theta(1)), exp(theta(2)), exp(theta(3)), mnKy);
    end
    rethrow(ME);
end
end

function mm_report_binding_diagnostics(theta, c_final, hyp_lb, hyp_ub, nC, nMono, ...
    enforce_upper_bound, enforce_data_fidelity, enforce_monotonicity, nData, active_tol)
% Report which Pensoneault constraint families and hyp-box bounds are active at theta.
% c_final block order must match pens_constraints: lower, upper (opt), data (opt), mono (opt).
if nargin < 11 || isempty(active_tol)
    active_tol = 1e-5;
end

nExp = nC;
if enforce_upper_bound
    nExp = nExp + nC;
end
if enforce_data_fidelity
    nExp = nExp + nData;
end
if enforce_monotonicity
    nExp = nExp + nMono;
end
assert(numel(c_final) == nExp, ...
    'mm_report_binding_diagnostics: expected %d constraints, got %d.', nExp, numel(c_final));

fprintf('\nBinding diagnostics (active_tol=%g):\n', active_tol);
idx = 0;

lower_c = c_final(idx + (1:nC));
idx = idx + nC;
fprintf('  lower: max(c)=%.8g, near_active=%d\n', max(lower_c), sum(lower_c > -active_tol));

if enforce_upper_bound
    upper_c = c_final(idx + (1:nC));
    idx = idx + nC;
    fprintf('  upper: max(c)=%.8g, near_active=%d\n', max(upper_c), sum(upper_c > -active_tol));
end

if enforce_data_fidelity
    data_c = c_final(idx + (1:nData));
    idx = idx + nData;
    fprintf('  data:  max(c)=%.8g, near_active=%d\n', max(data_c), sum(data_c > -active_tol));
end

if enforce_monotonicity
    mono_c = c_final(idx + (1:nMono));
    idx = idx + nMono;
    fprintf('  mono:  max(c)=%.8g, near_active=%d\n', max(mono_c), sum(mono_c > -active_tol));
end

ell = exp(theta(1));
sf  = exp(theta(2));
sn  = exp(theta(3));
ell_b = [exp(hyp_lb(1)), exp(hyp_ub(1))];
sf_b  = [exp(hyp_lb(2)), exp(hyp_ub(2))];
sn_b  = [exp(hyp_lb(3)), exp(hyp_ub(3))];

fprintf('Hyperparameter box (physical units):\n');
fprintf('  ell=%.4f  bounds [%.4f, %.4f]\n', ell, ell_b(1), ell_b(2));
fprintf('  sf =%.4f  bounds [%.4f, %.4f]\n', sf, sf_b(1), sf_b(2));
fprintf('  sn =%.4f  bounds [%.4f, %.4f]\n', sn, sn_b(1), sn_b(2));
fprintf('  ell at lower bound: %d\n', double(abs(theta(1) - hyp_lb(1)) < active_tol));
fprintf('  ell at upper bound: %d\n', double(abs(theta(1) - hyp_ub(1)) < active_tol));
fprintf('  sf  at lower bound: %d\n', double(abs(theta(2) - hyp_lb(2)) < active_tol));
fprintf('  sf  at upper bound: %d\n', double(abs(theta(2) - hyp_ub(2)) < active_tol));
fprintf('  sn  at lower bound: %d\n', double(abs(theta(3) - hyp_lb(3)) < active_tol));
fprintf('  sn  at upper bound: %d\n', double(abs(theta(3) - hyp_ub(3)) < active_tol));
end

function out = mm_constrained_ell_ub_run(ell_ub_max, ell_lo, cfg, theta_prev)
% Full constrained pipeline for one ell upper bound: random search, multi-start fmincon, metrics.
if nargin < 4
    theta_prev = [];
end
ell_bounds = [ell_lo, ell_ub_max];
hyp_lb = log([ell_bounds(1); cfg.sf_bounds(1); cfg.sn_bounds(1)]);
hyp_ub = log([ell_bounds(2); cfg.sf_bounds(2); cfg.sn_bounds(2)]);

nC = numel(cfg.X_c);
nMono = numel(cfg.X_c_mono);
nData = numel(cfg.x_col);

feasible_starts = zeros(3, 0);
best_v = inf;
best_feas_nlml = inf;
best_feas_theta = nan(3, 1);
best_feas_max_c = NaN;
fprintf('  Random search (%d trials), ell in [%.3g, %.3g]...\n', cfg.nTry, ell_bounds(1), ell_bounds(2));
% Common unit-cube design: same U(0,1)^3 draws each case, mapped through this ell_ub box.
rng(42);
for t = 1:cfg.nTry
    theta_try = hyp_lb + rand(3, 1) .* (hyp_ub - hyp_lb);
    [c_try, ~] = pens_constraints(theta_try, cfg.hyp_tpl, cfg.inffunc, cfg.meanfunc, cfg.covfunc, ...
        cfg.likfunc, cfg.x_col, cfg.y_col, cfg.X_c, cfg.X_c_mono, cfg.k, cfg.epsilon, cfg.y_max, ...
        cfg.enforce_upper_bound, cfg.enforce_data_fidelity, cfg.enforce_monotonicity, cfg.k_mono);
    v_try = max(c_try);
    if v_try < best_v
        best_v = v_try;
    end
    if v_try <= 0
        feasible_starts = [feasible_starts, theta_try];
        nlml_try = cfg.obj_con(theta_try);
        if nlml_try < best_feas_nlml
            best_feas_nlml = nlml_try;
            best_feas_theta = theta_try;
            best_feas_max_c = v_try;
        end
    end
end
nFeas = size(feasible_starts, 2);
fprintf('  Feasible random starts: %d / %d\n', nFeas, cfg.nTry);
if nFeas > 0
    fprintf('  Best feasible pre-fmincon: NLML=%.4f max(c)=%.6g ell=%.4f sf=%.4f sn=%.4f\n', ...
        best_feas_nlml, best_feas_max_c, exp(best_feas_theta(1)), exp(best_feas_theta(2)), exp(best_feas_theta(3)));
else
    fprintf('  No feasible random start (best_feas metrics set to NaN).\n');
end

theta_unc_box = min(max([cfg.hyp_unc.cov(1); cfg.hyp_unc.cov(2); cfg.hyp_unc.lik(1)], hyp_lb), hyp_ub);
if nFeas > 0
    nlml_feas = nan(nFeas, 1);
    for j = 1:nFeas
        nlml_feas(j) = cfg.obj_con(feasible_starts(:, j));
    end
    [~, ord] = sort(nlml_feas, 'ascend');
    starts_for_fmincon = feasible_starts(:, ord(1:min(cfg.nMultistart, nFeas)));
else
    starts_for_fmincon = theta_unc_box;
end
if ~isempty(theta_prev)
    theta_cont = min(max(theta_prev(:), hyp_lb), hyp_ub);
    starts_for_fmincon = mm_prepend_unique_start(starts_for_fmincon, theta_cont, 1e-6);
    fprintf('  Prepended continuation warm start from previous ell_ub solution.\n');
end

best_nlml = inf;
theta_opt = nan(3, 1);
nlml_opt = nan;
exitflag = -99;
fmincon_output = struct('message', 'no fmincon run');
nStarts = size(starts_for_fmincon, 2);
for j = 1:nStarts
    [theta_j, nlml_j, ef_j, out_j] = fmincon(cfg.objfun, starts_for_fmincon(:, j), [], [], [], [], ...
        hyp_lb, hyp_ub, cfg.nonlcon, cfg.opts);
    if isfinite(nlml_j) && nlml_j < best_nlml
        best_nlml = nlml_j;
        theta_opt = theta_j;
        nlml_opt = nlml_j;
        exitflag = ef_j;
        fmincon_output = out_j;
    end
end

if ~isfinite(best_nlml)
  if nFeas > 0
    theta_opt = best_feas_theta;
  else
    theta_opt = theta_unc_box;
  end
  nlml_opt = cfg.obj_con(theta_opt);
  exitflag = -99;
  fmincon_output = struct('message', 'no successful fmincon run; using fallback theta');
  fprintf('  Warning: no successful fmincon run; using fallback theta for diagnostics.\n');
end

[c_final, ~] = pens_constraints(theta_opt, cfg.hyp_tpl, cfg.inffunc, cfg.meanfunc, cfg.covfunc, ...
    cfg.likfunc, cfg.x_col, cfg.y_col, cfg.X_c, cfg.X_c_mono, cfg.k, cfg.epsilon, cfg.y_max, ...
    cfg.enforce_upper_bound, cfg.enforce_data_fidelity, cfg.enforce_monotonicity, cfg.k_mono);
cmx = mm_c_family_maxes(c_final, nC, nMono, nData, cfg.enforce_upper_bound, ...
    cfg.enforce_data_fidelity, cfg.enforce_monotonicity);
hyp_con_run = theta_to_hyp(theta_opt, cfg.hyp_tpl);
[m_con, s2_con] = gp(hyp_con_run, cfg.inffunc, cfg.meanfunc, cfg.covfunc, cfg.likfunc, ...
    cfg.x_col, cfg.y_col, cfg.x_grid(:));

out = struct();
out.ell_ub_max = ell_ub_max;
out.ell_bounds = ell_bounds;
out.hyp_lb = hyp_lb;
out.hyp_ub = hyp_ub;
out.theta_opt = theta_opt(:);
out.ell_opt = exp(theta_opt(1));
out.sf_opt = exp(theta_opt(2));
out.sn_opt = exp(theta_opt(3));
out.ell_ratio = out.ell_opt / ell_ub_max;
out.nlml_opt = nlml_opt;
out.delta_nlml = nlml_opt - cfg.nlml_unc;
out.exitflag = exitflag;
out.fmincon_message = mm_fmincon_message(fmincon_output);
out.max_c_final = max(c_final);
out.lower_max_c = cmx.lower;
out.upper_max_c = cmx.upper;
out.mono_max_c = cmx.mono;
out.n_feasible = nFeas;
out.n_try = cfg.nTry;
out.ell_at_ub = double(abs(theta_opt(1) - hyp_ub(1)) < cfg.active_tol);
out.in_box = all(theta_opt >= hyp_lb - 1e-9 & theta_opt <= hyp_ub + 1e-9);
out.m_con = m_con(:);
out.s2_con = s2_con(:);
out.best_feas_nlml = NaN;
out.best_feas_max_c = NaN;
out.best_feas_ell = NaN;
out.best_feas_sf = NaN;
out.best_feas_sn = NaN;
if nFeas > 0
    out.best_feas_nlml = best_feas_nlml;
    out.best_feas_max_c = best_feas_max_c;
    out.best_feas_ell = exp(best_feas_theta(1));
    out.best_feas_sf = exp(best_feas_theta(2));
    out.best_feas_sn = exp(best_feas_theta(3));
end
end

function msg = mm_fmincon_message(fmincon_output)
if isstruct(fmincon_output) && isfield(fmincon_output, 'message')
    msg = char(fmincon_output.message);
else
    msg = '';
end
end

function cmx = mm_c_family_maxes(c_final, nC, nMono, nData, enforce_upper_bound, enforce_data_fidelity, enforce_monotonicity)
idx = 0;
cmx.lower = max(c_final(idx + (1:nC)));
idx = idx + nC;
if enforce_upper_bound
    cmx.upper = max(c_final(idx + (1:nC)));
    idx = idx + nC;
else
    cmx.upper = NaN;
end
if enforce_data_fidelity
    cmx.data = max(c_final(idx + (1:nData)));
    idx = idx + nData;
else
    cmx.data = NaN;
end
if enforce_monotonicity
    cmx.mono = max(c_final(idx + (1:nMono)));
else
    cmx.mono = NaN;
end
end

function mm_print_ell_ub_sensitivity_table(sens, ell_ub_sweep)
fprintf('\nLength-scale upper-bound sensitivity summary (post-fmincon):\n');
fprintf('%-6s %7s %6s %7s %7s %10s %10s %5s %10s %10s %10s %10s %12s %5s\n', ...
    'ell_ub', 'ell_opt', 'e/e_ub', 'sf_opt', 'sn_opt', 'NLML', 'dNLML', 'exit', ...
    'max_c', 'low_max', 'up_max', 'mono_max', 'n_feas/nTry', 'e@ub');
for i = 1:numel(sens)
    fprintf('%-6.0f %7.4f %6.3f %7.4f %7.4f %10.4f %10.4f %5d %10.6g %10.6g %10.6g %10.6g %5d/%-5d %5d\n', ...
        ell_ub_sweep(i), sens(i).ell_opt, sens(i).ell_ratio, sens(i).sf_opt, sens(i).sn_opt, ...
        sens(i).nlml_opt, sens(i).delta_nlml, sens(i).exitflag, sens(i).max_c_final, ...
        sens(i).lower_max_c, sens(i).upper_max_c, sens(i).mono_max_c, ...
        sens(i).n_feasible, sens(i).n_try, sens(i).ell_at_ub);
end
fprintf('\nBest feasible random start before fmincon:\n');
fprintf('%-6s %10s %10s %7s %7s %7s\n', 'ell_ub', 'bf_NLML', 'bf_max_c', 'bf_ell', 'bf_sf', 'bf_sn');
for i = 1:numel(sens)
    fprintf('%-6.0f %10.4f %10.6g %7.4f %7.4f %7.4f\n', ell_ub_sweep(i), sens(i).best_feas_nlml, ...
        sens(i).best_feas_max_c, sens(i).best_feas_ell, sens(i).best_feas_sf, sens(i).best_feas_sn);
end
end

function starts = mm_prepend_unique_start(starts, theta_new, dedupe_tol)
% Prepend theta_new as first column unless already present (log-space norm).
theta_new = theta_new(:);
if isempty(starts)
    starts = theta_new;
    return;
end
for j = 1:size(starts, 2)
    if norm(starts(:, j) - theta_new) < dedupe_tol
        return;
    end
end
starts = [theta_new, starts];
end

function ylim_unc = mm_plot_ell_ub_sensitivity_grid(sens, ell_ub_sweep, x_grid, y_true, x_train, y_train, ...
    m_unc, s2_unc, f_upper_unc, f_lower_unc, x_lim_lo, x_max, Vmax, n_train, noise_sd_true)
% 2x3 grid: unconstrained + constrained posteriors for ell_ub = 4,6,8,10.
tlo = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo, 'ell_{ub} sensitivity: unconstrained vs constrained posteriors', 'Interpreter', 'none');

nexttile(1);
hold on; grid on;
fill([x_grid, fliplr(x_grid)], [f_upper_unc', fliplr(f_lower_unc')], [0.75, 0.75, 0.78], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.5);
plot(x_grid, m_unc, 'k--', 'LineWidth', 2);
plot(x_grid, y_true, 'b-', 'LineWidth', 1.5);
plot(x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 5);
yline(Vmax, 'k:', 'Alpha', 0.5);
xlabel('[S] (mM)');
ylabel('v_0 (\muM/s)');
title('Unconstrained', 'Interpreter', 'none');
set(gca, 'FontSize', 10);
xlim([x_lim_lo, x_max]);
ylim_unc = ylim;

tile_pos = [2, 3, 5, 6];
for k = 1:numel(ell_ub_sweep)
    r = sens(k);
    f_upper = r.m_con + 2 * sqrt(max(r.s2_con, 0));
    f_lower = r.m_con - 2 * sqrt(max(r.s2_con, 0));
    nexttile(tile_pos(k));
    hold on; grid on;
    fill([x_grid, fliplr(x_grid)], [f_upper', fliplr(f_lower')], [0.55, 0.72, 0.55], ...
        'EdgeColor', 'none', 'FaceAlpha', 0.5);
    plot(x_grid, r.m_con, 'k--', 'LineWidth', 2);
    plot(x_grid, y_true, 'b-', 'LineWidth', 1.5);
    plot(x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 5);
    yline(Vmax, 'k:', 'Alpha', 0.5);
    xlabel('[S] (mM)');
    ylabel('v_0 (\muM/s)');
    title(sprintf('ell_{ub}=%g: ell=%.3f NLML=%.2f e@ub=%d', ell_ub_sweep(k), r.ell_opt, r.nlml_opt, r.ell_at_ub), ...
        'Interpreter', 'none');
    set(gca, 'FontSize', 10);
    ylim(ylim_unc);
    xlim([x_lim_lo, x_max]);
end

nexttile(4);
axis off;
text(0.1, 0.85, 'Shared ylim from unconstrained', 'Units', 'normalized', 'FontSize', 10);
text(0.1, 0.70, sprintf('Data: n=%d, \\sigma=%.2g', n_train, noise_sd_true), 'Units', 'normalized', 'FontSize', 10);
text(0.1, 0.55, 'Blue: MM truth; dashed: GP mean', 'Units', 'normalized', 'FontSize', 10);
text(0.1, 0.40, 'Continuation warm start between ell_{ub} cases', 'Units', 'normalized', 'FontSize', 10);
end
