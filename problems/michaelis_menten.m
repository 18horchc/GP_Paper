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
% Toggle: false = experimental assay table; true = Trial 1 synthetic (see plan).
useTrial1Synthetic = true;
% Constraint grid sizes (value tails at X_c; monotonicity at X_c_mono) — used in %% Fit section.
m_bounds = 30;
m_mono = 15;

% x_max = upper [S] (mM): sampling upper bound (synthetic), x_grid / ground truth extent, and plot window.
S_lo = 1e-6;
x_max = 2;
n_samples = 12;
half_step = 0.015;   % Half-step grid (Option B); candidates S_lo : half_step : x_max.

if useTrial1Synthetic
    % ----- Synthetic: fixed [S] (mM); Gaussian noise on v_0 except no draw at [S]=0 (exact 0,0) -----
    x_train = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.8];
    x_max = 2;
    noise_level_gp = 0.10;
    rng(42);
    y_clean = mm_static(x_train);
    dz = zeros(size(x_train));
    idx = abs(x_train) > 1e-12;
    dz(idx) = noise_level_gp .* randn(1, sum(idx(:)));
    y_train = y_clean + dz;
    n_unique_S = numel(x_train);
    n_samples = n_unique_S;
    n_replicates = 1;
else
    % ----- DEFAULT: experimental assay (table + replicates) -----
    S_mM = [0.00; 0.03; 0.07; 0.15; 0.30; 0.60; 1.50];
    Y_runs = [ ...
        0.00, 0.00, 0.00; ...
        1.05, 0.98, 0.95; ...
        1.85, 1.96, 2.01; ...
        3.10, 2.95, 2.92; ...
        3.95, 4.10, 4.05; ...
        4.85, 4.70, 4.88; ...
        5.51, 5.40, 5.48];
    v0_std = std(Y_runs, 0, 2);
    n_unique_S = size(Y_runs, 1);
    n_replicates = size(Y_runs, 2);
    x_train = repelem(S_mM.', n_replicates);
    y_train = reshape(permute(Y_runs, [2, 1]), 1, []);
    noise_level_gp = mean(v0_std(v0_std > 0));
    x_max = max(2, max(S_mM) * 1.2);
end

% ----- OPTION A / B (reference): set useTrial1Synthetic false for assay, or paste logic into a new branch. -----
% ----- OPTION A: uniform [S] on [0, x_max] (n_samples points), Gaussian noise on v_0 -----
% Always includes (0, 0): equispaced linspace from 0 to x_max; no noise added at [S]=0 (matches Option B).
% noise_level_gp = 0.10;   % std dev of additive Gaussian noise on v_0 (μM/s); used by GP init (sn0) unchanged
% rng(42);                 % optional reproducibility
% x_train = linspace(0, x_max, n_samples);
% y_train = mm_static(x_train) + noise_level_gp .* randn(size(x_train));
% y_train(abs(x_train) <= 1e-12) = 0;
% n_unique_S = n_samples;
% n_replicates = 1;

% ----- OPTION B: n_samples random [S] from half-step grid, Gaussian noise -----
% Always includes (0, 0); remaining n_samples-1 are drawn without replacement from grid nodes with [S] > 0.
% Requires n_samples >= 1 and n_samples-1 <= numel(nonzero grid nodes); else widen x_max or shrink half_step.
% noise_level_gp = 0.10;
% rng(42);
% candidates = (S_lo:half_step:x_max).';
% pool = candidates(abs(candidates) > 1e-12);   % exclude [S]=0 so (0,0) is not duplicated
% n_pick = n_samples - 1;
% assert(n_samples >= 1, 'Option B: n_samples must be at least 1.');
% assert(n_pick <= numel(pool), ['Option B: need n_samples-1=%d random grid points plus (0,0); ', ...
%     'only %d grid nodes with [S]>0. Widen x_max or shrink half_step.'], n_pick, numel(pool));
% idx = randperm(numel(pool), n_pick);
% x_rand = pool(idx).';
% y_rand = mm_static(x_rand) + noise_level_gp .* randn(size(x_rand));
% x_train = [0, x_rand];
% y_train = [0, y_rand];
% [x_train, ord] = sort(x_train);
% y_train = y_train(ord);
% n_unique_S = n_samples;
% n_replicates = 1;

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
% Pre-calculating values for initialization
ell0 = std(x_train); 
sf0  = std(y_train);
sn0  = max(1e-3, noise_level_gp);

% Initialize the GPML hyperparameter structure
hyp = struct();
hyp.mean = [];              % @meanZero: no mean hyperparameters
hyp.cov  = log([ell0; sf0]); 
hyp.lik  = log(sn0);

% Define GP components
meanfunc = @meanZero;
%covfunc  = {@covMaterniso,5}; 
covfunc = @covSEiso;
likfunc = @likGauss;
inffunc = @infGaussLik;

x_col = x_train(:);
y_col = y_train(:);

%  Hyperparameter Optimization (unconstrained baseline)
% -100 specifies a maximum of 100 function evaluations
fprintf('Optimizing hyperparameters (unconstrained NLML)...\n');
hyp_unc = minimize(hyp, @gp, -100, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);

[m_unc, s2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
f_upper_unc = m_unc + 2 * sqrt(max(s2_unc, 0));
f_lower_unc = m_unc - 2 * sqrt(max(s2_unc, 0));

% --- Pensoneault constrained GP (direct math-to-code translation of Eqs. 12-14) ---
% Eq. (12):  argmin_theta  -log p(y | X, theta)
% Eq. (13):  0 <= y*(x_c^(i)) - k * s(x_c^(i)),  i = 1, ..., m
% Eq. (14):  0 <= epsilon  - |y^(j) - y*(x^(j))|, j = 1, ..., n
% y*(.) and s(.) are the GP posterior predictive mean and standard deviation
% (noisy predictive in GPML: ymu, sqrt(ys2)) at theta = (log_ell, log_sf, log_sn).

% theta packs [log(ell); log(sf); log(sn)] (mean is fixed at zero via @meanZero).
% Warm start: use the unconstrained NLML solution as both the template and
% the starting point. theta0 is therefore already in a well-conditioned
% region for chol() inside GPML's infGaussLik.
hyp_tpl = hyp_unc;

% Eq. (13) ingredients: m_bounds value-tail points; m_mono derivative (monotonicity) points.
% k = inverse normal CDF at eta=2.2%, paper's approximation k=2.
% Use S_lo (not 0) as the left end: if training includes [S]=0 (Options A/B or assay),
% xstar = [X_c; x] in pens_constraints must not repeat the same x (GPML chol on predictive cov).
% Monotonicity at X_c_mono: posterior on latent slope f' is Gaussian; require
% m_deriv - k * s_deriv >= 0 (equivalently k * s_deriv - m_deriv <= 0), matching
% the one-sided tail used for the nonnegative value bound in Eq. (13).
X_c = linspace(S_lo, max(x_grid), m_bounds)';
X_c_mono = linspace(S_lo, max(x_grid), m_mono)';
eta = 0.022;
k   = -sqrt(2) * erfinv(2*eta - 1); %def of invCDF(eta)

% Eq. (14) ingredient: epsilon widened for joint constraints (upper + data + mono).
% Tighter values (e.g. 0.165 per paper rescale) bind hard with monotonicity on.
epsilon = 0.5;

% Upper-tail counterpart to Eq. (13): y*(x_c) + k * s(x_c) <= y_max.
% y_max is set to Vmax (defined in Section 1, line 5) -- the known MM
% asymptote. In real applications without a known asymptote you'd pick
% y_max from a separate physical or experimental bound.
y_max = Vmax;

% Constraint toggles (nonnegative tail at X_c is always on when fmincon runs).
% Set true to re-enable the Pensoneault upper tail at X_c or the epsilon data tube.
enforce_upper_bound = false;
enforce_data_fidelity = false;

% If fmincon is infeasible or stagnates, the derivative tail can conflict with
% tight epsilon / replicate fidelity; set false to use value-only constraints.
enforce_monotonicity = true;
% Optional softer tail for derivative constraints only (empty = same k as values).
% If fmincon reports joint infeasibility with monotonicity on, try e.g. k_mono = 0.65*k.
k_mono = [];

% Warm start theta0 from the unconstrained NLML solution.
theta0 = [hyp_unc.cov(1); hyp_unc.cov(2); hyp_unc.lik(1)];

% No lower bound on log(sigma_n) (trial plan). With many replicate x rows, tiny sn can
% still make chol() fail inside GPML; if that happens, reintroduce e.g. lb(3)=log(1e-3).
lb = [-Inf; -Inf; -Inf];
ub = [];

% Objective (Eq. 12): GPML's gp() with no test inputs returns the NLML.
objfun  = @(theta) gp(theta_to_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
% Inequality constraints: optional upper tail (y_max), optional data tube (epsilon),
% nonnegative + monotonicity at X_c; see enforce_* flags above.
nonlcon = @(theta) pens_constraints(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, X_c, X_c_mono, k, epsilon, y_max, ...
    enforce_upper_bound, enforce_data_fidelity, enforce_monotonicity, k_mono);

opts = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, ...
    'Display', 'iter', ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);

fprintf(['Running constrained optimization (fmincon). m_bounds=%d m_mono=%d, k=%g; ', ...
    'upper=%d data_tube=%d mono=%d (epsilon=%.4f, y_max=%g when used).\n'], ...
    m_bounds, m_mono, k, enforce_upper_bound, enforce_data_fidelity, enforce_monotonicity, epsilon, y_max);
[theta_opt, nlml_opt, exitflag, output] = fmincon(objfun, theta0, [], [], [], [], lb, ub, nonlcon, opts);

hyp_con = theta_to_hyp(theta_opt, hyp_tpl);
[m_con, s2_con] = gp(hyp_con, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
f_upper_con = m_con + 2 * sqrt(max(s2_con, 0));
f_lower_con = m_con - 2 * sqrt(max(s2_con, 0));

% Plot x-axis must extend to min training [S] (e.g. 0 for Options A/B); x_grid may start at S_lo > 0.
x_lim_lo = min(S_lo, min(x_train(:)));

%% Visualization: unconstrained vs constrained
tlo = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
if enforce_monotonicity && ~enforce_upper_bound && ~enforce_data_fidelity
    title(tlo, 'Michaelis-Menten GP: unconstrained vs nonnegative + monotone (no V_{max} tail, no \epsilon tube)');
elseif enforce_monotonicity
    title(tlo, 'Michaelis-Menten GP: unconstrained vs constrained (see upper/data/mono flags)');
else
    title(tlo, 'Michaelis-Menten GP: unconstrained vs bounded (value constraints only)');
end

nexttile;
hold on; grid on;
fill([x_grid, fliplr(x_grid)], [f_upper_unc', fliplr(f_lower_unc')], [0.75, 0.75, 0.78], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', '95% CI');
plot(x_grid, m_unc, 'k--', 'LineWidth', 2, 'DisplayName', 'GP mean');
plot(x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
plot(x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', sprintf('Data (%d [S], %d runs)', n_unique_S, n_replicates));
yline(Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
xlabel('[S] (mM)');
ylabel('v_0 (\muM/s)');
title('Unconstrained GPML (minimize NLML)');
legend('Location', 'southeast');
set(gca, 'FontSize', 11);
xlim([x_lim_lo, x_max]);
ylim_unc = ylim;

nexttile;
hold on; grid on;
fill([x_grid, fliplr(x_grid)], [f_upper_con', fliplr(f_lower_con')], [0.55, 0.72, 0.55], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', '95% CI');
plot(x_grid, m_con, 'k--', 'LineWidth', 2, 'DisplayName', 'GP mean');
plot(x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
plot(x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', sprintf('Data (%d [S], %d runs)', n_unique_S, n_replicates));
yline(Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
xlabel('[S] (mM)');
ylabel('v_0 (\muM/s)');
if enforce_monotonicity && ~enforce_upper_bound && ~enforce_data_fidelity
    title(sprintf('Nonneg + monotone (m_{val}=%d, m_{mono}=%d, k=%g)', m_bounds, m_mono, k));
elseif enforce_monotonicity
    title(sprintf('Constrained (m_{val}=%d, m_{mono}=%d, k=%g, \\epsilon=%.2f, y_{max}=%g)', m_bounds, m_mono, k, epsilon, y_max));
else
    title(sprintf('Bounded (m_{val}=%d, m_{mono}=%d, k=%g, \\epsilon=%.2f, y_{max}=%g)', m_bounds, m_mono, k, epsilon, y_max));
end
legend('Location', 'southeast');
set(gca, 'FontSize', 11);
ylim(ylim_unc);
xlim([x_lim_lo, x_max]);

fprintf('GP optimization complete.\n');
fprintf('Unconstrained: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), ...
    gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col));
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
