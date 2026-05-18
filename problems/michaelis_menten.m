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

% Hyperparameter box (constrained fmincon only)
ell_bounds = [0.02, 4];
sf_bounds  = [0.1, 12];
sn_bounds  = [0.05, 2.5];
hyp_lb = log([ell_bounds(1); sf_bounds(1); sn_bounds(1)]);
hyp_ub = log([ell_bounds(2); sf_bounds(2); sn_bounds(2)]);

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

% Hyperparameter optimization (unconstrained NLML, unbounded)
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

% Baseline scientific constraints (Pensoneault tail at eta = 2.2%, k ~ 2):
%   mu*(S) - k*s*(S) >= 0,  mu*(S) + k*s*(S) <= Vmax,  mu*'(S) - k*s*'(S) >= 0
% on X_c (and X_c_mono for derivatives). No data-fidelity tube in baseline.
eta = 0.022;
k   = -sqrt(2) * erfinv(2*eta - 1);   % Phi^{-1}(eta) ~ -2

y_max = Vmax;   % L = 0 (lower tail), U = Vmax (upper tail)

% Baseline: lower + upper + monotonicity on; data tube off (add later as separate experiment).
enforce_upper_bound = true;
enforce_data_fidelity = false;
enforce_monotonicity = true;
k_mono = [];   % same k for value and derivative tails

% Reserved for data-fidelity tube experiments (Eq. 14); unused when enforce_data_fidelity = false.
epsilon = 0.165;

if debug_chol && enforce_monotonicity
    try
        gp_seiso_deriv_pred(hyp_unc, x_col, y_col, X_c_mono);
        fprintf('[CHOL debug hyp_unc] gp_seiso_deriv_pred OK at unconstrained hyp.\n');
    catch ME
        fprintf('[CHOL debug hyp_unc] gp_seiso_deriv_pred FAILED: %s\n', ME.message);
        rethrow(ME);
    end
end

% Warm start from unconstrained solution (projected onto hyperparameter box).
theta0 = min(max([hyp_unc.cov(1); hyp_unc.cov(2); hyp_unc.lik(1)], hyp_lb), hyp_ub);
if ~all(isfinite(theta0))
    error('michaelis_menten:theta0', 'theta0 has non-finite entries; check hyp_unc.');
end

[c0, ~] = pens_constraints(theta0, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x_col, y_col, X_c, X_c_mono, k, epsilon, y_max, ...
    enforce_upper_bound, enforce_data_fidelity, enforce_monotonicity, k_mono);
fprintf('theta0 max constraint violation = %.6g\n', max(c0));
fprintf('theta0: ell=%.4f sf=%.4f sn=%.4f\n', exp(theta0(1)), exp(theta0(2)), exp(theta0(3)));

% Objective (Eq. 12): GPML's gp() with no test inputs returns the NLML.
objfun_inner = @(theta) gp(theta_to_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
if debug_chol
    objfun = @(theta) mm_gp_wrap_chol(objfun_inner, theta, hyp_tpl, x_col, 'objfun');
else
    objfun = objfun_inner;
end
% Inequality constraints: optional upper tail (y_max), optional data tube (epsilon),
% nonnegative + monotonicity at X_c; see enforce_* flags above.
nonlcon_inner = @(theta) pens_constraints(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, X_c, X_c_mono, k, epsilon, y_max, ...
    enforce_upper_bound, enforce_data_fidelity, enforce_monotonicity, k_mono);
if debug_chol
    nonlcon = @(theta) mm_nonlcon_wrap_chol(nonlcon_inner, theta, hyp_tpl, x_col, 'nonlcon');
else
    nonlcon = nonlcon_inner;
end

opts = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, ...
    'Display', 'iter', ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);

fprintf(['Running constrained optimization (fmincon). |X_c|=%d |X_c_mono|=%d, k=%.4f; ', ...
    'L=0 U=%g; upper=%d data_tube=%d mono=%d.\n'], ...
    numel(X_c), numel(X_c_mono), k, y_max, enforce_upper_bound, enforce_data_fidelity, enforce_monotonicity);
fprintf(['  hyp box: ell [%.3g,%.3g], sf [%.3g,%.3g], sn [%.3g,%.3g].\n'], ...
    ell_bounds(1), ell_bounds(2), sf_bounds(1), sf_bounds(2), sn_bounds(1), sn_bounds(2));
[theta_opt, nlml_opt, exitflag, output] = fmincon(objfun, theta0, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts);

hyp_con = theta_to_hyp(theta_opt, hyp_tpl);
[m_con, s2_con] = gp(hyp_con, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
f_upper_con = m_con + 2 * sqrt(max(s2_con, 0));
f_lower_con = m_con - 2 * sqrt(max(s2_con, 0));

x_lim_lo = 0;

%% Visualization: unconstrained vs constrained (baseline)
tlo = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo, sprintf('Michaelis-Menten GP baseline: unconstrained vs constrained (L=0, U=%g, f'' >= 0; no data tube)', y_max), ...
    'Interpreter', 'none');

nexttile;
hold on; grid on;
fill([x_grid, fliplr(x_grid)], [f_upper_unc', fliplr(f_lower_unc')], [0.75, 0.75, 0.78], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', '95% CI');
plot(x_grid, m_unc, 'k--', 'LineWidth', 2, 'DisplayName', 'GP mean');
plot(x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
plot(x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', sprintf('Data (n=%d, \\sigma=%.2g)', n_train, noise_sd_true));
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
    'DisplayName', sprintf('Data (n=%d, \\sigma=%.2g)', n_train, noise_sd_true));
yline(Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
xlabel('[S] (mM)');
ylabel('v_0 (\muM/s)');
title(sprintf('Constrained (|X_c|=%d, k=%.3f, L=0, U=%g, mono)', numel(X_c), k, y_max));
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
