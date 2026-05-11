% GP Research Project: Acute Transient Equation Sampling
clear; clc; close all;

%% 1. Define Realistic Parameters 
% Parameters 
Vmax = 6;       % Maximum velocity (units/s) 
Km   = 0.15;        % Michaelis constant (units of [S]) 

% Define function
% v = (Vmax * [S]) / (Km + [S])
mm_static = @(S) (Vmax .* S) ./ (Km + S); 


%% 2. Replicate assay data ([S] in mM, v_0 in μM/s)
% Table layout: [Substrate] (mM), Run 1–3 (v_0), mean, std. dev.
% (0.15 mM row is near Km in the experiment; three runs plot as three y at the same x.)

S_mM = [0.00; 0.03; 0.07; 0.15; 0.30; 0.60; 1.50];

Y_runs = [ ...
    0.00, 0.00, 0.00; ...
    1.05, 0.98, 0.95; ...
    1.85, 1.96, 2.01; ...
    3.10, 2.95, 2.92; ...
    3.95, 4.10, 4.05; ...
    4.85, 4.70, 4.88; ...
    5.51, 5.40, 5.48];

%v0_mean = [0.00; 0.99; 1.94; 2.99; 4.03; 4.81; 5.46];
%v0_std  = [0.00; 0.05; 0.08; 0.09; 0.08; 0.10; 0.06];
v0_std = std(Y_runs, 0, 2); 

n_unique_S = size(Y_runs, 1);
n_replicates = size(Y_runs, 2);

% Training vectors: each [S] repeated n_replicates times; y in run order per substrate
x_train = repelem(S_mM.', n_replicates);
y_train = reshape(permute(Y_runs, [2, 1]), 1, []);

% Sanity check vs. reported means (table means are rounded)
%calc_mean = mean(reshape(y_train, n_replicates, n_unique_S), 1)';
%assert(max(abs(calc_mean - v0_mean)) < 0.02);

% Initial GP observation noise scale (μM/s) from typical within-[S] spread
noise_level_gp = mean(v0_std(v0_std > 0));

%% 3. Ground truth curve on an [S] grid (mM)
x_grid = linspace(1e-6, max(2, max(S_mM) * 1.2), 500);
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
hyp.mean = []; % Correct for meanZero
hyp.cov  = log([ell0; sf0]); 
hyp.lik  = log(sn0);

% Define GP components 
meanfunc = @meanZero; 
%meanfunc = @meanConst; 
%hyp.mean = mm_static(20);
%covfunc  = {@covMaterniso,5}; 
covfunc = @covSEiso;
likfunc = @likGauss;
inffunc = @infGaussLik;

x_col = x_train(:);
y_col = y_train(:);

% --- Pensoneault et al. (2020): constraint knobs (tune if fmincon is infeasible) ---
m_constraint = 30;                    % number of constraint locations X_c (includes [S]=0)
eta_quantile = 2.2 / 100;             % P(f<0) <= eta; Phi^{-1}(eta) ~ -2 for eta=2.2%
k_nonneg = -sqrt(2) * erfinv(2 * eta_quantile - 1);   % Phi^{-1}(eta); ~2 for 2.2%
k_epsilon = 1;                        % data fidelity: epsilon = k_epsilon * mean(v0_std) (tune k_epsilon)
% Nonnegativity on latent f(x) (noise-free v0): easier/feasible than y|x predictive; still Pensoneault-style tail bound
use_latent_nonneg = true;
v0_std_mean = mean(v0_std(v0_std > 0));
epsilon_data = k_epsilon * v0_std_mean;   % |y_j - m(x_j)| <= epsilon (muM/s)
x_hi_c = max(x_grid);
% X_c: exactly m_constraint points from 0 to x_hi_c (0 is always the first constraint point)
X_c = [0; linspace(x_hi_c / (m_constraint - 1), x_hi_c, m_constraint - 1)'];
fprintf('Pensoneault data fidelity: epsilon = k_epsilon * mean(v0_std) = %.2f * %.4f = %.4f (muM/s)\n', ...
    k_epsilon, v0_std_mean, epsilon_data);

%  Hyperparameter Optimization (unconstrained baseline)
% -100 specifies a maximum of 100 function evaluations
fprintf('Optimizing hyperparameters (unconstrained NLML)...\n');
hyp_unc = minimize(hyp, @gp, -100, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);

[m_unc, s2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
f_upper_unc = m_unc + 2 * sqrt(max(s2_unc, 0));
f_lower_unc = m_unc - 2 * sqrt(max(s2_unc, 0));

% --- Constrained NLML (Pensoneault et al., fmincon) ---
hyp_tpl = struct('mean', hyp_unc.mean, 'cov', hyp_unc.cov, 'lik', hyp_unc.lik);
theta0 = [hyp_unc.cov(1); hyp_unc.cov(2); hyp_unc.lik(1)];
% Floor sigma_n so the solver cannot collapse noise to ~0 to cheat constraints (stabilizes KKT / NLML)
sn_floor = max(1e-3, 0.35 * noise_level_gp);
lb = [-10; -10; log(sn_floor)];
ub = [10; 10; 5];
theta0 = min(max(theta0, lb), ub);

objfun = @(th) pens_nlml(th, hyp_tpl, x_col, y_col, meanfunc, covfunc, likfunc, inffunc);
nonlfun = @(th) pens_nonlcon(th, hyp_tpl, x_col, y_col, meanfunc, covfunc, likfunc, inffunc, ...
    X_c, k_nonneg, epsilon_data, use_latent_nonneg);

% SQP is often more stable than interior-point for this small dense NLP; relax tol vs 1e-6 to reduce false infeasible stops
opts_sqp = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'iter', ...
    'MaxIterations', 1500, 'MaxFunctionEvaluations', 3000, ...
    'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, 'StepTolerance', 1e-10);
opts_ip = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'iter', ...
    'MaxIterations', 1500, 'MaxFunctionEvaluations', 3000, ...
    'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, 'StepTolerance', 1e-10);
try
    opts_ip = optimoptions(opts_ip, 'EnableFeasibilityMode', true);
catch
end

if use_latent_nonneg
    nnz_mode = 'latent f (noise-free v_0)';
else
    nnz_mode = 'noisy predictive y';
end
fprintf('Running constrained optimization (fmincon, SQP). Nonneg on %s; sigma_n >= %.4f muM/s.\n', ...
    nnz_mode, sn_floor);
epsilon_used = epsilon_data;
[theta_opt, ~, exitflag, output] = fmincon(objfun, theta0, [], [], [], [], lb, ub, nonlfun, opts_sqp);

if ~ismember(exitflag, [1, 2])
    warning('SQP exit flag %d. Trying interior-point with feasibility mode.', exitflag);
    [theta_opt, ~, exitflag, output] = fmincon(objfun, theta0, [], [], [], [], lb, ub, nonlfun, opts_ip);
end

if ~ismember(exitflag, [1, 2])
    warning('fmincon exit flag %d (%s). Retrying with paper fallback initial theta.', ...
        exitflag, output.message);
    theta_fallback = [-3; -3; -10] + 0.1 * randn(3, 1);
    theta_fallback = min(max(theta_fallback, lb), ub);
    [theta_opt, ~, exitflag, output] = fmincon(objfun, theta_fallback, [], [], [], [], lb, ub, nonlfun, opts_sqp);
end

if ~ismember(exitflag, [1, 2])
    warning('Constrained optimization still not converged (exitflag=%d). Loosening epsilon_data once.', exitflag);
    epsilon_loose = max(epsilon_data, 1.5 * k_epsilon * v0_std_mean);
    nonlfun_loose = @(th) pens_nonlcon(th, hyp_tpl, x_col, y_col, meanfunc, covfunc, likfunc, inffunc, ...
        X_c, k_nonneg, epsilon_loose, use_latent_nonneg);
    [theta_opt, ~, exitflag, output] = fmincon(objfun, theta0, [], [], [], [], lb, ub, nonlfun_loose, opts_sqp);
    epsilon_used = epsilon_loose;
    fprintf('Retried with epsilon_data = %.4f (muM/s)\n', epsilon_loose);
end

hyp_con = pens_unpack_theta(theta_opt, hyp_tpl);
[m_con, s2_con] = gp(hyp_con, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
f_upper_con = m_con + 2 * sqrt(max(s2_con, 0));
f_lower_con = m_con - 2 * sqrt(max(s2_con, 0));

%% Visualization: unconstrained vs nonnegative-enforced
tlo = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo, 'Michaelis-Menten GP: unconstrained vs Pensoneault nonnegative-enforced');

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
title(sprintf('Nonnegative-enforced (m=%d, k=%.2f, \\epsilon=%.2f)', m_constraint, k_nonneg, epsilon_used));
legend('Location', 'southeast');
set(gca, 'FontSize', 11);
ylim(ylim_unc);

fprintf('GP optimization complete.\n');
fprintf('Unconstrained: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), ...
    pens_nlml([hyp_unc.cov(1); hyp_unc.cov(2); hyp_unc.lik(1)], hyp_tpl, x_col, y_col, meanfunc, covfunc, likfunc, inffunc));
fprintf('Constrained:   ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | fmincon exitflag=%d\n', ...
    exp(hyp_con.cov(1)), exp(hyp_con.cov(2)), exp(hyp_con.lik), ...
    pens_nlml(theta_opt, hyp_tpl, x_col, y_col, meanfunc, covfunc, likfunc, inffunc), exitflag);

%% Encodings to try:
% Non zero mean. Since MM kinetircs are strictly positive and approach
% asymptote at Vmax, large gaps in data points may suffer from
% prior-dominated corridor effect where the prediction will try to revert
% to zero rather than staying near saturation plateau when zero mean is
% used.

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

function hyp = pens_unpack_theta(theta, hyp_tpl)
hyp = hyp_tpl;
hyp.cov = theta(1:2);
hyp.lik = theta(3);
end

function nlZ = pens_nlml(theta, hyp_tpl, x, y, meanf, covf, likf, infMeth)
hyp = pens_unpack_theta(theta, hyp_tpl);
try
    nlZ = gp(hyp, infMeth, meanf, covf, likf, x, y);
catch
    nlZ = 1e10;
end
end

function [c, ceq] = pens_nonlcon(theta, hyp_tpl, x, y, meanf, covf, likf, infMeth, Xc, k, epsd, use_latent_nonneg)
ceq = [];
hyp = pens_unpack_theta(theta, hyp_tpl);
try
    xstar = [Xc(:); x(:)];
    [ymu, ys2, fmu, fs2] = gp(hyp, infMeth, meanf, covf, likf, x, y, xstar);
catch
    c = 1e6 * ones(numel(Xc) + numel(y), 1);
    return;
end
nC = numel(Xc);
if use_latent_nonneg
    mc = fmu(1:nC);
    sc = sqrt(max(fs2(1:nC), 1e-18));
else
    mc = ymu(1:nC);
    sc = sqrt(max(ys2(1:nC), 1e-18));
end
c_nonneg = k * sc - mc;
mj = ymu(nC+1:end);
c_data = abs(y(:) - mj) - epsd;
c = [c_nonneg; c_data];
end
