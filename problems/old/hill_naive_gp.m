% GP Research Project: Hill Equation -- Naive GP vs Pensoneault-constrained GP
clear; clc; close all;

%% 1. Define Realistic Parameters (Dose-Response)
% Parameters for a standard dose-response curve
E_min = 5.0;       % Minimum effect (baseline)
E_max = 10.0;      % Maximum effect (plateau)
EC50  = 12.5;       % Half-maximal effective concentration
n_hill = 3;      % Hill coefficient (slope / cooperativity)

x_max = 35;        % Upper end of the concentration domain

% Hill equation: E(C) = E_min + (E_max - E_min) / (1 + (EC50 / C)^n)
hill_func = @(C) E_min + (E_max - E_min) ./ (1 + (EC50 ./ C).^n_hill);

%% 2. Ground truth curve on a concentration grid
x_grid = linspace(1e-6, x_max, 500);
y_true = hill_func(x_grid);

%% 3. Sampling options (select ONE; comment the other two)
noise_frac = 0.05;   % homoscedastic: sigma = noise_frac * max E on [0, x_max]

% --- Option 1: Uniform sampling of n_samples points across the domain ---
%n_samples = 7;
%x_train = linspace(0, x_max, n_samples);

% --- Option 2: Randomly select n_samples points from a half-step grid ---
% n_samples = 12;
% half_step_grid = 0:0.5:x_max;             % candidate locations at 0.5 spacing
% idx_sel  = randperm(numel(half_step_grid), n_samples);
% x_train  = sort(half_step_grid(idx_sel));

% --- Option 3: Set specific sample locations ---
x_train   = [6.5, 12.5, 18.5, 21.5, 28.0, 30.0, 34.0];
n_samples = numel(x_train);

% Noisy observations at the chosen training locations
rng(100);
y_true_at_train = hill_func(x_train);
y_domain_max = hill_func(x_max);
noise_sd_true = noise_frac * y_domain_max;
y_train = y_true_at_train + noise_sd_true * randn(size(y_true_at_train));
noise_sd_est = noise_sd_true;
fprintf('Synthetic data: n=%d, homoscedastic noise sigma_n = %.4f (%.0f%% of E(x_max))\n', ...
    n_samples, noise_sd_true, 100 * noise_frac);

%% Pensoneault constraint grid and parameters
eta = 0.022;   % 2.2% tail probability
k   = -sqrt(2) * erfinv(2 * eta - 1);
n_constraint = 41;
X_c = linspace(0, x_max, n_constraint)';
epsilon = 0.5;   % data fidelity: |y - y*(x)| <= epsilon

%% 4. Fit the Naive GP
% Make sure GPML is on the path (looks for a core function like gp.m)
gpml_folder_name = "C:\Users\chorc\OneDrive\Documents\Stroke Research\Gaussian Processes\Old\gpml-matlab-master\gpml-matlab-master";
addpath(fileparts(mfilename('fullpath')));

if ~exist('gp', 'file')
    fprintf('GPML not found. Searching in local directory: %s...\n', gpml_folder_name);
    if exist(gpml_folder_name, 'dir')
        addpath(genpath(gpml_folder_name));
        fprintf('GPML successfully added to path.\n');
    else
        error(['GPML toolbox missing! Please download it and ensure ' ...
               'the folder "%s" is in your project directory.'], gpml_folder_name);
    end
else
    fprintf('GPML toolbox is already loaded and ready.\n');
end

try
    startup;
catch
    % Some GPML versions ship without startup.m; safe to ignore.
end

% Configuration & Initial Hyperparameters
ell0 = std(x_train);
sf0  = std(y_train);
sn0  = max(1e-3, noise_sd_true);

hyp = struct();
hyp.mean = [];                  % matches @meanZero
hyp.cov  = log([ell0; sf0]);
hyp.lik  = log(sn0);

meanfunc = @meanZero;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

x_col = x_train(:);
y_col = y_train(:);

% Hyperparameter Optimization (unconstrained NLML, naive GP)
fprintf('Optimizing hyperparameters (unconstrained NLML)...\n');
hyp_unc = minimize(hyp, @gp, -100, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
nlml_unc = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);

%% Pensoneault-constrained GP (fmincon on NLML)
hyp_tpl = hyp_unc;
objfun = @(theta) gp(theta_to_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
nonlcon = @(theta) pens_constraints_hill(theta, hyp_tpl, x_col, y_col, X_c, k, ...
    E_min, E_max, epsilon, inffunc, meanfunc, covfunc, likfunc);

opts_sqp = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'off', ...
    'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
    'MaxFunctionEvaluations', 30000, 'MaxIterations', 3000);
opts_ip = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, 'Display', 'off', ...
    'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
    'MaxFunctionEvaluations', 30000, 'MaxIterations', 3000);

ell_bounds_lo = 0.02;
ell_ub = x_max;
sf_bounds = [0.05, 15];
sn_bounds = [max(1e-4, noise_sd_true / 10), 2];
hyp_lb = log([ell_bounds_lo; sf_bounds(1); sn_bounds(1)]);
hyp_ub = log([ell_ub; sf_bounds(2); sn_bounds(2)]);
theta_unc_box = min(max([hyp_unc.cov(1); hyp_unc.cov(2); hyp_unc.lik(1)], hyp_lb), hyp_ub);

fprintf('\n=== Pensoneault-constrained GP ===\n');
fprintf('eta = %.3g%% | k = %.4f | X_c: %d | epsilon = %.4g | bounds [%.1f, %.1f]\n', ...
    100 * eta, k, numel(X_c), epsilon, E_min, E_max);

nTry = 2000;
nMultistart = 3;
feasible_starts = zeros(3, 0);
rng(43);
for t = 1:nTry
    theta_try = hyp_lb + rand(3, 1) .* (hyp_ub - hyp_lb);
    [c_try, ~] = nonlcon(theta_try);
    if max(c_try) <= 0
        feasible_starts = [feasible_starts, theta_try];
    end
end
nFeas = size(feasible_starts, 2);
fprintf('Feasible random starts: %d / %d\n', nFeas, nTry);

if nFeas > 0
    nlml_feas = arrayfun(@(j) objfun(feasible_starts(:, j)), 1:nFeas);
    [~, ord] = sort(nlml_feas, 'ascend');
    starts = feasible_starts(:, ord(1:min(nMultistart, nFeas)));
else
    fprintf('No feasible random start; using projected unconstrained theta.\n');
    starts = zeros(3, 0);
end
theta_hand = min(max(log([std(x_train); std(y_train); noise_sd_true]), hyp_lb), hyp_ub);
starts = [theta_unc_box, theta_hand, starts];
starts = starts(:, 1:min(nMultistart + 2, size(starts, 2)));

obj_feas = @(theta) max(nonlcon(theta));
opts_feas = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'off', ...
    'MaxFunctionEvaluations', 20000, 'MaxIterations', 2000, ...
    'ConstraintTolerance', 1e-6, 'OptimalityTolerance', 1e-6);
fprintf('Phase 1: minimizing constraint violation...\n');
theta_feas = theta_unc_box;
maxc_feas = obj_feas(theta_feas);
feas_starts = [theta_unc_box, theta_hand];
for j = 1:size(feas_starts, 2)
    [theta_j, maxc_j] = fmincon(obj_feas, feas_starts(:, j), [], [], [], [], hyp_lb, hyp_ub, [], opts_feas);
    if maxc_j < maxc_feas
        theta_feas = theta_j;
        maxc_feas = maxc_j;
    end
end
for t = 1:min(100, nTry)
    theta_try = hyp_lb + rand(3, 1) .* (hyp_ub - hyp_lb);
    [theta_j, maxc_j] = fmincon(obj_feas, theta_try, [], [], [], [], hyp_lb, hyp_ub, [], opts_feas);
    if maxc_j < maxc_feas
        theta_feas = theta_j;
        maxc_feas = maxc_j;
    end
end
fprintf('  Phase 1 best max(c)=%.6g\n', maxc_feas);
starts = [theta_feas, starts];

best_nlml = inf;
theta_opt = nan(3, 1);
exitflag = -99;
best_maxc = inf;
for j = 1:size(starts, 2)
    [theta_j, nlml_j, ef_j] = fmincon(objfun, starts(:, j), [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts_sqp);
    [c_j, ~] = nonlcon(theta_j);
    maxc_j = max(c_j);
    if isfinite(nlml_j) && pens_better(maxc_j, nlml_j, best_maxc, best_nlml)
        best_nlml = nlml_j;
        theta_opt = theta_j;
        exitflag = ef_j;
        best_maxc = maxc_j;
    end
end

if best_maxc > 1e-2
    fprintf('SQP did not reach feasibility (max(c)=%.4g); retrying with interior-point.\n', best_maxc);
    for j = 1:size(starts, 2)
        [theta_j, nlml_j, ef_j] = fmincon(objfun, starts(:, j), [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts_ip);
        [c_j, ~] = nonlcon(theta_j);
        maxc_j = max(c_j);
        if isfinite(nlml_j) && pens_better(maxc_j, nlml_j, best_maxc, best_nlml)
            best_nlml = nlml_j;
            theta_opt = theta_j;
            exitflag = ef_j;
            best_maxc = maxc_j;
        end
    end
end

if ~isfinite(best_nlml) || best_maxc > 1e-2
    fprintf('Trying paper-style cold start.\n');
    theta_cold = min(max(log([0.1; 1; noise_sd_true]), hyp_lb), hyp_ub);
    [theta_j, nlml_j, ef_j] = fmincon(objfun, theta_cold, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts_sqp);
    [c_j, ~] = nonlcon(theta_j);
    maxc_j = max(c_j);
    if isfinite(nlml_j) && pens_better(maxc_j, nlml_j, best_maxc, best_nlml)
        best_nlml = nlml_j;
        theta_opt = theta_j;
        exitflag = ef_j;
        best_maxc = maxc_j;
    end
end

fprintf('Phase 2: NLML minimization from best feasibility start...\n');
[theta_p2, nlml_p2, ef_p2] = fmincon(objfun, theta_feas, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts_sqp);
[c_p2, ~] = nonlcon(theta_p2);
maxc_p2 = max(c_p2);
if isfinite(nlml_p2) && pens_better(maxc_p2, nlml_p2, best_maxc, best_nlml)
    theta_opt = theta_p2;
    best_nlml = nlml_p2;
    exitflag = ef_p2;
    best_maxc = maxc_p2;
end

if best_maxc > maxc_feas + 1e-6
    fprintf('Using Phase 1 theta (lower constraint violation).\n');
    theta_opt = theta_feas;
    best_maxc = maxc_feas;
    best_nlml = objfun(theta_opt);
end

hyp_con = theta_to_hyp(theta_opt, hyp_tpl);
nlml_con = objfun(theta_opt);
[c_final, ~] = nonlcon(theta_opt);
nC = numel(X_c);
fprintf('fmincon exitflag = %d\n', exitflag);
fprintf('  lower max(c) = %.6g | upper max(c) = %.6g | data max(c) = %.6g\n', ...
    max(c_final(1:nC)), max(c_final(nC+1:2*nC)), max(c_final(2*nC+1:end)));

%% 5. Predictions and visualization
k_plot = 2;
[~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_unc = fmu_unc(:);
sf_unc = sqrt(max(fs2_unc(:), 0));

[~, ~, fmu_con, fs2_con] = gp(hyp_con, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_con = fmu_con(:);
sf_con = sqrt(max(fs2_con(:), 0));

band_label = sprintf('\\mu_f \\pm %g\\sigma_f (latent)', k_plot);
ylim_plot = [0, 11];

fig = figure('Color', 'w', 'Position', [80, 80, 920, 620], ...
    'Name', sprintf('Hill GP (n=%d, %.0f%% noise)', n_samples, 100 * noise_frac));
tg = uitabgroup(fig);

tab_unc = uitab(tg, 'Title', 'Naive GP');
plot_hill_gp_tab(tab_unc, x_grid, x_max, m_unc, sf_unc, k_plot, band_label, ...
    y_true, x_train, y_train, noise_sd_true, E_min, E_max, 'Naive GPML', ylim_plot);

tab_con = uitab(tg, 'Title', 'Pensoneault');
plot_hill_gp_tab(tab_con, x_grid, x_max, m_con, sf_con, k_plot, band_label, ...
    y_true, x_train, y_train, noise_sd_true, E_min, E_max, ...
    sprintf('Pensoneault GP (\\eta=2.2%%, \\epsilon=%.1f, NLML=%.2f)', epsilon, nlml_con), ...
    ylim_plot);

fprintf('\nGP optimization complete.\n');
fprintf('Naive GP:        ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);
fprintf('Pensoneault GP:  ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_con.cov(1)), exp(hyp_con.cov(2)), exp(hyp_con.lik), nlml_con);

function better = pens_better(maxc, nlml, best_maxc, best_nlml)
if maxc <= 1e-3 && best_maxc > 1e-3
    better = true;
elseif maxc <= 1e-3 && best_maxc <= 1e-3
    better = nlml < best_nlml;
elseif maxc < best_maxc - 1e-6
    better = true;
elseif abs(maxc - best_maxc) <= 1e-6
    better = nlml < best_nlml;
else
    better = false;
end
end

function hyp = theta_to_hyp(theta, hyp_tpl)
hyp = hyp_tpl;
hyp.cov = theta(1:2);
hyp.lik = theta(3);
hyp.mean = [];
end

function [c, ceq] = pens_constraints_hill(theta, hyp_tpl, x, y, X_c, k, ...
    E_min, E_max, epsilon, inffunc, meanfunc, covfunc, likfunc)
hyp = theta_to_hyp(theta, hyp_tpl);
nC = numel(X_c);
xstar = [X_c(:); x(:)];
[ymu, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xstar);
ymu = ymu(:);
fmu = fmu(:);
m_xc = fmu(1:nC);
s_xc = sqrt(max(fs2(1:nC), 0));
c_lower = k * s_xc - (m_xc - E_min);   % m - k*s >= E_min
c_upper = m_xc + k * s_xc - E_max;     % m + k*s <= E_max
y_star = ymu(nC+1:end);
c_data = abs(y - y_star) - epsilon;    % |y - y*(x)| <= epsilon
c = [c_lower; c_upper; c_data(:)];
ceq = [];
end

function plot_hill_gp_tab(tab, x_grid, x_max, m, sf, k_plot, band_label, ...
    y_true, x_train, y_train, noise_sd_true, E_min, E_max, panel_title, ylim_fixed)
ax = axes('Parent', tab, 'Position', [0.09, 0.11, 0.865, 0.815]);
ax.Layer = 'top';
hold(ax, 'on'); grid(ax, 'on');
fill(ax, [x_grid, fliplr(x_grid)], [m + k_plot * sf; flipud(m - k_plot * sf)]', ...
    [0.72, 0.72, 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
plot(ax, x_grid, m, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
plot(ax, x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (Hill)');
plot(ax, x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', sprintf('Data (\\sigma=%.2g)', noise_sd_true));
yline(ax, E_max, 'k:', 'E_{max}', 'Alpha', 0.5);
yline(ax, E_min, 'k:', 'E_{min}', 'Alpha', 0.5);
xlabel(ax, 'Drug Concentration (C)');
ylabel(ax, 'Biological Effect (E)');
title(ax, panel_title, 'Interpreter', 'none');
xlim(ax, [0, x_max]);
ylim(ax, ylim_fixed);
legend(ax, 'Location', 'southeast');
set(ax, 'FontSize', 11);
end
