% Paper figure: Michaelis-Menten GP comparison (baseline, bound, bound+deriv).
% Tab 1: unconstrained | bound only. Tab 2: bound+deriv @ S=1.2 | bound+deriv @ S=1.0,1.2.
% Requires gp_seiso_deriv_obs.m on the path (problems/).
clear; clc; close all;

%% MM parameters
Vmax = 6;
Km   = .15;
mm_static = @(S) (Vmax .* S) ./ (Km + S);

%% Training data ([S] in mM, v_0 in μM/s)
x_max = 2;
noise_frac = 0.05;
x_train = [0.0; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 1.8; 1.9; 2.0];

rng(100);
v_true_at_train = mm_static(x_train);
y_domain_max = mm_static(x_max);
noise_sd_true = noise_frac * y_domain_max;
y_train = v_true_at_train + noise_sd_true * randn(size(v_true_at_train));
fprintf('Synthetic data: n=%d on [0, %.1f], homoscedastic noise sigma_n = %.4f (%.0f%% of v(x_max))\n', ...
    numel(x_train), x_max, noise_sd_true, 100 * noise_frac);

x_obs = x_train(:);
y_obs = y_train(:);

%% Solak derivative-observation settings (soft)
y_deriv_target = 0.3;
sn_deriv = 0.2;
x_deriv_12 = 1.2;
x_deriv_10_12 = [1.0; 1.2];

%% Ground truth curve
x_grid = linspace(0, x_max, 500);
y_true = mm_static(x_grid);

%% Pensoneault upper-bound constraint grid at X_c
eta = 0.022;
k   = -sqrt(2) * erfinv(2 * eta - 1);
n_constraint = 41;
X_c = linspace(0, x_max, n_constraint)';
y_max = Vmax;
epsilon = 0.5;

%% GPML setup
gpml_folder_name = "C:\Users\chorc\OneDrive\Documents\Stroke Research\Gaussian Processes\Old\gpml-matlab-master\gpml-matlab-master";
if ~exist('gp', 'file')
    if exist(gpml_folder_name, 'dir')
        addpath(genpath(gpml_folder_name));
    else
        error('GPML toolbox missing at %s', gpml_folder_name);
    end
end
try
    startup;
catch
end
addpath(fileparts(fileparts(mfilename('fullpath'))));  % problems/

ell0 = std(x_train);
sf0  = std(y_train);
sn0  = max(1e-3, noise_sd_true);
hyp = struct('mean', [], 'cov', log([ell0; sf0]), 'lik', log(sn0));

ell_bounds_lo = 0.05;
ell_ub = 3;
sf_bounds = [0.05, 15];
sn_bounds = [noise_sd_true, 2];

meanfunc = @meanZero;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

x_col = x_train(:);
y_col = y_train(:);

hyp_lb = log([ell_bounds_lo; sf_bounds(1); sn_bounds(1)]);
hyp_ub = log([ell_ub; sf_bounds(2); sn_bounds(2)]);
to_hyp = @(theta) struct('mean', [], 'cov', theta(1:2), 'lik', theta(3));
opts_unc = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'off');
opts_pens = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, 'Display', 'off', ...
    'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);
nTry = 2000;
nMultistart = 10;

%% 1. Unconstrained GP (bounded NLML: sigma_n floor, ell cap)
obj_unc = @(theta) gp(to_hyp(theta), inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
theta0 = min(max([hyp.cov(:); hyp.lik], hyp_lb), hyp_ub);
fprintf('=== Unconstrained GP ===\n');
fprintf('Optimizing hyperparameters (bounded NLML: ell in [%.2g, %.2g], sn >= %.4f)...\n', ...
    ell_bounds_lo, ell_ub, sn_bounds(1));
[theta_unc, nlml_unc, exitflag_unc] = fmincon(obj_unc, theta0, [], [], [], [], hyp_lb, hyp_ub, [], opts_unc);
hyp_unc = to_hyp(theta_unc);

%% 2. Pensoneault bound only (upper bound + data fidelity; no deriv obs)
fprintf('\n=== Pensoneault bound only ===\n');
hyp_tpl = hyp_unc;
obj_bound = @(theta) gp(theta_to_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
nonlcon_bound = @(theta) pens_constraints_upper(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x_col, y_col, X_c, k, y_max, epsilon);
[hyp_bound, nlml_bound, exitflag_bound, c_bound] = fit_fmincon_multistart( ...
    obj_bound, nonlcon_bound, theta_unc, hyp_lb, hyp_ub, opts_pens, nTry, nMultistart, 42);
nC = numel(X_c);
fprintf('Final max(c) = %.6g | upper max(c) = %.6g | data max(c) = %.6g\n', ...
    max(c_bound), max(c_bound(1:nC)), max(c_bound(nC+1:end)));

%% 3. Upper bound + deriv obs at S = 1.2 (Solak NLML + Pensoneault fmincon)
fprintf('\n=== Bound + deriv (S = 1.2) ===\n');
y_deriv_12 = y_deriv_target;
obj_d12 = @(theta) gp_seiso_deriv_obs('nlml', theta_to_hyp(theta, hyp_tpl), ...
    x_col, y_col, x_deriv_12, y_deriv_12, [], sn_deriv);
nonlcon_d12 = @(theta) pens_constraints_upper_deriv(theta, hyp_tpl, ...
    x_col, y_col, x_deriv_12, y_deriv_12, sn_deriv, X_c, k, y_max, epsilon);
[hyp_d12, nlml_d12, exitflag_d12, c_d12] = fit_fmincon_multistart( ...
    obj_d12, nonlcon_d12, theta_unc, hyp_lb, hyp_ub, opts_pens, nTry, nMultistart, 43);
fprintf('Final max(c) = %.6g | upper max(c) = %.6g | data max(c) = %.6g\n', ...
    max(c_d12), max(c_d12(1:nC)), max(c_d12(nC+1:end)));

%% 4. Upper bound + deriv obs at S = 1.0 and 1.2
fprintf('\n=== Bound + deriv (S = 1.0, 1.2) ===\n');
y_deriv_10_12 = y_deriv_target * ones(numel(x_deriv_10_12), 1);
obj_d1012 = @(theta) gp_seiso_deriv_obs('nlml', theta_to_hyp(theta, hyp_tpl), ...
    x_col, y_col, x_deriv_10_12, y_deriv_10_12, [], sn_deriv);
nonlcon_d1012 = @(theta) pens_constraints_upper_deriv(theta, hyp_tpl, ...
    x_col, y_col, x_deriv_10_12, y_deriv_10_12, sn_deriv, X_c, k, y_max, epsilon);
[hyp_d1012, nlml_d1012, exitflag_d1012, c_d1012] = fit_fmincon_multistart( ...
    obj_d1012, nonlcon_d1012, theta_unc, hyp_lb, hyp_ub, opts_pens, nTry, nMultistart, 44);
fprintf('Final max(c) = %.6g | upper max(c) = %.6g | data max(c) = %.6g\n', ...
    max(c_d1012), max(c_d1012(1:nC)), max(c_d1012(nC+1:end)));

%% Predict on grid for all four models
k_plot = 2;
[~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_unc = fmu_unc(:);
sf_unc = sqrt(max(fs2_unc(:), 0));

[~, ~, fmu_bound, fs2_bound] = gp(hyp_bound, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_bound = fmu_bound(:);
sf_bound = sqrt(max(fs2_bound(:), 0));

[~, ~, fmu_d12, fs2_d12] = gp_seiso_deriv_obs('pred', hyp_d12, x_col, y_col, ...
    x_deriv_12, y_deriv_12, x_grid(:), sn_deriv);
m_d12 = fmu_d12(:);
sf_d12 = sqrt(max(fs2_d12(:), 0));

[~, ~, fmu_d1012, fs2_d1012] = gp_seiso_deriv_obs('pred', hyp_d1012, x_col, y_col, ...
    x_deriv_10_12, y_deriv_10_12, x_grid(:), sn_deriv);
m_d1012 = fmu_d1012(:);
sf_d1012 = sqrt(max(fs2_d1012(:), 0));

band_label = sprintf('\\mu_f \\pm %g\\sigma_f (latent)', k_plot);
ylim_all = [0, max([y_train(:); Vmax; ...
    m_unc + k_plot * sf_unc; m_bound + k_plot * sf_bound; ...
    m_d12 + k_plot * sf_d12; m_d1012 + k_plot * sf_d1012]) * 1.02];

fig = figure('Color', 'w', 'Position', [80, 80, 1100, 640], ...
    'Name', 'Michaelis-Menten GP: bound + derivative comparison');
tg = uitabgroup(fig);

tab_gp = uitab(tg, 'Title', 'Baseline & bound');
panels_gp(1) = struct('m', m_unc, 'sf', sf_unc, 'title', 'Unconstrained GP', 'x_deriv', []);
panels_gp(2) = struct('m', m_bound, 'sf', sf_bound, 'title', 'Bound only', 'x_deriv', []);
plot_tiled_gp_panels(tab_gp, panels_gp, x_grid, x_max, k_plot, band_label, ...
    y_true, x_obs, y_obs, Vmax, ylim_all);

tab_deriv = uitab(tg, 'Title', 'Bound & deriv');
panels_d = struct('m', {m_d12, m_d1012}, 'sf', {sf_d12, sf_d1012}, ...
    'title', {sprintf('Bound + deriv (S=1.2, f''=%.2g)', y_deriv_target), ...
    sprintf('Bound + deriv (S=1.0, 1.2, f''=%.2g)', y_deriv_target)}, ...
    'x_deriv', {x_deriv_12(:), x_deriv_10_12(:)});
plot_tiled_gp_panels(tab_deriv, panels_d, x_grid, x_max, k_plot, band_label, ...
    y_true, x_obs, y_obs, Vmax, ylim_all);

fprintf('\nUnconstrained:  ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc, exitflag_unc);
fprintf('Bound only:     ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(hyp_bound.cov(1)), exp(hyp_bound.cov(2)), exp(hyp_bound.lik), nlml_bound, exitflag_bound, max(c_bound));
fprintf('Bound+deriv S=1.2:  ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(hyp_d12.cov(1)), exp(hyp_d12.cov(2)), exp(hyp_d12.lik), nlml_d12, exitflag_d12, max(c_d12));
fprintf('Bound+deriv S=1,1.2: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(hyp_d1012.cov(1)), exp(hyp_d1012.cov(2)), exp(hyp_d1012.lik), nlml_d1012, exitflag_d1012, max(c_d1012));

function [hyp_fit, nlml_fit, exitflag_fit, c_final] = fit_fmincon_multistart( ...
    objfun, nonlcon, theta_unc, hyp_lb, hyp_ub, opts, nTry, nMultistart, rng_seed)
theta_unc_box = min(max(theta_unc, hyp_lb), hyp_ub);
fprintf('eta multistart: %d random starts\n', nTry);
feasible_starts = zeros(3, 0);
best_feas_nlml = inf;
best_feas_theta = nan(3, 1);
rng(rng_seed);
for t = 1:nTry
    theta_try = hyp_lb + rand(3, 1) .* (hyp_ub - hyp_lb);
    [c_try, ~] = nonlcon(theta_try);
    if max(c_try) <= 0
        feasible_starts = [feasible_starts, theta_try];
        nlml_try = objfun(theta_try);
        if nlml_try < best_feas_nlml
            best_feas_nlml = nlml_try;
            best_feas_theta = theta_try;
        end
    end
end
nFeas = size(feasible_starts, 2);
fprintf('Feasible random starts: %d / %d\n', nFeas, nTry);
if nFeas > 0
    nlml_feas = arrayfun(@(j) objfun(feasible_starts(:, j)), 1:nFeas);
    [~, ord] = sort(nlml_feas, 'ascend');
    starts_for_fmincon = feasible_starts(:, ord(1:min(nMultistart, nFeas)));
else
    fprintf('No feasible random start; using projected baseline theta.\n');
    starts_for_fmincon = theta_unc_box;
end
starts_for_fmincon = [theta_unc_box, starts_for_fmincon];
starts_for_fmincon = starts_for_fmincon(:, 1:min(nMultistart + 1, size(starts_for_fmincon, 2)));
best_nlml = inf;
theta_opt = nan(3, 1);
nlml_fit = nan;
exitflag_fit = -99;
nStarts = size(starts_for_fmincon, 2);
for j = 1:nStarts
    theta0_j = starts_for_fmincon(:, j);
    [theta_j, nlml_j, ef_j] = fmincon(objfun, theta0_j, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts);
    if isfinite(nlml_j) && nlml_j < best_nlml
        best_nlml = nlml_j;
        theta_opt = theta_j;
        nlml_fit = nlml_j;
        exitflag_fit = ef_j;
    end
end
if ~isfinite(best_nlml)
    if nFeas > 0
        theta_opt = best_feas_theta;
    else
        theta_opt = theta_unc_box;
    end
    nlml_fit = objfun(theta_opt);
    exitflag_fit = -99;
    fprintf('Warning: no successful fmincon run; using fallback theta.\n');
end
hyp_fit = struct('mean', [], 'cov', theta_opt(1:2), 'lik', theta_opt(3));
[c_final, ~] = nonlcon(theta_opt);
end

function hyp = theta_to_hyp(theta, hyp_tpl)
hyp = hyp_tpl;
hyp.cov = theta(1:2);
hyp.lik = theta(3);
hyp.mean = [];
end

function [c, ceq] = pens_constraints_upper(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k, y_max, epsilon)
hyp = theta_to_hyp(theta, hyp_tpl);
nC = numel(X_c);
xstar = [X_c(:); x(:)];
[ymu, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xstar);
m_xc = fmu(1:nC);
s_xc = sqrt(max(fs2(1:nC), 0));
c_upper = m_xc + k .* s_xc - y_max;
y_star = ymu(nC+1:end);
c_data = abs(y - y_star) - epsilon;
c = [c_upper; c_data(:)];
ceq = [];
end

function [c, ceq] = pens_constraints_upper_deriv(theta, hyp_tpl, x, y, x_d, y_d, sn_deriv, ...
    X_c, k, y_max, epsilon)
% Upper bound + data fidelity on Solak augmented posterior (function + deriv obs).
hyp = theta_to_hyp(theta, hyp_tpl);
nC = numel(X_c);
xstar = [X_c(:); x(:)];
[ymu, ~, fmu, fs2] = gp_seiso_deriv_obs('pred', hyp, x, y, x_d, y_d, xstar, sn_deriv);
m_xc = fmu(1:nC);
s_xc = sqrt(max(fs2(1:nC), 0));
c_upper = m_xc + k .* s_xc - y_max;
y_star = ymu(nC+1:end);
c_data = abs(y - y_star) - epsilon;
c = [c_upper(:); c_data(:)];
ceq = [];
end

function plot_tiled_gp_panels(tab, panels, x_grid, x_max, k_plot, band_label, ...
    y_true, x_train, y_train, Vmax, ylim_fixed)
tl = tiledlayout(tab, 1, numel(panels), 'Padding', 'compact', 'TileSpacing', 'compact');
for p = 1:numel(panels)
    ax = nexttile(tl);
    ax.Layer = 'top';
    hold(ax, 'on');
    grid(ax, 'on');
    m = panels(p).m;
    sf = panels(p).sf;
    fill(ax, [x_grid, fliplr(x_grid)], [m + k_plot * sf; flipud(m - k_plot * sf)]', ...
        [0.72, 0.72, 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
    plot(ax, x_grid, m, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
    plot(ax, x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
    plot(ax, x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
        'DisplayName', 'Observed data');
    yline(ax, Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
    xlim(ax, [0, x_max]);
    ylim(ax, ylim_fixed);
    x_deriv = panels(p).x_deriv(:);
    x_mark = x_deriv(isfinite(x_deriv) & x_deriv >= 0 & x_deriv <= x_max);
    if ~isempty(x_mark)
        y_mark = ylim_fixed(1);   % on x-axis (v_0 = 0)
        h_deriv = plot(ax, x_mark, repmat(y_mark, numel(x_mark), 1), '^', ...
            'LineStyle', 'none', 'MarkerSize', 9, 'LineWidth', 0.8, ...
            'MarkerFaceColor', [0.55, 0.25, 0.65], 'MarkerEdgeColor', 'k', ...
            'Clipping', 'off', 'DisplayName', 'Solak deriv obs locations');
        uistack(h_deriv, 'top');
    end
    xlabel(ax, '[S] (mM)');
    ylabel(ax, 'v_0 (\muM/s)');
    title(ax, panels(p).title, 'Interpreter', 'none');
    legend(ax, 'Location', 'southeast');
end
end
