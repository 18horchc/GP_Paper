% Paper figure: Michaelis-Menten GP with Pensoneault lower-bound constraint.
% Baseline SE-GP vs fmincon fit with mu_f(x_c) - k*sigma_f(x_c) >= 0 at 41 grid points
% % and data-fidelity tube |y - y*(x)| <= epsilon at training locations.
% eta = 2.2%% => k from erfinv; % epsilon = 0.5.
clear; clc; close all;

%% MM parameters
Vmax = 6;
Km   = .15;
mm_static = @(S) (Vmax .* S) ./ (Km + S);

%% Training data ([S] in mM, v_0 in μM/s)
x_max = 2;
noise_frac = 0.1;   % homoscedastic: sigma = noise_frac * max v on [0, x_max]
x_train = [0.0;0.2;0.4;0.6;0.8;2.0]; %crazy behavior
n_train = numel(x_train);

rng(100);
v_true_at_train = mm_static(x_train);
y_domain_max = mm_static(x_max);
noise_sd_true = noise_frac * y_domain_max;
y_train = v_true_at_train + noise_sd_true * randn(size(v_true_at_train));
fprintf('Synthetic data: n=%d on [0, %.1f], homoscedastic noise sigma_n = %.4f (%.0f%% of v(x_max))\n', ...
    n_train, x_max, noise_sd_true, 100 * noise_frac);

x_obs = x_train(:);
y_obs = y_train(:);

%% Ground truth curve
x_grid = linspace(0, x_max, 500);
y_true = mm_static(x_grid);

%% Pensoneault lower-bound constraint grid at X_c
eta = 0.022;   % 2.2% tail probability
k   = -sqrt(2) * erfinv(2 * eta - 1);
n_constraint = 41;
X_c = linspace(0, x_max, n_constraint)';
% epsilon = 0.5;   % data fidelity: |y - y*(x)| <= epsilon

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
ell_ub = 3;   % cap length scale at domain width
sf_bounds = [0.05, 15];
meanfunc = @meanZero;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

x_col = x_train(:);
y_col = y_train(:);

%% Baseline GP (sigma_n fixed at noise_sd_true; optimize ell, sf only)
sn_fixed = log(noise_sd_true);
fprintf('Optimizing baseline (ell, sf; sigma_n fixed at %.4f)...\n', noise_sd_true);
obj_unc = @(hyp_cov) gp_nlml_cov_only(hyp_cov, sn_fixed, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
hyp_cov_unc = minimize(hyp.cov, obj_unc, -100);
hyp_unc = struct('mean', [], 'cov', hyp_cov_unc(:), 'lik', sn_fixed);
nlml_unc = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
theta_unc = hyp_unc.cov(:);

%% Pensoneault-constrained GP (lower bound at 0; sigma_n fixed)
% %% Pensoneault-constrained GP (lower bound at 0 + data fidelity; sigma_n fixed)
hyp_lb = log([ell_bounds_lo; sf_bounds(1)]);
hyp_ub = log([ell_ub; sf_bounds(2)]);
hyp_tpl = hyp_unc;
objfun = @(theta) gp(theta_to_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
nonlcon = @(theta) pens_constraints_lower(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x_col, y_col, X_c, k);
% nonlcon = @(theta) pens_constraints_lower(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
%     x_col, y_col, X_c, k, epsilon);
opts_pens = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, 'Display', 'off', ...
    'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);
nTry = 2000;
nMultistart = 10;
theta_unc_box = min(max(theta_unc, hyp_lb), hyp_ub);

fprintf('\n=== Pensoneault GP (lower bound at 0) ===\n');
% fprintf('\n=== Pensoneault GP (lower bound at 0 + data fidelity, epsilon = %.4g) ===\n', epsilon);
fprintf('eta = %.3g%% | k = %.4f | X_c: %d points | random starts: %d\n', ...
    100 * eta, k, numel(X_c), nTry);

feasible_starts = zeros(2, 0);
best_feas_nlml = inf;
best_feas_theta = nan(2, 1);
rng(42);
for t = 1:nTry
    theta_try = hyp_lb + rand(2, 1) .* (hyp_ub - hyp_lb);
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
theta_opt = nan(2, 1);
nlml_con = nan;
exitflag_con = -99;
nStarts = size(starts_for_fmincon, 2);
for j = 1:nStarts
    theta0_j = starts_for_fmincon(:, j);
    [theta_j, nlml_j, ef_j] = fmincon(objfun, theta0_j, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts_pens);
    if isfinite(nlml_j) && nlml_j < best_nlml
        best_nlml = nlml_j;
        theta_opt = theta_j;
        nlml_con = nlml_j;
        exitflag_con = ef_j;
    end
end

if ~isfinite(best_nlml)
    if nFeas > 0
        theta_opt = best_feas_theta;
    else
        theta_opt = theta_unc_box;
    end
    nlml_con = objfun(theta_opt);
    exitflag_con = -99;
    fprintf('Warning: no successful fmincon run; using fallback theta.\n');
end

hyp_con = theta_to_hyp(theta_opt, hyp_tpl);
[c_final, ~] = nonlcon(theta_opt);
% nC = numel(X_c);
fprintf('Final max(c) = %.6g (feasible if <= 0)\n', max(c_final));
fprintf('  lower max(c) = %.6g\n', max(c_final));
% fprintf('  lower max(c) = %.6g | data max(c) = %.6g\n', ...
%     max(c_final(1:nC)), max(c_final(nC+1:end)));

%% Plot baseline vs lower-bound GP
k_plot = 2;
[~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_unc = fmu_unc(:);
sf_unc = sqrt(max(fs2_unc(:), 0));

[~, ~, fmu_con, fs2_con] = gp(hyp_con, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_con = fmu_con(:);
sf_con = sqrt(max(fs2_con(:), 0));

band_label = sprintf('\\mu_f \\pm %g\\sigma_f (latent)', k_plot);
ylim_shared = [-1, max([y_train(:); Vmax; m_unc + k_plot * sf_unc; m_con + k_plot * sf_con]) * 1.02];

figure('Color', 'w', 'Position', [80, 80, 1100, 520], ...
    'Name', 'Michaelis-Menten GP: lower bound');
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

panels(1) = struct('m', m_unc, 'sf', sf_unc, 'title', 'Baseline GP');
panels(2) = struct('m', m_con, 'sf', sf_con, 'title', 'Lower-bound GP');

for p = 1:2
    nexttile;
    ax = gca;
    ax.Layer = 'top';
    hold on; grid on;
    m = panels(p).m;
    sf = panels(p).sf;
    fill([x_grid, fliplr(x_grid)], [m + k_plot * sf; flipud(m - k_plot * sf)]', ...
        [0.72, 0.72, 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
    plot(x_grid, m, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
    plot(x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
    plot(x_obs, y_obs, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
        'DisplayName', 'Observed data');
    yline(0, 'k:', 'v_0 = 0', 'Alpha', 0.5);
    yline(Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
    xlabel('[S] (mM)');
    ylabel('v_0 (\muM/s)');
    title(panels(p).title, 'Interpreter', 'none');
    xlim([0, x_max]);
    ylim(ylim_shared);
    legend('Location', 'southeast');
end

fprintf('\nBaseline:      ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);
fprintf('Pensoneault:   ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(hyp_con.cov(1)), exp(hyp_con.cov(2)), exp(hyp_con.lik), nlml_con, exitflag_con, max(c_final));

function hyp = theta_to_hyp(theta, hyp_tpl)
hyp = hyp_tpl;
hyp.cov = theta(1:2);
hyp.mean = [];
end

function [c, ceq] = pens_constraints_lower(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k)
% function [c, ceq] = pens_constraints_lower(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
%     x, y, X_c, k, epsilon)
% Pensoneault-style constraints: lower bound on latent f at X_c.
% % Pensoneault-style constraints: lower bound on latent f at X_c; data fidelity tube.
% mu_f - k*sigma_f >= 0  <=>  c <= 0
hyp = theta_to_hyp(theta, hyp_tpl);
% nC = numel(X_c);
% xstar = [X_c(:); x(:)];
% [ymu, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xstar);
% m_xc = fmu(1:nC);
% s_xc = sqrt(max(fs2(1:nC), 0));
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, X_c(:));
m_xc = fmu(:);
s_xc = sqrt(max(fs2(:), 0));
c_lower = k .* s_xc - m_xc;
% y_star = ymu(nC+1:end);
% c_data = abs(y - y_star) - epsilon;
% c = [c_lower; c_data(:)];
c = c_lower;
ceq = [];
end
