% Michaelis-Menten GP: unconstrained vs Pensoneault bound-constrained [0, 6].
clear; clc; close all;

%% MM parameters
Vmax = 6;
Km   = 0.15;
mm_static = @(S) (Vmax .* S) ./ (Km + S);

%% Training data ([S] in mM, v_0 in μM/s)
x_max = 2;
n_train = 7;
noise_frac = 0.05;   % homoscedastic: sigma = noise_frac * max v on [0, x_max]
%x_train = [0.1;0.2;0.4;0.6;0.8;1;1.2;1.4;1.8;2]';%linspace(0, x_max, n_train)';
x_train = [0.01;0.08;0.2;0.3;0.5;0.8;1.8];

rng(100);
v_true_at_train = mm_static(x_train);
y_domain_max = mm_static(x_max);
noise_sd_true = noise_frac * y_domain_max;
y_train = v_true_at_train + noise_sd_true * randn(size(v_true_at_train));
noise_sd_est = noise_sd_true;
fprintf('Synthetic data: n=%d equally spaced on [0, %.1f], homoscedastic noise sigma_n = %.4f (%.0f%% of v(x_max))\n', ...
    n_train, x_max, noise_sd_true, 100 * noise_frac);

%% Derivative observations (Solak et al. 2002): df/dx at selected [S]
% x_deriv = [5;20;35];
% y_deriv = [.1;.1;0.1];
% sn_deriv = 0.2; %max(0.005, 0.5 * noise_sd_true);

%% Ground truth curve
x_grid = linspace(0, x_max, 500);
y_true = mm_static(x_grid);

%% Constraint grid: 41 evenly spaced points on [0, x_max]
% constraint_step = 1.0;
% X_c = (0:constraint_step:x_max)';
n_constraint = 41;
X_c = linspace(0, x_max, n_constraint)';

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

ell0 = std(x_train);
sf0  = std(y_train);
sn0  = max(1e-3, noise_sd_true);
hyp = struct('mean', [], 'cov', log([ell0; sf0]), 'lik', log(sn0));

ell_bounds_lo = 0.02;
ell_ub = 3;
sf_bounds = [0.05, 15];
sn_bounds = [max(1e-4, noise_sd_true / 10), 2];

meanfunc = @meanZero;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

x_col = x_train(:);
y_col = y_train(:);

%% Unconstrained GP (GPML minimize)
fprintf('Optimizing hyperparameters (unconstrained NLML)...\n');
hyp_unc = minimize(hyp, @gp, -100, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
nlml_unc = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);

%% Derivative-observation GP (Solak-style augmented NLML)
% obj_deriv = @(h) gp_seiso_deriv_obs('nlml', h, x_col, y_col, x_deriv, y_deriv, [], sn_deriv);
% fprintf('\nOptimizing hyperparameters (derivative-observation NLML)...\n');
% hyp_deriv = minimize(hyp, obj_deriv, -100);
% nlml_deriv = obj_deriv(hyp_deriv);

%% Probabilistic bound constraints (Pensoneault tails on latent f at X_c)
hyp_tpl = hyp_unc;
eta = 0.022;   % 2.2%
k   = -sqrt(2) * erfinv(2 * eta - 1);
y_min_bound = 0;   % lower tail: m - k*s >= 0
y_max_bound = 6;   % upper tail: m + k*s <= 6
epsilon = 0.5;     % data fidelity: |y - y*(x)| <= epsilon

objfun = @(theta) gp(theta_to_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
nonlcon = @(theta) pens_constraints(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x_col, y_col, X_c, k, y_max_bound, epsilon);
opts = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, 'Display', 'off', ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);

nTry = 5000;
nMultistart = 10;
hyp_lb = log([ell_bounds_lo; sf_bounds(1); sn_bounds(1)]);
hyp_ub = log([ell_ub; sf_bounds(2); sn_bounds(2)]);
theta_unc_box = min(max([hyp_unc.cov(1); hyp_unc.cov(2); hyp_unc.lik(1)], hyp_lb), hyp_ub);

fprintf('\n=== Constrained GP (probabilistic bounds [%.0f, %.0f] + data fidelity) ===\n', y_min_bound, y_max_bound);
fprintf('eta = %.3g%% | k = %.4f | epsilon = %.4g | X_c: %d points | random starts: %d\n', ...
    100 * eta, k, epsilon, numel(X_c), nTry);

feasible_starts = zeros(3, 0);
best_feas_nlml = inf;
best_feas_theta = nan(3, 1);
rng(42);
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
    fprintf('No feasible random start; using projected unconstrained theta.\n');
    starts_for_fmincon = theta_unc_box;
end

best_nlml = inf;
theta_opt = nan(3, 1);
nlml_opt = nan;
exitflag = -99;
nStarts = size(starts_for_fmincon, 2);
for j = 1:nStarts
    theta0_j = starts_for_fmincon(:, j);
    [theta_j, nlml_j, ef_j] = fmincon(objfun, theta0_j, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts);
    if isfinite(nlml_j) && nlml_j < best_nlml
        best_nlml = nlml_j;
        theta_opt = theta_j;
        nlml_opt = nlml_j;
        exitflag = ef_j;
    end
end

if ~isfinite(best_nlml)
    if nFeas > 0
        theta_opt = best_feas_theta;
    else
        theta_opt = theta_unc_box;
    end
    nlml_opt = objfun(theta_opt);
    exitflag = -99;
    fprintf('Warning: no successful fmincon run; using fallback theta.\n');
end

hyp_con = theta_to_hyp(theta_opt, hyp_tpl);
[c_final, ~] = nonlcon(theta_opt);
nC = numel(X_c);
fprintf('Final max(c) = %.6g (feasible if <= 0)\n', max(c_final));
fprintf('  lower max(c) = %.6g | upper max(c) = %.6g | data max(c) = %.6g\n', ...
    max(c_final(1:nC)), max(c_final(nC+1:2*nC)), max(c_final(2*nC+1:end)));

%% Combined GP: Pensoneault bounds + derivative observations (Solak augmented NLML)
% hyp_tpl_comb = hyp_deriv;
% objfun_comb = @(theta) gp_seiso_deriv_obs('nlml', theta_to_hyp(theta, hyp_tpl_comb), ...
%     x_col, y_col, x_deriv, y_deriv, [], sn_deriv);
% nonlcon_comb = @(theta) pens_constraints_deriv(theta, hyp_tpl_comb, ...
%     x_col, y_col, x_deriv, y_deriv, sn_deriv, X_c, k, y_max, epsilon);
% theta_deriv_box = min(max([hyp_deriv.cov(1); hyp_deriv.cov(2); hyp_deriv.lik(1)], hyp_lb), hyp_ub);
%
% fprintf('\n=== Combined GP (Pensoneault bounds + derivative observations) ===\n');
% fprintf('eta = %.3g%% | k = %.4f | epsilon = %.4g | X_c: %d | deriv obs: %d | random starts: %d\n', ...
%     100 * eta, k, epsilon, numel(X_c), numel(x_deriv), nTry);
%
% feasible_starts_comb = zeros(3, 0);
% best_feas_nlml_comb = inf;
% best_feas_theta_comb = nan(3, 1);
% rng(43);
% for t = 1:nTry
%     theta_try = hyp_lb + rand(3, 1) .* (hyp_ub - hyp_lb);
%     [c_try, ~] = nonlcon_comb(theta_try);
%     if max(c_try) <= 0
%         feasible_starts_comb = [feasible_starts_comb, theta_try];
%         nlml_try = objfun_comb(theta_try);
%         if nlml_try < best_feas_nlml_comb
%             best_feas_nlml_comb = nlml_try;
%             best_feas_theta_comb = theta_try;
%         end
%     end
% end
% nFeasComb = size(feasible_starts_comb, 2);
% fprintf('Feasible random starts: %d / %d\n', nFeasComb, nTry);
%
% if nFeasComb > 0
%     nlml_feas_comb = arrayfun(@(j) objfun_comb(feasible_starts_comb(:, j)), 1:nFeasComb);
%     [~, ord_comb] = sort(nlml_feas_comb, 'ascend');
%     starts_comb = feasible_starts_comb(:, ord_comb(1:min(nMultistart, nFeasComb)));
% else
%     fprintf('No feasible random start; using projected deriv-obs theta.\n');
%     starts_comb = theta_deriv_box;
% end
% starts_comb = [theta_deriv_box, starts_comb];
% starts_comb = starts_comb(:, 1:min(nMultistart + 1, size(starts_comb, 2)));
%
% best_nlml_comb = inf;
% theta_comb = nan(3, 1);
% nlml_comb = nan;
% exitflag_comb = -99;
% nStartsComb = size(starts_comb, 2);
% for j = 1:nStartsComb
%     theta0_j = starts_comb(:, j);
%     [theta_j, nlml_j, ef_j] = fmincon(objfun_comb, theta0_j, [], [], [], [], hyp_lb, hyp_ub, nonlcon_comb, opts);
%     if isfinite(nlml_j) && nlml_j < best_nlml_comb
%         best_nlml_comb = nlml_j;
%         theta_comb = theta_j;
%         nlml_comb = nlml_j;
%         exitflag_comb = ef_j;
%     end
% end
%
% if ~isfinite(best_nlml_comb)
%     if nFeasComb > 0
%         theta_comb = best_feas_theta_comb;
%     else
%         theta_comb = theta_deriv_box;
%     end
%     nlml_comb = objfun_comb(theta_comb);
%     exitflag_comb = -99;
%     fprintf('Warning: no successful combined fmincon run; using fallback theta.\n');
% end
%
% hyp_comb = theta_to_hyp(theta_comb, hyp_tpl_comb);
% [c_final_comb, ~] = nonlcon_comb(theta_comb);
% fprintf('Final max(c) = %.6g (feasible if <= 0)\n', max(c_final_comb));
% fprintf('  lower max(c) = %.6g | upper max(c) = %.6g | data max(c) = %.6g\n', ...
%     max(c_final_comb(1:nC)), max(c_final_comb(nC+1:2*nC)), max(c_final_comb(2*nC+1:end)));

%% Plot unconstrained vs bound-constrained GP
k_plot = 2;
[~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_unc = fmu_unc(:);
sf_unc = sqrt(max(fs2_unc(:), 0));

[~, ~, fmu_con, fs2_con] = gp(hyp_con, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_con = fmu_con(:);
sf_con = sqrt(max(fs2_con(:), 0));

% [~, ~, fmu_deriv, fs2_deriv] = gp_seiso_deriv_obs('pred', hyp_deriv, x_col, y_col, ...
%     x_deriv, y_deriv, x_grid(:), sn_deriv);
% m_deriv = fmu_deriv(:);
% sf_deriv = sqrt(max(fs2_deriv(:), 0));
%
% [~, ~, fmu_comb, fs2_comb] = gp_seiso_deriv_obs('pred', hyp_comb, x_col, y_col, ...
%     x_deriv, y_deriv, x_grid(:), sn_deriv);
% m_comb = fmu_comb(:);
% sf_comb = sqrt(max(fs2_comb(:), 0));
%
% [m_deriv_at_xd, s2_deriv_at_xd] = gp_seiso_deriv_obs('deriv', hyp_deriv, x_col, y_col, ...
%     x_deriv, y_deriv, x_deriv, sn_deriv);
% mm_deriv_true = Vmax * Km ./ (Km + x_deriv).^2;
%
% [m_deriv_at_xd_comb, s2_deriv_at_xd_comb] = gp_seiso_deriv_obs('deriv', hyp_comb, x_col, y_col, ...
%     x_deriv, y_deriv, x_deriv, sn_deriv);

band_label = sprintf('\\mu_f \\pm %g\\sigma_f (latent)', k_plot);
ylim_unc = [0, max([y_train(:); y_max_bound; m_unc + k_plot * sf_unc; m_con + k_plot * sf_con]) * 1.02];

fig = figure('Color', 'w', 'Position', [80, 80, 920, 620], ...
    'Name', sprintf('Michaelis-Menten GP (n=%d, %.0f%% noise)', n_train, 100 * noise_frac));
tg = uitabgroup(fig);

tab_unc = uitab(tg, 'Title', 'Unconstrained');
plot_mm_gp_tab(tab_unc, x_grid, x_max, m_unc, sf_unc, k_plot, band_label, ...
    y_true, x_train, y_train, noise_sd_true, Vmax, 'Unconstrained GPML', ylim_unc);

tab_con = uitab(tg, 'Title', 'Bound-constrained');
plot_mm_gp_tab(tab_con, x_grid, x_max, m_con, sf_con, k_plot, band_label, ...
    y_true, x_train, y_train, noise_sd_true, Vmax, ...
    sprintf('Bound-constrained (\\ell=%.2f, NLML=%.2f)', exp(theta_opt(1)), nlml_opt), ylim_unc);

% tab_deriv = uitab(tg, 'Title', 'Deriv-obs');
% plot_mm_gp_tab(tab_deriv, x_grid, x_max, m_deriv, sf_deriv, k_plot, band_label, ...
%     y_true, x_train, y_train, noise_sd_true, Vmax, ...
%     sprintf('Deriv-obs GP (\\ell=%.2f, NLML=%.2f)', exp(hyp_deriv.cov(1)), nlml_deriv), ...
%     ylim_unc, x_deriv);
%
% tab_comb = uitab(tg, 'Title', 'Combined');
% plot_mm_gp_tab(tab_comb, x_grid, x_max, m_comb, sf_comb, k_plot, band_label, ...
%     y_true, x_train, y_train, noise_sd_true, Vmax, ...
%     sprintf('Combined bounds+deriv (\\ell=%.2f, NLML=%.2f)', exp(theta_comb(1)), nlml_comb), ...
%     ylim_unc, x_deriv);

fprintf('\nUnconstrained: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);
fprintf('Constrained:   ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d\n', ...
    exp(hyp_con.cov(1)), exp(hyp_con.cov(2)), exp(hyp_con.lik), nlml_opt, exitflag);
% fprintf('Deriv-obs:     ell=%.4f, sf=%.4f, sn=%.4f, sn_deriv=%.4f | NLML=%.4f\n', ...
%     exp(hyp_deriv.cov(1)), exp(hyp_deriv.cov(2)), exp(hyp_deriv.lik), sn_deriv, nlml_deriv);
% fprintf('Combined:      ell=%.4f, sf=%.4f, sn=%.4f, sn_deriv=%.4f | NLML=%.4f | exitflag=%d\n', ...
%     exp(hyp_comb.cov(1)), exp(hyp_comb.cov(2)), exp(hyp_comb.lik), sn_deriv, nlml_comb, exitflag_comb);
% fprintf('\nPosterior f'' at derivative observation points — deriv-obs GP (sn_deriv = %.3g):\n', sn_deriv);
% fprintf('  [S]    target    post mean    post sd    MM analytic\n');
% for j = 1:numel(x_deriv)
%     fprintf('  %4.2f   %6.3f    %8.4f    %8.4f    %8.4f\n', ...
%         x_deriv(j), y_deriv(j), m_deriv_at_xd(j), sqrt(s2_deriv_at_xd(j)), mm_deriv_true(j));
% end
% fprintf('\nPosterior f'' at derivative observation points — combined GP (sn_deriv = %.3g):\n', sn_deriv);
% fprintf('  [S]    target    post mean    post sd    MM analytic\n');
% for j = 1:numel(x_deriv)
%     fprintf('  %4.2f   %6.3f    %8.4f    %8.4f    %8.4f\n', ...
%         x_deriv(j), y_deriv(j), m_deriv_at_xd_comb(j), sqrt(s2_deriv_at_xd_comb(j)), mm_deriv_true(j));
% end

function hyp = theta_to_hyp(theta, hyp_tpl)
hyp = hyp_tpl;
hyp.cov  = theta(1:2);
hyp.lik  = theta(3);
hyp.mean = [];
end

function [c, ceq] = pens_constraints(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k, y_max, epsilon)
% Lower/upper Pensoneault tails on latent f at X_c; data tube |y - y*(x)| <= epsilon.
hyp = theta_to_hyp(theta, hyp_tpl);
nC = numel(X_c);
xstar = [X_c(:); x(:)];
[ymu, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xstar);
m_xc = fmu(1:nC);
s_xc = sqrt(max(fs2(1:nC), 0));
c_lower = k * s_xc - m_xc;              % m - k*s >= 0
c_upper = m_xc + k * s_xc - y_max;      % m + k*s <= y_max
y_star = ymu(nC+1:end);
c_data = abs(y - y_star) - epsilon;     % |y - y*(x)| <= epsilon
c = [c_lower; c_upper; c_data(:)];
ceq = [];
end

% function [c, ceq] = pens_constraints_deriv(theta, hyp_tpl, x, y, x_d, y_d, sn_deriv, ...
%     X_c, k, y_max, epsilon)
% % Pensoneault tails on latent f at X_c; data tube — augmented deriv-obs posterior.
% hyp = theta_to_hyp(theta, hyp_tpl);
% nC = numel(X_c);
% xstar = [X_c(:); x(:)];
% [ymu, ~, fmu, fs2] = gp_seiso_deriv_obs('pred', hyp, x, y, x_d, y_d, xstar, sn_deriv);
% ymu = ymu(:); fmu = fmu(:);
% m_xc = fmu(1:nC);
% s_xc = sqrt(max(fs2(1:nC), 0));
% c_lower = k * s_xc - m_xc;
% c_upper = m_xc + k * s_xc - y_max;
% y_star = ymu(nC+1:end);
% c_data = abs(y - y_star) - epsilon;
% c = [c_lower; c_upper; c_data(:)];
% ceq = [];
% end

function plot_mm_gp_tab(tab, x_grid, x_max, m, sf, k_plot, band_label, ...
    y_true, x_train, y_train, noise_sd_true, Vmax, panel_title, ylim_fixed, x_deriv)
if nargin < 15
    x_deriv = [];
end
ax = axes('Parent', tab, 'Position', [0.09, 0.11, 0.865, 0.815]);
ax.Layer = 'top';
hold(ax, 'on'); grid(ax, 'on');
fill(ax, [x_grid, fliplr(x_grid)], [m + k_plot * sf; flipud(m - k_plot * sf)]', ...
    [0.72, 0.72, 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
plot(ax, x_grid, m, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
plot(ax, x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
plot(ax, x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', sprintf('Data (\\sigma=%.2g)', noise_sd_true));
yline(ax, Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
xlabel(ax, '[S] (mM)'); ylabel(ax, 'v_0 (\muM/s)');
title(ax, panel_title, 'Interpreter', 'none');
xlim(ax, [0, x_max]);
ylim(ax, ylim_fixed);
% if ~isempty(x_deriv)
%     x_mark = x_deriv(:);
%     x_mark = x_mark(x_mark >= 0 & x_mark <= x_max);
%     if ~isempty(x_mark)
%         y_axis = ax.YLim(1);
%         h_deriv = plot(ax, x_mark, repmat(y_axis, numel(x_mark), 1), '^', ...
%             'MarkerSize', 9, 'LineWidth', 0.8, ...
%             'MarkerFaceColor', [0.55, 0.25, 0.65], 'MarkerEdgeColor', 'k', ...
%             'Clipping', 'off', 'DisplayName', 'Derivative obs locations');
%         uistack(h_deriv, 'top');
%     end
% end
legend(ax, 'Location', 'southeast');
end
