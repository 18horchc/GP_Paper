% Michaelis-Menten GP: unconstrained NLML vs bound constraints vs derivative observations.
clear; clc; close all;

%% MM parameters
Vmax = 6;
Km   = 0.15;
mm_static = @(S) (Vmax .* S) ./ (Km + S);

%% Training data ([S] in mM, v_0 in μM/s)
x_max = 2;
noise_frac = 0.05;   % homoscedastic: sigma = noise_frac * max v on [0, x_max]
x_train = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.8]';

rng(100);
v_true = mm_static(x_train);
y_domain_max = mm_static(x_max);
noise_sd_true = noise_frac * y_domain_max;
y_train = v_true + noise_sd_true * randn(size(v_true));
n_train = numel(x_train);

%% Derivative observations (Solak et al. 2002): df/dx at selected [S]
x_deriv = [1; 1.5; 2];
y_deriv = 0.3 * ones(3, 1);
sn_deriv = 0.02;   % derivative observation noise (distinct from function noise sigma_n)

%% Ground truth curve
x_grid = linspace(0, x_max, 500);
y_true = mm_static(x_grid);

%% Constraint grid: 41 evenly spaced points on [0, x_max]
constraint_step = 0.05;
X_c = (0:constraint_step:x_max)';

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
sn_bounds = [1e-4, 2];

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
obj_deriv = @(h) gp_seiso_deriv_obs('nlml', h, x_col, y_col, x_deriv, y_deriv, [], sn_deriv);
fprintf('\nOptimizing hyperparameters (derivative-observation NLML)...\n');
hyp_deriv = minimize(hyp, obj_deriv, -100);
nlml_deriv = obj_deriv(hyp_deriv);

%% Probabilistic bound constraints (Pensoneault tails on latent f at X_c)
hyp_tpl = hyp_unc;
eta = 0.022;   % 2.2%
k   = -sqrt(2) * erfinv(2 * eta - 1);
y_max = Vmax;
epsilon = 0.5;

objfun = @(theta) gp(theta_to_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
nonlcon = @(theta) pens_constraints(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x_col, y_col, X_c, k, y_max, epsilon);
opts = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, 'Display', 'off', ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);

nTry = 5000;
nMultistart = 10;
hyp_lb = log([ell_bounds_lo; sf_bounds(1); sn_bounds(1)]);
hyp_ub = log([ell_ub; sf_bounds(2); sn_bounds(2)]);
theta_unc_box = min(max([hyp_unc.cov(1); hyp_unc.cov(2); hyp_unc.lik(1)], hyp_lb), hyp_ub);

fprintf('\n=== Constrained GP (probabilistic bounds + data fidelity) ===\n');
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

%% Plot unconstrained vs bound-constrained vs derivative-observation GP
k_plot = 2;
[~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_unc = fmu_unc(:);
sf_unc = sqrt(max(fs2_unc(:), 0));

[~, ~, fmu_con, fs2_con] = gp(hyp_con, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_con = fmu_con(:);
sf_con = sqrt(max(fs2_con(:), 0));

[~, ~, fmu_deriv, fs2_deriv] = gp_seiso_deriv_obs('pred', hyp_deriv, x_col, y_col, ...
    x_deriv, y_deriv, x_grid(:), sn_deriv);
m_deriv = fmu_deriv(:);
sf_deriv = sqrt(max(fs2_deriv(:), 0));

[m_deriv_at_xd, s2_deriv_at_xd] = gp_seiso_deriv_obs('deriv', hyp_deriv, x_col, y_col, ...
    x_deriv, y_deriv, x_deriv, sn_deriv);
mm_deriv_true = Vmax * Km ./ (Km + x_deriv).^2;

band_label = sprintf('\\mu_f \\pm %g\\sigma_f (latent)', k_plot);
tlo = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tlo, sprintf('Michaelis-Menten GP (n=%d, %.0f%% noise): unconstrained vs bounds vs deriv-obs', ...
    n_train, 100 * noise_frac), 'Interpreter', 'none');

nexttile;
hold on; grid on;
fill([x_grid, fliplr(x_grid)], [m_unc + k_plot * sf_unc; flipud(m_unc - k_plot * sf_unc)]', ...
    [0.75, 0.75, 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
plot(x_grid, m_unc, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
plot(x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
plot(x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', sprintf('Data (\\sigma=%.2g)', noise_sd_true));
yline(Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
xlabel('[S] (mM)'); ylabel('v_0 (\muM/s)');
title('Unconstrained GPML');
legend('Location', 'southeast');
xlim([0, x_max]);
ylim_unc = ylim;

nexttile;
hold on; grid on;
fill([x_grid, fliplr(x_grid)], [m_con + k_plot * sf_con; flipud(m_con - k_plot * sf_con)]', ...
    [0.55, 0.72, 0.55], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
plot(x_grid, m_con, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
plot(x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
plot(x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', sprintf('Data (\\sigma=%.2g)', noise_sd_true));
yline(Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
xlabel('[S] (mM)'); ylabel('v_0 (\muM/s)');
title(sprintf('Bound-constrained (\\ell=%.2f, NLML=%.2f)', exp(theta_opt(1)), nlml_opt), ...
    'Interpreter', 'none');
legend('Location', 'southeast');
ylim(ylim_unc);
xlim([0, x_max]);

nexttile;
hold on; grid on;
fill([x_grid, fliplr(x_grid)], [m_deriv + k_plot * sf_deriv; flipud(m_deriv - k_plot * sf_deriv)]', ...
    [0.72, 0.62, 0.82], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
plot(x_grid, m_deriv, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
plot(x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
plot(x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', sprintf('Data (\\sigma=%.2g)', noise_sd_true));
for j = 1:numel(x_deriv)
    xline(x_deriv(j), ':', 'Color', [0.45, 0.25, 0.55], 'LineWidth', 1, ...
        'HandleVisibility', 'off');
end
yline(Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
xlabel('[S] (mM)'); ylabel('v_0 (\muM/s)');
title(sprintf('Deriv-obs GP (\\ell=%.2f, NLML=%.2f)', exp(hyp_deriv.cov(1)), nlml_deriv), ...
    'Interpreter', 'none');
legend('Location', 'southeast');
ylim(ylim_unc);
xlim([0, x_max]);

fprintf('\nUnconstrained: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);
fprintf('Constrained:   ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d\n', ...
    exp(hyp_con.cov(1)), exp(hyp_con.cov(2)), exp(hyp_con.lik), nlml_opt, exitflag);
fprintf('Deriv-obs:     ell=%.4f, sf=%.4f, sn=%.4f, sn_deriv=%.4f | NLML=%.4f\n', ...
    exp(hyp_deriv.cov(1)), exp(hyp_deriv.cov(2)), exp(hyp_deriv.lik), sn_deriv, nlml_deriv);
fprintf('\nPosterior f'' at derivative constraint points (target = %.3g, sn_deriv = %.3g):\n', ...
    y_deriv(1), sn_deriv);
fprintf('  [S]    target    post mean    post sd    MM analytic\n');
for j = 1:numel(x_deriv)
    fprintf('  %4.2f   %6.3f    %8.4f    %8.4f    %8.4f\n', ...
        x_deriv(j), y_deriv(j), m_deriv_at_xd(j), sqrt(s2_deriv_at_xd(j)), mm_deriv_true(j));
end

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
