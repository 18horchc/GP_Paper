% Paper figure: Michaelis-Menten GP with Pensoneault positive-derivative bound.
% Baseline SE-GP vs fmincon fit with:
%   mu_f'(x_c) - k*sigma_f'(x_c) >= 0
% at n_constraint points.
% Data-fidelity tube currently disabled (code retained, commented).
% Uses analytic NLML / constraint gradients for fmincon.
% (Eta sweep and n_constraint sweep retained below, commented out.)
clear; clc; close all;

%% MM parameters
Vmax = 6;
Km   = .15;
mm_static = @(S) (Vmax .* S) ./ (Km + S);

%% Training data ([S] in mM, v_0 in μM/s)
x_max = 2;
noise_frac = 0.1;   % homoscedastic: sigma = noise_frac * max v on [0, x_max]
x_train = [0.0; 0.2; 0.4; 0.6; 0.8; 2.0];
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

%% Pensoneault constraint settings
eta = 0.05;
k   = -sqrt(2) * erfinv(2 * eta - 1);
epsilon = 0.5;   % data fidelity (currently disabled in constraints)
n_constraint = 11;
X_c = linspace(0.5, 2, n_constraint)';
% --- eta sweep (commented out; do not delete) ---
% eta_grid = 0.022:0.01:0.12;   % also tried 0.1:0.05:0.45
% --- n_constraint sweep (commented out; do not delete) ---
% n_constraint_grid = 20:24;

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

%% Shared fmincon / multistart settings
hyp_lb = log([ell_bounds_lo; sf_bounds(1)]);
hyp_ub = log([ell_ub; sf_bounds(2)]);
hyp_tpl = hyp_unc;
objfun = @(theta) obj_cov_grad(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
opts_pens = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, 'Display', 'off', ...
    'SpecifyObjectiveGradient', true, ...
    'SpecifyConstraintGradient', true, ...
    'ConstraintTolerance', 1e-2, 'OptimalityTolerance', 1e-2, ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 10000);
nTry = 2000;
nMultistart = 10;
theta_unc_box = min(max(theta_unc, hyp_lb), hyp_ub);

%% Pensoneault-constrained GP (positive derivative; n_constraint fixed)
nC = numel(X_c);
nonlcon = @(theta) pens_constraints_pos_deriv(theta, hyp_tpl, x_col, y_col, X_c, k, epsilon);

fprintf('\n=== Pensoneault GP (positive derivative; data fidelity off) ===\n');
fprintf('eta = %.4g (%.3g%%) | k = %.4f | n_constraint = %d | random starts: %d\n', ...
    eta, 100 * eta, k, n_constraint, nTry);
fprintf('Baseline: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);

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

[c_final, ~] = nonlcon(theta_opt);
deriv_max_c = max(c_final(1:nC));
% data_max_c  = max(c_final(nC+1:end));  % data fidelity off
fprintf('NLML=%.4f | exitflag=%d | deriv max(c)=%.6g\n', ...
    nlml_con, exitflag_con, deriv_max_c);
fprintf('Pensoneault: ell=%.4f, sf=%.4f, sn=%.4f\n', ...
    exp(theta_opt(1)), exp(theta_opt(2)), exp(hyp_tpl.lik));

%% Plot baseline vs constrained GP
hyp_con = theta_to_hyp(theta_opt, hyp_tpl);
k_plot = 2;
[~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_unc = fmu_unc(:);
sf_unc = sqrt(max(fs2_unc(:), 0));

[~, ~, fmu_con, fs2_con] = gp(hyp_con, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_con = fmu_con(:);
sf_con = sqrt(max(fs2_con(:), 0));

band_label = sprintf('\\mu_f \\pm %g\\sigma_f (latent)', k_plot);
ylim_shared = [-0.5, max([y_train(:); Vmax; m_unc + k_plot * sf_unc; m_con + k_plot * sf_con]) * 1.02];

figure('Color', 'w', 'Position', [100, 100, 1100, 520], ...
    'Name', 'Michaelis-Menten GP: positive derivative');
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

panels(1) = struct('m', m_unc, 'sf', sf_unc, 'title', 'Baseline GP');
panels(2) = struct('m', m_con, 'sf', sf_con, ...
    'title', sprintf('Positive derivative GP (\\eta=%.3g, n_c=%d)', eta, n_constraint));

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
    title(panels(p).title, 'Interpreter', 'tex');
    xlim([0, x_max]);
    ylim(ylim_shared);
    legend('Location', 'southeast');
end

% =============================================================================
% ETA SWEEP (commented out; do not delete)
% =============================================================================
% %% Pensoneault-constrained GP eta sweep (positive derivative; data fidelity off)
% nC = numel(X_c);
% n_eta = numel(eta_grid);
%
% sweep_deriv_max_c = nan(n_eta, 1);
% sweep_data_max_c  = nan(n_eta, 1);   % unused while data fidelity off
% sweep_nlml        = nan(n_eta, 1);
% sweep_exitflag    = nan(n_eta, 1);
% theta_opt_last = theta_unc_box;
% eta_last = eta_grid(end);
%
% fprintf('\n=== Eta sweep (positive derivative; data fidelity off) ===\n');
% fprintf('n_constraint = %d | X_c: %s | eta grid: %s | random starts: %d\n', ...
%     n_constraint, mat2str(X_c(:)'), mat2str(eta_grid), nTry);
% fprintf('Baseline: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
%     exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);
%
% for ie = 1:n_eta
%     eta = eta_grid(ie);
%     k = -sqrt(2) * erfinv(2 * eta - 1);
%     nonlcon = @(theta) pens_constraints_pos_deriv(theta, hyp_tpl, x_col, y_col, X_c, k, epsilon);
%
%     fprintf('\n--- eta = %.4g (%.3g%%), k = %.4f ---\n', eta, 100 * eta, k);
%
%     feasible_starts = zeros(2, 0);
%     best_feas_nlml = inf;
%     best_feas_theta = nan(2, 1);
%     rng(42);
%     for t = 1:nTry
%         theta_try = hyp_lb + rand(2, 1) .* (hyp_ub - hyp_lb);
%         [c_try, ~] = nonlcon(theta_try);
%         if max(c_try) <= 0
%             feasible_starts = [feasible_starts, theta_try];
%             nlml_try = objfun(theta_try);
%             if nlml_try < best_feas_nlml
%                 best_feas_nlml = nlml_try;
%                 best_feas_theta = theta_try;
%             end
%         end
%     end
%     nFeas = size(feasible_starts, 2);
%     fprintf('Feasible random starts: %d / %d\n', nFeas, nTry);
%
%     if nFeas > 0
%         nlml_feas = arrayfun(@(j) objfun(feasible_starts(:, j)), 1:nFeas);
%         [~, ord] = sort(nlml_feas, 'ascend');
%         starts_for_fmincon = feasible_starts(:, ord(1:min(nMultistart, nFeas)));
%     else
%         fprintf('No feasible random start; using projected baseline theta.\n');
%         starts_for_fmincon = theta_unc_box;
%     end
%     starts_for_fmincon = [theta_unc_box, starts_for_fmincon];
%     starts_for_fmincon = starts_for_fmincon(:, 1:min(nMultistart + 1, size(starts_for_fmincon, 2)));
%
%     best_nlml = inf;
%     theta_opt = nan(2, 1);
%     nlml_con = nan;
%     exitflag_con = -99;
%     nStarts = size(starts_for_fmincon, 2);
%     for j = 1:nStarts
%         theta0_j = starts_for_fmincon(:, j);
%         [theta_j, nlml_j, ef_j] = fmincon(objfun, theta0_j, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts_pens);
%         if isfinite(nlml_j) && nlml_j < best_nlml
%             best_nlml = nlml_j;
%             theta_opt = theta_j;
%             nlml_con = nlml_j;
%             exitflag_con = ef_j;
%         end
%     end
%
%     if ~isfinite(best_nlml)
%         if nFeas > 0
%             theta_opt = best_feas_theta;
%         else
%             theta_opt = theta_unc_box;
%         end
%         nlml_con = objfun(theta_opt);
%         exitflag_con = -99;
%         fprintf('Warning: no successful fmincon run; using fallback theta.\n');
%     end
%
%     [c_final, ~] = nonlcon(theta_opt);
%     deriv_max_c = max(c_final(1:nC));
%     data_max_c = nan;  % data fidelity off
%     sweep_deriv_max_c(ie) = deriv_max_c;
%     sweep_data_max_c(ie)  = data_max_c;
%     sweep_nlml(ie)        = nlml_con;
%     sweep_exitflag(ie)    = exitflag_con;
%     theta_opt_last = theta_opt;
%     eta_last = eta;
%
%     fprintf('NLML=%.4f | exitflag=%d | deriv max(c)=%.6g\n', ...
%         nlml_con, exitflag_con, deriv_max_c);
% end
%
% %% Plot eta sweep metrics
% figure('Color', 'w', 'Position', [80, 80, 900, 700], ...
%     'Name', 'Michaelis-Menten GP: eta sweep (positive derivative)');
% tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
%
% nexttile;
% plot(eta_grid, sweep_deriv_max_c, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
% hold on; yline(0, 'k--', 'HandleVisibility', 'off'); hold off;
% xlabel('\eta');
% ylabel('deriv max(c)');
% title('deriv max(c) vs \eta');
% grid on;
%
% nexttile;
% plot(eta_grid, sweep_data_max_c, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
% hold on; yline(0, 'k--', 'HandleVisibility', 'off'); hold off;
% xlabel('\eta');
% ylabel('data max(c)');
% title('data max(c) vs \eta (fidelity off)');
% grid on;
%
% nexttile;
% plot(eta_grid, sweep_nlml, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
% xlabel('\eta');
% ylabel('NLML');
% title('NLML vs \eta');
% grid on;
%
% nexttile;
% plot(eta_grid, sweep_exitflag, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
% xlabel('\eta');
% ylabel('exitflag');
% title('exitflag vs \eta');
% grid on;
%
% sgtitle(sprintf('Eta sweep (n_{constraint}=%d, data fidelity off, nTry=%d)', ...
%     n_constraint, nTry));
%
% %% Plot baseline vs constrained GP at last eta
% k_last = -sqrt(2) * erfinv(2 * eta_last - 1);
% hyp_con = theta_to_hyp(theta_opt_last, hyp_tpl);
% ... (GP side-by-side as above)
%
% =============================================================================
% N_CONSTRAINT SWEEP (commented out; do not delete)
% =============================================================================
% %% Pensoneault-constrained GP n_constraint sweep (positive derivative + data fidelity)
% n_nc = numel(n_constraint_grid);
% sweep_deriv_max_c = nan(n_nc, 1);
% sweep_data_max_c  = nan(n_nc, 1);
% sweep_nlml        = nan(n_nc, 1);
% sweep_exitflag    = nan(n_nc, 1);
% theta_opt_last = theta_unc_box;
% n_constraint_last = n_constraint_grid(end);
%
% fprintf('\n=== n_constraint sweep (positive derivative + data fidelity) ===\n');
% fprintf('eta = %.4g (%.3g%%) | k = %.4f | epsilon = %.4g | n_constraint grid: %s | random starts: %d\n', ...
%     eta, 100 * eta, k, epsilon, mat2str(n_constraint_grid), nTry);
% fprintf('Baseline: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
%     exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);
%
% for in = 1:n_nc
%     n_constraint = n_constraint_grid(in);
%     X_c = linspace(0, x_max, n_constraint)';
%     nC = numel(X_c);
%     nonlcon = @(theta) pens_constraints_pos_deriv(theta, hyp_tpl, x_col, y_col, X_c, k, epsilon);
%
%     fprintf('\n--- n_constraint = %d ---\n', n_constraint);
%
%     feasible_starts = zeros(2, 0);
%     best_feas_nlml = inf;
%     best_feas_theta = nan(2, 1);
%     rng(42);
%     for t = 1:nTry
%         theta_try = hyp_lb + rand(2, 1) .* (hyp_ub - hyp_lb);
%         [c_try, ~] = nonlcon(theta_try);
%         if max(c_try) <= 0
%             feasible_starts = [feasible_starts, theta_try];
%             nlml_try = objfun(theta_try);
%             if nlml_try < best_feas_nlml
%                 best_feas_nlml = nlml_try;
%                 best_feas_theta = theta_try;
%             end
%         end
%     end
%     nFeas = size(feasible_starts, 2);
%     fprintf('Feasible random starts: %d / %d\n', nFeas, nTry);
%
%     if nFeas > 0
%         nlml_feas = arrayfun(@(j) objfun(feasible_starts(:, j)), 1:nFeas);
%         [~, ord] = sort(nlml_feas, 'ascend');
%         starts_for_fmincon = feasible_starts(:, ord(1:min(nMultistart, nFeas)));
%     else
%         fprintf('No feasible random start; using projected baseline theta.\n');
%         starts_for_fmincon = theta_unc_box;
%     end
%     starts_for_fmincon = [theta_unc_box, starts_for_fmincon];
%     starts_for_fmincon = starts_for_fmincon(:, 1:min(nMultistart + 1, size(starts_for_fmincon, 2)));
%
%     best_nlml = inf;
%     theta_opt = nan(2, 1);
%     nlml_con = nan;
%     exitflag_con = -99;
%     nStarts = size(starts_for_fmincon, 2);
%     for j = 1:nStarts
%         theta0_j = starts_for_fmincon(:, j);
%         [theta_j, nlml_j, ef_j] = fmincon(objfun, theta0_j, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts_pens);
%         if isfinite(nlml_j) && nlml_j < best_nlml
%             best_nlml = nlml_j;
%             theta_opt = theta_j;
%             nlml_con = nlml_j;
%             exitflag_con = ef_j;
%         end
%     end
%
%     if ~isfinite(best_nlml)
%         if nFeas > 0
%             theta_opt = best_feas_theta;
%         else
%             theta_opt = theta_unc_box;
%         end
%         nlml_con = objfun(theta_opt);
%         exitflag_con = -99;
%         fprintf('Warning: no successful fmincon run; using fallback theta.\n');
%     end
%
%     [c_final, ~] = nonlcon(theta_opt);
%     deriv_max_c = max(c_final(1:nC));
%     data_max_c  = max(c_final(nC+1:end));
%     sweep_deriv_max_c(in) = deriv_max_c;
%     sweep_data_max_c(in)  = data_max_c;
%     sweep_nlml(in)        = nlml_con;
%     sweep_exitflag(in)    = exitflag_con;
%     theta_opt_last = theta_opt;
%     n_constraint_last = n_constraint;
%
%     fprintf('NLML=%.4f | exitflag=%d | deriv max(c)=%.6g | data max(c)=%.6g\n', ...
%         nlml_con, exitflag_con, deriv_max_c, data_max_c);
% end
%
% %% Plot n_constraint sweep metrics
% figure('Color', 'w', 'Position', [80, 80, 900, 700], ...
%     'Name', 'Michaelis-Menten GP: n_constraint sweep (positive derivative)');
% tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
%
% nexttile;
% plot(n_constraint_grid, sweep_deriv_max_c, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
% hold on; yline(0, 'k--', 'HandleVisibility', 'off'); hold off;
% xlabel('n_{constraint}');
% ylabel('deriv max(c)');
% title('deriv max(c) vs n_{constraint}');
% grid on;
%
% nexttile;
% plot(n_constraint_grid, sweep_data_max_c, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
% hold on; yline(0, 'k--', 'HandleVisibility', 'off'); hold off;
% xlabel('n_{constraint}');
% ylabel('data max(c)');
% title('data max(c) vs n_{constraint}');
% grid on;
%
% nexttile;
% plot(n_constraint_grid, sweep_nlml, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
% xlabel('n_{constraint}');
% ylabel('NLML');
% title('NLML vs n_{constraint}');
% grid on;
%
% nexttile;
% plot(n_constraint_grid, sweep_exitflag, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
% xlabel('n_{constraint}');
% ylabel('exitflag');
% title('exitflag vs n_{constraint}');
% grid on;
%
% sgtitle(sprintf('n_{constraint} sweep (\\eta=%.3g, \\epsilon=%.2g, nTry=%d)', ...
%     eta, epsilon, nTry));
%
% %% Plot baseline vs constrained GP at last n_constraint
% X_c = linspace(0, x_max, n_constraint_last)';
% hyp_con = theta_to_hyp(theta_opt_last, hyp_tpl);
% k_plot = 2;
% [~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
% m_unc = fmu_unc(:);
% sf_unc = sqrt(max(fs2_unc(:), 0));
%
% [~, ~, fmu_con, fs2_con] = gp(hyp_con, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
% m_con = fmu_con(:);
% sf_con = sqrt(max(fs2_con(:), 0));
%
% band_label = sprintf('\\mu_f \\pm %g\\sigma_f (latent)', k_plot);
% ylim_shared = [-0.5, max([y_train(:); Vmax; m_unc + k_plot * sf_unc; m_con + k_plot * sf_con]) * 1.02];
%
% figure('Color', 'w', 'Position', [100, 100, 1100, 520], ...
%     'Name', 'Michaelis-Menten GP: positive derivative (last n_constraint)');
% tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
%
% panels(1) = struct('m', m_unc, 'sf', sf_unc, 'title', 'Baseline GP');
% panels(2) = struct('m', m_con, 'sf', sf_con, ...
%     'title', sprintf('Positive derivative GP (\\eta=%.3g, n_c=%d)', eta, n_constraint_last));
%
% for p = 1:2
%     nexttile;
%     ax = gca;
%     ax.Layer = 'top';
%     hold on; grid on;
%     m = panels(p).m;
%     sf = panels(p).sf;
%     fill([x_grid, fliplr(x_grid)], [m + k_plot * sf; flipud(m - k_plot * sf)]', ...
%         [0.72, 0.72, 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
%     plot(x_grid, m, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
%     plot(x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
%     plot(x_obs, y_obs, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
%         'DisplayName', 'Observed data');
%     yline(0, 'k:', 'v_0 = 0', 'Alpha', 0.5);
%     yline(Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
%     xlabel('[S] (mM)');
%     ylabel('v_0 (\muM/s)');
%     title(panels(p).title, 'Interpreter', 'tex');
%     xlim([0, x_max]);
%     ylim(ylim_shared);
%     legend('Location', 'southeast');
% end

% =============================================================================
% ETA SWEEP (commented out; do not delete)
% =============================================================================
% %% Pensoneault-constrained GP eta sweep (positive derivative + data fidelity)
% nC = numel(X_c);
% n_eta = numel(eta_grid);
%
% sweep_deriv_max_c = nan(n_eta, 1);
% sweep_data_max_c  = nan(n_eta, 1);
% sweep_nlml        = nan(n_eta, 1);
% sweep_exitflag    = nan(n_eta, 1);
% theta_opt_last = theta_unc_box;
% eta_last = eta_grid(end);
%
% fprintf('\n=== Eta sweep (positive derivative + data fidelity) ===\n');
% fprintf('epsilon = %.4g | X_c: %d points | eta grid: %s | random starts: %d\n', ...
%     epsilon, nC, mat2str(eta_grid), nTry);
% fprintf('Baseline: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
%     exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);
%
% for ie = 1:n_eta
%     eta = eta_grid(ie);
%     k = -sqrt(2) * erfinv(2 * eta - 1);
%     nonlcon = @(theta) pens_constraints_pos_deriv(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
%         x_col, y_col, X_c, k, epsilon);
%
%     fprintf('\n--- eta = %.4g (%.3g%%), k = %.4f ---\n', eta, 100 * eta, k);
%
%     feasible_starts = zeros(2, 0);
%     best_feas_nlml = inf;
%     best_feas_theta = nan(2, 1);
%     rng(42);
%     for t = 1:nTry
%         theta_try = hyp_lb + rand(2, 1) .* (hyp_ub - hyp_lb);
%         [c_try, ~] = nonlcon(theta_try);
%         if max(c_try) <= 0
%             feasible_starts = [feasible_starts, theta_try];
%             nlml_try = objfun(theta_try);
%             if nlml_try < best_feas_nlml
%                 best_feas_nlml = nlml_try;
%                 best_feas_theta = theta_try;
%             end
%         end
%     end
%     nFeas = size(feasible_starts, 2);
%     fprintf('Feasible random starts: %d / %d\n', nFeas, nTry);
%
%     if nFeas > 0
%         nlml_feas = arrayfun(@(j) objfun(feasible_starts(:, j)), 1:nFeas);
%         [~, ord] = sort(nlml_feas, 'ascend');
%         starts_for_fmincon = feasible_starts(:, ord(1:min(nMultistart, nFeas)));
%     else
%         fprintf('No feasible random start; using projected baseline theta.\n');
%         starts_for_fmincon = theta_unc_box;
%     end
%     starts_for_fmincon = [theta_unc_box, starts_for_fmincon];
%     starts_for_fmincon = starts_for_fmincon(:, 1:min(nMultistart + 1, size(starts_for_fmincon, 2)));
%
%     best_nlml = inf;
%     theta_opt = nan(2, 1);
%     nlml_con = nan;
%     exitflag_con = -99;
%     nStarts = size(starts_for_fmincon, 2);
%     for j = 1:nStarts
%         theta0_j = starts_for_fmincon(:, j);
%         [theta_j, nlml_j, ef_j] = fmincon(objfun, theta0_j, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts_pens);
%         if isfinite(nlml_j) && nlml_j < best_nlml
%             best_nlml = nlml_j;
%             theta_opt = theta_j;
%             nlml_con = nlml_j;
%             exitflag_con = ef_j;
%         end
%     end
%
%     if ~isfinite(best_nlml)
%         if nFeas > 0
%             theta_opt = best_feas_theta;
%         else
%             theta_opt = theta_unc_box;
%         end
%         nlml_con = objfun(theta_opt);
%         exitflag_con = -99;
%         fprintf('Warning: no successful fmincon run; using fallback theta.\n');
%     end
%
%     [c_final, ~] = nonlcon(theta_opt);
%     deriv_max_c = max(c_final(1:nC));
%     data_max_c  = max(c_final(nC+1:end));
%     sweep_deriv_max_c(ie) = deriv_max_c;
%     sweep_data_max_c(ie)  = data_max_c;
%     sweep_nlml(ie)        = nlml_con;
%     sweep_exitflag(ie)    = exitflag_con;
%     theta_opt_last = theta_opt;
%     eta_last = eta;
%
%     fprintf('NLML=%.4f | exitflag=%d | deriv max(c)=%.6g | data max(c)=%.6g\n', ...
%         nlml_con, exitflag_con, deriv_max_c, data_max_c);
% end
%
% %% Plot eta sweep metrics
% figure('Color', 'w', 'Position', [80, 80, 900, 700], ...
%     'Name', 'Michaelis-Menten GP: eta sweep (positive derivative)');
% tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
%
% nexttile;
% plot(eta_grid, sweep_deriv_max_c, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
% hold on; yline(0, 'k--', 'HandleVisibility', 'off'); hold off;
% xlabel('\eta');
% ylabel('deriv max(c)');
% title('deriv max(c) vs \eta');
% grid on;
%
% nexttile;
% plot(eta_grid, sweep_data_max_c, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
% hold on; yline(0, 'k--', 'HandleVisibility', 'off'); hold off;
% xlabel('\eta');
% ylabel('data max(c)');
% title('data max(c) vs \eta');
% grid on;
%
% nexttile;
% plot(eta_grid, sweep_nlml, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
% xlabel('\eta');
% ylabel('NLML');
% title('NLML vs \eta');
% grid on;
%
% nexttile;
% plot(eta_grid, sweep_exitflag, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
% xlabel('\eta');
% ylabel('exitflag');
% title('exitflag vs \eta');
% grid on;
%
% sgtitle(sprintf('Eta sweep (n_{constraint}=%d, \\epsilon=%.2g, nTry=%d)', ...
%     n_constraint, epsilon, nTry));
%
% %% Plot baseline vs constrained GP at last eta
% k_last = -sqrt(2) * erfinv(2 * eta_last - 1);
% hyp_con = theta_to_hyp(theta_opt_last, hyp_tpl);
% k_plot = 2;
% [~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
% m_unc = fmu_unc(:);
% sf_unc = sqrt(max(fs2_unc(:), 0));
%
% [~, ~, fmu_con, fs2_con] = gp(hyp_con, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
% m_con = fmu_con(:);
% sf_con = sqrt(max(fs2_con(:), 0));
%
% band_label = sprintf('\\mu_f \\pm %g\\sigma_f (latent)', k_plot);
% ylim_shared = [-0.5, max([y_train(:); Vmax; m_unc + k_plot * sf_unc; m_con + k_plot * sf_con]) * 1.02];
%
% figure('Color', 'w', 'Position', [100, 100, 1100, 520], ...
%     'Name', 'Michaelis-Menten GP: positive derivative (last eta)');
% tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
%
% panels(1) = struct('m', m_unc, 'sf', sf_unc, 'title', 'Baseline GP');
% panels(2) = struct('m', m_con, 'sf', sf_con, ...
%     'title', sprintf('Positive derivative GP (\\eta=%.3g, k=%.3g)', eta_last, k_last));
%
% for p = 1:2
%     nexttile;
%     ax = gca;
%     ax.Layer = 'top';
%     hold on; grid on;
%     m = panels(p).m;
%     sf = panels(p).sf;
%     fill([x_grid, fliplr(x_grid)], [m + k_plot * sf; flipud(m - k_plot * sf)]', ...
%         [0.72, 0.72, 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
%     plot(x_grid, m, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
%     plot(x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
%     plot(x_obs, y_obs, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
%         'DisplayName', 'Observed data');
%     yline(0, 'k:', 'v_0 = 0', 'Alpha', 0.5);
%     yline(Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
%     xlabel('[S] (mM)');
%     ylabel('v_0 (\muM/s)');
%     title(panels(p).title, 'Interpreter', 'tex');
%     xlim([0, x_max]);
%     ylim(ylim_shared);
%     legend('Location', 'southeast');
% end

function hyp = theta_to_hyp(theta, hyp_tpl)
hyp = hyp_tpl;
hyp.cov = theta(1:2);
hyp.mean = [];
end

function [f, g] = obj_cov_grad(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, x, y)
% Analytic NLML and dNLML/d[log ell; log sf] via GPML (sn fixed in hyp_tpl.lik).
hyp = theta_to_hyp(theta, hyp_tpl);
if nargout > 1
    [f, dnlml] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y);
    g = dnlml.cov(:);
else
    f = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y);
end
end

function [c, ceq, GC, GCeq] = pens_constraints_pos_deriv(theta, hyp_tpl, x, y, X_c, k, epsilon)
% Positive-derivative Pensoneault + smooth data tube; analytic Jacobian if requested.
%   c_deriv = k*sigma_f' - mu_f' <= 0
%   c_hi    = ymu - y - epsilon <= 0
%   c_lo    = y - ymu - epsilon <= 0
hyp = theta_to_hyp(theta, hyp_tpl);
want_grad = (nargout > 2);
if want_grad
    [c_deriv, c_hi, c_lo, GC] = seiso_pens_c_and_jac(hyp, x, y, X_c, k, epsilon);
else
    [c_deriv, c_hi, c_lo] = seiso_pens_c_and_jac(hyp, x, y, X_c, k, epsilon);
end
c = [c_deriv; c_hi; c_lo];
ceq = [];
if want_grad
    GCeq = [];
end
end

function [c_deriv, c_hi, c_lo, GC] = seiso_pens_c_and_jac(hyp, x, y, X_c, k, epsilon)
% Function-obs SE-iso: constraint values and optional GC (2 x n_c).
x = x(:); y = y(:); X_c = X_c(:);
n = numel(x);
nC = numel(X_c);
ell = exp(hyp.cov(1));
sf2 = exp(2 * hyp.cov(2));
sn2 = exp(2 * hyp.lik(1));

[K, dK_dpsi1, dK_dpsi2, r2] = seiso_Kff_and_dpsi(x, x, ell, sf2); %#ok<ASGLU>
diag_mean = mean(diag(K));
if ~isfinite(diag_mean) || diag_mean <= 0
    diag_mean = 1;
end
jitter0 = max(1e-10, 1e-12 * diag_mean);
max_jitter = max(1e-4, 1e-4 * diag_mean);
jitter = jitter0;
L = [];
for attempt = 1:12
    Ky = K + (sn2 + jitter) * eye(n);
    [L, p] = chol(Ky, 'lower');
    if p == 0
        break;
    end
    jitter = min(jitter * 10, max_jitter);
    L = [];
end
if isempty(L)
    error('MM_ProbDeriv:CholFailed', 'Cholesky failed in seiso_pens_c_and_jac.');
end

alpha = L' \ (L \ y);
ymu = K * alpha;
c_hi = ymu - y - epsilon;
c_lo = y - ymu - epsilon;

[Kdf, dKdf_dpsi1, dKdf_dpsi2] = seiso_Kdf_and_dpsi(X_c, x, ell, sf2);
m_deriv = Kdf * alpha;
V = L \ Kdf';
k_dd = (sf2 / ell^2) * ones(nC, 1);
s2 = max(k_dd - sum(V.^2, 1).', 0);
s = sqrt(max(s2, 0));
eps_s = 1e-12;
s_safe = max(s, eps_s);
c_deriv = k .* s_safe - m_deriv;

if nargout < 4
    return;
end

% Parameter Jacobians: psi = [log ell; log sf]
dKy1 = dK_dpsi1;
dKy2 = dK_dpsi2;

% d alpha / d psi = -Ky \ (dKy * alpha)
dalpha1 = -(L' \ (L \ (dKy1 * alpha)));
dalpha2 = -(L' \ (L \ (dKy2 * alpha)));

dymu1 = dK_dpsi1 * alpha + K * dalpha1;
dymu2 = dK_dpsi2 * alpha + K * dalpha2;

dm1 = dKdf_dpsi1 * alpha + Kdf * dalpha1;
dm2 = dKdf_dpsi2 * alpha + Kdf * dalpha2;

% s2 = k_dd - diag(Kdf * Ky^{-1} * Kdf')
B = L' \ (L \ Kdf');                 % n x nC

dB1 = -(L' \ (L \ (dKy1 * B))) + (L' \ (L \ dKdf_dpsi1'));
dB2 = -(L' \ (L \ (dKy2 * B))) + (L' \ (L \ dKdf_dpsi2'));
dquad1 = sum(dKdf_dpsi1 .* B.', 2) + sum(Kdf .* dB1.', 2);
dquad2 = sum(dKdf_dpsi2 .* B.', 2) + sum(Kdf .* dB2.', 2);

dkdd1 = -2 * k_dd;                   % d(sf2/ell^2)/d log ell
dkdd2 = 2 * k_dd;                    % d(sf2/ell^2)/d log sf
ds2_1 = dkdd1 - dquad1;
ds2_2 = dkdd2 - dquad2;

ds1 = ds2_1 ./ (2 * s_safe);
ds2 = ds2_2 ./ (2 * s_safe);
% Zero gradient contribution where s was floored (numerically flat)
tiny = (s < eps_s);
ds1(tiny) = 0;
ds2(tiny) = 0;

dc_deriv1 = k .* ds1 - dm1;
dc_deriv2 = k .* ds2 - dm2;

% GC: 2 x (nC + 2*n); columns are grads of each constraint
GC = zeros(2, nC + 2 * n);
GC(1, 1:nC) = dc_deriv1.';
GC(2, 1:nC) = dc_deriv2.';
GC(1, nC + (1:n)) = dymu1.';          % c_hi
GC(2, nC + (1:n)) = dymu2.';
GC(1, nC + n + (1:n)) = -dymu1.';     % c_lo
GC(2, nC + n + (1:n)) = -dymu2.';
end

function [K, dK1, dK2, r2] = seiso_Kff_and_dpsi(xa, xb, ell, sf2)
% SE-iso K and dK/d[log ell, log sf]
R = xa(:) - xb(:).';
r2 = (R ./ ell).^2;
W = exp(-0.5 * r2);
K = sf2 * W;
dK1 = K .* r2;        % d / d log ell
dK2 = 2 * K;          % d / d log sf
end

function [Kdf, dKdf1, dKdf2] = seiso_Kdf_and_dpsi(xa, xb, ell, sf2)
% cov(f'(xa), f(xb)) and derivatives w.r.t. log ell, log sf
R = xa(:) - xb(:).';
r2 = (R ./ ell).^2;
W = exp(-0.5 * r2);
Kxc = sf2 * W;
Kdf = -Kxc .* (R ./ ell^2);
dKdf1 = Kdf .* (r2 - 2);   % d / d log ell
dKdf2 = 2 * Kdf;           % d / d log sf
end
