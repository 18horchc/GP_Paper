% LG_het.m — Logistic growth heteroscedastic GP: Models A–E (single run).
% Design: observation times with nRep replicates.
% Latent truth mu(t) = K / (1+exp(-r(t-t0))).
% Observations: y_ij ~ N(mu_i, sigma_i^2) with additive Gaussian noise:
%   sigma = 10% of SD of true curve at all times except Day 5, which has
%   sigma = 50% of SD of true curve.
%   A: homoscedastic GP (learn one sn)
%   B: oracle hetero diag(R) via gp_seiso_hetero_noise
%   C: empirical replicate s_i^2 on diagonal
%   D: learn one sn per unique time by NLML (with ell, sf)
%   E: VHGPR (Lazaro-Gredilla & Titsias, ICML 2011)
% % Pensoneault lower bound mu_f - k*sigma_f >= 0 at X_c is then enforced on
% % A (ell, sf; sn fixed), D (ell, sf; per-time sn fixed), and E (VHGPR).
% All models use squared-exponential kernels.
% Requires GPML, gp_seiso_hetero_noise, gp_seiso_nlml_time_noise,
% vhgpr_fit_predict (+ problems/vhgpr/).
clear; clc; close all;

%% Logistic growth parameters
K  = 1000;
r  = 1.0;
t0 = 5;
t_min = 0;
t_max = 14;

logistic = @(t) K ./ (1 + exp(-r .* (t - t0)));

%% Additive Gaussian heteroscedastic design (10% of SD of true curve; 50% at Day 5)
t_unique = [0;2;5;7;14];
t_obs = t_unique;
noise_frac_base = 0.1;
noise_frac_day5 = 0.7;
t_day5 = 5;
nRep  = 5;
R_rep = nRep;   % alias used by fprintf / models

rng(30);
mu_true = logistic(t_obs);
sd_true_curve = std(logistic(linspace(t_min, t_max, 500)'));
sigma_base = noise_frac_base * sd_true_curve;
sigma_day5 = noise_frac_day5 * sd_true_curve;
sigma_true_obs = sigma_base * ones(size(t_obs));
idx_day5 = abs(t_obs - t_day5) < 1e-12;
sigma_true_obs(idx_day5) = sigma_day5;
var_true = sigma_true_obs.^2;

Y = zeros(numel(t_obs), nRep);
for i = 1:numel(t_obs)
    Y(i,:) = mu_true(i) + sigma_true_obs(i) .* randn(1, nRep);
end

% n_neg = sum(Y(:) < 0);
% fprintf('Negative observations: %d / %d (no clipping applied)\n', n_neg, numel(Y));
% if n_neg > 0
%     warning('LG_het:NegativeObs', ...
%         '%d negative y values; consider reducing additive noise fractions.', n_neg);
% end

mean_rep        = mean(Y, 2);
var_rep         = var(Y, 0, 2);
sd_rep          = sqrt(var_rep);
var_mean_true   = var_true ./ nRep;
sigma_mean_true = sqrt(var_mean_true);

% GP training vectors: each t appears nRep times, responses row-major by replicate
x_train = repelem(t_obs, nRep);
y_train = reshape(Y.', [], 1);
var_train_true   = repelem(var_true, nRep);
sigma_train_true = repelem(sigma_true_obs, nRep);

assert(size(Y, 2) == nRep, 'Expected nRep=%d columns in Y.', nRep);
[~, i_peak] = max(sigma_true_obs);
fprintf('Logistic growth: K=%.0f, r=%.3g, t0=%.3g, t in [%.0f, %.0f]\n', ...
    K, r, t0, t_min, t_max);
fprintf('Additive Gaussian: Nu=%d times, nRep=%d, sigma=%.4g (%.0f%% of SD of true curve = %.4g) / %.4g (%.0f%% of SD) at t=%.0f (N=%d obs)\n', ...
    numel(t_obs), nRep, sigma_base, 100 * noise_frac_base, sd_true_curve, ...
    sigma_day5, 100 * noise_frac_day5, t_day5, numel(x_train));
fprintf('True sigma largest at t=%.3g, max sigma=%.4g\n', ...
    t_obs(i_peak), sigma_true_obs(i_peak));

%% Ground truth on plot grid + oracle noise for Model B
x_grid = linspace(t_min, t_max, 500)';
f_true = logistic(x_grid);
% Block-constant additive noise by nearest observation time (spike at Day 5)
sigma_true = zeros(size(x_grid));
for k = 1:numel(x_grid)
    [~, j] = min(abs(t_obs - x_grid(k)));
    sigma_true(k) = sigma_true_obs(j);
end
noise_var_oracle_tr = var_train_true;
noise_var_oracle_te = sigma_true.^2;

%% GPML / helpers path
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
if ~exist('gp_seiso_hetero_noise', 'file')
    addpath("C:\Users\chorc\OneDrive\Documents\Stroke Research\Gaussian Processes\Bio_Inf_GP_Code\problems");
end
if ~exist('gp_seiso_nlml_time_noise', 'file')
    error('LG_het:MissingTimeNoise', 'gp_seiso_nlml_time_noise.m not found on path.');
end
if ~exist('vhgpr_fit_predict', 'file')
    error('LG_het:MissingVHGPR', 'vhgpr_fit_predict.m not found on path.');
end

meanfunc = @meanZero;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

hyp0_cov = log([std(x_train); std(y_train)]);
minimize_n = -100;
vhgpr_iter = 40;
k_plot = 1.96;

% %% Pensoneault lower bound (applied to A, D, E)
% eta_pens = 0.022;   % 2.2% tail probability
% k_pens = -sqrt(2) * erfinv(2 * eta_pens - 1);
% n_constraint = 41;
% X_c = linspace(t_min, t_max, n_constraint)';
% ell_bounds_lo = 0.05;
% ell_ub = t_max - t_min;
% sf_bounds = [0.05, max(15, 1.5 * std(y_train))];
% hyp_lb = log([ell_bounds_lo; sf_bounds(1)]);
% hyp_ub = log([ell_ub; sf_bounds(2)]);
% opts_pens = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
%     'EnableFeasibilityMode', true, 'Display', 'off', ...
%     'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
%     'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);
% opts_pens_e = optimoptions(opts_pens, 'SpecifyObjectiveGradient', true);
% nTry = 2000;
% nMultistart = 10;

%% Model A: homoscedastic GP (learn sn)
fprintf('\n=== Model A: homoscedastic (learn sn) ===\n');
hyp_a = struct('mean', [], 'cov', hyp0_cov, 'lik', log(max(std(y_train)*0.1, 1e-3)));
hyp_a = minimize(hyp_a, @gp, minimize_n, inffunc, meanfunc, covfunc, likfunc, x_train, y_train);
[~, ~, fmu_a, fs2_a] = gp(hyp_a, inffunc, meanfunc, covfunc, likfunc, x_train, y_train, x_grid);
sigma_a = exp(hyp_a.lik) * ones(size(x_grid));
nlml_a = gp(hyp_a, inffunc, meanfunc, covfunc, likfunc, x_train, y_train);
fprintf('A: ell=%.4g, sf=%.4g, sn=%.4g | NLML=%.4f\n', ...
    exp(hyp_a.cov(1)), exp(hyp_a.cov(2)), exp(hyp_a.lik), nlml_a);

%% Model B: oracle heteroscedastic R
% fprintf('\n=== Model B: oracle hetero diag(R) ===\n');
% hyp_b = struct('mean', [], 'cov', hyp0_cov, 'lik', []);
% hyp_b = minimize(hyp_b, @(h) gp_seiso_hetero_noise('nlml', h, x_train, y_train, noise_var_oracle_tr), minimize_n);
% [~, ~, fmu_b, fs2_b] = gp_seiso_hetero_noise( ...
%     'pred', hyp_b, x_train, y_train, noise_var_oracle_tr, x_grid, noise_var_oracle_te);
% sigma_b = sigma_true;
% nlml_b = gp_seiso_hetero_noise('nlml', hyp_b, x_train, y_train, noise_var_oracle_tr);
% fprintf('B: ell=%.4g, sf=%.4g | NLML=%.4f\n', ...
%     exp(hyp_b.cov(1)), exp(hyp_b.cov(2)), nlml_b);

%% Model C: empirical replicate variance
% fprintf('\n=== Model C: empirical replicate s_i^2 ===\n');
% [noise_var_c_tr, s2_unique] = empirical_replicate_noise(x_train, y_train, t_unique);
% noise_var_c_te = nearest_noise_var(x_grid, t_unique, s2_unique);
% hyp_c = struct('mean', [], 'cov', hyp0_cov, 'lik', []);
% hyp_c = minimize(hyp_c, @(h) gp_seiso_hetero_noise('nlml', h, x_train, y_train, noise_var_c_tr), minimize_n);
% [~, ~, fmu_c, fs2_c] = gp_seiso_hetero_noise( ...
%     'pred', hyp_c, x_train, y_train, noise_var_c_tr, x_grid, noise_var_c_te);
% % Plot-only smooth sigma (fit still uses block-constant noise_var_c_tr / nearest te)
% sigma_c = exp(interp1(t_unique, 0.5 * log(s2_unique), x_grid, 'pchip', 'extrap'));
% nlml_c = gp_seiso_hetero_noise('nlml', hyp_c, x_train, y_train, noise_var_c_tr);
% fprintf('C: ell=%.4g, sf=%.4g | NLML=%.4f\n', ...
%     exp(hyp_c.cov(1)), exp(hyp_c.cov(2)), nlml_c);

%% Model D: NLML-learned noise at each unique time
fprintf('\n=== Model D: NLML per-time noise ===\n');
nU = numel(t_unique);
sn0_d = log(max(std(y_train) * 0.1, 1e-3));
p0_d = [hyp0_cov; sn0_d * ones(nU, 1)];
obj_d = @(p) gp_seiso_nlml_time_noise(p, x_train, y_train, t_unique);
p_d = minimize(p0_d, obj_d, minimize_n);
hyp_d = struct('mean', [], 'cov', p_d(1:2), 'lik', []);
s2_d_unique = exp(2 * p_d(3:end));
noise_var_d_tr = expand_unique_noise(x_train, t_unique, s2_d_unique);
noise_var_d_te = nearest_noise_var(x_grid, t_unique, s2_d_unique);
[~, ~, fmu_d, fs2_d] = gp_seiso_hetero_noise( ...
    'pred', hyp_d, x_train, y_train, noise_var_d_tr, x_grid, noise_var_d_te);
% Plot-only smooth sigma (fit still uses block-constant noise at unique times)
sigma_d = exp(interp1(t_unique, 0.5 * log(s2_d_unique), x_grid, 'pchip', 'extrap'));
nlml_d = gp_seiso_hetero_noise('nlml', hyp_d, x_train, y_train, noise_var_d_tr);
fprintf('D: ell=%.4g, sf=%.4g | NLML=%.4f\n', ...
    exp(hyp_d.cov(1)), exp(hyp_d.cov(2)), nlml_d);
fprintf('D: learned sn at unique t = %s\n', mat2str(exp(p_d(3:end))', 4));

%% Model E: VHGPR
fprintf('\n=== Model E: VHGPR ===\n');
vboth = vhgpr_fit_predict(x_train, y_train, x_grid, struct('iter', vhgpr_iter));
fmu_e = vboth.fmu;
fs2_e = vboth.fs2;
sigma_e = vboth.sigma_n;
fprintf('E: VHGPR done (iter=%d)\n', vhgpr_iter);

% %% Pensoneault lower bound on A, D, E
% fprintf('\n=== Pensoneault lower bound (eta=%.3g%%, k=%.4f, %d pts on [%.0f, %.0f]) ===\n', ...
%     100 * eta_pens, k_pens, n_constraint, t_min, t_max);
%
% fprintf('\n--- A: Homo GP + lower bound (sn fixed from unconstrained A) ---\n');
% hyp_tpl_a = struct('mean', [], 'cov', hyp_a.cov(:), 'lik', hyp_a.lik);
% obj_a_b = @(theta) gp(theta_to_hyp(theta, hyp_tpl_a), inffunc, meanfunc, covfunc, likfunc, ...
%     x_train, y_train);
% nonlcon_a = @(theta) pens_constraints_lower_homo(theta, hyp_tpl_a, inffunc, meanfunc, covfunc, likfunc, ...
%     x_train, y_train, X_c, k_pens);
% [theta_a_b, nlml_a_b, ef_a_b, c_a_b] = fit_pens_constrained( ...
%     obj_a_b, nonlcon_a, hyp_a.cov(:), hyp_lb, hyp_ub, opts_pens, nTry, nMultistart, 42);
% hyp_a_b = theta_to_hyp(theta_a_b, hyp_tpl_a);
% [~, ~, fmu_a_b, fs2_a_b] = gp(hyp_a_b, inffunc, meanfunc, covfunc, likfunc, ...
%     x_train, y_train, x_grid);
% fprintf('A bound: ell=%.4g, sf=%.4g, sn=%.4g | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
%     exp(hyp_a_b.cov(1)), exp(hyp_a_b.cov(2)), exp(hyp_a_b.lik), nlml_a_b, ef_a_b, max(c_a_b));
%
% fprintf('\n--- D: NLML per-time sn + lower bound (per-time sn fixed from unconstrained D) ---\n');
% hyp_tpl_d = struct('mean', [], 'cov', hyp_d.cov(:), 'lik', []);
% obj_d_b = @(theta) gp_seiso_hetero_noise('nlml', theta_to_hyp(theta, hyp_tpl_d), ...
%     x_train, y_train, noise_var_d_tr);
% nonlcon_d = @(theta) pens_constraints_lower_hetero(theta, hyp_tpl_d, ...
%     x_train, y_train, noise_var_d_tr, X_c, k_pens);
% [theta_d_b, nlml_d_b, ef_d_b, c_d_b] = fit_pens_constrained( ...
%     obj_d_b, nonlcon_d, hyp_d.cov(:), hyp_lb, hyp_ub, opts_pens, nTry, nMultistart, 43);
% hyp_d_b = theta_to_hyp(theta_d_b, hyp_tpl_d);
% [~, ~, fmu_d_b, fs2_d_b] = gp_seiso_hetero_noise( ...
%     'pred', hyp_d_b, x_train, y_train, noise_var_d_tr, x_grid, noise_var_d_te);
% fprintf('D bound: ell=%.4g, sf=%.4g | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
%     exp(hyp_d_b.cov(1)), exp(hyp_d_b.cov(2)), nlml_d_b, ef_d_b, max(c_d_b));
%
% fprintf('\n--- E: VHGPR + lower bound ---\n');
% vbound = vhgpr_fit_predict(x_train, y_train, x_grid, struct( ...
%     'iter', vhgpr_iter, 'warm', vboth, 'X_c', X_c, 'k', k_pens, ...
%     'opts_fmincon', opts_pens_e));
% fmu_e_b = vbound.fmu;
% fs2_e_b = vbound.fs2;
% fprintf('E bound: VHGPR | MV bound=%.4f | exitflag=%d | max(c)=%.4g\n', ...
%     vbound.mv_bound, vbound.exitflag, vbound.max_c);

%% Data-generation check: truth + replicates + empirical mean ± SD
% figure('Color', 'w', 'Position', [60, 80, 720, 420], ...
%     'Name', 'LG_het: additive-noise data check');
% hold on; grid on;
% plot(x_grid, f_true, 'k-', 'LineWidth', 2.0, 'DisplayName', 'True \mu(t)');
% plot(x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4, ...
%     'DisplayName', 'Replicates y_{ij}');
% errorbar(t_obs, mean_rep, sd_rep, 'b-', 'LineWidth', 1.4, 'Marker', 's', ...
%     'MarkerFaceColor', 'b', 'MarkerSize', 6, ...
%     'DisplayName', 'Replicate mean \pm emp. SD');
% xlabel('t'); ylabel('\mu(t) / observations');
% title(sprintf('Additive hetero Gaussian (%.0f%% / %.0f%% of SD of true curve at Day %.0f, nRep=%d)', ...
%     100 * noise_frac_base, 100 * noise_frac_day5, t_day5, nRep));
% legend('Location', 'southeast');
% set(gca, 'FontSize', 12);
% xlim([t_min, t_max]);

%% Figure 2: Trajectory figure (one model per tab; shared legend separate)
fig2 = figure('Color', 'w', 'Position', [80, 80, 900, 560], ...
    'Name', 'LG_het: logistic trajectories A-E');
tg = uitabgroup(fig2);

tab_a = uitab(tg, 'Title', 'A: Homo GP');
ax_a = axes('Parent', tab_a);
plot_traj(ax_a, x_grid, f_true, fmu_a, fs2_a, x_train, y_train, k_plot, 'Baseline');

% tab_b = uitab(tg, 'Title', 'B: Oracle Hetero');
% ax_b = axes('Parent', tab_b);
% plot_traj(ax_b, x_grid, f_true, fmu_b, fs2_b, x_train, y_train, k_plot, 'B: Oracle Heteroscedastic');

% tab_c = uitab(tg, 'Title', 'C: Empirical s^2');
% ax_c = axes('Parent', tab_c);
% plot_traj(ax_c, x_grid, f_true, fmu_c, fs2_c, x_train, y_train, k_plot, 'C: Empirical Replicate Variance');

tab_d = uitab(tg, 'Title', 'D: NLML per-time sn');
ax_d = axes('Parent', tab_d);
plot_traj(ax_d, x_grid, f_true, fmu_d, fs2_d, x_train, y_train, k_plot, 'Learned Heteroscedastic GP');

tab_e = uitab(tg, 'Title', 'E: VHGPR');
ax_e = axes('Parent', tab_e);
plot_traj(ax_e, x_grid, f_true, fmu_e, fs2_e, x_train, y_train, k_plot, 'VHGPR');

%% Standalone shared legend for Figure 2 (for LaTeX / Inkscape)
band_label = sprintf('\\mu_f \\pm %.2g\\sigma_f', k_plot);
fig2L = figure('Color', 'w', 'Position', [100, 100, 900, 80], ...
    'Name', 'LG_het: traj shared legend');
ax2L = axes('Parent', fig2L, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
    'Position', [0 0 1 1]);
hold(ax2L, 'on');
h2L = gobjects(4, 1);
h2L(1) = fill(ax2L, nan, nan, [0.75, 0.85, 0.95], 'EdgeColor', 'none', ...
    'FaceAlpha', 0.55, 'DisplayName', band_label);
h2L(2) = plot(ax2L, nan, nan, 'k-', 'LineWidth', 1.8, 'DisplayName', 'True f');
h2L(3) = plot(ax2L, nan, nan, 'b-', 'LineWidth', 1.5, 'DisplayName', 'GP mean');
h2L(4) = plot(ax2L, nan, nan, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', 'Train y');
lgd2 = legend(ax2L, h2L, 'Orientation', 'horizontal');
lgd2.FontSize = 14;
lgd2.ItemTokenSize = [20, 12];
lgd2.Box = 'on';
drawnow;
fig2L.Units = 'pixels';
lgd2.Units = 'pixels';
lp2 = lgd2.Position;
margin2 = 4;
fig2L.Position(3:4) = [lp2(3) + 2 * margin2, lp2(4) + 2 * margin2];
lgd2.Position = [margin2, margin2, lp2(3), lp2(4)];
ax2L.Position = [0 0 1 1];
drawnow;

% %% Figure 3: Pensoneault lower bound on A, D, E (unconstrained | bound)
% fig4 = figure('Color', 'w', 'Position', [80, 80, 1100, 480], ...
%     'Name', 'LG_het: Pensoneault lower bound A/D/E');
% tg4 = uitabgroup(fig4);
% bound_tabs = { ...
%     'A: Homo GP', 'Baseline', fmu_a, fs2_a, fmu_a_b, fs2_a_b; ...
%     'D: NLML per-time sn', 'Learned Heteroscedastic GP', fmu_d, fs2_d, fmu_d_b, fs2_d_b; ...
%     'E: VHGPR', 'VHGPR', fmu_e, fs2_e, fmu_e_b, fs2_e_b};
% tab4_list = cell(1, size(bound_tabs, 1));
% ax4_list = cell(1, 2 * size(bound_tabs, 1));
% for bi = 1:size(bound_tabs, 1)
%     tab_b = uitab(tg4, 'Title', bound_tabs{bi, 1});
%     tab4_list{bi} = tab_b;
%     tg4.SelectedTab = tab_b;
%     tl_b = tiledlayout(tab_b, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
%     fmu_u = bound_tabs{bi, 3};
%     fs2_u = bound_tabs{bi, 4};
%     fmu_b = bound_tabs{bi, 5};
%     fs2_b = bound_tabs{bi, 6};
%     yl_b = traj_ylim(f_true, fmu_u, fs2_u, fmu_b, fs2_b, y_train, k_pens);
%     ax_u = nexttile(tl_b);
%     plot_traj(ax_u, x_grid, f_true, fmu_u, fs2_u, x_train, y_train, k_pens, ...
%         sprintf('%s (unconstrained)', bound_tabs{bi, 2}), yl_b, true);
%     ax_c = nexttile(tl_b);
%     plot_traj(ax_c, x_grid, f_true, fmu_b, fs2_b, x_train, y_train, k_pens, ...
%         'Pensoneault lower bound', yl_b, true);
%     ax4_list{2*bi-1} = ax_u;
%     ax4_list{2*bi} = ax_c;
% end

% %% Standalone shared legend for Figure 3
% band_label_b = sprintf('\\mu_f \\pm %.2g\\sigma_f', k_pens);
% fig4L = figure('Color', 'w', 'Position', [100, 100, 900, 80], ...
%     'Name', 'LG_het: bound traj shared legend');
% ax4L = axes('Parent', fig4L, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
%     'Position', [0 0 1 1]);
% hold(ax4L, 'on');
% h4L = gobjects(5, 1);
% h4L(1) = fill(ax4L, nan, nan, [0.75, 0.85, 0.95], 'EdgeColor', 'none', ...
%     'FaceAlpha', 0.55, 'DisplayName', band_label_b);
% h4L(2) = plot(ax4L, nan, nan, 'k-', 'LineWidth', 1.8, 'DisplayName', 'True f');
% h4L(3) = plot(ax4L, nan, nan, 'b-', 'LineWidth', 1.5, 'DisplayName', 'GP mean');
% % h4L(4) = plot(ax4L, nan, nan, 'r--', 'LineWidth', 1.2, 'DisplayName', '\mu_f - k\sigma_f');
% h4L(4) = plot(ax4L, nan, nan, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
%     'DisplayName', 'Train y');
% h4L(5) = plot(ax4L, nan, nan, 'k:', 'LineWidth', 1.2, 'DisplayName', 'N = 0');
% lgd4 = legend(ax4L, h4L, 'Orientation', 'horizontal');
% lgd4.FontSize = 14;
% lgd4.ItemTokenSize = [20, 12];
% lgd4.Box = 'on';
% drawnow;
% fig4L.Units = 'pixels';
% lgd4.Units = 'pixels';
% lp4 = lgd4.Position;
% margin4 = 4;
% fig4L.Position(3:4) = [lp4(3) + 2 * margin4, lp4(4) + 2 * margin4];
% lgd4.Position = [margin4, margin4, lp4(3), lp4(4)];
% ax4L.Position = [0 0 1 1];
% drawnow;

%% Figure 3: Noise SD comparison (legend separate)
% fig3 = figure('Color', 'w', 'Position', [80, 120, 700, 420], ...
%     'Name', 'LG_het: noise SD');
% ax3 = axes('Parent', fig3);
% hold(ax3, 'on'); grid(ax3, 'on');
% plot(ax3, x_grid, sigma_true, 'k-', 'LineWidth', 2.2, ...
%     'DisplayName', 'True \sigma(t) (additive)');
% plot(ax3, x_grid, sigma_a, 'r-.', 'LineWidth', 1.5, 'DisplayName', 'A (homo sn)');
% plot(ax3, x_grid, sigma_b, 'g--', 'LineWidth', 1.5, 'DisplayName', 'B (oracle)');
% plot(ax3, x_grid, sigma_c, 'm:', 'LineWidth', 1.6, 'DisplayName', 'C (empirical)');
% plot(ax3, x_grid, sigma_d, 'c-', 'LineWidth', 1.5, 'DisplayName', 'D (NLML per-time)');
% plot(ax3, x_grid, sigma_e, 'b-', 'LineWidth', 1.5, 'DisplayName', 'E (VHGPR)');
% xlabel(ax3, 't'); ylabel(ax3, '\sigma_n(t)');
% title(ax3, 'Observation noise SD: additive truth vs models');
% set(ax3, 'FontSize', 12);
% xlim(ax3, [t_min, t_max]);
% 
% %% Standalone shared legend for Figure 3 (for LaTeX / Inkscape)
% fig3L = figure('Color', 'w', 'Position', [100, 100, 1000, 80], ...
%     'Name', 'LG_het: noise SD shared legend');
% ax3L = axes('Parent', fig3L, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
%     'Position', [0 0 1 1]);
% hold(ax3L, 'on');
% h3L = gobjects(6, 1);
% h3L(1) = plot(ax3L, nan, nan, 'k-', 'LineWidth', 2.2, ...
%     'DisplayName', 'True \sigma(t) (additive)');
% h3L(2) = plot(ax3L, nan, nan, 'r-.', 'LineWidth', 1.5, 'DisplayName', 'A (homo sn)');
% h3L(3) = plot(ax3L, nan, nan, 'g--', 'LineWidth', 1.5, 'DisplayName', 'B (oracle)');
% h3L(4) = plot(ax3L, nan, nan, 'm:', 'LineWidth', 1.6, 'DisplayName', 'C (empirical)');
% h3L(5) = plot(ax3L, nan, nan, 'c-', 'LineWidth', 1.5, 'DisplayName', 'D (NLML per-time)');
% h3L(6) = plot(ax3L, nan, nan, 'b-', 'LineWidth', 1.5, 'DisplayName', 'E (VHGPR)');
% lgd3 = legend(ax3L, h3L, 'Orientation', 'horizontal', 'NumColumns', 6);
% lgd3.FontSize = 14;
% lgd3.ItemTokenSize = [24, 12];
% lgd3.Box = 'on';
% drawnow;
% fig3L.Units = 'pixels';
% lgd3.Units = 'pixels';
% lp3 = lgd3.Position;
% margin3 = 4;
% fig3L.Position(3:4) = [lp3(3) + 2 * margin3, lp3(4) + 2 * margin3];
% lgd3.Position = [margin3, margin3, lp3(3), lp3(4)];
% ax3L.Position = [0 0 1 1];
% drawnow;

%% Save Figure 2 panels (A, D, E) and shared legend as EPS
plot_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
    'results', 'plots', 'Paper Draft 2', 'Log Growth');
if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end
tab_list = {tab_a, tab_d, tab_e};
ax_list  = {ax_a, ax_d, ax_e};
name_list = {'LG_het_A_Homo_GP.eps', 'LG_het_D_NLML_per_time.eps', 'LG_het_E_VHGPR.eps'};
for i = 1:numel(tab_list)
    tg.SelectedTab = tab_list{i};
    ax_list{i}.Toolbar.Visible = 'off';
    disableDefaultInteractivity(ax_list{i});
    drawnow;
    out_path = fullfile(plot_dir, name_list{i});
    exportgraphics(ax_list{i}, out_path, 'ContentType', 'image');
    fprintf('Saved %s\n', out_path);
end
legend2_path = fullfile(plot_dir, 'LG_het_traj_legend.eps');
exportgraphics(fig2L, legend2_path, 'ContentType', 'image', 'BackgroundColor', 'white');
fprintf('Saved %s\n', legend2_path);

% tab_list = {tab_a, tab_b, tab_c, tab_d, tab_e};
% ax_list  = {ax_a, ax_b, ax_c, ax_d, ax_e};
% name_list = {'LG_het_A_Homo_GP.eps', 'LG_het_B_Oracle_Hetero.eps', ...
%     'LG_het_C_Empirical.eps', 'LG_het_D_NLML_per_time.eps', 'LG_het_E_VHGPR.eps'};
%
% ax3.Toolbar.Visible = 'off';
% disableDefaultInteractivity(ax3);
% drawnow;
% noise_path = fullfile(plot_dir, 'LG_het_noise_SD.eps');
% exportgraphics(ax3, noise_path, 'ContentType', 'image');
% fprintf('Saved %s\n', noise_path);
% legend3_path = fullfile(plot_dir, 'LG_het_noise_legend.eps');
% exportgraphics(fig3L, legend3_path, 'ContentType', 'image', 'BackgroundColor', 'white');
% fprintf('Saved %s\n', legend3_path);
%
% bound_name_list = {'LG_het_bound_A_uncon.eps', 'LG_het_bound_A.eps', ...
%     'LG_het_bound_D_uncon.eps', 'LG_het_bound_D.eps', ...
%     'LG_het_bound_E_uncon.eps', 'LG_het_bound_E.eps'};
% for i = 1:numel(tab4_list)
%     tg4.SelectedTab = tab4_list{i};
%     drawnow;
%     for j = 1:2
%         axij = ax4_list{2*(i-1)+j};
%         axij.Toolbar.Visible = 'off';
%         disableDefaultInteractivity(axij);
%         out_path = fullfile(plot_dir, bound_name_list{2*(i-1)+j});
%         exportgraphics(axij, out_path, 'ContentType', 'image');
%         fprintf('Saved %s\n', out_path);
%     end
% end
% legend4_path = fullfile(plot_dir, 'LG_het_bound_legend.eps');
% exportgraphics(fig4L, legend4_path, 'ContentType', 'image', 'BackgroundColor', 'white');
% fprintf('Saved %s\n', legend4_path);

fprintf('\nDone.\n');

%% ==================== Local functions ====================
function noise_var = expand_unique_noise(x, t_unique, var_unique)
x = x(:);
t_unique = t_unique(:);
var_unique = var_unique(:);
noise_var = zeros(size(x));
for i = 1:numel(t_unique)
    noise_var(abs(x - t_unique(i)) < 1e-12) = var_unique(i);
end
end

function [noise_var, s2_unique] = empirical_replicate_noise(x, y, t_unique)
x = x(:); y = y(:); t_unique = t_unique(:);
n = numel(x);
nu = numel(t_unique);
s2_unique = zeros(nu, 1);
eps_floor = 1e-6 * max(var(y), 1);
for i = 1:nu
    idx = abs(x - t_unique(i)) < 1e-12;
    yi = y(idx);
    m = numel(yi);
    if m >= 2
        s2_unique(i) = sum((yi - mean(yi)).^2) / (m - 1);
    else
        s2_unique(i) = eps_floor;
    end
    s2_unique(i) = max(s2_unique(i), eps_floor);
end
noise_var = zeros(n, 1);
for i = 1:nu
    idx = abs(x - t_unique(i)) < 1e-12;
    noise_var(idx) = s2_unique(i);
end
end

function noise_var_star = nearest_noise_var(xs, t_unique, s2_unique)
xs = xs(:); t_unique = t_unique(:); s2_unique = s2_unique(:);
noise_var_star = zeros(size(xs));
for k = 1:numel(xs)
    [~, j] = min(abs(t_unique - xs(k)));
    noise_var_star(k) = s2_unique(j);
end
end

function plot_traj(ax, xg, f_true, fmu, fs2, x_train, y_train, k_plot, title_str, ylims, show_zero)
if nargin < 10
    ylims = [];
end
if nargin < 11
    show_zero = false;
end
hold(ax, 'on'); grid(ax, 'on');
fmu = fmu(:);
sf = sqrt(max(fs2(:), 0));
fill(ax, [xg; flipud(xg)], [fmu + k_plot * sf; flipud(fmu - k_plot * sf)], ...
    [0.75, 0.85, 0.95], 'EdgeColor', 'none', 'FaceAlpha', 0.55, ...
    'DisplayName', sprintf('\\mu_f \\pm %.2g\\sigma_f', k_plot));
plot(ax, xg, f_true, 'k-', 'LineWidth', 1.8, 'DisplayName', 'True f');
plot(ax, xg, fmu, 'b-', 'LineWidth', 1.5, 'DisplayName', 'GP mean');
plot(ax, x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4, ...
    'DisplayName', 'Train y');
if show_zero
    % plot(ax, xg, fmu - k_plot * sf, 'r--', 'LineWidth', 1.2, ...
    %     'DisplayName', '\mu_f - k\sigma_f');
    yline(ax, 0, 'k:', 'LineWidth', 1.2, 'DisplayName', 'N = 0');
end
xlabel(ax, 't', 'FontSize', 18);
ylabel(ax, 'N(t)', 'FontSize', 18);
title(ax, title_str, 'Interpreter', 'none', 'FontSize', 18);
xlim(ax, [xg(1), xg(end)]);
if ~isempty(ylims)
    ylim(ax, ylims);
end
set(ax, 'FontSize', 18);
ax.Layer = 'top';
end

function ylims = traj_ylim(f_true, fmu1, fs21, fmu2, fs22, y_train, k)
sf1 = sqrt(max(fs21(:), 0));
sf2 = sqrt(max(fs22(:), 0));
lo = min([0; y_train(:); f_true(:); fmu1(:) - k * sf1; fmu2(:) - k * sf2]);
hi = max([y_train(:); f_true(:); fmu1(:) + k * sf1; fmu2(:) + k * sf2]);
pad = 0.04 * (hi - lo + eps);
ylims = [lo - pad, hi + pad];
end

function hyp = theta_to_hyp(theta, hyp_tpl)
hyp = hyp_tpl;
hyp.cov = theta(1:2);
hyp.mean = [];
end

function [c, ceq] = pens_constraints_lower_homo(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k)
hyp = theta_to_hyp(theta, hyp_tpl);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, X_c(:));
c = k .* sqrt(max(fs2(:), 0)) - fmu(:);
ceq = [];
end

function [c, ceq] = pens_constraints_lower_hetero(theta, hyp_tpl, x, y, noise_var, X_c, k)
hyp = theta_to_hyp(theta, hyp_tpl);
[~, ~, fmu, fs2] = gp_seiso_hetero_noise('pred', hyp, x, y, noise_var, X_c(:));
c = k .* sqrt(max(fs2(:), 0)) - fmu(:);
ceq = [];
end

function [theta_opt, nlml_con, exitflag_con, c_final] = fit_pens_constrained( ...
    objfun, nonlcon, theta_unc, hyp_lb, hyp_ub, opts_pens, nTry, nMultistart, rng_seed)
n_theta = numel(hyp_lb);
theta_unc = theta_unc(:);
theta_unc_box = min(max(theta_unc, hyp_lb), hyp_ub);
fprintf('  Multistart: %d random starts\n', nTry);
feasible_starts = zeros(n_theta, 0);
best_feas_nlml = inf;
best_feas_theta = nan(n_theta, 1);
rng(rng_seed);
for t = 1:nTry
    theta_try = hyp_lb + rand(n_theta, 1) .* (hyp_ub - hyp_lb);
    [c_try, ~] = nonlcon(theta_try);
    if max(c_try) <= 0
        feasible_starts = [feasible_starts, theta_try]; %#ok<AGROW>
        nlml_try = objfun(theta_try);
        if nlml_try < best_feas_nlml
            best_feas_nlml = nlml_try;
            best_feas_theta = theta_try;
        end
    end
end
nFeas = size(feasible_starts, 2);
fprintf('  Feasible random starts: %d / %d\n', nFeas, nTry);
if nFeas > 0
    nlml_feas = arrayfun(@(j) objfun(feasible_starts(:, j)), 1:nFeas);
    [~, ord] = sort(nlml_feas, 'ascend');
    starts_for_fmincon = feasible_starts(:, ord(1:min(nMultistart, nFeas)));
else
    fprintf('  No feasible random start; using projected baseline theta.\n');
    starts_for_fmincon = theta_unc_box;
end
starts_for_fmincon = [theta_unc_box, starts_for_fmincon];
starts_for_fmincon = unique(starts_for_fmincon', 'rows', 'stable')';
starts_for_fmincon = starts_for_fmincon(:, 1:min(nMultistart + 1, size(starts_for_fmincon, 2)));

best_nlml = inf;
theta_opt = nan(n_theta, 1);
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
    fprintf('  Warning: no successful fmincon run; using fallback theta.\n');
end
[c_final, ~] = nonlcon(theta_opt);
end
