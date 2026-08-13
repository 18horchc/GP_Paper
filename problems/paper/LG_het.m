% LG_het.m — Logistic growth heteroscedastic GP: Models A–E (single run).
% Design: observation times with nRep=5 replicates.
% Latent truth mu(t) = K / (1+exp(-r(t-t0))).
% Observations: y_ij ~ N(mu_i, sigma_i^2) with NB-inspired Var = mu + alpha*mu^2
% (Gaussian likelihood only; not sampled from a negative-binomial distribution).
%   A: homoscedastic GP (learn one sn)
%   B: oracle hetero diag(R) via gp_seiso_hetero_noise
%   C: empirical replicate s_i^2 on diagonal
%   D: learn one sn per unique time by NLML (with ell, sf)
%   E: VHGPR (Lazaro-Gredilla & Titsias, ICML 2011)
% All models use squared-exponential kernels.
% Requires GPML, gp_seiso_hetero_noise, gp_seiso_nlml_time_noise,
% vhgpr_fit_predict (+ problems/vhgpr/).
clear; clc; close all;

%% Logistic growth parameters
K  = 1000;
r  = 0.6;
t0 = 5;
t_min = 0;
t_max = 14;

logistic = @(t) K ./ (1 + exp(-r .* (t - t0)));

%% NB-inspired heteroscedastic Gaussian design
t_unique = [0;2;4;6;8;10;12;14];
t_obs = t_unique;
alpha = 0.01;
nRep  = 5;
R_rep = nRep;   % alias used by fprintf / models

rng(42);
mu_true        = logistic(t_obs);
var_true       = mu_true + alpha .* mu_true.^2;
sigma_true_obs = sqrt(var_true);

Y = zeros(numel(t_obs), nRep);
for i = 1:numel(t_obs)
    Y(i,:) = mu_true(i) + sigma_true_obs(i) .* randn(1, nRep);
end

n_neg = sum(Y(:) < 0);
fprintf('Negative observations: %d / %d (no clipping applied)\n', n_neg, numel(Y));
if n_neg > 0
    warning('LG_het:NegativeObs', ...
        '%d negative y values; consider reducing alpha (currently %.4g).', n_neg, alpha);
end

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

assert(size(Y, 2) == 5, 'Expected nRep=5 columns in Y.');
[~, i_peak] = max(sigma_true_obs);
fprintf('Logistic growth: K=%.0f, r=%.3g, t0=%.3g, t in [%.0f, %.0f]\n', ...
    K, r, t0, t_min, t_max);
fprintf('NB-inspired Gaussian: Nu=%d times, nRep=%d, alpha=%.4g (N=%d obs)\n', ...
    numel(t_obs), nRep, alpha, numel(x_train));
fprintf('True sigma largest at t=%.3g, max sigma=%.4g (Var=mu+alpha*mu^2)\n', ...
    t_obs(i_peak), sigma_true_obs(i_peak));

%% Ground truth on plot grid + oracle noise for Model B
x_grid = linspace(t_min, t_max, 500)';
f_true = logistic(x_grid);
sigma_true = sqrt(f_true + alpha .* f_true.^2);
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
fprintf('\n=== Model B: oracle hetero diag(R) ===\n');
hyp_b = struct('mean', [], 'cov', hyp0_cov, 'lik', []);
hyp_b = minimize(hyp_b, @(h) gp_seiso_hetero_noise('nlml', h, x_train, y_train, noise_var_oracle_tr), minimize_n);
[~, ~, fmu_b, fs2_b] = gp_seiso_hetero_noise( ...
    'pred', hyp_b, x_train, y_train, noise_var_oracle_tr, x_grid, noise_var_oracle_te);
sigma_b = sigma_true;
nlml_b = gp_seiso_hetero_noise('nlml', hyp_b, x_train, y_train, noise_var_oracle_tr);
fprintf('B: ell=%.4g, sf=%.4g | NLML=%.4f\n', ...
    exp(hyp_b.cov(1)), exp(hyp_b.cov(2)), nlml_b);

%% Model C: empirical replicate variance
fprintf('\n=== Model C: empirical replicate s_i^2 ===\n');
[noise_var_c_tr, s2_unique] = empirical_replicate_noise(x_train, y_train, t_unique);
noise_var_c_te = nearest_noise_var(x_grid, t_unique, s2_unique);
hyp_c = struct('mean', [], 'cov', hyp0_cov, 'lik', []);
hyp_c = minimize(hyp_c, @(h) gp_seiso_hetero_noise('nlml', h, x_train, y_train, noise_var_c_tr), minimize_n);
[~, ~, fmu_c, fs2_c] = gp_seiso_hetero_noise( ...
    'pred', hyp_c, x_train, y_train, noise_var_c_tr, x_grid, noise_var_c_te);
% Plot-only smooth sigma (fit still uses block-constant noise_var_c_tr / nearest te)
sigma_c = exp(interp1(t_unique, 0.5 * log(s2_unique), x_grid, 'pchip', 'extrap'));
nlml_c = gp_seiso_hetero_noise('nlml', hyp_c, x_train, y_train, noise_var_c_tr);
fprintf('C: ell=%.4g, sf=%.4g | NLML=%.4f\n', ...
    exp(hyp_c.cov(1)), exp(hyp_c.cov(2)), nlml_c);

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

%% Data-generation check: truth + replicates + empirical mean ± SD
figure('Color', 'w', 'Position', [60, 80, 720, 420], ...
    'Name', 'LG_het: NB-inspired data check');
hold on; grid on;
plot(x_grid, f_true, 'k-', 'LineWidth', 2.0, 'DisplayName', 'True \mu(t)');
plot(x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4, ...
    'DisplayName', 'Replicates y_{ij}');
errorbar(t_obs, mean_rep, sd_rep, 'b-', 'LineWidth', 1.4, 'Marker', 's', ...
    'MarkerFaceColor', 'b', 'MarkerSize', 6, ...
    'DisplayName', 'Replicate mean \pm emp. SD');
xlabel('t'); ylabel('\mu(t) / observations');
title(sprintf('NB-inspired hetero Gaussian (\\alpha=%.3g, nRep=%d)', alpha, nRep));
legend('Location', 'southeast');
set(gca, 'FontSize', 12);
xlim([t_min, t_max]);

%% Figure 2: Trajectory figure (one model per tab; shared legend separate)
fig2 = figure('Color', 'w', 'Position', [80, 80, 900, 560], ...
    'Name', 'LG_het: logistic trajectories A-E');
tg = uitabgroup(fig2);

tab_a = uitab(tg, 'Title', 'A: Homo GP');
ax_a = axes('Parent', tab_a);
plot_traj(ax_a, x_grid, f_true, fmu_a, fs2_a, x_train, y_train, k_plot, 'A: Homoscedastic GP');

tab_b = uitab(tg, 'Title', 'B: Oracle Hetero');
ax_b = axes('Parent', tab_b);
plot_traj(ax_b, x_grid, f_true, fmu_b, fs2_b, x_train, y_train, k_plot, 'B: Oracle Heteroscedastic');

tab_c = uitab(tg, 'Title', 'C: Empirical s^2');
ax_c = axes('Parent', tab_c);
plot_traj(ax_c, x_grid, f_true, fmu_c, fs2_c, x_train, y_train, k_plot, 'C: Empirical Replicate Variance');

tab_d = uitab(tg, 'Title', 'D: NLML per-time sn');
ax_d = axes('Parent', tab_d);
plot_traj(ax_d, x_grid, f_true, fmu_d, fs2_d, x_train, y_train, k_plot, 'D: NLML Per-Time Noise');

tab_e = uitab(tg, 'Title', 'E: VHGPR');
ax_e = axes('Parent', tab_e);
plot_traj(ax_e, x_grid, f_true, fmu_e, fs2_e, x_train, y_train, k_plot, 'E: VHGPR');

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

%% Figure 3: Noise SD comparison (legend separate)
fig3 = figure('Color', 'w', 'Position', [80, 120, 700, 420], ...
    'Name', 'LG_het: noise SD');
ax3 = axes('Parent', fig3);
hold(ax3, 'on'); grid(ax3, 'on');
plot(ax3, x_grid, sigma_true, 'k-', 'LineWidth', 2.2, ...
    'DisplayName', 'True \sigma(t) (NB-inspired)');
plot(ax3, x_grid, sigma_a, 'r-.', 'LineWidth', 1.5, 'DisplayName', 'A (homo sn)');
plot(ax3, x_grid, sigma_b, 'g--', 'LineWidth', 1.5, 'DisplayName', 'B (oracle)');
plot(ax3, x_grid, sigma_c, 'm:', 'LineWidth', 1.6, 'DisplayName', 'C (empirical)');
plot(ax3, x_grid, sigma_d, 'c-', 'LineWidth', 1.5, 'DisplayName', 'D (NLML per-time)');
plot(ax3, x_grid, sigma_e, 'b-', 'LineWidth', 1.5, 'DisplayName', 'E (VHGPR)');
xlabel(ax3, 't'); ylabel(ax3, '\sigma_n(t)');
title(ax3, 'Observation noise SD: NB-inspired truth vs models');
set(ax3, 'FontSize', 12);
xlim(ax3, [t_min, t_max]);

%% Standalone shared legend for Figure 3 (for LaTeX / Inkscape)
fig3L = figure('Color', 'w', 'Position', [100, 100, 1000, 80], ...
    'Name', 'LG_het: noise SD shared legend');
ax3L = axes('Parent', fig3L, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
    'Position', [0 0 1 1]);
hold(ax3L, 'on');
h3L = gobjects(6, 1);
h3L(1) = plot(ax3L, nan, nan, 'k-', 'LineWidth', 2.2, ...
    'DisplayName', 'True \sigma(t) (NB-inspired)');
h3L(2) = plot(ax3L, nan, nan, 'r-.', 'LineWidth', 1.5, 'DisplayName', 'A (homo sn)');
h3L(3) = plot(ax3L, nan, nan, 'g--', 'LineWidth', 1.5, 'DisplayName', 'B (oracle)');
h3L(4) = plot(ax3L, nan, nan, 'm:', 'LineWidth', 1.6, 'DisplayName', 'C (empirical)');
h3L(5) = plot(ax3L, nan, nan, 'c-', 'LineWidth', 1.5, 'DisplayName', 'D (NLML per-time)');
h3L(6) = plot(ax3L, nan, nan, 'b-', 'LineWidth', 1.5, 'DisplayName', 'E (VHGPR)');
lgd3 = legend(ax3L, h3L, 'Orientation', 'horizontal', 'NumColumns', 6);
lgd3.FontSize = 14;
lgd3.ItemTokenSize = [24, 12];
lgd3.Box = 'on';
drawnow;
fig3L.Units = 'pixels';
lgd3.Units = 'pixels';
lp3 = lgd3.Position;
margin3 = 4;
fig3L.Position(3:4) = [lp3(3) + 2 * margin3, lp3(4) + 2 * margin3];
lgd3.Position = [margin3, margin3, lp3(3), lp3(4)];
ax3L.Position = [0 0 1 1];
drawnow;

% %% Save Figure 2 panels, Figure 3, and shared legends as EPS
% plot_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
%     'results', 'plots', 'Paper Draft 2', 'Log Growth');
% if ~exist(plot_dir, 'dir')
%     mkdir(plot_dir);
% end
% tab_list = {tab_a, tab_b, tab_c, tab_d, tab_e};
% ax_list  = {ax_a, ax_b, ax_c, ax_d, ax_e};
% name_list = {'LG_het_A_Homo_GP.eps', 'LG_het_B_Oracle_Hetero.eps', ...
%     'LG_het_C_Empirical.eps', 'LG_het_D_NLML_per_time.eps', 'LG_het_E_VHGPR.eps'};
% for i = 1:numel(tab_list)
%     tg.SelectedTab = tab_list{i};
%     ax_list{i}.Toolbar.Visible = 'off';
%     disableDefaultInteractivity(ax_list{i});
%     drawnow;
%     out_path = fullfile(plot_dir, name_list{i});
%     exportgraphics(ax_list{i}, out_path, 'ContentType', 'image');
%     fprintf('Saved %s\n', out_path);
% end
% legend2_path = fullfile(plot_dir, 'LG_het_traj_legend.eps');
% exportgraphics(fig2L, legend2_path, 'ContentType', 'image', 'BackgroundColor', 'white');
% fprintf('Saved %s\n', legend2_path);
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

function plot_traj(ax, xg, f_true, fmu, fs2, x_train, y_train, k_plot, title_str)
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
xlabel(ax, 't', 'FontSize', 18);
ylabel(ax, 'N(t)', 'FontSize', 18);
title(ax, title_str, 'Interpreter', 'none', 'FontSize', 18);
xlim(ax, [xg(1), xg(end)]);
set(ax, 'FontSize', 18);
ax.Layer = 'top';
end
