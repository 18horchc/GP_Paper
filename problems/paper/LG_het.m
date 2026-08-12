% LG_het.m — Logistic growth heteroscedastic GP: Models A–D (single run).
% Design: t in [0,12], times {0,1,2,3,4,5,6,8,10,12}, R=5 replicates.
% Noise: sigma(t) = sigma_min + A*exp(-(t-tc)^2/(2w^2)) (unreliable middle).
%   A: homoscedastic GP (learn one sn)
%   B: oracle hetero diag(R) via gp_seiso_hetero_noise
%   C: empirical replicate s_i^2 on diagonal
%   D: VHGPR (Lazaro-Gredilla & Titsias, ICML 2011)
% All models use squared-exponential kernels.
% Requires GPML, gp_seiso_hetero_noise, vhgpr_fit_predict (+ problems/vhgpr/).
clear; clc; close all;

%% Logistic growth parameters
K  = 1000;
r  = 0.6;
t0 = 5;
t_min = 0;
t_max = 14; %12;

logistic = @(t) K ./ (1 + exp(-r .* (t - t0)));

%% Heteroscedastic bump noise (peak in middle)
sigma_min = 25;
A_bump = 125;
tc = 5; %5;
w_bump = 1.5;
sigma_fn = @(t) sigma_min + A_bump .* exp(-(t - tc).^2 ./ (2 * w_bump^2));

%% Training data
t_unique = [0; 1; 2; 3; 4; 5; 10; 11; 14];
%t_unique = [0;1;2;3;4;7;9;14];
R_rep = 5;
rng(100);
x_train = repmat(t_unique, R_rep, 1);
f_train = logistic(x_train);
sn_train = sigma_fn(x_train);
y_train = f_train + sn_train .* randn(size(x_train));

fprintf('Logistic growth: K=%.0f, r=%.3g, t0=%.3g, t in [%.0f, %.0f]\n', ...
    K, r, t0, t_min, t_max);
fprintf('Design: Nu=%d unique times, R=%d replicates (N=%d)\n', ...
    numel(t_unique), R_rep, numel(x_train));
fprintf('Bump noise: sigma_min=%.0f, A=%.0f, tc=%.3g, w=%.3g\n', ...
    sigma_min, A_bump, tc, w_bump);

%% Ground truth on plot grid
x_grid = linspace(t_min, t_max, 500)';
f_true = logistic(x_grid);
sigma_true = sigma_fn(x_grid);
noise_var_oracle_tr = sn_train.^2;
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

%% Model D: VHGPR
fprintf('\n=== Model D: VHGPR ===\n');
vboth = vhgpr_fit_predict(x_train, y_train, x_grid, struct('iter', vhgpr_iter));
fmu_d = vboth.fmu;
fs2_d = vboth.fs2;
sigma_d = vboth.sigma_n;
fprintf('D: VHGPR done (iter=%d)\n', vhgpr_iter);

%% Trajectory figure (one model per tab)
fig = figure('Color', 'w', 'Position', [80, 80, 900, 560], ...
    'Name', 'LG_het: logistic trajectories A-D');
tg = uitabgroup(fig);

tab_a = uitab(tg, 'Title', 'A: Homo GP');
ax_a = axes('Parent', tab_a);
plot_traj(ax_a, x_grid, f_true, fmu_a, fs2_a, x_train, y_train, k_plot, 'A: homoscedastic');

tab_b = uitab(tg, 'Title', 'B: Oracle Hetero');
ax_b = axes('Parent', tab_b);
plot_traj(ax_b, x_grid, f_true, fmu_b, fs2_b, x_train, y_train, k_plot, 'B: oracle hetero');

tab_c = uitab(tg, 'Title', 'C: Empirical s^2');
ax_c = axes('Parent', tab_c);
plot_traj(ax_c, x_grid, f_true, fmu_c, fs2_c, x_train, y_train, k_plot, 'C: empirical s^2');

tab_d = uitab(tg, 'Title', 'D: VHGPR');
ax_d = axes('Parent', tab_d);
plot_traj(ax_d, x_grid, f_true, fmu_d, fs2_d, x_train, y_train, k_plot, 'D: VHGPR');

%% Noise SD comparison
figure('Color', 'w', 'Position', [80, 120, 700, 420], ...
    'Name', 'LG_het: noise SD');
hold on; grid on;
plot(x_grid, sigma_true, 'k-', 'LineWidth', 2.2, 'DisplayName', 'True \sigma_n(t)');
plot(x_grid, sigma_a, 'r-.', 'LineWidth', 1.5, 'DisplayName', 'A (homo sn)');
plot(x_grid, sigma_b, 'g--', 'LineWidth', 1.5, 'DisplayName', 'B (oracle)');
plot(x_grid, sigma_c, 'm:', 'LineWidth', 1.6, 'DisplayName', 'C (empirical)');
plot(x_grid, sigma_d, 'b-', 'LineWidth', 1.5, 'DisplayName', 'D (VHGPR)');
xlabel('t'); ylabel('\sigma_n(t)');
title('Observation noise SD: true vs models');
legend('Location', 'best');
set(gca, 'FontSize', 12);

fprintf('\nDone.\n');

%% ==================== Local functions ====================
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
xlabel(ax, 't'); ylabel(ax, 'N(t)');
title(ax, title_str, 'Interpreter', 'none', 'FontSize', 16);
xlim(ax, [xg(1), xg(end)]);
legend(ax, 'Location', 'southeast');
set(ax, 'FontSize', 12);
ax.Layer = 'top';
end
