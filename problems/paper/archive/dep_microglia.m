% dep_microglia.m — Method 2: virtual observations on log-ratio r(t) = log1p(M1) - log1p(M2).
% Encodes M2 > M1 early and M1 > M2 late via soft virtual anchors (SE kernel, heteroscedastic).
clear; close all; clc;

%% Data (same as microglia.m; M2 day 5 excluded)

newtime = [0, 1, 2, 3, 5, 7, 14];
datapointsM1 = [5, 27.5, 122.5, 139.8, 325, 445, 816.67];
newtimeM2 = [0, 1, 2, 3, 7, 14];
datapointsM2 = [5, 78.33, 179.5, 126.4, 319, 136.67];

timeM1 = [0, 1, 3, 5, 7, 14, ...
          3, 7, ...
          2, ...
          14, ...
          3, ...
          2, ...
          0, 1, 3, 7, 14, ...
          3, 7];
dataM1 = [0, 5, 375, 325, 600, 750, ...
          62, 55, ...
          120, ...
          400, ...
          102, ...
          125, ...
          10, 50, 100, 900, 1300, ...
          60, 225];

timeM2 = [0, 1, 3, 7, 14, ...
          1, 3, 7, ...
          2, ...
          14, ...
          3, ...
          2, ...
          0, 1, 3, 7, 14, ...
          3, 7];
dataM2 = [0, 170, 300, 600, 200, ...
          15, 15, 6, ...
          90, ...
          110, ...
          57, ...
          269, ...
          10, 50, 100, 400, 100, ...
          160, 270];

%% Virtual-observation parameters (log-ratio prior)

t_cross = 5;              % crossover: r(t_cross) = 0
delta_r = 0.5;            % virtual |r| early/late (log-scale dominance)
n_virt_early = 6;         % anchors on [0, t_cross)
n_virt_late  = 6;         % anchors on (t_cross, 14]
sigma_virt_soft = 0.4;    % std of early/late virtual obs (larger = softer)
sigma_virt_cross = 0.05;  % tight anchor at r(t_cross) = 0
sigma_real = [];          % if empty, set from baseline GP on real r
n_mc_coupled = 3000;      % MC samples for coupled cells/mm^2 uncertainty bands

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
addpath(fileparts(fileparts(mfilename('fullpath'))));  % problems/ (gp_seiso_hetero_noise)

meanfunc = @meanZero;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

tgrid = (0:0.1:14)';
k_plot = 1.96;

%% Averaged data: log-ratio GP with virtual observations

fprintf('\n=== Averaged data: log-ratio GP ===\n');
[t_avg, r_avg] = build_paired_log_ratio_avg(newtime, datapointsM1, newtimeM2, datapointsM2);
run_ratio_experiment(t_avg, r_avg, tgrid, t_cross, delta_r, n_virt_early, n_virt_late, ...
    sigma_virt_soft, sigma_virt_cross, sigma_real, ...
    inffunc, meanfunc, covfunc, likfunc, k_plot, n_mc_coupled, ...
    newtime, datapointsM1, newtimeM2, datapointsM2, 201, 203, 'Averaged');

%% Full data: log-ratio GP with virtual observations

fprintf('\n=== Full data: log-ratio GP ===\n');
[t_full, r_full] = build_paired_log_ratio_full(timeM1, dataM1, timeM2, dataM2);
run_ratio_experiment(t_full, r_full, tgrid, t_cross, delta_r, n_virt_early, n_virt_late, ...
    sigma_virt_soft, sigma_virt_cross, sigma_real, ...
    inffunc, meanfunc, covfunc, likfunc, k_plot, n_mc_coupled, ...
    timeM1, dataM1, timeM2, dataM2, 202, 204, 'Full');


%% Local helpers

function [t, r] = build_paired_log_ratio_avg(timeM1, yM1, timeM2, yM2)
% Paired log-ratio at times where both M1 and M2 averaged observations exist.
timeM1 = timeM1(:);
yM1 = yM1(:);
timeM2 = timeM2(:);
yM2 = yM2(:);
shared = intersect(timeM1, timeM2);
t = shared(:);
r = zeros(numel(t), 1);
for i = 1:numel(t)
    m1 = yM1(timeM1 == t(i));
    m2 = yM2(timeM2 == t(i));
    r(i) = log1p(m1(1)) - log1p(m2(1));
end
end

function [t, r] = build_paired_log_ratio_full(timeM1, yM1, timeM2, yM2)
% Paired log-ratio at shared times using per-time means of full data.
timeM1 = timeM1(:);
yM1 = yM1(:);
timeM2 = timeM2(:);
yM2 = yM2(:);
shared = intersect(unique(timeM1), unique(timeM2));
t = sort(shared(:));
r = zeros(numel(t), 1);
for i = 1:numel(t)
    m1 = mean(yM1(timeM1 == t(i)));
    m2 = mean(yM2(timeM2 == t(i)));
    r(i) = log1p(m1) - log1p(m2);
end
end

function run_ratio_experiment(t_real, r_real, tgrid, t_cross, delta_r, n_virt_early, n_virt_late, ...
    sigma_virt_soft, sigma_virt_cross, sigma_real, ...
    inffunc, meanfunc, covfunc, likfunc, k_plot, n_mc, ...
    timeM1_data, yM1_data, timeM2_data, yM2_data, fig_num_ratio, fig_num_cells, label)
% Baseline GP on real r; augmented GP with virtual ordering anchors; plot comparison.

t_real = t_real(:);
r_real = r_real(:);
n_real = numel(t_real);
timeM1_data = timeM1_data(:);
yM1_data = yM1_data(:);
timeM2_data = timeM2_data(:);
yM2_data = yM2_data(:);

% Baseline fit on real paired observations only
[hyp_base, mu_base, s2_base, nlml_base] = fit_ratio_baseline( ...
    t_real, r_real, tgrid, inffunc, meanfunc, covfunc, likfunc);

if isempty(sigma_real)
    sigma_real_use = exp(hyp_base.lik);
else
    sigma_real_use = sigma_real;
end

% Virtual anchor times and targets
if n_virt_early > 0
    t_virt_early = linspace(0, max(0, t_cross - 0.1), n_virt_early)';
else
    t_virt_early = zeros(0, 1);
end
if n_virt_late > 0
    t_virt_late = linspace(min(14, t_cross + 0.1), 14, n_virt_late)';
else
    t_virt_late = zeros(0, 1);
end
t_virt_cross = t_cross;
r_virt_early = -delta_r * ones(numel(t_virt_early), 1);
r_virt_cross = 0;
r_virt_late  =  delta_r * ones(numel(t_virt_late), 1);

x_aug = [t_real; t_virt_early; t_virt_cross; t_virt_late];
y_aug = [r_real; r_virt_early; r_virt_cross; r_virt_late];
noise_var_aug = [sigma_real_use^2 * ones(n_real, 1); ...
    sigma_virt_soft^2 * ones(numel(t_virt_early) + numel(t_virt_late), 1); ...
    sigma_virt_cross^2];

[hyp_aug, mu_aug, s2_aug, nlml_aug] = fit_ratio_virtual_obs( ...
    x_aug, y_aug, noise_var_aug, tgrid, hyp_base);

% Unconstrained original-scale fits set the overall M1/M2 level for the
% symmetric reconstruction; the augmented ratio controls their separation.
[~, mu_M1_unc, s2_M1_unc] = fit_gp(timeM1_data, yM1_data, tgrid, ...
    inffunc, meanfunc, covfunc, likfunc);
[~, mu_M2_unc, s2_M2_unc] = fit_gp(timeM2_data, yM2_data, tgrid, ...
    inffunc, meanfunc, covfunc, likfunc);

z_mu = 0.5 * (log1p(max(0, mu_M1_unc)) + log1p(max(0, mu_M2_unc)));
mu_M1_coupled = expm1(z_mu + 0.5 * mu_aug);
mu_M2_coupled = expm1(z_mu - 0.5 * mu_aug);

sd_M1_unc = sqrt(max(s2_M1_unc, 0));
sd_M2_unc = sqrt(max(s2_M2_unc, 0));
sd_aug = sqrt(max(s2_aug, 0));
[lo_M1_coupled, hi_M1_coupled, lo_M2_coupled, hi_M2_coupled] = coupled_band_mc( ...
    mu_M1_unc, sd_M1_unc, mu_M2_unc, sd_M2_unc, mu_aug, sd_aug, n_mc);

fprintf('%s: n_real=%d | virtual early=%d cross=1 late=%d\n', ...
    label, n_real, numel(t_virt_early), numel(t_virt_late));
fprintf('  t_cross=%.2f | delta_r=%.3g | sigma_real=%.4g | sigma_virt_soft=%.4g | sigma_virt_cross=%.4g\n', ...
    t_cross, delta_r, sigma_real_use, sigma_virt_soft, sigma_virt_cross);
fprintf('  Baseline: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_base.cov(1)), exp(hyp_base.cov(2)), exp(hyp_base.lik), nlml_base);
fprintf('  Augmented: ell=%.4f, sf=%.4f | NLML=%.4f (heteroscedastic, n_aug=%d)\n', ...
    exp(hyp_aug.cov(1)), exp(hyp_aug.cov(2)), nlml_aug, numel(y_aug));

sd_base = sqrt(max(s2_base, 0));
ylim_r = [min([r_real; mu_base - k_plot * sd_base; mu_aug - k_plot * sd_aug]) - 0.15, ...
          max([r_real; mu_base + k_plot * sd_base; mu_aug + k_plot * sd_aug]) + 0.15];

virt_meta = struct( ...
    't_early', t_virt_early, 'r_early', r_virt_early, ...
    't_cross', t_virt_cross, 'r_cross', r_virt_cross, ...
    't_late', t_virt_late, 'r_late', r_virt_late);

figure(fig_num_ratio)
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

plot_ratio_panel(nexttile, tgrid, mu_base, sd_base, t_real, r_real, virt_meta, ...
    t_cross, k_plot, ylim_r, sprintf('%s: baseline log-ratio GP', label), false);
plot_ratio_panel(nexttile, tgrid, mu_aug, sd_aug, t_real, r_real, virt_meta, ...
    t_cross, k_plot, ylim_r, sprintf('%s: virtual-obs log-ratio GP', label), true);

ylim_cells = [0, max([yM1_data; yM2_data; mu_M1_unc; mu_M2_unc; ...
    mu_M1_coupled; mu_M2_coupled; hi_M1_coupled; hi_M2_coupled]) * 1.05];

figure(fig_num_cells)
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

plot_cells_panel(nexttile, tgrid, mu_M1_coupled, lo_M1_coupled, hi_M1_coupled, mu_M1_unc, ...
    timeM1_data, yM1_data, 'k', ...
    {sprintf('%s: coupled M1 from virtual-obs ratio (SE)', label), ...
    'Symmetric reconstruction: z + r/2'}, k_plot, ylim_cells);
plot_cells_panel(nexttile, tgrid, mu_M2_coupled, lo_M2_coupled, hi_M2_coupled, mu_M2_unc, ...
    timeM2_data, yM2_data, 'r', ...
    {sprintf('%s: coupled M2 from virtual-obs ratio (SE)', label), ...
    'Symmetric reconstruction: z - r/2'}, k_plot, ylim_cells);
end

function [lo_M1, hi_M1, lo_M2, hi_M2] = coupled_band_mc( ...
    mu_M1, sd_M1, mu_M2, sd_M2, mu_r, sd_r, n_mc)
% Approximate 95% coupled bands via MC: sample unc M1/M2 and augmented r, apply z +/- r/2.
mu_M1 = mu_M1(:);
mu_M2 = mu_M2(:);
mu_r = mu_r(:);
sd_M1 = sd_M1(:);
sd_M2 = sd_M2(:);
sd_r = sd_r(:);
n_t = numel(mu_M1);

s1 = mu_M1 + sd_M1 .* randn(n_t, n_mc);
s2 = mu_M2 + sd_M2 .* randn(n_t, n_mc);
sr = mu_r + sd_r .* randn(n_t, n_mc);
z_s = 0.5 * (log1p(max(0, s1)) + log1p(max(0, s2)));
m1_s = expm1(z_s + 0.5 * sr);
m2_s = expm1(z_s - 0.5 * sr);

lo_M1 = quantile(m1_s, 0.025, 2);
hi_M1 = quantile(m1_s, 0.975, 2);
lo_M2 = quantile(m2_s, 0.025, 2);
hi_M2 = quantile(m2_s, 0.975, 2);
lo_M1 = max(0, lo_M1(:));
lo_M2 = max(0, lo_M2(:));
hi_M1 = hi_M1(:);
hi_M2 = hi_M2(:);
end

function plot_ratio_panel(ax, tgrid, mu, sd, t_real, r_real, virt, t_cross, k_plot, ylim_r, title_str, show_virt)
axes(ax);
hold on
fill([tgrid; flipud(tgrid)], [mu + k_plot * sd; flipud(mu - k_plot * sd)], ...
    [0.72, 0.72, 0.78], 'FaceAlpha', 0.5, 'EdgeColor', 'none', ...
    'DisplayName', sprintf('95%% band (\\mu_f \\pm %.2g\\sigma_f)', k_plot));
plot(tgrid, mu, 'k--', 'LineWidth', 2.0, 'DisplayName', 'Posterior mean r(t)');
scatter(t_real, r_real, 90, 'ko', 'filled', 'DisplayName', 'Paired data');
if show_virt
    if ~isempty(virt.t_early)
        scatter(virt.t_early, virt.r_early, 90, 's', ...
            'MarkerFaceColor', [0.85, 0.85, 0.85], 'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.2, 'DisplayName', 'Virtual: M2 > M1 (early)');
    end
    scatter(virt.t_cross, virt.r_cross, 110, 'd', ...
        'MarkerFaceColor', [0.85, 0.85, 0.85], 'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.2, 'DisplayName', 'Virtual: crossover r=0');
    if ~isempty(virt.t_late)
        scatter(virt.t_late, virt.r_late, 90, 's', ...
            'MarkerFaceColor', [0.85, 0.85, 0.85], 'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.2, 'DisplayName', 'Virtual: M1 > M2 (late)');
    end
end
yline(0, 'k:', 'HandleVisibility', 'off');
xline(t_cross, 'k:', 'HandleVisibility', 'off');
hold off
xlabel('Time (Days)', 'fontsize', 18)
ylabel('r(t) = log1p(M_1) - log1p(M_2)', 'fontsize', 18)
title({title_str, 'r < 0: M_2 > M_1  |  r > 0: M_1 > M_2'}, 'fontsize', 16)
legend('Location', 'best')
xlim([0, 14])
ylim(ylim_r)
grid on
set(gca, 'fontsize', 16)
end

function plot_cells_panel(ax, tgrid, mu_coupled, lo_coupled, hi_coupled, mu_unc, time_data, y_data, line_color, title_str, k_plot, ylim_cells)
axes(ax);
hold on
if line_color == 'r'
    band_color = [1.0, 0.75, 0.75];
else
    band_color = [0.75, 0.75, 0.75];
end
fill([tgrid; flipud(tgrid)], [hi_coupled; flipud(lo_coupled)], ...
    band_color, 'FaceAlpha', 0.35, 'EdgeColor', 'none', ...
    'DisplayName', 'Coupled 95% band (MC)');
plot(tgrid, mu_unc, '--', 'Color', line_color, 'LineWidth', 1.2, ...
    'DisplayName', 'Unconstrained GP mean');
plot(tgrid, mu_coupled, '-', 'Color', line_color, 'LineWidth', 2.4, ...
    'DisplayName', 'Coupled mean');
s = scatter(time_data, y_data, line_color, 'filled', 'DisplayName', 'Actual data');
s.Marker = 'hexagram';
s.SizeData = 150;
yline(0, 'k:', 'HandleVisibility', 'off');
hold off
xlabel('Time (Days)', 'fontsize', 18)
ylabel('cells/mm^2', 'fontsize', 18)
title(title_str, 'fontsize', 16)
legend('Location', 'best')
xlim([0, 14])
ylim(ylim_cells)
grid on
set(gca, 'fontsize', 16)
end

function [hyp, mu, s2, nlml] = fit_ratio_baseline(x, y, xs, inffunc, meanfunc, covfunc, likfunc)
[hyp, mu, s2] = fit_gp(x, y, xs, inffunc, meanfunc, covfunc, likfunc);
nlml = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x(:), y(:));
end

function [hyp, mu, s2, nlml] = fit_ratio_virtual_obs(x, y, noise_var, xs, hyp0)
x = x(:); y = y(:); noise_var = noise_var(:); xs = xs(:);
hyp_tpl = struct('mean', [], 'cov', hyp0.cov(:), 'lik', []);
obj = @(h) gp_seiso_hetero_noise('nlml', h, x, y, noise_var);
hyp = minimize(hyp_tpl, obj, -100);
nlml = obj(hyp);
[~, ~, mu, s2] = gp_seiso_hetero_noise('pred', hyp, x, y, noise_var, xs);
mu = mu(:);
s2 = s2(:);
end

function [hyp, mu, s2] = fit_gp(x, y, xs, inffunc, meanfunc, covfunc, likfunc)
x = x(:); y = y(:); xs = xs(:);
ell0 = max(std(x), 1e-3);
sf0  = max(std(y), 1e-3);
sn0  = max(0.1 * std(y), 1e-3);
hyp.mean = [];
hyp.cov  = log([ell0; sf0]);
hyp.lik  = log(sn0);
hyp = minimize(hyp, @gp, -100, inffunc, meanfunc, covfunc, likfunc, x, y);
[mu, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
end
