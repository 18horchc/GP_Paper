% microglia_VO.m — Independent NN GPs with soft virtual observations.
% Kernel: covNNone (neural-network / arcsin) throughout, including Solak deriv VOs.
% Tab 1: Independent NN vs Level VO at (t=0, y=5) on M1 and M2.
% Tab 2: Independent NN vs M1 Solak derivative VOs (soft rise prior); M2 independent.
% Tab 3: Independent NN vs Level VO + M1 Solak deriv VOs; M2 level VO only.
% Tab 4: Independent NN vs M2 Solak unimodal deriv VOs (early rise / late decline); M1 independent.
% Tab 5: Independent NN vs Level VO + M2 Solak unimodal deriv; M1 level VO only.
% Each tab: 2x2 — left = independent, right = method; top = full, bottom = averaged.
% Day 5 included in both M1 and M2.

clear; close all; clc;

%% ===== Configuration =====
k_plot    = 1.96;
max_iters = -200;
t_vo      = 0;
y_vo      = 5;
sigma_vo  = 0.5;   % low noise on level VO (count units)

% Soft Solak derivative VOs on M1 (cells/mm^2 per day) — soft rise prior
t_deriv_M1  = (1:2:13)';
y_deriv_M1  = 30 * ones(size(t_deriv_M1));
sn_deriv_M1 = 80;     % fixed; larger => weaker monotonicity pull

% Soft Solak derivative VOs on M2 — unimodal (early +, late -)
t_deriv_M2_early = (0:0.5:2)';
t_deriv_M2_late  = [8; 10; 12; 14];
t_deriv_M2  = [t_deriv_M2_early; t_deriv_M2_late];
y_deriv_M2  = [30 * ones(size(t_deriv_M2_early)); -80 * ones(size(t_deriv_M2_late))];
sn_deriv_M2 = 80;

%% ===== Data (day 5 restored) =====
newtime = [0, 1, 2, 3, 5, 7, 14];
datapointsM1 = [5, 27.5, 122.5, 139.8, 325, 445, 816.67];
newtimeM2 = [0, 1, 2, 3, 5, 7, 14];
datapointsM2 = [5, 78.33, 179.5, 126.4, 800, 319, 136.67];

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

timeM2 = [0, 1, 3, 5, 7, 14, ...
          1, 3, 7, ...
          2, ...
          14, ...
          3, ...
          2, ...
          0, 1, 3, 7, 14, ...
          3, 7];
dataM2 = [0, 170, 300, 800, 600, 200, ...
          15, 15, 6, ...
          90, ...
          110, ...
          57, ...
          269, ...
          10, 50, 100, 400, 100, ...
          160, 270];

datasets = struct( ...
    'name', {'full', 'averaged'}, ...
    'timeM1', {timeM1(:), newtime(:)}, ...
    'dataM1', {dataM1(:), datapointsM1(:)}, ...
    'timeM2', {timeM2(:), newtimeM2(:)}, ...
    'dataM2', {dataM2(:), datapointsM2(:)});

fprintf('=== Independent NN vs Level VO / M1 Deriv / M2 Deriv ===\n');
fprintf('Kernel: covNNone (NN Solak for derivative VOs)\n');
fprintf('Level VO: t=%.3g, y=%.3g, sigma=%.3g\n', t_vo, y_vo, sigma_vo);
fprintf('M1 Solak deriv: %d sites in [%.3g, %.3g] | y''=%.3g | sn=%.3g\n', ...
    numel(t_deriv_M1), t_deriv_M1(1), t_deriv_M1(end), y_deriv_M1(1), sn_deriv_M1);
fprintf('M2 Solak unimodal: early %s (y''=+%.3g) | late %s (y''=%.3g) | sn=%.3g\n', ...
    mat2str(t_deriv_M2_early(:)'), y_deriv_M2(1), ...
    mat2str(t_deriv_M2_late(:)'), y_deriv_M2(end), sn_deriv_M2);

%% ===== GPML setup =====
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
addpath(fileparts(fileparts(mfilename('fullpath'))));  % problems/ for gp_nn_*

meanfunc = @meanZero;
likfunc  = @likGauss;
tgrid    = (0:0.1:14)';
temporalKernel = @covNNone;

%% ===== Fit =====
results = struct([]);
for didx = 1:numel(datasets)
    ds = datasets(didx);
    fprintf('\n--- Dataset: %s ---\n', ds.name);

    naive = fit_naive_gp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot);
    naive_vo = fit_naive_gp_vo(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        t_vo, y_vo, sigma_vo, naive, tgrid, max_iters, k_plot);
    naive_deriv_M1 = fit_naive_gp_deriv_M1(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        t_deriv_M1, y_deriv_M1, sn_deriv_M1, naive, tgrid, temporalKernel, meanfunc, likfunc, ...
        max_iters, k_plot);
    naive_vo_deriv_M1 = fit_naive_gp_vo_deriv_M1(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        t_vo, y_vo, sigma_vo, t_deriv_M1, y_deriv_M1, sn_deriv_M1, naive, tgrid, max_iters, k_plot);
    naive_deriv_M2 = fit_naive_gp_deriv_M2(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        t_deriv_M2, y_deriv_M2, sn_deriv_M2, naive, tgrid, temporalKernel, meanfunc, likfunc, ...
        max_iters, k_plot);
    naive_vo_deriv_M2 = fit_naive_gp_vo_deriv_M2(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        t_vo, y_vo, sigma_vo, t_deriv_M2, y_deriv_M2, sn_deriv_M2, naive, tgrid, max_iters, k_plot);

    results(didx).name = ds.name;
    results(didx).naive = naive;
    results(didx).vo = naive_vo;
    results(didx).deriv_M1 = naive_deriv_M1;
    results(didx).vo_deriv_M1 = naive_vo_deriv_M1;
    results(didx).deriv_M2 = naive_deriv_M2;
    results(didx).vo_deriv_M2 = naive_vo_deriv_M2;
    results(didx).timeM1 = ds.timeM1;
    results(didx).dataM1 = ds.dataM1;
    results(didx).timeM2 = ds.timeM2;
    results(didx).dataM2 = ds.dataM2;

    report_fit(ds.name, 'Independent NN', naive.report);
    report_fit(ds.name, 'Level VO', naive_vo.report);
    report_fit(ds.name, 'M1 Deriv', naive_deriv_M1.report);
    report_fit(ds.name, 'Level VO + M1 Deriv', naive_vo_deriv_M1.report);
    report_fit(ds.name, 'M2 Deriv', naive_deriv_M2.report);
    report_fit(ds.name, 'Level VO + M2 Deriv', naive_vo_deriv_M2.report);
end

%% ===== Tabbed figure =====
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];
col_vo = [0.20, 0.45, 0.80];
col_deriv = [0.55, 0.25, 0.65];

fig = figure('Color', 'w', 'Position', [60, 60, 1240, 900], ...
    'Name', 'Microglia NN: Level VO / M1 Deriv / M2 Deriv');
tg = uitabgroup(fig);

tab1 = plot_comparison_tab(tg, 'Level VO', results, tgrid, ...
    'vo', 'Independent NN GP', sprintf('Level VO (0, %.3g)', y_vo), ...
    col_M1, col_M2, true, false, t_vo, y_vo, [], 'deriv VO times', col_vo, col_deriv);
plot_comparison_tab(tg, 'M1 Deriv', results, tgrid, ...
    'deriv_M1', 'Independent NN GP', 'M1 Solak deriv VOs', ...
    col_M1, col_M2, false, true, t_vo, y_vo, t_deriv_M1, 'M1 deriv VO times', col_vo, col_deriv);
plot_comparison_tab(tg, 'VO + M1 Deriv', results, tgrid, ...
    'vo_deriv_M1', 'Independent NN GP', sprintf('Level VO + M1 Deriv (0, %.3g)', y_vo), ...
    col_M1, col_M2, true, true, t_vo, y_vo, t_deriv_M1, 'M1 deriv VO times', col_vo, col_deriv);
plot_comparison_tab(tg, 'M2 Deriv', results, tgrid, ...
    'deriv_M2', 'Independent NN GP', 'M2 Solak unimodal deriv VOs', ...
    col_M1, col_M2, false, true, t_vo, y_vo, t_deriv_M2, 'M2 deriv VO times', col_vo, col_deriv);
plot_comparison_tab(tg, 'VO + M2 Deriv', results, tgrid, ...
    'vo_deriv_M2', 'Independent NN GP', sprintf('Level VO + M2 Deriv (0, %.3g)', y_vo), ...
    col_M1, col_M2, true, true, t_vo, y_vo, t_deriv_M2, 'M2 deriv VO times', col_vo, col_deriv);

tg.SelectedTab = tab1;
drawnow;

fprintf('\nDone.\n');

%% ===== Local functions =====

function tab = plot_comparison_tab(tg, tab_title, results, tgrid, ...
    right_field, left_title, right_title, ...
    col_M1, col_M2, mark_vo, mark_deriv, t_vo, y_vo, t_deriv, deriv_label, col_vo, col_deriv)

tab = uitab(tg, 'Title', tab_title);
tg.SelectedTab = tab;
tl = tiledlayout(tab, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
method_titles = {left_title, right_title};

for didx = 1:numel(results)
    ds = results(didx);
    methods = {ds.naive, ds.(right_field)};
    for midx = 1:2
        ax = nexttile(tl);
        ax.Layer = 'top';
        hold(ax, 'on'); grid(ax, 'on');
        fit = methods{midx};
        plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1');
        plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2');
        if midx == 2 && mark_vo
            scatter(ax, t_vo, y_vo, 110, 's', ...
                'MarkerFaceColor', col_vo, 'MarkerEdgeColor', 'k', ...
                'LineWidth', 1.2, 'DisplayName', sprintf('VO (0, %.3g)', y_vo));
        end
        if midx == 2 && mark_deriv && ~isempty(t_deriv)
            for iv = 1:numel(t_deriv)
                xline(ax, t_deriv(iv), ':', 'Color', col_deriv, 'LineWidth', 1.0, ...
                    'HandleVisibility', 'off');
            end
            plot(ax, NaN, NaN, ':', 'Color', col_deriv, 'LineWidth', 1.2, ...
                'DisplayName', deriv_label);
        end
        xlabel(ax, 'Time (days)');
        ylabel(ax, 'cells/mm^2');
        title(ax, sprintf('%s — %s', ds.name, method_titles{midx}), 'Interpreter', 'none');
        xlim(ax, [0, 14]);
        if midx == 2 && mark_vo
            ylim_auto_from_fit(ax, fit.M1, fit.M2, [ds.dataM1; y_vo], [ds.dataM2; y_vo]);
        else
            ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
        end
        legend(ax, 'Location', 'northwest', 'FontSize', 8);
    end
end
end

function [nTemp, hasAlpha] = temporal_hyp_layout(covfunc)
if iscell(covfunc) && strcmp(func2str(covfunc{1}), 'covMaterniso')
    nTemp = 2;
    hasAlpha = false;
elseif isa(covfunc, 'function_handle') && strcmp(func2str(covfunc), 'covRQiso')
    nTemp = 3;
    hasAlpha = true;
else
    nTemp = 2;
    hasAlpha = false;
end
end

function out = fit_naive_gp(timeM1, dataM1, timeM2, dataM2, tgrid, ...
    temporalKernel, meanfunc, likfunc, max_iters, k_plot)

y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);

fit1 = fit_single_gp(timeM1, y_M1, tgrid, temporalKernel, meanfunc, likfunc, max_iters);
fit2 = fit_single_gp(timeM2, y_M2, tgrid, temporalKernel, meanfunc, likfunc, max_iters);

out.M1 = pack_raw_fit(fit1.mu_y, fit1.sf_y, k_plot);
out.M2 = pack_raw_fit(fit2.mu_y, fit2.sf_y, k_plot);
out.hyp_M1 = fit1.hyp;
out.hyp_M2 = fit2.hyp;
out.nlml = fit1.nlml + fit2.nlml;

out.report.nlml = out.nlml;
out.report.ell_M1 = exp(fit1.hyp.cov(1));
out.report.ell_M2 = exp(fit2.hyp.cov(1));
out.report.sn_M1 = exp(fit1.hyp.lik);
out.report.sn_M2 = exp(fit2.hyp.lik);
end

function out = fit_naive_gp_vo(timeM1, dataM1, timeM2, dataM2, ...
    t_vo, y_vo, sigma_vo, naive_unc, tgrid, max_iters, k_plot)

y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);

fit1 = fit_single_gp_vo(timeM1, y_M1, t_vo, y_vo, sigma_vo, ...
    naive_unc.hyp_M1, tgrid, max_iters);
fit2 = fit_single_gp_vo(timeM2, y_M2, t_vo, y_vo, sigma_vo, ...
    naive_unc.hyp_M2, tgrid, max_iters);

out.M1 = pack_raw_fit(fit1.mu_y, fit1.sf_y, k_plot);
out.M2 = pack_raw_fit(fit2.mu_y, fit2.sf_y, k_plot);
out.hyp_M1 = fit1.hyp;
out.hyp_M2 = fit2.hyp;
out.nlml = fit1.nlml + fit2.nlml;

out.report.nlml = out.nlml;
out.report.ell_M1 = exp(fit1.hyp.cov(1));
out.report.ell_M2 = exp(fit2.hyp.cov(1));
out.report.sn_M1 = exp(naive_unc.hyp_M1.lik);
out.report.sn_M2 = exp(naive_unc.hyp_M2.lik);
end

function out = fit_naive_gp_deriv_M1(timeM1, dataM1, timeM2, dataM2, ...
    t_deriv, y_deriv, sn_deriv, naive_unc, tgrid, temporalKernel, meanfunc, likfunc, ...
    max_iters, k_plot)

y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);

fit1 = fit_single_gp_deriv(timeM1, y_M1, t_deriv, y_deriv, sn_deriv, ...
    naive_unc.hyp_M1, tgrid, max_iters);
fit2 = fit_single_gp(timeM2, y_M2, tgrid, temporalKernel, meanfunc, likfunc, max_iters);

out.M1 = pack_raw_fit(fit1.mu_y, fit1.sf_y, k_plot);
out.M2 = pack_raw_fit(fit2.mu_y, fit2.sf_y, k_plot);
out.hyp_M1 = fit1.hyp;
out.hyp_M2 = fit2.hyp;
out.nlml = fit1.nlml + fit2.nlml;

out.report.nlml = out.nlml;
out.report.ell_M1 = exp(fit1.hyp.cov(1));
out.report.ell_M2 = exp(fit2.hyp.cov(1));
out.report.sn_M1 = exp(naive_unc.hyp_M1.lik);
out.report.sn_M2 = exp(fit2.hyp.lik);
end

function out = fit_naive_gp_vo_deriv_M1(timeM1, dataM1, timeM2, dataM2, ...
    t_vo, y_vo, sigma_vo, t_deriv, y_deriv, sn_deriv, naive_unc, tgrid, max_iters, k_plot)

y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);

fit1 = fit_single_gp_vo_deriv(timeM1, y_M1, t_vo, y_vo, sigma_vo, ...
    t_deriv, y_deriv, sn_deriv, naive_unc.hyp_M1, tgrid, max_iters);
fit2 = fit_single_gp_vo(timeM2, y_M2, t_vo, y_vo, sigma_vo, ...
    naive_unc.hyp_M2, tgrid, max_iters);

out.M1 = pack_raw_fit(fit1.mu_y, fit1.sf_y, k_plot);
out.M2 = pack_raw_fit(fit2.mu_y, fit2.sf_y, k_plot);
out.hyp_M1 = fit1.hyp;
out.hyp_M2 = fit2.hyp;
out.nlml = fit1.nlml + fit2.nlml;

out.report.nlml = out.nlml;
out.report.ell_M1 = exp(fit1.hyp.cov(1));
out.report.ell_M2 = exp(fit2.hyp.cov(1));
out.report.sn_M1 = exp(naive_unc.hyp_M1.lik);
out.report.sn_M2 = exp(naive_unc.hyp_M2.lik);
end

function out = fit_naive_gp_deriv_M2(timeM1, dataM1, timeM2, dataM2, ...
    t_deriv, y_deriv, sn_deriv, naive_unc, tgrid, temporalKernel, meanfunc, likfunc, ...
    max_iters, k_plot)

y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);

fit1 = fit_single_gp(timeM1, y_M1, tgrid, temporalKernel, meanfunc, likfunc, max_iters);
fit2 = fit_single_gp_deriv(timeM2, y_M2, t_deriv, y_deriv, sn_deriv, ...
    naive_unc.hyp_M2, tgrid, max_iters);

out.M1 = pack_raw_fit(fit1.mu_y, fit1.sf_y, k_plot);
out.M2 = pack_raw_fit(fit2.mu_y, fit2.sf_y, k_plot);
out.hyp_M1 = fit1.hyp;
out.hyp_M2 = fit2.hyp;
out.nlml = fit1.nlml + fit2.nlml;

out.report.nlml = out.nlml;
out.report.ell_M1 = exp(fit1.hyp.cov(1));
out.report.ell_M2 = exp(fit2.hyp.cov(1));
out.report.sn_M1 = exp(fit1.hyp.lik);
out.report.sn_M2 = exp(naive_unc.hyp_M2.lik);
end

function out = fit_naive_gp_vo_deriv_M2(timeM1, dataM1, timeM2, dataM2, ...
    t_vo, y_vo, sigma_vo, t_deriv, y_deriv, sn_deriv, naive_unc, tgrid, max_iters, k_plot)

y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);

fit1 = fit_single_gp_vo(timeM1, y_M1, t_vo, y_vo, sigma_vo, ...
    naive_unc.hyp_M1, tgrid, max_iters);
fit2 = fit_single_gp_vo_deriv(timeM2, y_M2, t_vo, y_vo, sigma_vo, ...
    t_deriv, y_deriv, sn_deriv, naive_unc.hyp_M2, tgrid, max_iters);

out.M1 = pack_raw_fit(fit1.mu_y, fit1.sf_y, k_plot);
out.M2 = pack_raw_fit(fit2.mu_y, fit2.sf_y, k_plot);
out.hyp_M1 = fit1.hyp;
out.hyp_M2 = fit2.hyp;
out.nlml = fit1.nlml + fit2.nlml;

out.report.nlml = out.nlml;
out.report.ell_M1 = exp(fit1.hyp.cov(1));
out.report.ell_M2 = exp(fit2.hyp.cov(1));
out.report.sn_M1 = exp(naive_unc.hyp_M1.lik);
out.report.sn_M2 = exp(naive_unc.hyp_M2.lik);
end

function fit = fit_single_gp_vo(x, y, t_vo, y_vo, sigma_vo, hyp_unc, tgrid, max_iters)
x = x(:); y = y(:);
x_aug = [x; t_vo];
y_aug = [y; y_vo];
sn_real = exp(hyp_unc.lik);
noise_var = [sn_real^2 * ones(numel(y), 1); sigma_vo^2];

hyp0 = struct('mean', [], 'cov', hyp_unc.cov(:), 'lik', []);
obj = @(h) gp_nn_hetero_noise('nlml', h, x_aug, y_aug, noise_var);
hyp = minimize(hyp0, obj, max_iters);
nlml = obj(hyp);
[~, ~, fmu, fs2] = gp_nn_hetero_noise('pred', hyp, x_aug, y_aug, noise_var, tgrid(:));

fit.hyp = hyp;
fit.nlml = nlml;
fit.mu_y = fmu(:);
fit.sf_y = sqrt(max(fs2(:), 0));
end

function fit = fit_single_gp_deriv(x, y, t_deriv, y_deriv, sn_deriv, hyp_unc, tgrid, max_iters)
x = x(:); y = y(:);
t_deriv = t_deriv(:);
y_deriv = y_deriv(:);
sn_fixed = hyp_unc.lik;

obj = @(hyp_cov) gp_nn_deriv_obs_nlml_cov_only(hyp_cov, sn_fixed, ...
    x, y, t_deriv, y_deriv, sn_deriv);
hyp_cov = minimize(hyp_unc.cov(:), obj, max_iters);
hyp = struct('mean', [], 'cov', hyp_cov(:), 'lik', sn_fixed);
nlml = obj(hyp_cov);
[~, ~, fmu, fs2] = gp_nn_deriv_obs('pred', hyp, ...
    x, y, t_deriv, y_deriv, tgrid(:), sn_deriv);

fit.hyp = hyp;
fit.nlml = nlml;
fit.mu_y = fmu(:);
fit.sf_y = sqrt(max(fs2(:), 0));
end

function fit = fit_single_gp_vo_deriv(x, y, t_vo, y_vo, sigma_vo, ...
    t_deriv, y_deriv, sn_deriv, hyp_unc, tgrid, max_iters)
x = x(:); y = y(:);
t_deriv = t_deriv(:);
y_deriv = y_deriv(:);
x_aug = [x; t_vo];
y_aug = [y; y_vo];
sn_real = exp(hyp_unc.lik);
sn_fixed = hyp_unc.lik;
noise_var = [sn_real^2 * ones(numel(y), 1); sigma_vo^2];

obj = @(hyp_cov) gp_nn_deriv_obs_nlml_cov_only(hyp_cov, sn_fixed, ...
    x_aug, y_aug, t_deriv, y_deriv, sn_deriv, noise_var);
hyp_cov = minimize(hyp_unc.cov(:), obj, max_iters);
hyp = struct('mean', [], 'cov', hyp_cov(:), 'lik', sn_fixed);
nlml = obj(hyp_cov);
[~, ~, fmu, fs2] = gp_nn_deriv_obs('pred', hyp, ...
    x_aug, y_aug, t_deriv, y_deriv, tgrid(:), sn_deriv, true, noise_var);

fit.hyp = hyp;
fit.nlml = nlml;
fit.mu_y = fmu(:);
fit.sf_y = sqrt(max(fs2(:), 0));
end

function fit = fit_single_gp(x, y, tgrid, temporalKernel, meanfunc, likfunc, max_iters)
x = x(:); y = y(:);
inffunc = @infGaussLik;
ell0 = max(std(x), 0.5);
sf0 = max(std(y), 0.1);
sn0 = 0.1 * sf0;

[~, hasAlpha] = temporal_hyp_layout(temporalKernel);
if hasAlpha
    hyp.mean = [];
    hyp.cov = log([ell0; sf0; 1]);
else
    hyp.mean = [];
    hyp.cov = log([ell0; sf0]);
end
hyp.lik = log(sn0);

hyp = minimize(hyp, @gp, max_iters, inffunc, meanfunc, temporalKernel, likfunc, x, y);
nlml = gp(hyp, inffunc, meanfunc, temporalKernel, likfunc, x, y);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, temporalKernel, likfunc, x, y, tgrid);

fit.hyp = hyp;
fit.nlml = nlml;
fit.mu_y = fmu(:);
fit.sf_y = sqrt(max(fs2(:), 0));
end

function pheno = pack_raw_fit(mu, sf, k_plot)
pheno.mu = mu(:);
pheno.sf = sf(:);
pheno.lo = mu - k_plot .* sf;
pheno.hi = mu + k_plot .* sf;
end

function report_fit(dataset_name, method_name, report)
fprintf('[%s | %s] NLML=%.4f, ell_M1=%.4f, ell_M2=%.4f, sn_M1=%.4f, sn_M2=%.4f\n', ...
    dataset_name, method_name, report.nlml, report.ell_M1, report.ell_M2, ...
    report.sn_M1, report.sn_M2);
end

function plot_phenotype(ax, tgrid, fit, t_data, y_data, col, name)
tg = tgrid(:)';
fill(ax, [tg, fliplr(tg)], [fit.hi(:)', fliplr(fit.lo(:)')], col, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', sprintf('%s band', name));
plot(ax, tgrid, fit.mu, '--', 'Color', col, 'LineWidth', 2, ...
    'DisplayName', sprintf('%s mean', name));
scatter(ax, t_data, y_data, 36, 'filled', ...
    'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', ...
    'DisplayName', sprintf('%s data', name));
end

function ylim_auto_from_fit(ax, fitM1, fitM2, dataM1, dataM2)
vals = [fitM1.lo(:); fitM1.hi(:); fitM1.mu(:); ...
        fitM2.lo(:); fitM2.hi(:); fitM2.mu(:); ...
        dataM1(:); dataM2(:)];
pad = 0.05 * max(range(vals), 1);
ylim(ax, [min(vals) - pad, max(vals) + pad]);
end
