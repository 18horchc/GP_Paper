% microglia.m — Independent GPML GPs for M1/M2 across kernels.
% Kernels: SE, Matern 3/2, Matern 5/2, Neural Network (covNNone).
% Figure 1: outer tabs = full / averaged; inner tabs = kernel.
% Extra inner tab per dataset: NN extrapolated to day 60 (no re-fit).
% Figure 2: NN Pensoneault lower bound (f>0); outer tabs = full / averaged.
% Figure 3: NN noise comparison (homoscedastic sn | empirical sigma^2(t) | VHGPR);
%   extra inner tabs: empirical NN to day 60; empirical NN + Pensoneault bound (to day 60).
% M1 and M2 plotted on the same axes. Day 5 included in both phenotypes.
% Shared legend printed once at the end. Per-panel EPS exports follow LV_Periodic.m.

clear; close all; clc;

%% ===== Configuration =====
k_plot    = 1.96;
max_iters = -200;
tgrid     = (0:0.1:14)';
tgrid_ext = (0:0.1:60)';   % NN extrapolation only; training window unchanged

% Pensoneault lower bound (Figure 2, NN only): mu_f - k*sigma_f >= 0 at X_c
eta_pens      = 0.022;
k_pens        = -sqrt(2) * erfinv(2 * eta_pens - 1);
n_constraint  = 41;
X_c           = linspace(0, 14, n_constraint)';
ell_bounds_lo = 0.05;
ell_ub        = 14;
nTry          = 2000;
nMultistart   = 10;
opts_pens = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, 'Display', 'off', ...
    'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);

%% ===== Data  =====
newtime = [0, 1, 2, 3, 5, 7, 14];
datapointsM1 = [5, 27.5, 122.5, 139.8, 325, 445, 816.67];
newtimeM2 = [0, 1, 2, 3, 5, 7, 14];
datapointsM2 = [5, 78.33, 179.5, 126.4, 800, 319, 136.67];

timeM1 = [0, 1, 3, 5, 7, 14, ...
          3, 7, ...
          2, ...
          14, ...
          3, ...
          3, ...
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
          3, ...
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
    'title', {'Full', 'Averaged'}, ...
    'timeM1', {timeM1(:), newtime(:)}, ...
    'dataM1', {dataM1(:), datapointsM1(:)}, ...
    'timeM2', {timeM2(:), newtimeM2(:)}, ...
    'dataM2', {dataM2(:), datapointsM2(:)});

kernels = struct( ...
    'name',  {'se', 'matern32', 'matern52', 'nn'}, ...
    'title', {'SE', 'Matern 3/2', 'Matern 5/2', 'NN'}, ...
    'cov',   {@covSEiso, {@covMaterniso, 3}, {@covMaterniso, 5}, @covNNone});

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

meanfunc = @meanZero;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

fprintf('=== Microglia: independent GP kernel comparison ===\n');
fprintf('Kernels: SE | Matern 3/2 | Matern 5/2 | NN\n');

%% ===== Fit =====
results = struct([]);
for didx = 1:numel(datasets)
    ds = datasets(didx);
    fprintf('\n--- Dataset: %s ---\n', ds.name);
    results(didx).name = ds.name;
    results(didx).title = ds.title;
    results(didx).timeM1 = ds.timeM1;
    results(didx).dataM1 = ds.dataM1;
    results(didx).timeM2 = ds.timeM2;
    results(didx).dataM2 = ds.dataM2;
    results(didx).fits = struct([]);

    for kidx = 1:numel(kernels)
        ker = kernels(kidx);
        fit = fit_naive_gp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
            tgrid, inffunc, meanfunc, ker.cov, likfunc, max_iters, k_plot);
        results(didx).fits(kidx).name = ker.name;
        results(didx).fits(kidx).title = ker.title;
        results(didx).fits(kidx).fit = fit;
        if strcmp(ker.name, 'nn')
            results(didx).fits(kidx).fit_ext = predict_naive_gp(fit, ...
                ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, tgrid_ext, ...
                inffunc, meanfunc, ker.cov, likfunc, k_plot);
        end
        report_fit(ds.name, ker.title, fit.report);
    end
end

%% ===== Tabbed figure: outer = dataset, inner = kernel =====
% Per-panel legends omitted; one shared legend is drawn after all figures.
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];
band_label = '95% CI';

fig = figure('Color', 'w', 'Position', [60, 60, 1240, 900], ...
    'Name', 'Microglia: GP kernel comparison (M1 & M2)');
tg_outer = uitabgroup(fig);

n_panels = numel(results) * (numel(kernels) + 1);  % +1 NN-to-day-60 tab per dataset
ax_list  = gobjects(n_panels, 1);
tab_list = gobjects(n_panels, 1);
fname_list = cell(n_panels, 1);
pidx = 0;

for didx = 1:numel(results)
    ds = results(didx);
    tab_ds = uitab(tg_outer, 'Title', ds.title);
    tg_inner = uitabgroup(tab_ds);

    for kidx = 1:numel(ds.fits)
        kf = ds.fits(kidx);
        tab_k = uitab(tg_inner, 'Title', kf.title);
        ax = axes('Parent', tab_k);
        ax.Layer = 'top';
        ax.FontSize = 24;
        hold(ax, 'on'); grid(ax, 'on');
        fit = kf.fit;
        plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1', band_label);
        plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2', band_label);
        xlabel(ax, 'Time (days)', 'FontSize', 24);
        ylabel(ax, 'cells/mm^2', 'FontSize', 24);
        title(ax, sprintf('%s — %s', ds.title, kf.title), 'Interpreter', 'none', 'FontSize', 24);
        xlim(ax, [0, 14]);
        ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);

        pidx = pidx + 1;
        ax_list(pidx) = ax;
        tab_list(pidx) = tab_k;
        fname_list{pidx} = sprintf('Microglia_%s_%s.eps', ds.title, ...
            regexprep(kf.title, '[^A-Za-z0-9]+', ''));
    end

    nn_idx = find(strcmp({ds.fits.name}, 'nn'), 1);
    kf = ds.fits(nn_idx);
    tab_k = uitab(tg_inner, 'Title', 'NN (to day 60)');
    ax = axes('Parent', tab_k);
    ax.Layer = 'top';
    ax.FontSize = 24;
    hold(ax, 'on'); grid(ax, 'on');
    fit = kf.fit_ext;
    plot_phenotype(ax, tgrid_ext, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1', band_label);
    plot_phenotype(ax, tgrid_ext, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2', band_label);
    xlabel(ax, 'Time (days)', 'FontSize', 24);
    ylabel(ax, 'cells/mm^2', 'FontSize', 24);
    title(ax, sprintf('%s — NN (to day 60)', ds.title), 'Interpreter', 'none', 'FontSize', 24);
    xlim(ax, [0, 60]);
    ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);

    pidx = pidx + 1;
    ax_list(pidx) = ax;
    tab_list(pidx) = tab_k;
    fname_list{pidx} = sprintf('Microglia_%s_NNday60.eps', ds.title);
end

% %% ===== Save each panel as EPS =====
% plot_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
%     'results', 'plots', 'Paper Draft 2', 'Microglia', 'Kernels');
% if ~exist(plot_dir, 'dir')
%     mkdir(plot_dir);
% end
% for i = 1:n_panels
%     % Select the containing outer/inner tabs so the axes are visible for export
%     tg_outer.SelectedTab = tab_list(i).Parent.Parent;
%     tab_list(i).Parent.SelectedTab = tab_list(i);
%     ax_list(i).Toolbar.Visible = 'off';
%     disableDefaultInteractivity(ax_list(i));
%     drawnow;
%     out_path = fullfile(plot_dir, fname_list{i});
%     exportgraphics(ax_list(i), out_path, 'ContentType', 'image');
%     fprintf('Saved %s\n', out_path);
% end

%% ===== Fit NN Pensoneault lower bound (Figure 2) =====
nn_idx = find(strcmp({kernels.name}, 'nn'), 1);
cov_nn = kernels(nn_idx).cov;
fprintf('\n=== NN Pensoneault lower bound (f > 0) ===\n');
fprintf('eta = %.3g%% | k = %.4f | X_c: %d points on [0, 14]\n', ...
    100 * eta_pens, k_pens, n_constraint);

for didx = 1:numel(results)
    ds = results(didx);
    naive = ds.fits(nn_idx).fit;
    sf_bounds_M1 = [0.05, max(15, 1.5 * std(ds.dataM1))];
    sf_bounds_M2 = [0.05, max(15, 1.5 * std(ds.dataM2))];
    hyp_lb_M1 = log([ell_bounds_lo; sf_bounds_M1(1)]);
    hyp_ub_M1 = log([ell_ub; sf_bounds_M1(2)]);
    hyp_lb_M2 = log([ell_bounds_lo; sf_bounds_M2(1)]);
    hyp_ub_M2 = log([ell_ub; sf_bounds_M2(2)]);

    bound = fit_naive_gp_lower_bound(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        naive.hyp_M1, naive.hyp_M2, tgrid, X_c, k_pens, k_plot, ...
        inffunc, meanfunc, cov_nn, likfunc, hyp_lb_M1, hyp_ub_M1, hyp_lb_M2, hyp_ub_M2, ...
        opts_pens, nTry, nMultistart, 40 + 2*didx, 41 + 2*didx);
    results(didx).nn_bound = bound;
    results(didx).nn_bound_ext = predict_naive_gp(bound, ...
        ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, tgrid_ext, ...
        inffunc, meanfunc, cov_nn, likfunc, k_plot);
    report_bound(ds.name, 'NN', bound.report);
end

%% ===== Figure 2: NN lower bound; outer = dataset, inner = horizon =====
fig2 = figure('Color', 'w', 'Position', [80, 80, 1240, 900], ...
    'Name', 'Microglia: NN Pensoneault lower bound (M1 & M2)');
tg_outer2 = uitabgroup(fig2);

for didx = 1:numel(results)
    ds = results(didx);
    tab_ds = uitab(tg_outer2, 'Title', ds.title);
    tg_inner = uitabgroup(tab_ds);

    tab_k = uitab(tg_inner, 'Title', 'NN');
    ax = axes('Parent', tab_k);
    ax.Layer = 'top';
    ax.FontSize = 24;
    hold(ax, 'on'); grid(ax, 'on');
    fit = ds.nn_bound;
    plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1', band_label);
    plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2', band_label);
    yline(ax, 0, 'k:', 'HandleVisibility', 'off');
    xlabel(ax, 'Time (days)', 'FontSize', 24);
    ylabel(ax, 'cells/mm^2', 'FontSize', 24);
    title(ax, sprintf('%s — NN (lower bound)', ds.title), 'Interpreter', 'none', 'FontSize', 24);
    xlim(ax, [0, 14]);
    ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);

    tab_k = uitab(tg_inner, 'Title', 'NN (to day 60)');
    ax = axes('Parent', tab_k);
    ax.Layer = 'top';
    ax.FontSize = 24;
    hold(ax, 'on'); grid(ax, 'on');
    fit = ds.nn_bound_ext;
    plot_phenotype(ax, tgrid_ext, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1', band_label);
    plot_phenotype(ax, tgrid_ext, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2', band_label);
    yline(ax, 0, 'k:', 'HandleVisibility', 'off');
    xlabel(ax, 'Time (days)', 'FontSize', 24);
    ylabel(ax, 'cells/mm^2', 'FontSize', 24);
    title(ax, sprintf('%s — NN (lower bound, to day 60)', ds.title), ...
        'Interpreter', 'none', 'FontSize', 24);
    xlim(ax, [0, 60]);
    ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
end

%% ===== Fit NN noise comparison (Figure 3; does not refit Figures 1–2) =====
% Homoscedastic: reuse unconstrained NN from Figure 1 (single learned sn).
% Empirical: diagonal noise = replicate sample variance at each t (n=1 uses
% pooled variance from times with n>=2, else var(y)); ell/sf by NLML.
% VHGPR: Lazaro-Gredilla & Titsias (independent M1/M2).
vhgpr_iter = 40;
problems_dir = fileparts(fileparts(mfilename('fullpath')));
addpath(problems_dir);
if ~exist('gp_nn_hetero_noise', 'file')
    error('microglia:MissingHelper', 'gp_nn_hetero_noise.m not found on path.');
end
if ~exist('vhgpr_fit_predict', 'file')
    error('microglia:MissingVHGPR', 'vhgpr_fit_predict.m not found on path.');
end

fprintf('\n=== NN noise comparison: homoscedastic sn | empirical sigma^2(t) | VHGPR ===\n');
fprintf('Empirical: n=1 times use pooled replicate variance (*). VHGPR iter=%d\n', vhgpr_iter);

for didx = 1:numel(results)
    ds = results(didx);
    fprintf('\n--- Dataset: %s (NN noise) ---\n', ds.name);
    results(didx).nn_homo = ds.fits(nn_idx).fit;
    results(didx).nn_emp = fit_emp_nn_pair(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, max_iters, k_plot);
    results(didx).nn_emp_ext = predict_emp_nn_pair(results(didx).nn_emp, ...
        ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, tgrid_ext, k_plot);
    sf_bounds_M1 = [0.05, max(15, 1.5 * std(ds.dataM1))];
    sf_bounds_M2 = [0.05, max(15, 1.5 * std(ds.dataM2))];
    hyp_lb_M1 = log([ell_bounds_lo; sf_bounds_M1(1)]);
    hyp_ub_M1 = log([ell_ub; sf_bounds_M1(2)]);
    hyp_lb_M2 = log([ell_bounds_lo; sf_bounds_M2(1)]);
    hyp_ub_M2 = log([ell_ub; sf_bounds_M2(2)]);
    X_c_ext = linspace(14, 60, n_constraint)';
    fprintf('  Empirical NN + Pensoneault lower bound to day 60 (no data tube)\n');
    fprintf('  X_c: %d points on [0, 60]\n', n_constraint);
    results(didx).nn_emp_bound = fit_emp_nn_lower_bound_pair( ...
        ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, results(didx).nn_emp, ...
        tgrid_ext, X_c_ext, k_pens, k_plot, hyp_lb_M1, hyp_ub_M1, hyp_lb_M2, hyp_ub_M2, ...
        opts_pens, nTry, nMultistart, 60 + 2*didx, 61 + 2*didx);
    results(didx).nn_vhgpr = fit_vhgpr_pair(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, vhgpr_iter, k_plot);
    report_fit(ds.name, 'NN homo sn', results(didx).nn_homo.report);
    report_emp_nn(ds.name, results(didx).nn_emp);
    report_emp_bound(ds.name, results(didx).nn_emp_bound.report);
    report_vhgpr_nn(ds.name, results(didx).nn_vhgpr);
end

%% ===== Figure 3: NN noise models; outer = dataset, inner = noise method =====
fig3 = figure('Color', 'w', 'Position', [100, 100, 1240, 900], ...
    'Name', 'Microglia: NN homoscedastic sn vs empirical vs VHGPR');
tg_outer3 = uitabgroup(fig3);

noise_tab_titles = {'Homoscedastic sn', 'Empirical sigma^2(t)', 'VHGPR'};
noise_panel_titles = {'NN (homoscedastic sn)', 'NN (empirical sigma^2(t))', 'VHGPR'};

for didx = 1:numel(results)
    ds = results(didx);
    tab_ds = uitab(tg_outer3, 'Title', ds.title);
    tg_inner = uitabgroup(tab_ds);
    noise_fits = {ds.nn_homo, ds.nn_emp, ds.nn_vhgpr};

    for midx = 1:numel(noise_fits)
        tab_k = uitab(tg_inner, 'Title', noise_tab_titles{midx});
        ax = axes('Parent', tab_k);
        ax.Layer = 'top';
        ax.FontSize = 24;
        hold(ax, 'on'); grid(ax, 'on');
        fit = noise_fits{midx};
        plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1', band_label);
        plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2', band_label);
        xlabel(ax, 'Time (days)', 'FontSize', 24);
        ylabel(ax, 'cells/mm^2', 'FontSize', 24);
        title(ax, sprintf('%s — %s', ds.title, noise_panel_titles{midx}), ...
            'Interpreter', 'none', 'FontSize', 24);
        xlim(ax, [0, 14]);
        ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
    end

    tab_k = uitab(tg_inner, 'Title', 'Empirical sigma^2(t) (to day 60)');
    ax = axes('Parent', tab_k);
    ax.Layer = 'top';
    ax.FontSize = 24;
    hold(ax, 'on'); grid(ax, 'on');
    fit = ds.nn_emp_ext;
    plot_phenotype(ax, tgrid_ext, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1', band_label);
    plot_phenotype(ax, tgrid_ext, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2', band_label);
    xlabel(ax, 'Time (days)', 'FontSize', 24);
    ylabel(ax, 'cells/mm^2', 'FontSize', 24);
    title(ax, sprintf('%s — NN (empirical sigma^2(t), to day 60)', ds.title), ...
        'Interpreter', 'none', 'FontSize', 24);
    xlim(ax, [0, 60]);
    ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);

    tab_k = uitab(tg_inner, 'Title', 'Empirical sigma^2(t) + bound (to day 60)');
    ax = axes('Parent', tab_k);
    ax.Layer = 'top';
    ax.FontSize = 24;
    hold(ax, 'on'); grid(ax, 'on');
    fit = ds.nn_emp_bound;
    plot_phenotype(ax, tgrid_ext, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1', band_label);
    plot_phenotype(ax, tgrid_ext, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2', band_label);
    yline(ax, 0, 'k:', 'HandleVisibility', 'off');
    xlabel(ax, 'Time (days)', 'FontSize', 24);
    ylabel(ax, 'cells/mm^2', 'FontSize', 24);
    title(ax, sprintf('%s — NN (empirical sigma^2(t), lower bound, to day 60)', ds.title), ...
        'Interpreter', 'none', 'FontSize', 24);
    xlim(ax, [0, 60]);
    ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
end

%% ===== Standalone shared legend (once, after all figures) =====
make_shared_legend(col_M1, col_M2, band_label, 'Microglia shared legend');

%% ===== Local functions =====

function out = fit_naive_gp(timeM1, dataM1, timeM2, dataM2, tgrid, ...
    inffunc, meanfunc, covfunc, likfunc, max_iters, k_plot)

y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);

[hyp1, mu1, s21] = fit_gp(timeM1, y_M1, tgrid, inffunc, meanfunc, covfunc, likfunc, max_iters);
[hyp2, mu2, s22] = fit_gp(timeM2, y_M2, tgrid, inffunc, meanfunc, covfunc, likfunc, max_iters);

sf1 = sqrt(max(s21, 0));
sf2 = sqrt(max(s22, 0));
out.M1 = pack_raw_fit(mu1, sf1, k_plot);
out.M2 = pack_raw_fit(mu2, sf2, k_plot);
out.hyp_M1 = hyp1;
out.hyp_M2 = hyp2;

nlml1 = gp(hyp1, inffunc, meanfunc, covfunc, likfunc, timeM1(:), y_M1);
nlml2 = gp(hyp2, inffunc, meanfunc, covfunc, likfunc, timeM2(:), y_M2);
out.nlml = nlml1 + nlml2;

out.report.nlml = out.nlml;
out.report.ell_M1 = exp(hyp1.cov(1));
out.report.ell_M2 = exp(hyp2.cov(1));
out.report.sf_M1 = exp(hyp1.cov(2));
out.report.sf_M2 = exp(hyp2.cov(2));
out.report.sn_M1 = exp(hyp1.lik);
out.report.sn_M2 = exp(hyp2.lik);
end

function out = predict_naive_gp(fit, timeM1, dataM1, timeM2, dataM2, tgrid, ...
    inffunc, meanfunc, covfunc, likfunc, k_plot)
% Reuse fitted hyps; no re-optimization.
y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);
[~, ~, mu1, s21] = gp(fit.hyp_M1, inffunc, meanfunc, covfunc, likfunc, ...
    timeM1(:), y_M1, tgrid(:));
[~, ~, mu2, s22] = gp(fit.hyp_M2, inffunc, meanfunc, covfunc, likfunc, ...
    timeM2(:), y_M2, tgrid(:));
out = fit;
out.M1 = pack_raw_fit(mu1, sqrt(max(s21(:), 0)), k_plot);
out.M2 = pack_raw_fit(mu2, sqrt(max(s22(:), 0)), k_plot);
end

function [hyp, mu, s2] = fit_gp(x, y, xs, inffunc, meanfunc, covfunc, likfunc, max_iters)
x = x(:); y = y(:); xs = xs(:);
if nargin < 8
    max_iters = -100;
end
ell0 = max(std(x), 0.5);
sf0  = max(std(y), 0.1);
sn0  = 0.1 * sf0;
if sn0 <= 0
    sn0 = 0.1;
end
hyp.mean = [];
hyp.cov  = log([ell0; sf0]);
hyp.lik  = log(sn0);
hyp = minimize(hyp, @gp, max_iters, inffunc, meanfunc, covfunc, likfunc, x, y);
% Latent predictive mean/variance (fmu, fs2).
[~, ~, mu, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
mu = mu(:);
s2 = s2(:);
end

function pheno = pack_raw_fit(mu, sf, k_plot)
pheno.mu = mu(:);
pheno.sf = sf(:);
pheno.lo = mu - k_plot .* sf;
pheno.hi = mu + k_plot .* sf;
end

function report_fit(dataset_name, kernel_name, report)
fprintf('[%s | %s] NLML=%.4f | ell_M1=%.4f, ell_M2=%.4f | sf_M1=%.4f, sf_M2=%.4f | sn_M1=%.4f, sn_M2=%.4f\n', ...
    dataset_name, kernel_name, report.nlml, ...
    report.ell_M1, report.ell_M2, report.sf_M1, report.sf_M2, ...
    report.sn_M1, report.sn_M2);
end

function report_bound(dataset_name, kernel_name, report)
fprintf('[%s | %s | Pensoneault] NLML=%.4f | ell_M1=%.4f, ell_M2=%.4f | sf_M1=%.4f, sf_M2=%.4f | sn_M1=%.4f, sn_M2=%.4f\n', ...
    dataset_name, kernel_name, report.nlml, ...
    report.ell_M1, report.ell_M2, report.sf_M1, report.sf_M2, ...
    report.sn_M1, report.sn_M2);
fprintf('  M1: exitflag=%d, max(c)=%.4g | M2: exitflag=%d, max(c)=%.4g\n', ...
    report.exitflag_M1, report.max_c_M1, report.exitflag_M2, report.max_c_M2);
end

function out = fit_naive_gp_lower_bound(timeM1, dataM1, timeM2, dataM2, ...
    hyp_unc_M1, hyp_unc_M2, tgrid, X_c, k_pens, k_plot, ...
    inffunc, meanfunc, covfunc, likfunc, hyp_lb_M1, hyp_ub_M1, hyp_lb_M2, hyp_ub_M2, ...
    opts_pens, nTry, nMultistart, rng_seed_M1, rng_seed_M2)

y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);

[hyp1, mu1, s21, nlml1, ef1, max_c1] = fit_gp_lower_bound( ...
    timeM1(:), y_M1, hyp_unc_M1, X_c, k_pens, tgrid, ...
    inffunc, meanfunc, covfunc, likfunc, hyp_lb_M1, hyp_ub_M1, ...
    opts_pens, nTry, nMultistart, rng_seed_M1);
[hyp2, mu2, s22, nlml2, ef2, max_c2] = fit_gp_lower_bound( ...
    timeM2(:), y_M2, hyp_unc_M2, X_c, k_pens, tgrid, ...
    inffunc, meanfunc, covfunc, likfunc, hyp_lb_M2, hyp_ub_M2, ...
    opts_pens, nTry, nMultistart, rng_seed_M2);

out.M1 = pack_raw_fit(mu1, sqrt(max(s21, 0)), k_plot);
out.M2 = pack_raw_fit(mu2, sqrt(max(s22, 0)), k_plot);
out.hyp_M1 = hyp1;
out.hyp_M2 = hyp2;
out.nlml = nlml1 + nlml2;
out.report.nlml = out.nlml;
out.report.ell_M1 = exp(hyp1.cov(1));
out.report.ell_M2 = exp(hyp2.cov(1));
out.report.sf_M1 = exp(hyp1.cov(2));
out.report.sf_M2 = exp(hyp2.cov(2));
out.report.sn_M1 = exp(hyp1.lik);
out.report.sn_M2 = exp(hyp2.lik);
out.report.max_c_M1 = max_c1;
out.report.max_c_M2 = max_c2;
out.report.exitflag_M1 = ef1;
out.report.exitflag_M2 = ef2;
end

function [hyp, mu, s2, nlml, exitflag, max_c] = fit_gp_lower_bound( ...
    x, y, hyp_unc, X_c, k, xs, inffunc, meanfunc, covfunc, likfunc, ...
    hyp_lb, hyp_ub, opts, nTry, nMultistart, rng_seed)
% Pensoneault lower-bound GP: minimize NLML subject to mu_f - k*sigma_f >= 0 at X_c.
x = x(:); y = y(:); xs = xs(:);
sn_fixed = hyp_unc.lik;
hyp_tpl = struct('mean', [], 'cov', hyp_unc.cov(:), 'lik', sn_fixed);
theta_unc = hyp_unc.cov(:);

objfun = @(theta) gp(theta_to_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x, y);
nonlcon = @(theta) pens_constraints_lower(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k);

theta_unc_box = min(max(theta_unc, hyp_lb), hyp_ub);
fprintf('  Multistart: %d random starts\n', nTry);
feasible_starts = zeros(2, 0);
best_feas_nlml = inf;
best_feas_theta = nan(2, 1);
rng(rng_seed);
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
starts_for_fmincon = starts_for_fmincon(:, 1:min(nMultistart + 1, size(starts_for_fmincon, 2)));

best_nlml = inf;
theta_opt = nan(2, 1);
nlml = nan;
exitflag = -99;
nStarts = size(starts_for_fmincon, 2);
for j = 1:nStarts
    theta0_j = starts_for_fmincon(:, j);
    [theta_j, nlml_j, ef_j] = fmincon(objfun, theta0_j, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts);
    if isfinite(nlml_j) && nlml_j < best_nlml
        best_nlml = nlml_j;
        theta_opt = theta_j;
        nlml = nlml_j;
        exitflag = ef_j;
    end
end
if ~isfinite(best_nlml)
    if nFeas > 0
        theta_opt = best_feas_theta;
    else
        theta_opt = theta_unc_box;
    end
    nlml = objfun(theta_opt);
    exitflag = -99;
    fprintf('  Warning: no successful fmincon run; using fallback theta.\n');
end

hyp = struct('mean', [], 'cov', theta_opt(:), 'lik', sn_fixed);
[c_final, ~] = nonlcon(theta_opt);
max_c = max(c_final);
[~, ~, mu, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
mu = mu(:);
s2 = s2(:);
end

function hyp = theta_to_hyp(theta, hyp_tpl)
hyp = hyp_tpl;
hyp.cov = theta(1:2);
hyp.mean = [];
end

function [c, ceq] = pens_constraints_lower(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k)
hyp = theta_to_hyp(theta, hyp_tpl);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, X_c(:));
m_xc = fmu(:);
s_xc = sqrt(max(fs2(:), 0));
c = k .* s_xc - m_xc;
ceq = [];
end

function figL = make_shared_legend(col_M1, col_M2, band_label, fig_name)
% Standalone horizontal legend for LaTeX / Inkscape (LV_Periodic convention).
figL = figure('Color', 'w', 'Position', [100, 100, 900, 80], 'Name', fig_name);
axL = axes('Parent', figL, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
    'Position', [0 0 1 1]);
hold(axL, 'on');
hL = gobjects(6, 1);
hL(1) = fill(axL, nan, nan, col_M1, 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', sprintf('M1 %s', band_label));
hL(2) = plot(axL, nan, nan, '--', 'Color', col_M1, 'LineWidth', 2, ...
    'DisplayName', 'M1 GP Mean');
hL(3) = plot(axL, nan, nan, 'o', 'Color', col_M1, 'MarkerFaceColor', col_M1, ...
    'MarkerEdgeColor', 'k', 'MarkerSize', 5, 'DisplayName', 'M1 Obs Data');
hL(4) = fill(axL, nan, nan, col_M2, 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', sprintf('M2 %s', band_label));
hL(5) = plot(axL, nan, nan, '--', 'Color', col_M2, 'LineWidth', 2, ...
    'DisplayName', 'M2 GP Mean');
hL(6) = plot(axL, nan, nan, 'o', 'Color', col_M2, 'MarkerFaceColor', col_M2, ...
    'MarkerEdgeColor', 'k', 'MarkerSize', 5, 'DisplayName', 'M2 Obs Data');
lgd = legend(axL, hL, 'Orientation', 'horizontal', 'NumColumns', 3);
lgd.FontSize = 16;
lgd.ItemTokenSize = [20, 12];
drawnow;
figL.Units = 'pixels';
lgd.Units = 'pixels';
lp = lgd.Position;
margin = 6;
figL.Position(3:4) = [lp(3) + 2 * margin, lp(4) + 2 * margin];
lgd.Position = [margin, margin, lp(3), lp(4)];
end

function plot_phenotype(ax, tgrid, fit, t_data, y_data, col, name, band_label)
tg = tgrid(:)';
fill(ax, [tg, fliplr(tg)], [fit.hi(:)', fliplr(fit.lo(:)')], col, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', sprintf('%s %s', name, band_label));
plot(ax, tgrid, fit.mu, '--', 'Color', col, 'LineWidth', 2, ...
    'DisplayName', sprintf('%s GP Mean', name));
scatter(ax, t_data, y_data, 36, 'filled', ...
    'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', ...
    'DisplayName', sprintf('%s Obs Data', name));
end

function ylim_auto_from_fit(ax, fitM1, fitM2, dataM1, dataM2)
vals = [fitM1.lo(:); fitM1.hi(:); fitM1.mu(:); ...
        fitM2.lo(:); fitM2.hi(:); fitM2.mu(:); ...
        dataM1(:); dataM2(:)];
pad = 0.05 * max(range(vals), 1);
ylim(ax, [min(vals) - pad, max(vals) + pad]);
end

function out = fit_emp_nn_pair(timeM1, dataM1, timeM2, dataM2, tgrid, max_iters, k_plot)
y1 = max(dataM1(:), 0);
y2 = max(dataM2(:), 0);
fit1 = fit_emp_nn_gp(timeM1(:), y1, tgrid, max_iters);
fit2 = fit_emp_nn_gp(timeM2(:), y2, tgrid, max_iters);

out.M1 = pack_raw_fit(fit1.mu, fit1.sf, k_plot);
out.M2 = pack_raw_fit(fit2.mu, fit2.sf, k_plot);
out.hyp_M1 = fit1.hyp;
out.hyp_M2 = fit2.hyp;
out.nlml = fit1.nlml + fit2.nlml;
out.emp_M1 = fit1;
out.emp_M2 = fit2;
end

function out = predict_emp_nn_pair(fit, timeM1, dataM1, timeM2, dataM2, tgrid, k_plot)
% Reuse fitted ell/sf and training-diagonal empirical noise; no re-optimization.
y1 = max(dataM1(:), 0);
y2 = max(dataM2(:), 0);
[~, ~, fmu1, fs21] = gp_nn_hetero_noise('pred', fit.hyp_M1, timeM1(:), y1, ...
    fit.emp_M1.noise_var, tgrid(:));
[~, ~, fmu2, fs22] = gp_nn_hetero_noise('pred', fit.hyp_M2, timeM2(:), y2, ...
    fit.emp_M2.noise_var, tgrid(:));
out = fit;
out.M1 = pack_raw_fit(fmu1, sqrt(max(fs21(:), 0)), k_plot);
out.M2 = pack_raw_fit(fmu2, sqrt(max(fs22(:), 0)), k_plot);
end

function out = fit_emp_nn_lower_bound_pair(timeM1, dataM1, timeM2, dataM2, emp_fit, ...
    tgrid, X_c, k_pens, k_plot, hyp_lb_M1, hyp_ub_M1, hyp_lb_M2, hyp_ub_M2, ...
    opts_pens, nTry, nMultistart, rng_seed_M1, rng_seed_M2)
% Empirical noise_var held fixed; ell/sf re-fit with Pensoneault mu-k*sigma >= 0.
% No data-fidelity epsilon tube.

y1 = max(dataM1(:), 0);
y2 = max(dataM2(:), 0);

[hyp1, mu1, s21, nlml1, ef1, max_c1] = fit_emp_nn_lower_bound( ...
    timeM1(:), y1, emp_fit.hyp_M1, emp_fit.emp_M1.noise_var, X_c, k_pens, tgrid, ...
    hyp_lb_M1, hyp_ub_M1, opts_pens, nTry, nMultistart, rng_seed_M1);
[hyp2, mu2, s22, nlml2, ef2, max_c2] = fit_emp_nn_lower_bound( ...
    timeM2(:), y2, emp_fit.hyp_M2, emp_fit.emp_M2.noise_var, X_c, k_pens, tgrid, ...
    hyp_lb_M2, hyp_ub_M2, opts_pens, nTry, nMultistart, rng_seed_M2);

out.M1 = pack_raw_fit(mu1, sqrt(max(s21, 0)), k_plot);
out.M2 = pack_raw_fit(mu2, sqrt(max(s22, 0)), k_plot);
out.hyp_M1 = hyp1;
out.hyp_M2 = hyp2;
out.nlml = nlml1 + nlml2;
out.report.nlml = out.nlml;
out.report.ell_M1 = exp(hyp1.cov(1));
out.report.ell_M2 = exp(hyp2.cov(1));
out.report.sf_M1 = exp(hyp1.cov(2));
out.report.sf_M2 = exp(hyp2.cov(2));
out.report.max_c_M1 = max_c1;
out.report.max_c_M2 = max_c2;
out.report.exitflag_M1 = ef1;
out.report.exitflag_M2 = ef2;
end

function [hyp, mu, s2, nlml, exitflag, max_c] = fit_emp_nn_lower_bound( ...
    x, y, hyp_unc, noise_var, X_c, k, xs, hyp_lb, hyp_ub, opts, nTry, nMultistart, rng_seed)
% Minimize NN-hetero NLML subject to mu_f - k*sigma_f >= 0 at X_c. noise_var fixed.
x = x(:); y = y(:); xs = xs(:); noise_var = noise_var(:);
hyp_tpl = struct('mean', [], 'cov', hyp_unc.cov(:), 'lik', []);
theta_unc = hyp_unc.cov(:);

objfun = @(theta) gp_nn_hetero_noise('nlml', theta_to_hyp(theta, hyp_tpl), x, y, noise_var);
nonlcon = @(theta) pens_constraints_lower_nn_hetero(theta, hyp_tpl, x, y, noise_var, X_c, k);

theta_unc_box = min(max(theta_unc, hyp_lb), hyp_ub);
fprintf('  Multistart: %d random starts\n', nTry);
feasible_starts = zeros(2, 0);
best_feas_nlml = inf;
best_feas_theta = nan(2, 1);
rng(rng_seed);
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
starts_for_fmincon = starts_for_fmincon(:, 1:min(nMultistart + 1, size(starts_for_fmincon, 2)));

best_nlml = inf;
theta_opt = nan(2, 1);
nlml = nan;
exitflag = -99;
nStarts = size(starts_for_fmincon, 2);
for j = 1:nStarts
    theta0_j = starts_for_fmincon(:, j);
    [theta_j, nlml_j, ef_j] = fmincon(objfun, theta0_j, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts);
    if isfinite(nlml_j) && nlml_j < best_nlml
        best_nlml = nlml_j;
        theta_opt = theta_j;
        nlml = nlml_j;
        exitflag = ef_j;
    end
end
if ~isfinite(best_nlml)
    if nFeas > 0
        theta_opt = best_feas_theta;
    else
        theta_opt = theta_unc_box;
    end
    nlml = objfun(theta_opt);
    exitflag = -99;
    fprintf('  Warning: no successful fmincon run; using fallback theta.\n');
end

hyp = struct('mean', [], 'cov', theta_opt(:), 'lik', []);
[c_final, ~] = nonlcon(theta_opt);
max_c = max(c_final);
[~, ~, mu, s2] = gp_nn_hetero_noise('pred', hyp, x, y, noise_var, xs);
mu = mu(:);
s2 = s2(:);
end

function [c, ceq] = pens_constraints_lower_nn_hetero(theta, hyp_tpl, x, y, noise_var, X_c, k)
hyp = theta_to_hyp(theta, hyp_tpl);
[~, ~, fmu, fs2] = gp_nn_hetero_noise('pred', hyp, x, y, noise_var, X_c(:));
c = k .* sqrt(max(fs2(:), 0)) - fmu(:);
ceq = [];
end

function fit = fit_emp_nn_gp(x, y, xs, max_iters)
% ell/sf via NLML; diagonal noise = empirical replicate variance at each t.
x = x(:); y = y(:); xs = xs(:);
[noise_var, t_unique, s2_unique, n_per_t, used_fallback] = empirical_time_noise(x, y);

ell0 = max(std(x), 0.5);
sf0  = max(std(y), 0.1);
hyp0 = struct('mean', [], 'cov', log([ell0; sf0]), 'lik', []);
obj = @(h) gp_nn_hetero_noise('nlml', h, x, y, noise_var);
hyp = minimize(hyp0, obj, max_iters);
nlml = obj(hyp);
[~, ~, fmu, fs2] = gp_nn_hetero_noise('pred', hyp, x, y, noise_var, xs);

fit.hyp = hyp;
fit.nlml = nlml;
fit.mu = fmu(:);
fit.sf = sqrt(max(fs2(:), 0));
fit.noise_var = noise_var;
fit.t_unique = t_unique;
fit.s2_unique = s2_unique;
fit.n_per_t = n_per_t;
fit.used_fallback = used_fallback;
end

function [noise_var, t_unique, s2_unique, n_per_t, used_fallback] = empirical_time_noise(x, y)
% Sample variance of y at each unique x. n=1 times use pooled variance
% from times with n>=2 (else var(y)), floored away from zero.
x = x(:); y = y(:);
t_unique = unique(x);
nU = numel(t_unique);
s2_unique = zeros(nU, 1);
n_per_t = zeros(nU, 1);
eps_floor = 1e-6 * max(var(y), 1);

for i = 1:nU
    yi = y(abs(x - t_unique(i)) < 1e-12);
    n_per_t(i) = numel(yi);
    if n_per_t(i) >= 2
        s2_unique(i) = var(yi, 0);
    else
        s2_unique(i) = NaN;
    end
end

used_fallback = isnan(s2_unique);
if any(~used_fallback)
    pooled = mean(s2_unique(~used_fallback));
else
    pooled = max(var(y, 0), eps_floor);
end
s2_unique(used_fallback) = pooled;
s2_unique = max(s2_unique, eps_floor);

noise_var = zeros(size(x));
for i = 1:nU
    noise_var(abs(x - t_unique(i)) < 1e-12) = s2_unique(i);
end
end

function report_emp_nn(dataset_name, fit)
fprintf(['[%s | NN emp] NLML=%.4f | ' ...
    'ell_M1=%.4f, ell_M2=%.4f | sf_M1=%.4f, sf_M2=%.4f\n'], ...
    dataset_name, fit.nlml, ...
    exp(fit.hyp_M1.cov(1)), exp(fit.hyp_M2.cov(1)), ...
    exp(fit.hyp_M1.cov(2)), exp(fit.hyp_M2.cov(2)));
print_emp_sn(dataset_name, 'M1', fit.emp_M1);
print_emp_sn(dataset_name, 'M2', fit.emp_M2);
end

function report_emp_bound(dataset_name, report)
fprintf('[%s | NN emp + Pensoneault] NLML=%.4f | ell_M1=%.4f, ell_M2=%.4f | sf_M1=%.4f, sf_M2=%.4f\n', ...
    dataset_name, report.nlml, report.ell_M1, report.ell_M2, report.sf_M1, report.sf_M2);
fprintf('  M1: exitflag=%d, max(c)=%.4g | M2: exitflag=%d, max(c)=%.4g\n', ...
    report.exitflag_M1, report.max_c_M1, report.exitflag_M2, report.max_c_M2);
end

function print_emp_sn(dataset_name, pheno, emp)
sn = sqrt(emp.s2_unique);
mark = repmat(' ', size(sn));
mark(emp.used_fallback) = '*';
txt = arrayfun(@(t, s, n, m) sprintf('t=%.0f:sn=%.3g(n=%d)%s', t, s, n, m), ...
    emp.t_unique, sn, emp.n_per_t, mark, 'UniformOutput', false);
fprintf('  [%s | emp %s] %s  (* = n=1 pooled fallback)\n', ...
    dataset_name, pheno, strjoin(txt, ', '));
end

function out = fit_vhgpr_pair(timeM1, dataM1, timeM2, dataM2, tgrid, vhgpr_iter, k_plot)
y1 = max(dataM1(:), 0);
y2 = max(dataM2(:), 0);
fit1 = vhgpr_fit_predict(timeM1(:), y1, tgrid(:), struct('iter', vhgpr_iter));
fit2 = vhgpr_fit_predict(timeM2(:), y2, tgrid(:), struct('iter', vhgpr_iter));

out.M1 = pack_raw_fit(fit1.fmu, sqrt(max(fit1.fs2, 0)), k_plot);
out.M2 = pack_raw_fit(fit2.fmu, sqrt(max(fit2.fs2, 0)), k_plot);
out.raw_M1 = fit1;
out.raw_M2 = fit2;
out.mean_sigma_n_M1 = mean(fit1.sigma_n);
out.mean_sigma_n_M2 = mean(fit2.sigma_n);
end

function report_vhgpr_nn(dataset_name, fit)
fprintf(['[%s | vhgpr] done | mean sigma_n_M1=%.4g, mean sigma_n_M2=%.4g ' ...
    '(latent fmu/fs2 bands)\n'], ...
    dataset_name, fit.mean_sigma_n_M1, fit.mean_sigma_n_M2);
end
