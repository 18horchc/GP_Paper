% microglia.m — Independent GPML GPs for M1/M2 across kernels.
% Kernels: RBF (SE), Matern 3/2, Matern 5/2, Neural Network (covNNone).
% Figure: outer tabs = full / averaged; inner tabs = kernel.
% M1 and M2 plotted on the same axes. Day 5 included in both phenotypes.

clear; close all; clc;

%% ===== Configuration =====
k_plot    = 1.96;
max_iters = -200;
tgrid     = (0:0.1:14)';

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
    'title', {'Full', 'Averaged'}, ...
    'timeM1', {timeM1(:), newtime(:)}, ...
    'dataM1', {dataM1(:), datapointsM1(:)}, ...
    'timeM2', {timeM2(:), newtimeM2(:)}, ...
    'dataM2', {dataM2(:), datapointsM2(:)});

kernels = struct( ...
    'name',  {'rbf', 'matern32', 'matern52', 'nn'}, ...
    'title', {'RBF', 'Matern 3/2', 'Matern 5/2', 'NN'}, ...
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
fprintf('Kernels: RBF | Matern 3/2 | Matern 5/2 | NN\n');

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
        report_fit(ds.name, ker.title, fit.report);
    end
end

%% ===== Tabbed figure: outer = dataset, inner = kernel =====
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];

fig = figure('Color', 'w', 'Position', [60, 60, 1240, 900], ...
    'Name', 'Microglia: GP kernel comparison (M1 & M2)');
tg_outer = uitabgroup(fig);

for didx = 1:numel(results)
    ds = results(didx);
    tab_ds = uitab(tg_outer, 'Title', ds.title);
    tg_inner = uitabgroup(tab_ds);

    for kidx = 1:numel(ds.fits)
        kf = ds.fits(kidx);
        tab_k = uitab(tg_inner, 'Title', kf.title);
        ax = axes('Parent', tab_k);
        ax.Layer = 'top';
        hold on; grid on;
        fit = kf.fit;
        plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1');
        plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2');
        xlabel('Time (days)');
        ylabel('cells/mm^2');
        title(sprintf('%s — %s', ds.title, kf.title), 'Interpreter', 'none');
        xlim([0, 14]);
        ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
        legend('Location', 'northwest', 'FontSize', 8);
    end
end

fprintf('\nDone.\n');

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
