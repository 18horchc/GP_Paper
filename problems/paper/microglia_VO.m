% microglia_VO.m — Independent SE GPs vs independent SE GPs with a virtual
% observation at (t=0, y=5) with low noise on both M1 and M2.
% Figure: 2x2 — left = independent, right = independent+VO; top = full, bottom = averaged.
% Day 5 included in both M1 and M2.

clear; close all; clc;

%% ===== Configuration =====
k_plot    = 1.96;
max_iters = -200;
t_vo      = 0;
y_vo      = 5;
sigma_vo  = 0.5;   % low noise on VO (count units)

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

fprintf('=== Independent SE GP vs independent + VO (t=0, y=5) ===\n');
fprintf('VO: t=%.3g, y=%.3g, sigma=%.3g\n', t_vo, y_vo, sigma_vo);

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
addpath(fileparts(fileparts(mfilename('fullpath'))));  % problems/ for gp_seiso_hetero_noise

meanfunc = @meanZero;
likfunc  = @likGauss;
tgrid    = (0:0.1:14)';
temporalKernel = @covSEiso;

%% ===== Fit =====
results = struct([]);
for didx = 1:numel(datasets)
    ds = datasets(didx);
    fprintf('\n--- Dataset: %s ---\n', ds.name);

    naive = fit_naive_gp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot);
    naive_vo = fit_naive_gp_vo(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        t_vo, y_vo, sigma_vo, naive, tgrid, max_iters, k_plot);

    results(didx).name = ds.name;
    results(didx).naive = naive;
    results(didx).method = naive_vo;
    results(didx).timeM1 = ds.timeM1;
    results(didx).dataM1 = ds.dataM1;
    results(didx).timeM2 = ds.timeM2;
    results(didx).dataM2 = ds.dataM2;

    report_fit(ds.name, 'Independent SE', naive.report);
    report_fit(ds.name, 'Independent SE + VO', naive_vo.report);
end

%% ===== Figure: 2x2 (independent | VO) x (full | averaged) =====
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];
col_vo = [0.20, 0.45, 0.80];

figure('Color', 'w', 'Position', [60, 60, 1240, 900], ...
    'Name', 'Microglia: independent SE vs independent + VO');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for didx = 1:numel(results)
    ds = results(didx);
    methods = {ds.naive, ds.method};
    method_titles = {'Independent SE GP', sprintf('Independent + VO (0, %.3g)', y_vo)};
    for midx = 1:2
        nexttile;
        ax = gca; ax.Layer = 'top';
        hold on; grid on;
        fit = methods{midx};
        plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1');
        plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2');
        if midx == 2
            scatter(ax, t_vo, y_vo, 110, 's', ...
                'MarkerFaceColor', col_vo, 'MarkerEdgeColor', 'k', ...
                'LineWidth', 1.2, 'DisplayName', sprintf('VO (0, %.3g)', y_vo));
        end
        xlabel('Time (days)');
        ylabel('cells/mm^2');
        title(sprintf('%s — %s', ds.name, method_titles{midx}), 'Interpreter', 'none');
        xlim([0, 14]);
        if midx == 2
            ylim_auto_from_fit(ax, fit.M1, fit.M2, [ds.dataM1; y_vo], [ds.dataM2; y_vo]);
        else
            ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
        end
        legend('Location', 'northwest', 'FontSize', 8);
    end
end

fprintf('\nDone.\n');

%% ===== Local functions =====

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

function fit = fit_single_gp_vo(x, y, t_vo, y_vo, sigma_vo, hyp_unc, tgrid, max_iters)
x = x(:); y = y(:);
x_aug = [x; t_vo];
y_aug = [y; y_vo];
sn_real = exp(hyp_unc.lik);
noise_var = [sn_real^2 * ones(numel(y), 1); sigma_vo^2];

hyp0 = struct('mean', [], 'cov', hyp_unc.cov(:), 'lik', []);
obj = @(h) gp_seiso_hetero_noise('nlml', h, x_aug, y_aug, noise_var);
hyp = minimize(hyp0, obj, max_iters);
nlml = obj(hyp);
[~, ~, fmu, fs2] = gp_seiso_hetero_noise('pred', hyp, x_aug, y_aug, noise_var, tgrid(:));

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
