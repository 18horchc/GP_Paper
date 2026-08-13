% microglia_bound.m — Independent GPs vs Pensoneault lower-bound GPs.
% Single tabbed figure: SE | Matern 3/2 | Matern 5/2 | NN | Output transforms | Time warp.
% Kernel tabs: 2x2 independent vs Pensoneault (full / averaged).
% Tab Output transforms: 2x3 softplus / log / sqrt (full top, averaged bottom).
% Tab Time warp: 1x2 tau = log(1+t) (full | averaged).
% Day 5 included in both M1 and M2. No data-fidelity epsilon tube.

clear; close all; clc;

%% ===== Configuration =====
k_plot       = 1.96;
max_iters    = -100;
eta_pens     = 0.022;
k_pens       = -sqrt(2) * erfinv(2 * eta_pens - 1);
n_constraint = 41;
X_c          = linspace(0, 14, n_constraint)';
ell_bounds_lo = 0.05;
ell_ub        = 14;
nTry          = 2000;
nMultistart   = 10;
opts_pens = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, 'Display', 'off', ...
    'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);

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

% Figure tabs: SE | Matern 3/2 | Matern 5/2 | NN (same 2x2 layout)
kernels = struct( ...
    'name', {'se', 'matern32', 'matern52', 'nn'}, ...
    'label', {'SE', 'Matern 3/2', 'Matern 5/2', 'NN'}, ...
    'cov', {@covSEiso, {@covMaterniso, 3}, {@covMaterniso, 5}, @covNNone}, ...
    'indep_title', { ...
        'Independent SE GP', ...
        'Independent Matern 3/2 GP', ...
        'Independent Matern 5/2 GP', ...
        'Independent NN GP'}, ...
    'fig_name', { ...
        'Microglia: independent vs Pensoneault lower bound', ...
        'Microglia: independent Matern 3/2 vs Pensoneault', ...
        'Microglia: independent Matern 5/2 vs Pensoneault', ...
        'Microglia: independent NN vs Pensoneault lower bound'});

fprintf('=== Independent GP vs Pensoneault lower bound ===\n');
fprintf('Kernels: SE | Matern 3/2 | Matern 5/2 | NN\n');
fprintf('eta = %.3g%% | k = %.4f | X_c: %d points on [0, 14]\n', ...
    100 * eta_pens, k_pens, n_constraint);

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
tgrid    = (0:0.1:14)';

%% ===== Fit SE / NN (Figures 1–2 content; plotted later as tabs) =====
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];

kernel_results = struct([]);
for kidx = 1:numel(kernels)
    ker = kernels(kidx);
    covfunc = ker.cov;
    fprintf('\n========== Kernel: %s ==========\n', ker.label);

    results = struct([]);
    for didx = 1:numel(datasets)
        ds = datasets(didx);
        fprintf('\n--- Dataset: %s ---\n', ds.name);

        naive = fit_naive_gp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
            tgrid, inffunc, meanfunc, covfunc, likfunc, max_iters, k_plot);

        sf_bounds_M1 = [0.05, max(15, 1.5 * std(ds.dataM1))];
        sf_bounds_M2 = [0.05, max(15, 1.5 * std(ds.dataM2))];
        hyp_lb_M1 = log([ell_bounds_lo; sf_bounds_M1(1)]);
        hyp_ub_M1 = log([ell_ub; sf_bounds_M1(2)]);
        hyp_lb_M2 = log([ell_bounds_lo; sf_bounds_M2(1)]);
        hyp_ub_M2 = log([ell_ub; sf_bounds_M2(2)]);

        bound = fit_naive_gp_lower_bound(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
            naive.hyp_M1, naive.hyp_M2, tgrid, X_c, k_pens, k_plot, ...
            inffunc, meanfunc, covfunc, likfunc, hyp_lb_M1, hyp_ub_M1, hyp_lb_M2, hyp_ub_M2, ...
            opts_pens, nTry, nMultistart, 40 + 2*didx + 100*kidx, 41 + 2*didx + 100*kidx);

        results(didx).name = ds.name;
        results(didx).naive = naive;
        results(didx).method = bound;
        results(didx).timeM1 = ds.timeM1;
        results(didx).dataM1 = ds.dataM1;
        results(didx).timeM2 = ds.timeM2;
        results(didx).dataM2 = ds.dataM2;

        report_naive(ds.name, ker.label, naive.report);
        report_bound(ds.name, ker.label, bound.report);
    end

    kernel_results(kidx).label = ker.label;
    kernel_results(kidx).tab_title = ker.label;
    kernel_results(kidx).indep_title = ker.indep_title;
    kernel_results(kidx).results = results;
end

%% ===== Output transforms (SE, unconstrained independent) =====
% softplus / log1p / sqrt on y; back-transform to cells/mm^2 for plotting.
y_transforms = struct( ...
    'name', {'softplus', 'log', 'sqrt'}, ...
    'title', {'softplus', 'log', 'sqrt'});

cov_se = @covSEiso;
fprintf('\n========== Output transforms (SE independent) ==========\n');

xform_results = struct([]);
for didx = 1:numel(datasets)
    ds = datasets(didx);
    xform_results(didx).name = ds.name;
    xform_results(didx).timeM1 = ds.timeM1;
    xform_results(didx).dataM1 = ds.dataM1;
    xform_results(didx).timeM2 = ds.timeM2;
    xform_results(didx).dataM2 = ds.dataM2;
    xform_results(didx).fits = struct([]);
    for tidx = 1:numel(y_transforms)
        tr = y_transforms(tidx);
        fit = fit_naive_gp_ytransform(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
            tgrid, tr.name, inffunc, meanfunc, cov_se, likfunc, max_iters, k_plot);
        xform_results(didx).fits(tidx).name = tr.name;
        xform_results(didx).fits(tidx).title = tr.title;
        xform_results(didx).fits(tidx).fit = fit;
        report_transform(ds.name, tr.title, fit.report);
    end
end

%% ===== Time warp tau = log(1+t) (SE, unconstrained independent) =====
fprintf('\n========== Time warp tau = log(1+t) (SE independent) ==========\n');
warp_results = struct([]);
for didx = 1:numel(datasets)
    ds = datasets(didx);
    fit = fit_naive_gp_timewarp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, inffunc, meanfunc, cov_se, likfunc, max_iters, k_plot);
    warp_results(didx).name = ds.name;
    warp_results(didx).timeM1 = ds.timeM1;
    warp_results(didx).dataM1 = ds.dataM1;
    warp_results(didx).timeM2 = ds.timeM2;
    warp_results(didx).dataM2 = ds.dataM2;
    warp_results(didx).fit = fit;
    report_transform(ds.name, 'tau=log(1+t)', fit.report);
end

%% ===== Single tabbed figure =====
fig = figure('Color', 'w', 'Position', [60, 60, 1240, 900], ...
    'Name', 'Microglia bound: kernels / transforms / time warp');
tg = uitabgroup(fig);

% Tabs 1–2: SE and NN bound comparisons (2x2)
tab_first = [];
for kidx = 1:numel(kernel_results)
    kr = kernel_results(kidx);
    tab = uitab(tg, 'Title', kr.tab_title);
    if isempty(tab_first)
        tab_first = tab;
    end
    tg.SelectedTab = tab;
    method_titles = {kr.indep_title, 'Pensoneault lower bound'};
    tl = tiledlayout(tab, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    for didx = 1:numel(kr.results)
        ds = kr.results(didx);
        methods = {ds.naive, ds.method};
        for midx = 1:2
            ax = nexttile(tl);
            ax.Layer = 'top';
            hold(ax, 'on'); grid(ax, 'on');
            fit = methods{midx};
            plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1');
            plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2');
            if midx == 2
                yline(ax, 0, 'k:', 'HandleVisibility', 'off');
            end
            xlabel(ax, 'Time (days)');
            ylabel(ax, 'cells/mm^2');
            title(ax, sprintf('%s — %s', ds.name, method_titles{midx}), 'Interpreter', 'none');
            xlim(ax, [0, 14]);
            ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
            legend(ax, 'Location', 'northwest', 'FontSize', 8);
        end
    end
end

% Tab 3: output transforms (2x3 — full top, averaged bottom)
tab_xf = uitab(tg, 'Title', 'Output transforms');
tg.SelectedTab = tab_xf;
tl_xf = tiledlayout(tab_xf, 2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
for didx = 1:numel(xform_results)
    ds = xform_results(didx);
    for tidx = 1:numel(y_transforms)
        kf = ds.fits(tidx);
        ax = nexttile(tl_xf);
        ax.Layer = 'top';
        hold(ax, 'on'); grid(ax, 'on');
        fit = kf.fit;
        plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1');
        plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2');
        xlabel(ax, 'Time (days)');
        ylabel(ax, 'cells/mm^2');
        title(ax, sprintf('%s — %s', ds.name, kf.title), 'Interpreter', 'none');
        xlim(ax, [0, 14]);
        ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
        legend(ax, 'Location', 'northwest', 'FontSize', 8);
    end
end

% Tab 4: time warp (1x2 — full | averaged)
tab_tw = uitab(tg, 'Title', 'Time warp');
tg.SelectedTab = tab_tw;
tl_tw = tiledlayout(tab_tw, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
for didx = 1:numel(warp_results)
    ds = warp_results(didx);
    ax = nexttile(tl_tw);
    ax.Layer = 'top';
    hold(ax, 'on'); grid(ax, 'on');
    fit = ds.fit;
    plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1');
    plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2');
    xlabel(ax, 'Time (days)');
    ylabel(ax, 'cells/mm^2');
    title(ax, sprintf('%s — tau = log(1+t)', ds.name), 'Interpreter', 'none');
    xlim(ax, [0, 14]);
    ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
    legend(ax, 'Location', 'northwest', 'FontSize', 8);
end

tg.SelectedTab = tab_first;  % show SE tab first
drawnow;

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
out.report.sn_M1 = exp(hyp1.lik);
out.report.sn_M2 = exp(hyp2.lik);
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
out.report.sn_M1 = exp(hyp1.lik);
out.report.sn_M2 = exp(hyp2.lik);
out.report.max_c_M1 = max_c1;
out.report.max_c_M2 = max_c2;
out.report.exitflag_M1 = ef1;
out.report.exitflag_M2 = ef2;
end

function [hyp, mu, s2] = fit_gp(x, y, xs, inffunc, meanfunc, covfunc, likfunc, max_iters)
x = x(:); y = y(:); xs = xs(:);
if nargin < 8
    max_iters = -100;
end
ell0 = std(x);
sf0  = std(y);
sn0  = 0.1 * std(y);
if sn0 <= 0
    sn0 = 0.1;
end
hyp.mean = [];
hyp.cov  = log([ell0; sf0]);
hyp.lik  = log(sn0);
hyp = minimize(hyp, @gp, max_iters, inffunc, meanfunc, covfunc, likfunc, x, y);
% Latent predictive mean/variance (fmu, fs2) — matches microglia_ICM/VO/LMC.
% Two-output gp() would return ymu/ys2 (includes observation noise).
[~, ~, mu, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
mu = mu(:);
s2 = s2(:);
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

function pheno = pack_raw_fit(mu, sf, k_plot)
pheno.mu = mu(:);
pheno.sf = sf(:);
pheno.lo = mu - k_plot .* sf;
pheno.hi = mu + k_plot .* sf;
end

function report_naive(dataset_name, kernel_label, report)
fprintf('[%s | %s | Independent] NLML=%.4f, ell_M1=%.4f, ell_M2=%.4f, sn_M1=%.4f, sn_M2=%.4f\n', ...
    dataset_name, kernel_label, report.nlml, report.ell_M1, report.ell_M2, report.sn_M1, report.sn_M2);
end

function report_bound(dataset_name, kernel_label, report)
fprintf('[%s | %s | Pensoneault] NLML=%.4f, ell_M1=%.4f, ell_M2=%.4f\n', ...
    dataset_name, kernel_label, report.nlml, report.ell_M1, report.ell_M2);
fprintf('  M1: exitflag=%d, max(c)=%.4g | M2: exitflag=%d, max(c)=%.4g\n', ...
    report.exitflag_M1, report.max_c_M1, report.exitflag_M2, report.max_c_M2);
end

function report_transform(dataset_name, transform_label, report)
fprintf('[%s | SE | %s] NLML=%.4f, ell_M1=%.4f, ell_M2=%.4f, sn_M1=%.4f, sn_M2=%.4f\n', ...
    dataset_name, transform_label, report.nlml, ...
    report.ell_M1, report.ell_M2, report.sn_M1, report.sn_M2);
end

function out = fit_naive_gp_ytransform(timeM1, dataM1, timeM2, dataM2, tgrid, ...
    transform_name, inffunc, meanfunc, covfunc, likfunc, max_iters, k_plot)
y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);
z_M1 = forward_ytransform(y_M1, transform_name);
z_M2 = forward_ytransform(y_M2, transform_name);

[hyp1, mu1, s21] = fit_gp(timeM1, z_M1, tgrid, inffunc, meanfunc, covfunc, likfunc, max_iters);
[hyp2, mu2, s22] = fit_gp(timeM2, z_M2, tgrid, inffunc, meanfunc, covfunc, likfunc, max_iters);

out.M1 = pack_ytransform_fit(mu1, sqrt(max(s21, 0)), k_plot, transform_name);
out.M2 = pack_ytransform_fit(mu2, sqrt(max(s22, 0)), k_plot, transform_name);
out.hyp_M1 = hyp1;
out.hyp_M2 = hyp2;
nlml1 = gp(hyp1, inffunc, meanfunc, covfunc, likfunc, timeM1(:), z_M1);
nlml2 = gp(hyp2, inffunc, meanfunc, covfunc, likfunc, timeM2(:), z_M2);
out.nlml = nlml1 + nlml2;
out.report.nlml = out.nlml;
out.report.ell_M1 = exp(hyp1.cov(1));
out.report.ell_M2 = exp(hyp2.cov(1));
out.report.sn_M1 = exp(hyp1.lik);
out.report.sn_M2 = exp(hyp2.lik);
end

function out = fit_naive_gp_timewarp(timeM1, dataM1, timeM2, dataM2, tgrid, ...
    inffunc, meanfunc, covfunc, likfunc, max_iters, k_plot)
% Fit on tau = log(1+t); predict on warped grid; plot vs original t.
y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);
tau_M1 = log1p(timeM1(:));
tau_M2 = log1p(timeM2(:));
tau_grid = log1p(tgrid(:));

[hyp1, mu1, s21] = fit_gp(tau_M1, y_M1, tau_grid, inffunc, meanfunc, covfunc, likfunc, max_iters);
[hyp2, mu2, s22] = fit_gp(tau_M2, y_M2, tau_grid, inffunc, meanfunc, covfunc, likfunc, max_iters);

sf1 = sqrt(max(s21, 0));
sf2 = sqrt(max(s22, 0));
out.M1 = pack_raw_fit(mu1, sf1, k_plot);
out.M2 = pack_raw_fit(mu2, sf2, k_plot);
out.hyp_M1 = hyp1;
out.hyp_M2 = hyp2;
nlml1 = gp(hyp1, inffunc, meanfunc, covfunc, likfunc, tau_M1, y_M1);
nlml2 = gp(hyp2, inffunc, meanfunc, covfunc, likfunc, tau_M2, y_M2);
out.nlml = nlml1 + nlml2;
out.report.nlml = out.nlml;
out.report.ell_M1 = exp(hyp1.cov(1));
out.report.ell_M2 = exp(hyp2.cov(1));
out.report.sn_M1 = exp(hyp1.lik);
out.report.sn_M2 = exp(hyp2.lik);
end

function z = forward_ytransform(y, name)
y = y(:);
switch lower(name)
    case 'softplus'
        z = softplus_fwd(y);
    case 'log'
        z = log1p(y);
    case 'sqrt'
        z = sqrt(max(y, 0));
    otherwise
        error('Unknown y-transform: %s', name);
end
end

function y = inverse_ytransform(z, name)
z = z(:);
switch lower(name)
    case 'softplus'
        y = softplus_inv(z);
    case 'log'
        y = expm1(z);
    case 'sqrt'
        y = z.^2;
    otherwise
        error('Unknown y-transform: %s', name);
end
end

function z = softplus_fwd(y)
% Stable softplus: log(1+exp(y)); ≈ y for large y.
y = y(:);
z = y;
mask = y < 40;
z(mask) = log1p(exp(y(mask)));
end

function y = softplus_inv(z)
% Inverse softplus: log(expm1(z)); ≈ z for large z.
z = z(:);
y = z;
mask = z < 40;
y(mask) = log(max(expm1(z(mask)), realmin('double')));
end

function pheno = pack_ytransform_fit(mu_z, sf_z, k_plot, transform_name)
mu_z = mu_z(:);
sf_z = sf_z(:);
lo_z = mu_z - k_plot .* sf_z;
hi_z = mu_z + k_plot .* sf_z;
pheno.mu = inverse_ytransform(mu_z, transform_name);
pheno.lo = inverse_ytransform(lo_z, transform_name);
pheno.hi = inverse_ytransform(hi_z, transform_name);
% Floor lower band at 0 for positivity-friendly transforms
if any(strcmpi(transform_name, {'log', 'sqrt', 'softplus'}))
    pheno.lo = max(pheno.lo, 0);
end
pheno.sf = (pheno.hi - pheno.lo) / (2 * k_plot);
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
