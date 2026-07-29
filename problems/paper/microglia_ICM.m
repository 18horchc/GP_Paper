% microglia_ICM.m — Independent SE GPs vs ICM multi-output GP for M1/M2.
% Joint GP on raw counts with ICM covariance cov{M_d(t), M_d'(t')} = B_{dd'} k(t,t').
% Optional figures: ICM+VO and ICM+Pensoneault (toggled below).
% Each figure: 2x2 — left = independent, right = method; top = full, bottom = averaged.
% Day 5 included in both M1 and M2.

clear; close all; clc;

%% ===== Configuration =====
% Which figures to produce
run_fig_icm       = true;    % independent vs unconstrained ICM
run_fig_icm_vo    = false;   % independent vs ICM + VO at (t=0, y=5)
run_fig_icm_bound = true;   % independent vs ICM + Pensoneault lower bound

k_plot    = 1.96;
max_iters = -200;
tgrid     = (0:0.1:14)';

% VO (used if run_fig_icm_vo)
t_vo     = 0;
y_vo     = 5;
sigma_vo = 0.5;   % count-unit noise on VO (same as microglia_VO.m)

% Pensoneault (used if run_fig_icm_bound)
eta_pens      = 0.022;
k_pens        = -sqrt(2) * erfinv(2 * eta_pens - 1);
n_constraint  = 41;
X_c           = linspace(0, 14, n_constraint)';
nTry_pens     = 2000;
nMultistart   = 10;
opts_pens = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, 'Display', 'off', ...
    'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);

if ~(run_fig_icm || run_fig_icm_vo || run_fig_icm_bound)
    fprintf('No figures selected (all run_fig_* = false). Exiting.\n');
    return;
end

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
temporalKernel = @covSEiso;

fprintf('=== Independent SE GP vs ICM MOGP ===\n');
fprintf('Figures: ICM=%d | ICM+VO=%d | ICM+bound=%d\n', ...
    run_fig_icm, run_fig_icm_vo, run_fig_icm_bound);

%% ===== Fit =====
results = struct([]);
for didx = 1:numel(datasets)
    ds = datasets(didx);
    fprintf('\n--- Dataset: %s ---\n', ds.name);

    naive = fit_naive_gp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot);
    icm = fit_icm_mogp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot);

    results(didx).name = ds.name;
    results(didx).naive = naive;
    results(didx).icm = icm;
    results(didx).timeM1 = ds.timeM1;
    results(didx).dataM1 = ds.dataM1;
    results(didx).timeM2 = ds.timeM2;
    results(didx).dataM2 = ds.dataM2;

    report_naive(ds.name, naive.report);
    report_icm(ds.name, 'ICM', icm.report);

    if run_fig_icm_vo
        icm_vo = fit_icm_mogp_vo(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
            t_vo, y_vo, sigma_vo, icm, tgrid, temporalKernel, max_iters, k_plot);
        results(didx).icm_vo = icm_vo;
        report_icm(ds.name, 'ICM+VO', icm_vo.report);
    end

    if run_fig_icm_bound
        icm_bound = fit_icm_mogp_lower_bound(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
            tgrid, temporalKernel, meanfunc, likfunc, icm.hyp, X_c, k_pens, k_plot, ...
            opts_pens, nTry_pens, nMultistart, 40 + didx);
        results(didx).icm_bound = icm_bound;
        report_icm_bound(ds.name, icm_bound.report);
    end
end

%% ===== Figures =====
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];
col_vo = [0.20, 0.45, 0.80];

if run_fig_icm
    plot_comparison_2x2(results, 'icm', ...
        {'Independent SE GP', 'ICM MOGP'}, ...
        'Microglia: independent SE vs ICM', tgrid, col_M1, col_M2, ...
        struct('mark_vo', false));
end

if run_fig_icm_vo
    plot_comparison_2x2(results, 'icm_vo', ...
        {'Independent SE GP', sprintf('ICM + VO (0, %.3g)', y_vo)}, ...
        'Microglia: independent SE vs ICM + VO', tgrid, col_M1, col_M2, ...
        struct('mark_vo', true, 't_vo', t_vo, 'y_vo', y_vo, 'col_vo', col_vo));
end

if run_fig_icm_bound
    plot_comparison_2x2(results, 'icm_bound', ...
        {'Independent SE GP', 'ICM + Pensoneault lower bound'}, ...
        'Microglia: independent SE vs ICM + Pensoneault', tgrid, col_M1, col_M2, ...
        struct('mark_vo', false, 'yline0', true));
end

fprintf('\nDone.\n');

%% ===== Local functions =====

function plot_comparison_2x2(results, method_field, method_titles, fig_name, ...
    tgrid, col_M1, col_M2, opts)
if ~isfield(opts, 'mark_vo'), opts.mark_vo = false; end
if ~isfield(opts, 'yline0'), opts.yline0 = false; end

figure('Color', 'w', 'Position', [60, 60, 1240, 900], 'Name', fig_name);
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for didx = 1:numel(results)
    ds = results(didx);
    methods = {ds.naive, ds.(method_field)};
    for midx = 1:2
        nexttile;
        ax = gca; ax.Layer = 'top';
        hold on; grid on;
        fit = methods{midx};
        plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1');
        plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2');
        if midx == 2 && opts.mark_vo
            scatter(ax, opts.t_vo, opts.y_vo, 110, 's', ...
                'MarkerFaceColor', opts.col_vo, 'MarkerEdgeColor', 'k', ...
                'LineWidth', 1.2, 'DisplayName', sprintf('VO (0, %.3g)', opts.y_vo));
        end
        if midx == 2 && opts.yline0
            yline(ax, 0, 'k:', 'HandleVisibility', 'off');
        end
        xlabel('Time (days)');
        ylabel('cells/mm^2');
        title(sprintf('%s — %s', ds.name, method_titles{midx}), 'Interpreter', 'none');
        xlim([0, 14]);
        if midx == 2 && opts.mark_vo
            ylim_auto_from_fit(ax, fit.M1, fit.M2, [ds.dataM1; opts.y_vo], [ds.dataM2; opts.y_vo]);
        else
            ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
        end
        legend('Location', 'northwest', 'FontSize', 8);
    end
end
end

function B = chol2cov(hyp)
L = zeros(2);
L(triu(true(2))) = hyp(:);
L(1:3:end) = exp(diag(L));
B = L' * L;
end

function rho = corr_from_B(B)
rho = B(1, 2) / sqrt(max(B(1, 1) * B(2, 2), eps));
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

function covICM = build_icm_kernel(temporalKernel)
covICM = {@covProd, { ...
    {@covMask, {1, temporalKernel}}, ...
    {@covMask, {2, {@covDiscrete, 2}}} }};
end

function [hyp0, prior, inffunc] = init_icm_hyp(temporalKernel, x_aug, y_aug)
meanfunc = @meanZero;
likfunc  = @likGauss;
covICM   = build_icm_kernel(temporalKernel);

[nTemp, hasAlpha] = temporal_hyp_layout(temporalKernel);
t_all = x_aug(:, 1);
ell0 = max(std(t_all), 0.5);
Lchol0 = [log(sqrt(0.5)); 0; log(sqrt(0.5))];

hyp0.mean = [];
if hasAlpha
    hyp0.cov = [log(ell0); 0; log(1); Lchol0];
else
    hyp0.cov = [log(ell0); 0; Lchol0];
end
hyp0.lik = log(0.1);

prior.cov = cell(1, nTemp + 3);
prior.cov{1} = {@priorGauss, log(ell0), 0.5^2};
prior.cov{2} = @priorClamped;
if hasAlpha
    prior.cov{3} = {@priorGauss, 0, 0.5^2};
end
inffunc = {@infPrior, @infGaussLik, prior};

gp(hyp0, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);
end

function out = fit_icm_mogp(timeM1, dataM1, timeM2, dataM2, tgrid, ...
    temporalKernel, meanfunc, likfunc, max_iters, k_plot)

LABEL_M1 = 1;
LABEL_M2 = 2;

y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);
mu1 = mean(y_M1); sd1 = std(y_M1);
mu2 = mean(y_M2); sd2 = std(y_M2);
if sd1 < eps, sd1 = 1; end
if sd2 < eps, sd2 = 1; end

x_aug = [timeM1(:), LABEL_M1 * ones(numel(timeM1), 1); ...
         timeM2(:), LABEL_M2 * ones(numel(timeM2), 1)];
y_aug = [ (y_M1 - mu1) / sd1; (y_M2 - mu2) / sd2 ];

covICM = build_icm_kernel(temporalKernel);
[hyp0, ~, inffunc] = init_icm_hyp(temporalKernel, x_aug, y_aug);

hyp = minimize(hyp0, @gp, max_iters, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);
nlml = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);

x_te_M1 = [tgrid, LABEL_M1 * ones(size(tgrid))];
x_te_M2 = [tgrid, LABEL_M2 * ones(size(tgrid))];
[~, ~, fmu1, fs21] = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug, x_te_M1);
[~, ~, fmu2, fs22] = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug, x_te_M2);

mu_y1 = mu1 + sd1 * fmu1(:);
sf_y1 = sd1 * sqrt(max(fs21(:), 0));
mu_y2 = mu2 + sd2 * fmu2(:);
sf_y2 = sd2 * sqrt(max(fs22(:), 0));

out.M1 = pack_raw_fit(mu_y1, sf_y1, k_plot);
out.M2 = pack_raw_fit(mu_y2, sf_y2, k_plot);
out.hyp = hyp;
out.nlml = nlml;
out.mu1 = mu1; out.sd1 = sd1;
out.mu2 = mu2; out.sd2 = sd2;

[nTemp, ~] = temporal_hyp_layout(temporalKernel);
B = chol2cov(hyp.cov(nTemp + (1:3)));
out.report.nlml = nlml;
out.report.ell = exp(hyp.cov(1));
out.report.B = B;
out.report.rho = corr_from_B(B);
out.report.sn = exp(hyp.lik);
end

function out = fit_icm_mogp_vo(timeM1, dataM1, timeM2, dataM2, ...
    t_vo, y_vo, sigma_vo, icm_unc, tgrid, temporalKernel, max_iters, k_plot)
% ICM with heteroscedastic VO: real sn from unconstrained ICM; VO sigma fixed.

LABEL_M1 = 1;
LABEL_M2 = 2;

y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);
mu1 = icm_unc.mu1; sd1 = icm_unc.sd1;
mu2 = icm_unc.mu2; sd2 = icm_unc.sd2;

n1 = numel(timeM1);
n2 = numel(timeM2);
x_real = [timeM1(:), LABEL_M1 * ones(n1, 1); ...
          timeM2(:), LABEL_M2 * ones(n2, 1)];
y_real = [ (y_M1 - mu1) / sd1; (y_M2 - mu2) / sd2 ];

x_vo = [t_vo, LABEL_M1; t_vo, LABEL_M2];
y_vo_std = [(y_vo - mu1) / sd1; (y_vo - mu2) / sd2];

x_aug = [x_real; x_vo];
y_aug = [y_real; y_vo_std];

sn_real = exp(icm_unc.hyp.lik);
noise_var = [sn_real^2 * ones(n1 + n2, 1); ...
             (sigma_vo / sd1)^2; (sigma_vo / sd2)^2];

covICM = build_icm_kernel(temporalKernel);
[nTemp, ~] = temporal_hyp_layout(temporalKernel);

hyp0 = icm_unc.hyp;
hyp0.lik = [];
hyp0.mean = [];

obj = @(h) gp_icm_hetero_noise('nlml', h, covICM, x_aug, y_aug, noise_var, nTemp);
hyp = minimize(hyp0, obj, max_iters);
hyp.cov(2) = 0;
nlml = gp_icm_hetero_noise('nlml', hyp, covICM, x_aug, y_aug, noise_var, nTemp);

x_te_M1 = [tgrid, LABEL_M1 * ones(size(tgrid))];
x_te_M2 = [tgrid, LABEL_M2 * ones(size(tgrid))];
[~, ~, fmu1, fs21] = gp_icm_hetero_noise('pred', hyp, covICM, x_aug, y_aug, noise_var, nTemp, x_te_M1);
[~, ~, fmu2, fs22] = gp_icm_hetero_noise('pred', hyp, covICM, x_aug, y_aug, noise_var, nTemp, x_te_M2);

mu_y1 = mu1 + sd1 * fmu1(:);
sf_y1 = sd1 * sqrt(max(fs21(:), 0));
mu_y2 = mu2 + sd2 * fmu2(:);
sf_y2 = sd2 * sqrt(max(fs22(:), 0));

out.M1 = pack_raw_fit(mu_y1, sf_y1, k_plot);
out.M2 = pack_raw_fit(mu_y2, sf_y2, k_plot);
out.hyp = hyp;
out.nlml = nlml;

B = chol2cov(hyp.cov(nTemp + (1:3)));
out.report.nlml = nlml;
out.report.ell = exp(hyp.cov(1));
out.report.B = B;
out.report.rho = corr_from_B(B);
out.report.sn = sn_real;
end

function varargout = gp_icm_hetero_noise(mode, hyp, covICM, x, y, noise_var, nTemp, xs)
switch lower(mode)
    case 'nlml'
        [varargout{1:nargout}] = icm_hetero_nlml(hyp, covICM, x, y, noise_var, nTemp);
    case 'pred'
        [varargout{1}, varargout{2}, varargout{3}, varargout{4}] = ...
            icm_hetero_pred(hyp, covICM, x, y, noise_var, xs);
    otherwise
        error('gp_icm_hetero_noise:UnknownMode', 'Unknown mode: %s', mode);
end
end

function [nlml, dnlml] = icm_hetero_nlml(hyp, covICM, x, y, noise_var, nTemp)
hyp.cov(2) = 0;
[Ky, z, nTot] = icm_build_Ky(hyp, covICM, x, y, noise_var);
L = chol(Ky, 'lower');
alpha = L' \ (L \ z);
nlml = 0.5 * (z' * alpha) + sum(log(diag(L))) + 0.5 * nTot * log(2 * pi);

if nargout > 1
    nCov = numel(hyp.cov);
    dnll_vec = zeros(nCov, 1);
    step = 1e-4;
    free_idx = [1, (nTemp + 1):(nTemp + 3)];
    for ii = 1:numel(free_idx)
        i = free_idx(ii);
        hp = hyp;
        hp.cov(i) = hp.cov(i) + step;
        hp.cov(2) = 0;
        nlml_p = icm_hetero_nlml_value(hp, covICM, x, y, noise_var);
        hp.cov(i) = hp.cov(i) - 2 * step;
        hp.cov(2) = 0;
        nlml_m = icm_hetero_nlml_value(hp, covICM, x, y, noise_var);
        dnll_vec(i) = (nlml_p - nlml_m) / (2 * step);
    end
    dnlml = hyp;
    dnlml.cov = dnll_vec;
    dnlml.lik = [];
    dnlml.mean = [];
end
end

function nlml = icm_hetero_nlml_value(hyp, covICM, x, y, noise_var)
[Ky, z, nTot] = icm_build_Ky(hyp, covICM, x, y, noise_var);
L = chol(Ky, 'lower');
alpha = L' \ (L \ z);
nlml = 0.5 * (z' * alpha) + sum(log(diag(L))) + 0.5 * nTot * log(2 * pi);
end

function [ymu, ys2, fmu, fs2] = icm_hetero_pred(hyp, covICM, x, y, noise_var, xs)
hyp.cov(2) = 0;
[Ky, z] = icm_build_Ky(hyp, covICM, x, y, noise_var);
L = chol(Ky, 'lower');
alpha = L' \ (L \ z);

K_star = feval(covICM{:}, hyp.cov, x, xs);
fmu = K_star' * alpha;
V = L \ K_star;
k_diag = feval(covICM{:}, hyp.cov, xs, 'diag');
fs2 = max(k_diag - sum(V.^2, 1).', 0);
fmu = fmu(:);
fs2 = fs2(:);
ymu = fmu;
ys2 = fs2;
end

function [Ky, z, nTot] = icm_build_Ky(hyp, covICM, x, y, noise_var)
y = y(:);
noise_var = noise_var(:);
nTot = size(x, 1);
K_f = feval(covICM{:}, hyp.cov, x);
jitter = 1e-8 * mean(diag(K_f));
Ky = K_f + diag(noise_var + jitter);
z = y;
end

function out = fit_icm_mogp_lower_bound(timeM1, dataM1, timeM2, dataM2, tgrid, ...
    temporalKernel, meanfunc, likfunc, hyp_unc, X_c, k_pens, k_plot, ...
    opts_pens, nTry, nMultistart, rng_seed)
% ICM MOGP with Pensoneault lower bound at 0 on both M1 and M2 at X_c.

LABEL_M1 = 1;
LABEL_M2 = 2;

y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);
mu1 = mean(y_M1); sd1 = std(y_M1);
mu2 = mean(y_M2); sd2 = std(y_M2);
if sd1 < eps, sd1 = 1; end
if sd2 < eps, sd2 = 1; end

x_aug = [timeM1(:), LABEL_M1 * ones(numel(timeM1), 1); ...
         timeM2(:), LABEL_M2 * ones(numel(timeM2), 1)];
y_aug = [ (y_M1 - mu1) / sd1; (y_M2 - mu2) / sd2 ];

covICM = build_icm_kernel(temporalKernel);
[~, ~, inffunc] = init_icm_hyp(temporalKernel, x_aug, y_aug);

[nTemp, hasAlpha] = temporal_hyp_layout(temporalKernel);
[theta_unc, hyp_tpl] = icm_hyp_to_theta(hyp_unc, nTemp, hasAlpha);
sn_fixed = hyp_unc.lik;
hyp_tpl.lik = sn_fixed;

ell_bounds_lo = 0.05;
ell_ub = 14;
chol_lo = -5;
chol_hi = 5;
if hasAlpha
    hyp_lb = [log(ell_bounds_lo); log(0.01); chol_lo; chol_lo; chol_lo];
    hyp_ub = [log(ell_ub); log(10); chol_hi; chol_hi; chol_hi];
else
    hyp_lb = [log(ell_bounds_lo); chol_lo; chol_lo; chol_lo];
    hyp_ub = [log(ell_ub); chol_hi; chol_hi; chol_hi];
end

objfun = @(theta) gp(icm_theta_to_hyp(theta, hyp_tpl, nTemp, hasAlpha), ...
    inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);
nonlcon = @(theta) pens_constraints_lower_icm(theta, hyp_tpl, inffunc, covICM, ...
    meanfunc, likfunc, x_aug, y_aug, X_c, k_pens, mu1, sd1, mu2, sd2, nTemp, hasAlpha);

theta_unc_box = min(max(theta_unc, hyp_lb), hyp_ub);

fprintf('  Pensoneault ICM multistart: %d random starts\n', nTry);
feasible_starts = zeros(numel(theta_unc), 0);
best_feas_nlml = inf;
best_feas_theta = nan(size(theta_unc));
rng(rng_seed);
for t = 1:nTry
    theta_try = hyp_lb + rand(size(theta_unc)) .* (hyp_ub - hyp_lb);
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
theta_opt = nan(size(theta_unc));
nlml = nan;
exitflag = -99;
nStarts = size(starts_for_fmincon, 2);
for j = 1:nStarts
    theta0_j = starts_for_fmincon(:, j);
    [theta_j, nlml_j, ef_j] = fmincon(objfun, theta0_j, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts_pens);
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

hyp = icm_theta_to_hyp(theta_opt, hyp_tpl, nTemp, hasAlpha);
[c_final, ~] = nonlcon(theta_opt);
max_c = max(c_final);

x_te_M1 = [tgrid, LABEL_M1 * ones(size(tgrid))];
x_te_M2 = [tgrid, LABEL_M2 * ones(size(tgrid))];
[~, ~, fmu1, fs21] = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug, x_te_M1);
[~, ~, fmu2, fs22] = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug, x_te_M2);

mu_y1 = mu1 + sd1 * fmu1(:);
sf_y1 = sd1 * sqrt(max(fs21(:), 0));
mu_y2 = mu2 + sd2 * fmu2(:);
sf_y2 = sd2 * sqrt(max(fs22(:), 0));

out.M1 = pack_raw_fit(mu_y1, sf_y1, k_plot);
out.M2 = pack_raw_fit(mu_y2, sf_y2, k_plot);
out.hyp = hyp;
out.nlml = nlml;

B = chol2cov(hyp.cov(nTemp + (1:3)));
out.report.nlml = nlml;
out.report.ell = exp(hyp.cov(1));
out.report.B = B;
out.report.rho = corr_from_B(B);
out.report.sn = exp(hyp.lik);
out.report.max_c = max_c;
out.report.exitflag = exitflag;
end

function [theta, hyp_tpl] = icm_hyp_to_theta(hyp_unc, nTemp, hasAlpha)
hyp_cov = hyp_unc.cov(:);
chol_idx = nTemp + (1:3);
if hasAlpha
    theta = [hyp_cov(1); hyp_cov(3); hyp_cov(chol_idx)];
else
    theta = [hyp_cov(1); hyp_cov(chol_idx)];
end
hyp_tpl = struct('mean', [], 'cov', hyp_cov, 'lik', hyp_unc.lik);
end

function hyp = icm_theta_to_hyp(theta, hyp_tpl, nTemp, hasAlpha)
hyp = hyp_tpl;
hyp.mean = [];
hyp_cov = hyp_tpl.cov(:);
chol_idx = nTemp + (1:3);
hyp_cov(1) = theta(1);
if hasAlpha
    hyp_cov(3) = theta(2);
    hyp_cov(chol_idx) = theta(3:5);
else
    hyp_cov(chol_idx) = theta(2:4);
end
hyp.cov = hyp_cov;
end

function [c, ceq] = pens_constraints_lower_icm(theta, hyp_tpl, inffunc, covICM, ...
    meanfunc, likfunc, x_aug, y_aug, X_c, k_pens, mu1, sd1, mu2, sd2, nTemp, hasAlpha)
hyp = icm_theta_to_hyp(theta, hyp_tpl, nTemp, hasAlpha);

LABEL_M1 = 1;
LABEL_M2 = 2;
nC = numel(X_c);
x_te = [X_c(:), LABEL_M1 * ones(nC, 1); X_c(:), LABEL_M2 * ones(nC, 1)];

[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug, x_te);
fmu1 = fmu(1:nC);
fs21 = fs2(1:nC);
fmu2 = fmu(nC+1:end);
fs22 = fs2(nC+1:end);

mu_count1 = mu1 + sd1 * fmu1(:);
sf_count1 = sd1 * sqrt(max(fs21(:), 0));
mu_count2 = mu2 + sd2 * fmu2(:);
sf_count2 = sd2 * sqrt(max(fs22(:), 0));

c = [k_pens .* sf_count1 - mu_count1; k_pens .* sf_count2 - mu_count2];
ceq = [];
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

function report_naive(dataset_name, report)
fprintf('[%s | Independent] NLML=%.4f, ell_M1=%.4f, ell_M2=%.4f, sn_M1=%.4f, sn_M2=%.4f\n', ...
    dataset_name, report.nlml, report.ell_M1, report.ell_M2, report.sn_M1, report.sn_M2);
end

function report_icm(dataset_name, method_name, report)
fprintf('[%s | %s] NLML=%.4f, ell=%.4f, rho=%.4f, sn=%.4f\n', ...
    dataset_name, method_name, report.nlml, report.ell, report.rho, report.sn);
fprintf('  B = [%.4f %.4f; %.4f %.4f]\n', ...
    report.B(1,1), report.B(1,2), report.B(2,1), report.B(2,2));
end

function report_icm_bound(dataset_name, report)
fprintf('[%s | ICM+Pensoneault] NLML=%.4f, ell=%.4f, rho=%.4f, sn=%.4f', ...
    dataset_name, report.nlml, report.ell, report.rho, report.sn);
fprintf(', exitflag=%d, max(c)=%.4g\n', report.exitflag, report.max_c);
fprintf('  B = [%.4f %.4f; %.4f %.4f]\n', ...
    report.B(1,1), report.B(1,2), report.B(2,1), report.B(2,2));
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
