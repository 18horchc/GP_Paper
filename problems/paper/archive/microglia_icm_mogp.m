% microglia_icm_mogp.m — Baseline 1: raw-count ICM multi-output GP for M1/M2.
% Joint GP on raw counts M_d(t) with ICM covariance
%   cov{M_d(t), M_d'(t')} = B_{dd'} k(t,t').
% Compares ICM vs independent raw-scale GPs on full and averaged data.
% Active raw path: fully unconstrained Gaussian predictions (negative mean/bands allowed).
%
% Log-count path (z = log(1 + M/count_delta), back-transform via expm1) is
% preserved in %{ ... %} blocks below for easy restoration.

% Optional overrides from caller workspace (before run): run_se_smoke_test
overrideSe = [];
if evalin('caller', 'exist(''run_se_smoke_test'',''var'')')
    overrideSe = evalin('caller', 'run_se_smoke_test');
end
%{
overrideSweep = [];
if evalin('caller', 'exist(''run_delta_sweep'',''var'')')
    overrideSweep = evalin('caller', 'run_delta_sweep');
end
%}

clearvars -except overrideSe;
close all; clc;

%% ===== Configuration =====
kernel_name       = 'se';  % 'matern32' | 'matern52' | 'se' | 'rq'
k_plot            = 1.96;        % ~95% band multiplier
% Pensoneault lower-bound (Figures 2-3)
eta_pens          = 0.022;       % 2.2% tail probability
k_pens            = -sqrt(2) * erfinv(2 * eta_pens - 1);
n_constraint      = 41;
X_c               = linspace(0, 14, n_constraint)';
ell_bounds_lo     = 0.05;
ell_ub            = 14;
opts_pens = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, 'Display', 'off', ...
    'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);
nTry_pens         = 2000;
nMultistart_pens  = 10;
%{
t_impute          = 5;           % day for M2 imputation highlight (M2 has no obs here)
%}
max_iters         = -200;        % GPML minimize budget (<0 = function evals)
run_se_smoke_test = false;       % set true to verify SE temporal kernel on full data
if ~isempty(overrideSe)
    run_se_smoke_test = overrideSe;
end
clear overrideSe;

%{
% --- log-count path config (commented out) ---
count_delta_override = [];
run_delta_sweep = true;
if ~isempty(overrideSweep)
    run_delta_sweep = overrideSweep;
end
%}

%% ===== Data (same as microglia.m) =====
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

timeM2 = [0, 1, 3, 5, 7, 14, ...   % Hu2012 (day 5 restored)
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

fprintf('=== Raw-count ICM MOGP baseline ===\n');
fprintf('kernel = %s\n', kernel_name);
fprintf('Pensoneault lower bound: eta = %.3g%%, k = %.4f, X_c: %d points on [0, 14]\n', ...
    100 * eta_pens, k_pens, n_constraint);

%{
% --- log-count path: count_delta preprocessing (commented out) ---
pos_all = [dataM1(dataM1 > 0); dataM2(dataM2 > 0)];
count_delta_default = 0.5 * min(pos_all, [], 'all');
count_delta = count_delta_default;
if ~isempty(count_delta_override)
    count_delta = count_delta_override;
end
fprintf('=== Log-count ICM MOGP baseline ===\n');
fprintf('count_delta = %.4g (default: half of min positive count = %.4g)\n', ...
    count_delta, count_delta_default);
fprintf('kernel = %s\n', kernel_name);
%}

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
addpath(fileparts(fileparts(mfilename('fullpath'))));  % problems/

meanfunc = @meanZero;
likfunc  = @likGauss;
tgrid    = (0:0.1:14)';
temporalKernel = build_temporal_kernel(kernel_name);

%% ===== Fit naive + ICM for each dataset =====
results = struct([]);
for didx = 1:numel(datasets)
    ds = datasets(didx);
    fprintf('\n--- Dataset: %s ---\n', ds.name);

    naive = fit_naive_gp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot);
    naive_bound = fit_naive_gp_lower_bound(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        naive.hyp_M1, naive.hyp_M2, tgrid, temporalKernel, meanfunc, likfunc, ...
        X_c, k_pens, k_plot, opts_pens, nTry_pens, nMultistart_pens, 50 + 2*didx, 51 + 2*didx);
    icm = fit_icm_mogp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot);
    icm_bound = fit_icm_mogp_lower_bound(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, temporalKernel, meanfunc, likfunc, icm.hyp, X_c, k_pens, k_plot, ...
        opts_pens, nTry_pens, nMultistart_pens, 40 + didx);

    %{
    naive = fit_naive_log_gp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        count_delta, tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot);
    icm = fit_icm_mogp_log(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        count_delta, tgrid, temporalKernel, meanfunc, likfunc, max_iters, t_impute, k_plot);
    %}

    results(didx).name = ds.name;
    results(didx).naive = naive;
    results(didx).naive_bound = naive_bound;
    results(didx).icm = icm;
    results(didx).icm_bound = icm_bound;
    results(didx).timeM1 = ds.timeM1;
    results(didx).dataM1 = ds.dataM1;
    results(didx).timeM2 = ds.timeM2;
    results(didx).dataM2 = ds.dataM2;

    report_fit(ds.name, 'Naive GP (independent)', naive.report);
    report_fit_naive_bounded(ds.name, 'Naive GP (Pensoneault lower bound)', naive_bound.report);
    report_fit(ds.name, 'ICM MOGP (coupled)', icm.report);
    report_fit_bounded(ds.name, 'ICM MOGP (Pensoneault lower bound)', icm_bound.report);
    %{
    report_fit(ds.name, 'Naive log-GP', naive.report);
    report_fit(ds.name, 'ICM MOGP (log)', icm.report);
    %}
end

%% ===== Comparison figure: 2 x 2 (dataset x method) =====
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];

figure('Color', 'w', 'Position', [60, 60, 1240, 900], ...
    'Name', 'Microglia: naive GP vs raw-count ICM MOGP');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for didx = 1:numel(results)
    ds = results(didx);
    methods = {ds.naive, ds.icm};
    method_titles = {'Naive GP (independent)', 'ICM MOGP (coupled)'};

    for midx = 1:2
        nexttile;
        ax = gca; ax.Layer = 'top';
        hold on; grid on;
        fit = methods{midx};
        plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, k_plot, 'M1');
        plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, k_plot, 'M2');
        %{
        if strcmp(ds.name, 'full') && midx == 2
            xline(t_impute, ':', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1.2, ...
                'DisplayName', sprintf('M2 impute t=%g', t_impute));
        end
        %}
        xlabel('Time (days)');
        ylabel('cells/mm^2');
        title(sprintf('%s — %s', ds.name, method_titles{midx}), 'Interpreter', 'none');
        xlim([0, 14]);
        % ylim([0, inf]);  % clamped view (commented out)
        ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
        legend('Location', 'northwest', 'FontSize', 8);
    end
end

%% ===== Bounded ICM figure: 2 x 2 (dataset x method) =====
figure(2);
set(gcf, 'Color', 'w', 'Position', [100, 40, 1240, 900], ...
    'Name', 'Microglia: naive GP vs Pensoneault lower-bound ICM MOGP');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for didx = 1:numel(results)
    ds = results(didx);
    methods = {ds.naive, ds.icm_bound};
    method_titles = {'Naive GP (independent)', 'ICM MOGP (Pensoneault lower bound)'};

    for midx = 1:2
        nexttile;
        ax = gca; ax.Layer = 'top';
        hold on; grid on;
        fit = methods{midx};
        plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, k_plot, 'M1');
        plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, k_plot, 'M2');
        xlabel('Time (days)');
        ylabel('cells/mm^2');
        title(sprintf('%s — %s', ds.name, method_titles{midx}), 'Interpreter', 'none');
        xlim([0, 14]);
        ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
        legend('Location', 'northwest', 'FontSize', 8);
    end
end

%% ===== Bounded naive + ICM figure: 2 x 2 (dataset x method) =====
figure(3);
set(gcf, 'Color', 'w', 'Position', [140, 20, 1240, 900], ...
    'Name', 'Microglia: Pensoneault lower-bound naive GP vs ICM MOGP');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for didx = 1:numel(results)
    ds = results(didx);
    methods = {ds.naive_bound, ds.icm_bound};
    method_titles = {'Naive GP (Pensoneault lower bound)', 'ICM MOGP (Pensoneault lower bound)'};

    for midx = 1:2
        nexttile;
        ax = gca; ax.Layer = 'top';
        hold on; grid on;
        fit = methods{midx};
        plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, k_plot, 'M1');
        plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, k_plot, 'M2');
        xlabel('Time (days)');
        ylabel('cells/mm^2');
        title(sprintf('%s — %s', ds.name, method_titles{midx}), 'Interpreter', 'none');
        xlim([0, 14]);
        ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
        legend('Location', 'northwest', 'FontSize', 8);
    end
end

%{
%% ===== Delta sensitivity (full data, ICM only) — log-count path =====
if run_delta_sweep
    fprintf('\n=== Delta sensitivity (full data, ICM) ===\n');
    delta_sweep = count_delta_default * [0.5, 1, 2];
    if ~isempty(count_delta_override)
        delta_sweep = unique([delta_sweep(:); count_delta], 'stable');
    end

    n_sweep = numel(delta_sweep);
    sweep = struct('count_delta', num2cell(delta_sweep(:)), ...
        'nlml', cell(n_sweep, 1), ...
        'rho', cell(n_sweep, 1), ...
        'M2_day5', cell(n_sweep, 1), ...
        'M2_day5_sd', cell(n_sweep, 1));

    fprintf('%-10s  %-10s  %-8s  %-12s  %-12s\n', 'delta', 'NLML', 'rho', 'M2@day5', 'M2@day5_sd');
    for sidx = 1:n_sweep
        dval = delta_sweep(sidx);
        fit_s = fit_icm_mogp_log(timeM1, dataM1, timeM2, dataM2, ...
            dval, tgrid, temporalKernel, meanfunc, likfunc, max_iters, t_impute, k_plot);
        sweep(sidx).nlml = fit_s.report.nlml;
        sweep(sidx).rho = fit_s.report.rho;
        sweep(sidx).M2_day5 = fit_s.report.M2_at_t;
        sweep(sidx).M2_day5_sd = fit_s.report.M2_at_t_sd;
        fprintf('%-10.4g  %-10.4f  %-8.4f  %-12.2f  %-12.2f\n', ...
            dval, fit_s.report.nlml, fit_s.report.rho, ...
            fit_s.report.M2_at_t, fit_s.report.M2_at_t_sd);
    end

    figure('Color', 'w', 'Position', [120, 120, 900, 360], ...
        'Name', 'Delta sensitivity summary');
    tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

    deltas = delta_sweep(:);
    nexttile;
    plot(deltas, cellfun(@(x) x, {sweep.nlml}), 'o-', 'LineWidth', 1.5);
    xlabel('\delta'); ylabel('NLML'); title('NLML vs \delta'); grid on;

    nexttile;
    plot(deltas, cellfun(@(x) x, {sweep.rho}), 'o-', 'LineWidth', 1.5);
    xlabel('\delta'); ylabel('\rho'); title('Output correlation \rho'); grid on;

    nexttile;
    plot(deltas, cellfun(@(x) x, {sweep.M2_day5}), 'o-', 'LineWidth', 1.5);
    xlabel('\delta'); ylabel('M2 mean at day 5'); title('M2 imputation'); grid on;
end
%}

%% ===== Optional SE kernel smoke test =====
if run_se_smoke_test
    fprintf('\n=== Kernel smoke test: SE (full data) ===\n');
    se_kernel = build_temporal_kernel('se');
    se_fit = fit_icm_mogp(timeM1, dataM1, timeM2, dataM2, tgrid, ...
        se_kernel, meanfunc, likfunc, -100, k_plot);
    fprintf('SE ICM: NLML=%.4f, ell=%.4f, rho=%.4f\n', ...
        se_fit.report.nlml, se_fit.report.ell, se_fit.report.rho);
    % fprintf('SE ICM: NLML=%.4f, ell=%.4f, rho=%.4f, M2@day5=%.2f\n', ...
    %     se_fit.report.nlml, se_fit.report.ell, se_fit.report.rho, se_fit.report.M2_at_t);
    %{
    se_fit = fit_icm_mogp_log(timeM1, dataM1, timeM2, dataM2, count_delta, tgrid, ...
        se_kernel, meanfunc, likfunc, -100, t_impute, k_plot);
    %}
end

fprintf('\nDone.\n');

%% ===== Local functions =====

function covfunc = build_temporal_kernel(name)
switch lower(name)
    case 'matern32'
        covfunc = {@covMaterniso, 3};   % nu = d/2 = 3/2
    case 'matern52'
        covfunc = {@covMaterniso, 5};   % nu = d/2 = 5/2
    case 'se'
        covfunc = @covSEiso;
    case 'rq'
        covfunc = @covRQiso;
    otherwise
        error('Unknown kernel: %s (use matern32, matern52, se, rq)', name);
end
end

% converts 3 stored numbers (Cholesky factors) intot eh 2x2 matrix B that
% controls M1/M2 coupling. Builds a lower-triangular L, exponentiates
% diagonals, then B = L' * L (which is always positive semidefinite)
function B = chol2cov(hyp)
L = zeros(2);
L(triu(true(2))) = hyp(:);
L(1:3:end) = exp(diag(L));
B = L' * L;
end

function rho = corr_from_B(B)
rho = B(1, 2) / sqrt(max(B(1, 1) * B(2, 2), eps));
end

%Tells other functions how many hyperparameters the temporal kernel has
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

%Assembles the coupled covariance: time kernel × discrete label kernel. 
% Conceptually: cov(i,j) = B(label_i, label_j) × k(time_i, time_j)
function covICM = build_icm_kernel(temporalKernel)
covICM = {@covProd, { ...
    {@covMask, {1, temporalKernel}}, ...
    {@covMask, {2, {@covDiscrete, 2}}} }};
end

%Sets starting values and priors for the coupled model: 
% Initial ell, fixed temporal sf (=1, clamped), initial B via Cholesky
% Initial noise sn
% Wraps inference in infPrior (Gaussian prior on ell, etc.)
%Also runs one gp(...) call to validate the setup.
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

%The full ICM multi-output GP on raw counts.
% Steps: 
%  1. Floor counts at 0; rescale M1 and M2 separately (mean/std).
%  2. Stack inputs as [time, label] and standardized counts.
%  3. build_icm_kernel + init_icm_hyp.
%  4. minimize → learn ell, B, sn.
%  5. Predict M1 and M2 on tgrid; convert back to count scale.
%  6. pack_raw_fit for bands; build report (NLML, ell, B, rho, sn).
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

[nTemp, ~] = temporal_hyp_layout(temporalKernel);
B = chol2cov(hyp.cov(nTemp + (1:3)));
out.report.nlml = nlml;
out.report.ell = exp(hyp.cov(1));
out.report.B = B;
out.report.rho = corr_from_B(B);
out.report.sn = exp(hyp.lik);
%{
out.report.M2_at_t = interp1(tgrid, out.M2.mu, t_impute, 'linear', 'extrap');
out.report.M2_at_t_sd = interp1(tgrid, out.M2.sf, t_impute, 'linear', 'extrap');
%}
end

% ICM MOGP with Pensoneault lower bound at 0 on both M1 and M2 at X_c.
% Fixes sigma_n from unconstrained ICM fit; optimizes ell (and alpha for RQ) + B via fmincon.
function out = fit_icm_mogp_lower_bound(timeM1, dataM1, timeM2, dataM2, tgrid, ...
    temporalKernel, meanfunc, likfunc, hyp_unc, X_c, k_pens, k_plot, ...
    opts_pens, nTry, nMultistart, rng_seed)

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
% Pensoneault lower bound at 0 in count units: mu - k*sigma >= 0  <=>  c <= 0.
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

%Fits two separate GPs — one for M1, one for M2 — with no coupling.
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
out.report.B = [];
out.report.rho = NaN;
%{
out.report.M2_at_t = interp1(tgrid, out.M2.mu, 5, 'linear', 'extrap');
out.report.M2_at_t_sd = interp1(tgrid, out.M2.sf, 5, 'linear', 'extrap');
%}
end

% Independent naive GPs with Pensoneault lower bound at 0 on M1 and M2 separately.
function out = fit_naive_gp_lower_bound(timeM1, dataM1, timeM2, dataM2, ...
    hyp_unc_M1, hyp_unc_M2, tgrid, temporalKernel, meanfunc, likfunc, ...
    X_c, k_pens, k_plot, opts_pens, nTry, nMultistart, rng_seed_M1, rng_seed_M2)

inffunc = @infGaussLik;
y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);

ell_bounds_lo = 0.05;
ell_ub = 14;
sf_bounds_M1 = [0.05, max(15, 1.5 * std(y_M1))];
sf_bounds_M2 = [0.05, max(15, 1.5 * std(y_M2))];

[~, hasAlpha] = temporal_hyp_layout(temporalKernel);
if hasAlpha
    hyp_lb_M1 = log([ell_bounds_lo; sf_bounds_M1(1); 0.01]);
    hyp_ub_M1 = log([ell_ub; sf_bounds_M1(2); 10]);
    hyp_lb_M2 = log([ell_bounds_lo; sf_bounds_M2(1); 0.01]);
    hyp_ub_M2 = log([ell_ub; sf_bounds_M2(2); 10]);
else
    hyp_lb_M1 = log([ell_bounds_lo; sf_bounds_M1(1)]);
    hyp_ub_M1 = log([ell_ub; sf_bounds_M1(2)]);
    hyp_lb_M2 = log([ell_bounds_lo; sf_bounds_M2(1)]);
    hyp_ub_M2 = log([ell_ub; sf_bounds_M2(2)]);
end

fprintf('  Pensoneault naive GP (M1) multistart: %d random starts\n', nTry);
[fit1.hyp, fit1.mu_y, fit1.sf_y, fit1.nlml, fit1.exitflag, fit1.max_c] = ...
    fit_scalar_gp_lower_bound(timeM1(:), y_M1, hyp_unc_M1, X_c, k_pens, tgrid, ...
    inffunc, meanfunc, temporalKernel, likfunc, hyp_lb_M1, hyp_ub_M1, ...
    opts_pens, nTry, nMultistart, rng_seed_M1, false);

fprintf('  Pensoneault naive GP (M2) multistart: %d random starts\n', nTry);
[fit2.hyp, fit2.mu_y, fit2.sf_y, fit2.nlml, fit2.exitflag, fit2.max_c] = ...
    fit_scalar_gp_lower_bound(timeM2(:), y_M2, hyp_unc_M2, X_c, k_pens, tgrid, ...
    inffunc, meanfunc, temporalKernel, likfunc, hyp_lb_M2, hyp_ub_M2, ...
    opts_pens, nTry, nMultistart, rng_seed_M2, false);

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
out.report.max_c_M1 = fit1.max_c;
out.report.max_c_M2 = fit2.max_c;
out.report.exitflag_M1 = fit1.exitflag;
out.report.exitflag_M2 = fit2.exitflag;
out.report.B = [];
out.report.rho = NaN;
end

function [hyp, mu_y, sf_y, nlml, exitflag, max_c] = fit_scalar_gp_lower_bound( ...
    x, y, hyp_unc, X_c, k_pens, tgrid, inffunc, meanfunc, covfunc, likfunc, ...
    hyp_lb, hyp_ub, opts, nTry, nMultistart, rng_seed, verbose)
% Pensoneault lower-bound scalar GP: minimize NLML subject to mu_f - k*sigma_f >= 0 at X_c.
if nargin < 17
    verbose = true;
end
x = x(:); y = y(:);
tgrid = tgrid(:);

sn_fixed = hyp_unc.lik;
hyp_tpl = struct('mean', [], 'cov', hyp_unc.cov(:), 'lik', sn_fixed);
theta_unc = hyp_unc.cov(:);
nTheta = numel(theta_unc);

objfun = @(theta) gp(scalar_theta_to_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x, y);
nonlcon = @(theta) pens_constraints_lower_scalar(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k_pens);

theta_unc_box = min(max(theta_unc, hyp_lb), hyp_ub);
if verbose
    fprintf('    Multistart: %d random starts\n', nTry);
end
feasible_starts = zeros(nTheta, 0);
best_feas_nlml = inf;
best_feas_theta = nan(nTheta, 1);
rng(rng_seed);
for t = 1:nTry
    theta_try = hyp_lb + rand(nTheta, 1) .* (hyp_ub - hyp_lb);
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
if verbose
    fprintf('    Feasible random starts: %d / %d\n', nFeas, nTry);
end
if nFeas > 0
    nlml_feas = arrayfun(@(j) objfun(feasible_starts(:, j)), 1:nFeas);
    [~, ord] = sort(nlml_feas, 'ascend');
    starts_for_fmincon = feasible_starts(:, ord(1:min(nMultistart, nFeas)));
else
    if verbose
        fprintf('    No feasible random start; using projected baseline theta.\n');
    end
    starts_for_fmincon = theta_unc_box;
end
starts_for_fmincon = [theta_unc_box, starts_for_fmincon];
starts_for_fmincon = starts_for_fmincon(:, 1:min(nMultistart + 1, size(starts_for_fmincon, 2)));

best_nlml = inf;
theta_opt = nan(nTheta, 1);
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
    if verbose
        fprintf('    Warning: no successful fmincon run; using fallback theta.\n');
    end
end

hyp = scalar_theta_to_hyp(theta_opt, hyp_tpl);
[c_final, ~] = nonlcon(theta_opt);
max_c = max(c_final);

[~, ~, mu_y, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, tgrid);
mu_y = mu_y(:);
sf_y = sqrt(max(s2(:), 0));
end

function hyp = scalar_theta_to_hyp(theta, hyp_tpl)
hyp = hyp_tpl;
hyp.cov = theta(:);
hyp.mean = [];
end

function [c, ceq] = pens_constraints_lower_scalar(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k_pens)
% Pensoneault lower bound at 0 on latent f: mu_f - k*sigma_f >= 0  <=>  c <= 0.
hyp = scalar_theta_to_hyp(theta, hyp_tpl);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, X_c(:));
m_xc = fmu(:);
s_xc = sqrt(max(fs2(:), 0));
c = k_pens .* s_xc - m_xc;
ceq = [];
end

% Fits one scalar GP on time → count:
%  Initializes ell, sf, sn
%  minimize + predict on tgrid
function fit = fit_single_gp(x, y, tgrid, temporalKernel, meanfunc, likfunc, max_iters)
x = x(:); y = y(:);
inffunc = @infGaussLik;
ell0 = max(std(x), 0.5);
sf0 = max(std(y), 0.1);
sn0 = 0.1 * sf0;

[nTemp, hasAlpha] = temporal_hyp_layout(temporalKernel);
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

% Packages predictions into a small struct for plotting:
function pheno = pack_raw_fit(mu, sf, k_plot)
pheno.mu = mu(:);
pheno.sf = sf(:);
pheno.lo = mu - k_plot .* sf;
pheno.hi = mu + k_plot .* sf;
% pheno.lo = max(0, mu - k_plot .* sf);  % clamped lower band (commented out)
end

function report_fit(dataset_name, method_name, report)
fprintf('[%s | %s] NLML=%.4f', dataset_name, method_name, report.nlml);
if ~isempty(report.B)
    fprintf(', ell=%.4f, rho=%.4f, sn=%.4f', report.ell, report.rho, report.sn);
    fprintf('\n  B = [%.4f %.4f; %.4f %.4f]', ...
        report.B(1,1), report.B(1,2), report.B(2,1), report.B(2,2));
else
    fprintf(', ell_M1=%.4f, ell_M2=%.4f', report.ell_M1, report.ell_M2);
end
%{
fprintf('\n  M2 at t=5: mean=%.2f, sd=%.4f\n', report.M2_at_t, report.M2_at_t_sd);
%}
fprintf('\n');
end

function report_fit_bounded(dataset_name, method_name, report)
fprintf('[%s | %s] NLML=%.4f', dataset_name, method_name, report.nlml);
fprintf(', ell=%.4f, rho=%.4f, sn=%.4f', report.ell, report.rho, report.sn);
fprintf(', exitflag=%d, max(c)=%.4g', report.exitflag, report.max_c);
fprintf('\n  B = [%.4f %.4f; %.4f %.4f]', ...
    report.B(1,1), report.B(1,2), report.B(2,1), report.B(2,2));
fprintf('\n');
end

function report_fit_naive_bounded(dataset_name, method_name, report)
fprintf('[%s | %s] NLML=%.4f', dataset_name, method_name, report.nlml);
fprintf(', ell_M1=%.4f, ell_M2=%.4f, sn_M1=%.4f, sn_M2=%.4f', ...
    report.ell_M1, report.ell_M2, report.sn_M1, report.sn_M2);
fprintf('\n  M1: exitflag=%d, max(c)=%.4g | M2: exitflag=%d, max(c)=%.4g\n', ...
    report.exitflag_M1, report.max_c_M1, report.exitflag_M2, report.max_c_M2);
end

function plot_phenotype(ax, tgrid, fit, t_data, y_data, col, ~, name)
tg = tgrid(:)';
lo = fit.lo(:)';
hi = fit.hi(:)';
mu = fit.mu(:)';
fill(ax, [tg, fliplr(tg)], [hi, fliplr(lo)], col, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', sprintf('%s band', name));
plot(ax, tgrid, mu, '--', 'Color', col, 'LineWidth', 2, ...
    'DisplayName', sprintf('%s mean', name));
scatter(ax, t_data, y_data, 36, 'filled', ...
    'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', ...
    'DisplayName', sprintf('%s data', name));
end

%  Sets y-axis limits from all means, bands, and data (with 5% padding). Allows negative values to show.
function ylim_auto_from_fit(ax, fitM1, fitM2, dataM1, dataM2)
vals = [fitM1.lo(:); fitM1.hi(:); fitM1.mu(:); ...
        fitM2.lo(:); fitM2.hi(:); fitM2.mu(:); ...
        dataM1(:); dataM2(:)];
pad = 0.05 * max(range(vals), 1);
ylim(ax, [min(vals) - pad, max(vals) + pad]);
end

%{
% ===== Log-count path local functions (commented out) =====

function z = count_to_log(M, count_delta)
z = log(1 + max(M(:), 0) ./ count_delta);
end

function M = log_to_count(z, count_delta)
M = count_delta .* expm1(z(:));
end

% Coupled fit on log-transformed counts
function out = fit_icm_mogp_log(timeM1, dataM1, timeM2, dataM2, count_delta, tgrid, ...
    temporalKernel, meanfunc, likfunc, max_iters, t_impute, k_plot)

LABEL_M1 = 1;
LABEL_M2 = 2;

z_M1 = count_to_log(dataM1, count_delta);
z_M2 = count_to_log(dataM2, count_delta);
mu1 = mean(z_M1); sd1 = std(z_M1);
mu2 = mean(z_M2); sd2 = std(z_M2);
if sd1 < eps, sd1 = 1; end
if sd2 < eps, sd2 = 1; end

x_aug = [timeM1(:), LABEL_M1 * ones(numel(timeM1), 1); ...
         timeM2(:), LABEL_M2 * ones(numel(timeM2), 1)];
y_aug = [ (z_M1(:) - mu1) / sd1; (z_M2(:) - mu2) / sd2 ];

covICM = build_icm_kernel(temporalKernel);
[hyp0, ~, inffunc] = init_icm_hyp(temporalKernel, x_aug, y_aug);

hyp = minimize(hyp0, @gp, max_iters, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);
nlml = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);

x_te_M1 = [tgrid, LABEL_M1 * ones(size(tgrid))];
x_te_M2 = [tgrid, LABEL_M2 * ones(size(tgrid))];
[~, ~, fmu1, fs21] = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug, x_te_M1);
[~, ~, fmu2, fs22] = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug, x_te_M2);

mu_z1 = mu1 + sd1 * fmu1(:);
sf_z1 = sd1 * sqrt(max(fs21(:), 0));
mu_z2 = mu2 + sd2 * fmu2(:);
sf_z2 = sd2 * sqrt(max(fs22(:), 0));

out.M1 = pack_count_fit(mu_z1, sf_z1, count_delta, k_plot);
out.M2 = pack_count_fit(mu_z2, sf_z2, count_delta, k_plot);
out.hyp = hyp;
out.nlml = nlml;

[nTemp, ~] = temporal_hyp_layout(temporalKernel);
B = chol2cov(hyp.cov(nTemp + (1:3)));
out.report.count_delta = count_delta;
out.report.nlml = nlml;
out.report.ell = exp(hyp.cov(1));
out.report.B = B;
out.report.rho = corr_from_B(B);
out.report.sn = exp(hyp.lik);
out.report.M2_at_t = interp1(tgrid, out.M2.mu, t_impute, 'linear', 'extrap');
out.report.M2_at_t_sd = interp1(tgrid, out.M2.sf, t_impute, 'linear', 'extrap');
end

% Independent log-GP baseline
function out = fit_naive_log_gp(timeM1, dataM1, timeM2, dataM2, count_delta, tgrid, ...
    temporalKernel, meanfunc, likfunc, max_iters, k_plot)

z_M1 = count_to_log(dataM1, count_delta);
z_M2 = count_to_log(dataM2, count_delta);

fit1 = fit_single_log_gp(timeM1, z_M1, tgrid, temporalKernel, meanfunc, likfunc, max_iters);
fit2 = fit_single_log_gp(timeM2, z_M2, tgrid, temporalKernel, meanfunc, likfunc, max_iters);

out.M1 = pack_count_fit(fit1.mu_z, fit1.sf_z, count_delta, k_plot);
out.M2 = pack_count_fit(fit2.mu_z, fit2.sf_z, count_delta, k_plot);
out.hyp_M1 = fit1.hyp;
out.hyp_M2 = fit2.hyp;
out.nlml = fit1.nlml + fit2.nlml;

out.report.count_delta = count_delta;
out.report.nlml = out.nlml;
out.report.ell_M1 = exp(fit1.hyp.cov(1));
out.report.ell_M2 = exp(fit2.hyp.cov(1));
out.report.sn_M1 = exp(fit1.hyp.lik);
out.report.sn_M2 = exp(fit2.hyp.lik);
out.report.B = [];
out.report.rho = NaN;
out.report.M2_at_t = interp1(tgrid, out.M2.mu, 5, 'linear', 'extrap');
out.report.M2_at_t_sd = interp1(tgrid, out.M2.sf, 5, 'linear', 'extrap');
end

% One log-GP (like fit_single_gp but on z)
function fit = fit_single_log_gp(x, z, tgrid, temporalKernel, meanfunc, likfunc, max_iters)
x = x(:); z = z(:);
inffunc = @infGaussLik;
ell0 = max(std(x), 0.5);
sf0 = max(std(z), 0.1);
sn0 = 0.1 * sf0;

[nTemp, hasAlpha] = temporal_hyp_layout(temporalKernel);
if hasAlpha
    hyp.mean = [];
    hyp.cov = log([ell0; sf0; 1]);
else
    hyp.mean = [];
    hyp.cov = log([ell0; sf0]);
end
hyp.lik = log(sn0);

hyp = minimize(hyp, @gp, max_iters, inffunc, meanfunc, temporalKernel, likfunc, x, z);
nlml = gp(hyp, inffunc, meanfunc, temporalKernel, likfunc, x, z);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, temporalKernel, likfunc, x, z, tgrid);

fit.hyp = hyp;
fit.nlml = nlml;
fit.mu_z = fmu(:);
fit.sf_z = sqrt(max(fs2(:), 0));
end

%Back-transform predictions to count scale; clamps lo at 0
function pheno = pack_count_fit(mu_z, sf_z, count_delta, k_plot)
pheno.mu_z = mu_z(:);
pheno.sf_z = sf_z(:);
pheno.mu = log_to_count(mu_z, count_delta);
pheno.sf = sf_z;
pheno.lo = max(0, log_to_count(mu_z - k_plot .* sf_z, count_delta));
pheno.hi = log_to_count(mu_z + k_plot .* sf_z, count_delta);
end

% Console reporting for log path (includes M2@day5)
function report_fit_log(dataset_name, method_name, report)
fprintf('[%s | %s] count_delta=%.4g, NLML=%.4f', ...
    dataset_name, method_name, report.count_delta, report.nlml);
if ~isempty(report.B)
    fprintf(', ell=%.4f, rho=%.4f, sn=%.4f', report.ell, report.rho, report.sn);
    fprintf('\n  B = [%.4f %.4f; %.4f %.4f]', ...
        report.B(1,1), report.B(1,2), report.B(2,1), report.B(2,2));
else
    fprintf(', ell_M1=%.4f, ell_M2=%.4f', report.ell_M1, report.ell_M2);
end
fprintf('\n  M2 at t=5: mean=%.2f, latent-sd=%.4f\n', report.M2_at_t, report.M2_at_t_sd);
end

%}
