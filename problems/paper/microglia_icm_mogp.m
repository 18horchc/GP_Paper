% microglia_icm_mogp.m — Baseline 1: log-count ICM multi-output GP for M1/M2.
% Joint GP on z_d(t) = log(1 + M_d(t)/count_delta) with ICM covariance
%   cov{z_d(t), z_d'(t')} = B_{dd'} k(t,t'),
% back-transform M_d(t) = count_delta * expm1(z_d(t)).
% Compares ICM vs independent log-scale GPs on full and averaged data.

% Optional overrides from caller workspace (before run): run_se_smoke_test, run_delta_sweep
overrideSe = [];
overrideSweep = [];
if evalin('caller', 'exist(''run_se_smoke_test'',''var'')')
    overrideSe = evalin('caller', 'run_se_smoke_test');
end
if evalin('caller', 'exist(''run_delta_sweep'',''var'')')
    overrideSweep = evalin('caller', 'run_delta_sweep');
end

clearvars -except overrideSe overrideSweep;
close all; clc;

%% ===== Configuration =====
count_delta_override = [];    % scalar to override default; [] uses default rule
kernel_name          = 'matern52';  % 'matern32' | 'matern52' | 'se' | 'rq'
k_plot         = 1.96;        % ~95% band multiplier on latent scale
t_impute       = 5;           % day for M2 imputation highlight (M2 has no obs here)
max_iters      = -200;        % GPML minimize budget (<0 = function evals)
run_delta_sweep = true;       % sensitivity over count_delta_default * [0.5, 1, 2]
run_se_smoke_test = false;    % set true to verify SE temporal kernel on full data
if ~isempty(overrideSweep)
    run_delta_sweep = overrideSweep;
end
if ~isempty(overrideSe)
    run_se_smoke_test = overrideSe;
end
clear overrideSe overrideSweep;

%% ===== Data (same as microglia.m) =====
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

datasets = struct( ...
    'name', {'full', 'averaged'}, ...
    'timeM1', {timeM1(:), newtime(:)}, ...
    'dataM1', {dataM1(:), datapointsM1(:)}, ...
    'timeM2', {timeM2(:), newtimeM2(:)}, ...
    'dataM2', {dataM2(:), datapointsM2(:)});

%% ===== count_delta (fixed preprocessing, user-visible) =====
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

    naive = fit_naive_log_gp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        count_delta, tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot);
    icm = fit_icm_mogp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        count_delta, tgrid, temporalKernel, meanfunc, likfunc, max_iters, t_impute, k_plot);

    results(didx).name = ds.name;
    results(didx).naive = naive;
    results(didx).icm = icm;
    results(didx).timeM1 = ds.timeM1;
    results(didx).dataM1 = ds.dataM1;
    results(didx).timeM2 = ds.timeM2;
    results(didx).dataM2 = ds.dataM2;

    report_fit(ds.name, 'Naive log-GP', naive.report);
    report_fit(ds.name, 'ICM MOGP', icm.report);
end

%% ===== Comparison figure: 2 x 2 (dataset x method) =====
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];

figure('Color', 'w', 'Position', [60, 60, 1240, 900], ...
    'Name', 'Microglia: naive log-GP vs log-count ICM MOGP');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for didx = 1:numel(results)
    ds = results(didx);
    methods = {ds.naive, ds.icm};
    method_titles = {'Naive log-GP (independent)', 'ICM MOGP (coupled)'};

    for midx = 1:2
        nexttile;
        ax = gca; ax.Layer = 'top';
        hold on; grid on;
        fit = methods{midx};
        plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, k_plot, 'M1');
        plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, k_plot, 'M2');
        if strcmp(ds.name, 'full') && midx == 2
            xline(t_impute, ':', 'Color', [0.4, 0.4, 0.4], 'LineWidth', 1.2, ...
                'DisplayName', sprintf('M2 impute t=%g', t_impute));
        end
        xlabel('Time (days)');
        ylabel('cells/mm^2');
        title(sprintf('%s — %s', ds.name, method_titles{midx}), 'Interpreter', 'none');
        xlim([0, 14]);
        ylim([0, inf]);
        legend('Location', 'northwest', 'FontSize', 8);
    end
end

%% ===== Delta sensitivity (full data, ICM only) =====
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
        fit_s = fit_icm_mogp(timeM1, dataM1, timeM2, dataM2, ...
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

%% ===== Optional SE kernel smoke test =====
if run_se_smoke_test
    fprintf('\n=== Kernel smoke test: SE (full data) ===\n');
    se_kernel = build_temporal_kernel('se');
    se_fit = fit_icm_mogp(timeM1, dataM1, timeM2, dataM2, count_delta, tgrid, ...
        se_kernel, meanfunc, likfunc, -100, t_impute, k_plot);
    fprintf('SE ICM: NLML=%.4f, ell=%.4f, rho=%.4f, M2@day5=%.2f\n', ...
        se_fit.report.nlml, se_fit.report.ell, se_fit.report.rho, se_fit.report.M2_at_t);
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

function z = count_to_log(M, count_delta)
z = log(1 + max(M(:), 0) ./ count_delta);
end

function M = log_to_count(z, count_delta)
M = count_delta .* expm1(z(:));
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

% Touch GPML to validate hyperparameter layout before optimization.
gp(hyp0, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);
end

function out = fit_icm_mogp(timeM1, dataM1, timeM2, dataM2, count_delta, tgrid, ...
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

function pheno = pack_count_fit(mu_z, sf_z, count_delta, k_plot)
% Approximate count-scale bands via count_delta*expm1 on latent mean +/- k*sf.
pheno.mu_z = mu_z(:);
pheno.sf_z = sf_z(:);
pheno.mu = log_to_count(mu_z, count_delta);
pheno.sf = sf_z;
pheno.lo = max(0, log_to_count(mu_z - k_plot .* sf_z, count_delta));
pheno.hi = log_to_count(mu_z + k_plot .* sf_z, count_delta);
end

function report_fit(dataset_name, method_name, report)
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
