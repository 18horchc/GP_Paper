% microglia_ICM_het.m — Naive SE-GP vs ICM MOGP, with and without fixed
% low noise (sigma=0.05 in count units) on all real observations at t=0.
% Datasets: full and averaged raw counts. No Pensoneault bounds.
%
% Figure 1: naive SE vs ICM SE (coregionalization B).
% Figure 2: same comparison with heteroscedastic t=0 noise (real sn from
%         unconstrained fit; t=0 points use sigma=0.05 in count units).

clear; close all; clc;

%% ===== Configuration =====
kernel_name = 'se';
k_plot      = 1.96;
max_iters   = -200;
sigma_t0    = 0.05;   % count-unit noise for observations at t=0

%% ===== Data (same as microglia_icm_mogp.m) =====
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

fprintf('=== Microglia: naive SE vs ICM SE (+ t=0 low noise) ===\n');
fprintf('kernel = %s | t=0 noise sigma=%.3g (count units)\n', ...
    kernel_name, sigma_t0);

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

%% ===== Fit naive + ICM (+ t=0 het noise) for each dataset =====
results = struct([]);
for didx = 1:numel(datasets)
    ds = datasets(didx);
    fprintf('\n--- Dataset: %s ---\n', ds.name);

    naive = fit_naive_gp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot);
    icm = fit_icm_mogp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot);
    naive_het = fit_naive_gp_het(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        sigma_t0, naive, tgrid, max_iters, k_plot);
    icm_het = fit_icm_mogp_het(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        sigma_t0, icm, tgrid, temporalKernel, max_iters, k_plot);

    results(didx).name = ds.name;
    results(didx).naive = naive;
    results(didx).icm = icm;
    results(didx).naive_het = naive_het;
    results(didx).icm_het = icm_het;
    results(didx).timeM1 = ds.timeM1;
    results(didx).dataM1 = ds.dataM1;
    results(didx).timeM2 = ds.timeM2;
    results(didx).dataM2 = ds.dataM2;

    report_fit(ds.name, 'Naive GP (SE)', naive.report);
    report_fit(ds.name, 'ICM MOGP (SE)', icm.report);
    report_fit(ds.name, 'Naive GP + t0 sigma', naive_het.report);
    report_fit(ds.name, 'ICM MOGP + t0 sigma', icm_het.report);
end

%% ===== Figure 1: naive vs ICM (homoscedastic) =====
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];

figure('Color', 'w', 'Position', [60, 60, 1240, 900], ...
    'Name', 'Microglia: naive SE vs ICM SE');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for didx = 1:numel(results)
    ds = results(didx);
    methods = {ds.naive, ds.icm};
    method_titles = {'Naive GP (SE)', 'ICM MOGP (SE)'};
    for midx = 1:2
        nexttile;
        ax = gca; ax.Layer = 'top';
        hold on; grid on;
        fit = methods{midx};
        plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1');
        plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2');
        xlabel('Time (days)');
        ylabel('cells/mm^2');
        title(sprintf('%s — %s', ds.name, method_titles{midx}), 'Interpreter', 'none');
        xlim([0, 14]);
        ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
        legend('Location', 'northwest', 'FontSize', 8);
    end
end

%% ===== Figure 2: naive vs ICM with t=0 low noise =====
figure(2);
set(gcf, 'Color', 'w', 'Position', [100, 40, 1240, 900], ...
    'Name', sprintf('Microglia: naive SE vs ICM SE with t=0 sigma=%.3g', sigma_t0));
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for didx = 1:numel(results)
    ds = results(didx);
    methods = {ds.naive_het, ds.icm_het};
    method_titles = {'Naive GP + t0 sigma', 'ICM MOGP + t0 sigma'};
    for midx = 1:2
        nexttile;
        ax = gca; ax.Layer = 'top';
        hold on; grid on;
        fit = methods{midx};
        plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1');
        plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2');
        xlabel('Time (days)');
        ylabel('cells/mm^2');
        title(sprintf('%s — %s', ds.name, method_titles{midx}), 'Interpreter', 'none');
        xlim([0, 14]);
        ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
        legend('Location', 'northwest', 'FontSize', 8);
    end
end

fprintf('\nDone.\n');

%% ===== Local functions =====

function covfunc = build_temporal_kernel(name)
switch lower(name)
    case 'matern32'
        covfunc = {@covMaterniso, 3};
    case 'matern52'
        covfunc = {@covMaterniso, 5};
    case 'se'
        covfunc = @covSEiso;
    case 'rq'
        covfunc = @covRQiso;
    otherwise
        error('Unknown kernel: %s (use matern32, matern52, se, rq)', name);
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

function out = fit_icm_mogp_het(timeM1, dataM1, timeM2, dataM2, ...
    sigma_t0, icm_unc, tgrid, temporalKernel, max_iters, k_plot)
% ICM with heteroscedastic noise: real sn from unconstrained ICM; t=0 sigma fixed.

LABEL_M1 = 1;
LABEL_M2 = 2;

y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);
mu1 = icm_unc.mu1; sd1 = icm_unc.sd1;
mu2 = icm_unc.mu2; sd2 = icm_unc.sd2;

n1 = numel(timeM1);
n2 = numel(timeM2);
x_aug = [timeM1(:), LABEL_M1 * ones(n1, 1); ...
         timeM2(:), LABEL_M2 * ones(n2, 1)];
y_aug = [ (y_M1 - mu1) / sd1; (y_M2 - mu2) / sd2 ];

sn_real = exp(icm_unc.hyp.lik);
noise_var = sn_real^2 * ones(n1 + n2, 1);
sd_per_row = [sd1 * ones(n1, 1); sd2 * ones(n2, 1)];
is_t0 = (x_aug(:, 1) == 0);
noise_var(is_t0) = (sigma_t0 ./ sd_per_row(is_t0)).^2;

covICM = build_icm_kernel(temporalKernel);
[nTemp, ~] = temporal_hyp_layout(temporalKernel);

hyp0 = icm_unc.hyp;
hyp0.lik = [];   % unused in hetero path
hyp0.mean = [];

obj = @(h) gp_icm_hetero_noise('nlml', h, covICM, x_aug, y_aug, noise_var, nTemp);
hyp = minimize(hyp0, obj, max_iters);
% Keep temporal sf clamped at log(1)
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
out.report.sn = sn_real;   % fixed non-t0 noise (std units)
end

function varargout = gp_icm_hetero_noise(mode, hyp, covICM, x, y, noise_var, nTemp, xs)
% ICM GP with fixed per-row observation noise. Optimizes hyp.cov; hyp.lik unused.
% Temporal sf (hyp.cov(2)) is clamped: gradient zeroed, value forced to 0.
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
    free_idx = [1, (nTemp + 1):(nTemp + 3)];  % ell + B; skip clamped sf
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
end

function out = fit_naive_gp_het(timeM1, dataM1, timeM2, dataM2, ...
    sigma_t0, naive_unc, tgrid, max_iters, k_plot)
% Independent SE GPs with heteroscedastic t=0 noise via gp_seiso_hetero_noise.

y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);

fit1 = fit_single_gp_het(timeM1, y_M1, sigma_t0, ...
    naive_unc.hyp_M1, tgrid, max_iters);
fit2 = fit_single_gp_het(timeM2, y_M2, sigma_t0, ...
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
out.report.B = [];
out.report.rho = NaN;
end

function fit = fit_single_gp_het(x, y, sigma_t0, hyp_unc, tgrid, max_iters)
x = x(:); y = y(:);
sn_real = exp(hyp_unc.lik);
noise_var = sn_real^2 * ones(numel(y), 1);
noise_var(x == 0) = sigma_t0^2;

hyp0 = struct('mean', [], 'cov', hyp_unc.cov(:), 'lik', []);
obj = @(h) gp_seiso_hetero_noise('nlml', h, x, y, noise_var);
hyp = minimize(hyp0, obj, max_iters);
nlml = obj(hyp);
[~, ~, fmu, fs2] = gp_seiso_hetero_noise('pred', hyp, x, y, noise_var, tgrid(:));

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
fprintf('[%s | %s] NLML=%.4f', dataset_name, method_name, report.nlml);
if ~isempty(report.B)
    fprintf(', ell=%.4f, rho=%.4f, sn=%.4f', report.ell, report.rho, report.sn);
    fprintf('\n  B = [%.4f %.4f; %.4f %.4f]', ...
        report.B(1,1), report.B(1,2), report.B(2,1), report.B(2,2));
else
    fprintf(', ell_M1=%.4f, ell_M2=%.4f', report.ell_M1, report.ell_M2);
    if isfield(report, 'sn_M1')
        fprintf(', sn_M1=%.4f, sn_M2=%.4f', report.sn_M1, report.sn_M2);
    end
end
fprintf('\n');
end

function plot_phenotype(ax, tgrid, fit, t_data, y_data, col, name)
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

function ylim_auto_from_fit(ax, fitM1, fitM2, dataM1, dataM2)
vals = [fitM1.lo(:); fitM1.hi(:); fitM1.mu(:); ...
        fitM2.lo(:); fitM2.hi(:); fitM2.mu(:); ...
        dataM1(:); dataM2(:)];
pad = 0.05 * max(range(vals), 1);
ylim(ax, [min(vals) - pad, max(vals) + pad]);
end
