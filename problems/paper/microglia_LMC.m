% microglia_LMC.m — Independent Matérn 3/2 GPs vs LMC multi-output GP for M1/M2.
% Joint GP on log1p counts with LMC (Matérn 3/2 + RQ latent processes).
% Figure: 2x2 — left = independent, right = LMC; top = full, bottom = averaged.
% Day 5 included in both M1 and M2.

clearvars; close all; clc;

%% ===== Configuration =====
k_plot    = 1.96;
max_iters = -200;
tgrid     = (0:0.1:14)';

LABEL_M1 = 1;
LABEL_M2 = 2;

meanfunc = @meanZero;
likfunc  = @likGauss;

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

fprintf('=== Independent Matérn 3/2 (log1p) vs LMC MOGP ===\n');

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

covLMC = build_lmc_kernel();
naiveKernel = {@covMaterniso, 3};
inffunc_naive = @infGaussLik;

%% ===== Fit =====
results = struct([]);
for didx = 1:numel(datasets)
    ds = datasets(didx);
    fprintf('\n--- Dataset: %s ---\n', ds.name);

    naive = fit_naive_gp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, naiveKernel, meanfunc, likfunc, inffunc_naive, max_iters, k_plot);
    lmc = fit_lmc_mogp(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, covLMC, meanfunc, likfunc, max_iters, k_plot, LABEL_M1, LABEL_M2);

    results(didx).name = ds.name;
    results(didx).naive = naive;
    results(didx).method = lmc;
    results(didx).timeM1 = ds.timeM1;
    results(didx).dataM1 = ds.dataM1;
    results(didx).timeM2 = ds.timeM2;
    results(didx).dataM2 = ds.dataM2;

    report_naive_fit(ds.name, naive.report);
    report_lmc_fit(ds.name, lmc.report);
end

%% ===== Figure: 2x2 (independent | LMC) x (full | averaged) =====
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];

figure('Color', 'w', 'Position', [60, 60, 1240, 900], ...
    'Name', 'Microglia: independent vs LMC (log1p)');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for didx = 1:numel(results)
    ds = results(didx);
    methods = {ds.naive, ds.method};
    method_titles = {'Independent Matérn 3/2 (log1p)', 'LMC MOGP (log1p)'};
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

function covLMC = build_lmc_kernel()
covLMC = {@covSum, { ...
    {@covProd, { {@covMask, {1, {@covMaterniso, 3}}}, {@covMask, {2, {@covDiscrete, 2}}} }}, ...
    {@covProd, { {@covMask, {1, @covRQiso}},              {@covMask, {2, {@covDiscrete, 2}}} }} }};
end

function [hyp0, prior, inffunc] = init_lmc_hyp(covLMC, x_aug, y_aug)
meanfunc = @meanZero;
likfunc  = @likGauss;

t_all = x_aug(:, 1);
y_aug = y_aug(:);
ell0 = max(std(t_all), 0.5);
Lchol0 = [log(sqrt(0.5)); 0; log(sqrt(0.5))];
sn0 = max(0.1 * std(y_aug), 1e-3);

hyp0.mean = [];
hyp0.cov  = [log(ell0); 0; Lchol0; ...
             log(ell0); 0; log(1);  Lchol0];
hyp0.lik  = log(sn0);

s2_ell = 0.5^2;
prior.cov = { {@priorGauss, log(ell0), s2_ell}, @priorClamped, [], [], [], ...
              {@priorGauss, log(ell0), s2_ell}, @priorClamped, {@priorGauss, 0, s2_ell}, [], [], [] };
inffunc = {@infPrior, @infGaussLik, prior};

gp(hyp0, inffunc, meanfunc, covLMC, likfunc, x_aug, y_aug);
end

function out = fit_naive_gp(timeM1, dataM1, timeM2, dataM2, tgrid, ...
    temporalKernel, meanfunc, likfunc, inffunc, max_iters, k_plot)

z_M1 = count_to_log1p(dataM1);
z_M2 = count_to_log1p(dataM2);

fit1 = fit_single_gp(timeM1, z_M1, tgrid, temporalKernel, meanfunc, likfunc, inffunc, max_iters);
fit2 = fit_single_gp(timeM2, z_M2, tgrid, temporalKernel, meanfunc, likfunc, inffunc, max_iters);

out.M1 = pack_log1p_fit(fit1.mu_z, fit1.sf_z, k_plot);
out.M2 = pack_log1p_fit(fit2.mu_z, fit2.sf_z, k_plot);
out.hyp_M1 = fit1.hyp;
out.hyp_M2 = fit2.hyp;
out.nlml = fit1.nlml + fit2.nlml;
out.report.nlml = out.nlml;
out.report.ell_M1 = exp(fit1.hyp.cov(1));
out.report.ell_M2 = exp(fit2.hyp.cov(1));
out.report.sn_M1 = exp(fit1.hyp.lik);
out.report.sn_M2 = exp(fit2.hyp.lik);
end

function fit = fit_single_gp(x, z, tgrid, temporalKernel, meanfunc, likfunc, inffunc, max_iters)
x = x(:); z = z(:);
ell0 = max(std(x), 0.5);
sf0 = max(std(z), 0.1);
sn0 = 0.1 * sf0;

hyp.mean = [];
hyp.cov = log([ell0; sf0]);
hyp.lik = log(sn0);

hyp = minimize(hyp, @gp, max_iters, inffunc, meanfunc, temporalKernel, likfunc, x, z);
nlml = gp(hyp, inffunc, meanfunc, temporalKernel, likfunc, x, z);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, temporalKernel, likfunc, x, z, tgrid);

fit.hyp = hyp;
fit.nlml = nlml;
fit.mu_z = fmu(:);
fit.sf_z = sqrt(max(fs2(:), 0));
end

function fit = fit_lmc_mogp(timeM1, dataM1, timeM2, dataM2, tgrid, ...
    covLMC, meanfunc, likfunc, max_iters, k_plot, LABEL_M1, LABEL_M2)

z_M1 = count_to_log1p(dataM1);
z_M2 = count_to_log1p(dataM2);

x_aug = [timeM1(:), LABEL_M1 * ones(numel(timeM1), 1); ...
         timeM2(:), LABEL_M2 * ones(numel(timeM2), 1)];
y_aug = [z_M1(:); z_M2(:)];

[hyp0, ~, inffunc] = init_lmc_hyp(covLMC, x_aug, y_aug);

hyp = minimize(hyp0, @gp, max_iters, inffunc, meanfunc, covLMC, likfunc, x_aug, y_aug);
nlml = gp(hyp, inffunc, meanfunc, covLMC, likfunc, x_aug, y_aug);

x_te_M1 = [tgrid, LABEL_M1 * ones(size(tgrid))];
x_te_M2 = [tgrid, LABEL_M2 * ones(size(tgrid))];
[~, ~, fmu1, fs21] = gp(hyp, inffunc, meanfunc, covLMC, likfunc, x_aug, y_aug, x_te_M1);
[~, ~, fmu2, fs22] = gp(hyp, inffunc, meanfunc, covLMC, likfunc, x_aug, y_aug, x_te_M2);

fit.M1 = pack_log1p_fit(fmu1(:), sqrt(max(fs21(:), 0)), k_plot);
fit.M2 = pack_log1p_fit(fmu2(:), sqrt(max(fs22(:), 0)), k_plot);
fit.hyp = hyp;
fit.nlml = nlml;
fit.report = build_lmc_report(hyp, nlml);
end

function report = build_lmc_report(hyp, nlml)
hyp_cov = hyp.cov(:);
B1 = chol2cov(hyp_cov(3:5));
B2 = chol2cov(hyp_cov(9:11));

report.nlml = nlml;
report.ell_matern = exp(hyp_cov(1));
report.sf_matern  = 1.0;
report.ell_rq     = exp(hyp_cov(6));
report.sf_rq      = 1.0;
report.alpha_rq   = exp(hyp_cov(8));
report.B1 = B1;
report.B2 = B2;
report.rho_B1 = corr_from_B(B1);
report.rho_B2 = corr_from_B(B2);
report.sn = exp(hyp.lik);
end

function z = count_to_log1p(M)
z = log1p(max(M(:), 0));
end

function pheno = pack_log1p_fit(mu_z, sf_z, k_plot)
pheno.mu_z = mu_z(:);
pheno.sf_z = sf_z(:);
pheno.mu = expm1(mu_z);
pheno.sf = sf_z;
pheno.lo = max(0, expm1(mu_z - k_plot .* sf_z));
pheno.hi = expm1(mu_z + k_plot .* sf_z);
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

function report_naive_fit(dataset_name, report)
fprintf('[%s | Independent log1p] NLML=%.4f, ell_M1=%.4f, ell_M2=%.4f, sn_M1=%.4f, sn_M2=%.4f\n', ...
    dataset_name, report.nlml, report.ell_M1, report.ell_M2, report.sn_M1, report.sn_M2);
end

function report_lmc_fit(dataset_name, report)
fprintf('[%s | LMC log1p] NLML=%.4f\n', dataset_name, report.nlml);
fprintf('  Matérn 3/2: ell=%.4f | RQ: ell=%.4f, alpha=%.4f | sn=%.4f\n', ...
    report.ell_matern, report.ell_rq, report.alpha_rq, report.sn);
fprintf('  B1 rho=%.4f | B2 rho=%.4f\n', report.rho_B1, report.rho_B2);
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
