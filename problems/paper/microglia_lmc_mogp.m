% microglia_lmc_mogp.m — Two-output Linear Model of Coregionalization (LMC) for M1/M2.
%
% Joint GP on raw microglia counts with LMC covariance (GPML composed kernel):
%   K_ab = B^(1)_{o_a,o_b} k_Matérn3/2(t_a,t_b)
%        + B^(2)_{o_a,o_b} k_RQ(t_a,t_b),
% where o_a ∈ {M1,M2} is the output label at observation a.
% Observation noise: single shared likGauss sigma_n (ICM-consistent).
%
% Latent interpretation:
%   u_1(t) ~ GP(0, k_Matérn3/2)  — shared rough post-stroke inflammatory response
%   u_2(t) ~ GP(0, k_RQ)         — broader multi-scale temporal variation
%   B^(1), B^(2) (covDiscrete, B = L'L) control M1/M2 loading on each latent process.
%   Base-kernel signal variances are clamped to 1; cross-output magnitude lives in B_q.
%
% Inference via GPML @gp + analytic derivatives (covSum/covProd/covMask/covDiscrete).

clearvars; close all; clc;

%% ===== Configuration =====
k_plot    = 1.96;          % ~95% posterior band multiplier
max_iters = -200;          % GPML minimize budget (<0 = function evaluations)
tgrid     = (0:0.1:14)';   % dense prediction grid [days]

LABEL_M1 = 1;
LABEL_M2 = 2;

meanfunc = @meanZero;
likfunc  = @likGauss;

%% ===== Data (same as microglia.m / microglia_icm_mogp.m) =====
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

fprintf('=== Raw-count LMC MOGP (Matérn 3/2 + RQ, Q = 2, GPML @gp) ===\n');

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
naiveKernel = {@covMaterniso, 3};   % independent baseline: Matérn 3/2 per output
inffunc_naive = @infGaussLik;

%% ===== Fit naive + LMC for each dataset =====
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
    results(didx).lmc = lmc;
    results(didx).timeM1 = ds.timeM1;
    results(didx).dataM1 = ds.dataM1;
    results(didx).timeM2 = ds.timeM2;
    results(didx).dataM2 = ds.dataM2;

    report_naive_fit(ds.name, naive.report);
    report_lmc_fit(ds.name, lmc.report);
end

%% ===== Comparison figure: 2 x 2 (dataset x method) =====
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];

figure('Color', 'w', 'Position', [60, 60, 1240, 900], ...
    'Name', 'Microglia: naive GP vs LMC MOGP');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for didx = 1:numel(results)
    ds = results(didx);
    methods = {ds.naive, ds.lmc};
    method_titles = {'Naive GP (independent)', 'LMC MOGP (coupled)'};

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

fprintf('\nDone.\n');

%% ===== Local functions =====

function covLMC = build_lmc_kernel()
% LMC with Q=2 latent processes: Matérn 3/2 + rational quadratic over time,
% each multiplied by a 2x2 coregionalization matrix (covDiscrete) over label.
covLMC = {@covSum, { ...
    {@covProd, { {@covMask, {1, {@covMaterniso, 3}}}, {@covMask, {2, {@covDiscrete, 2}}} }}, ...
    {@covProd, { {@covMask, {1, @covRQiso}},              {@covMask, {2, {@covDiscrete, 2}}} }} }};
end

function [hyp0, prior, inffunc] = init_lmc_hyp(covLMC, x_aug, y_aug)
% hyp.cov layout (11 params):
%   [log ell_m; 0 (clamped sf_m); Lchol B1 (3);
%    log ell_rq; 0 (clamped sf_rq); log alpha_rq; Lchol B2 (3)]
meanfunc = @meanZero;
likfunc  = @likGauss;

t_all = x_aug(:, 1);
ell0 = max(std(t_all), 0.5);
Lchol0 = [log(sqrt(0.5)); 0; log(sqrt(0.5))];

hyp0.mean = [];
hyp0.cov  = [log(ell0); 0; Lchol0; ...
             log(ell0); 0; log(1);  Lchol0];
hyp0.lik  = log(0.1);

s2_ell = 0.5^2;
prior.cov = { {@priorGauss, log(ell0), s2_ell}, @priorClamped, [], [], [], ...
              {@priorGauss, log(ell0), s2_ell}, @priorClamped, {@priorGauss, 0, s2_ell}, [], [], [] };
inffunc = {@infPrior, @infGaussLik, prior};

gp(hyp0, inffunc, meanfunc, covLMC, likfunc, x_aug, y_aug);
end

function out = fit_naive_gp(timeM1, dataM1, timeM2, dataM2, tgrid, ...
    temporalKernel, meanfunc, likfunc, inffunc, max_iters, k_plot)

% Independent scalar GPs — one for M1, one for M2 — on raw counts.
y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);

fit1 = fit_single_gp(timeM1, y_M1, tgrid, temporalKernel, meanfunc, likfunc, inffunc, max_iters);
fit2 = fit_single_gp(timeM2, y_M2, tgrid, temporalKernel, meanfunc, likfunc, inffunc, max_iters);

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

function fit = fit_single_gp(x, y, tgrid, temporalKernel, meanfunc, likfunc, inffunc, max_iters)
x = x(:); y = y(:);
ell0 = max(std(x), 0.5);
sf0 = max(std(y), 0.1);
sn0 = 0.1 * sf0;

hyp.mean = [];
hyp.cov = log([ell0; sf0]);
hyp.lik = log(sn0);

hyp = minimize(hyp, @gp, max_iters, inffunc, meanfunc, temporalKernel, likfunc, x, y);
nlml = gp(hyp, inffunc, meanfunc, temporalKernel, likfunc, x, y);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, temporalKernel, likfunc, x, y, tgrid);

fit.hyp = hyp;
fit.nlml = nlml;
fit.mu_y = fmu(:);
fit.sf_y = sqrt(max(fs2(:), 0));
end

function fit = fit_lmc_mogp(timeM1, dataM1, timeM2, dataM2, tgrid, ...
    covLMC, meanfunc, likfunc, max_iters, k_plot, LABEL_M1, LABEL_M2)

% Floor negative counts; per-output standardization stabilizes optimization.
y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);
mu1 = mean(y_M1); sd1 = std(y_M1);
mu2 = mean(y_M2); sd2 = std(y_M2);
if sd1 < eps, sd1 = 1; end
if sd2 < eps, sd2 = 1; end

x_aug = [timeM1(:), LABEL_M1 * ones(numel(timeM1), 1); ...
         timeM2(:), LABEL_M2 * ones(numel(timeM2), 1)];
y_aug = [(y_M1 - mu1) / sd1; (y_M2 - mu2) / sd2];

[hyp0, ~, inffunc] = init_lmc_hyp(covLMC, x_aug, y_aug);

hyp = minimize(hyp0, @gp, max_iters, inffunc, meanfunc, covLMC, likfunc, x_aug, y_aug);
nlml = gp(hyp, inffunc, meanfunc, covLMC, likfunc, x_aug, y_aug);

x_te_M1 = [tgrid, LABEL_M1 * ones(size(tgrid))];
x_te_M2 = [tgrid, LABEL_M2 * ones(size(tgrid))];
[~, ~, fmu1, fs21] = gp(hyp, inffunc, meanfunc, covLMC, likfunc, x_aug, y_aug, x_te_M1);
[~, ~, fmu2, fs22] = gp(hyp, inffunc, meanfunc, covLMC, likfunc, x_aug, y_aug, x_te_M2);

mu_y1 = mu1 + sd1 * fmu1(:);
sf_y1 = sd1 * sqrt(max(fs21(:), 0));
mu_y2 = mu2 + sd2 * fmu2(:);
sf_y2 = sd2 * sqrt(max(fs22(:), 0));

fit.M1 = pack_raw_fit(mu_y1, sf_y1, k_plot);
fit.M2 = pack_raw_fit(mu_y2, sf_y2, k_plot);
fit.hyp = hyp;
fit.nlml = nlml;
fit.report = build_lmc_report(hyp, nlml, sd1, sd2);
end

function report = build_lmc_report(hyp, nlml, sd1, sd2)
% hyp.cov: [ell_m(1), sf_m(2), B1(3:5), ell_rq(6), sf_rq(7), alpha(8), B2(9:11)]
hyp_cov = hyp.cov(:);
B1 = chol2cov(hyp_cov(3:5));
B2 = chol2cov(hyp_cov(9:11));
sn_std = exp(hyp.lik);
sd_avg = 0.5 * (sd1 + sd2);

report.nlml = nlml;
report.ell_matern = exp(hyp_cov(1));
report.sf_matern  = 1.0;   % clamped
report.ell_rq     = exp(hyp_cov(6));
report.sf_rq      = 1.0;   % clamped
report.alpha_rq   = exp(hyp_cov(8));
report.B1 = B1;
report.B2 = B2;
report.rho_B1 = corr_from_B(B1);
report.rho_B2 = corr_from_B(B2);
report.sn_std = sn_std;
report.sn_count = sn_std * sd_avg;
end

function B = chol2cov(hyp)
% Reconstruct 2x2 covDiscrete coregionalization matrix B = L'*L.
L = zeros(2);
L(triu(true(2))) = hyp(:);
L(1:3:end) = exp(diag(L));
B = L' * L;
end

function pheno = pack_raw_fit(mu, sf, k_plot)
pheno.mu = mu(:);
pheno.sf = sf(:);
pheno.lo = mu - k_plot .* sf;
pheno.hi = mu + k_plot .* sf;
end

function rho = corr_from_B(B)
rho = B(1, 2) / sqrt(max(B(1, 1) * B(2, 2), eps));
end

function report_naive_fit(dataset_name, report)
fprintf('[%s | Naive GP] NLML = %.4f, ell_M1 = %.4f, ell_M2 = %.4f, sn_M1 = %.4f, sn_M2 = %.4f\n', ...
    dataset_name, report.nlml, report.ell_M1, report.ell_M2, report.sn_M1, report.sn_M2);
end

function report_lmc_fit(dataset_name, report)
fprintf('[%s | LMC MOGP] NLML = %.4f\n', dataset_name, report.nlml);
fprintf('  Matérn 3/2: ell = %.4f, sf = %.4f (clamped)\n', report.ell_matern, report.sf_matern);
fprintf('  RQ:         ell = %.4f, sf = %.4f (clamped), alpha = %.4f\n', ...
    report.ell_rq, report.sf_rq, report.alpha_rq);
fprintf('  sn = %.4f (std scale), %.4f (approx count scale)\n', report.sn_std, report.sn_count);
fprintf('  B^(1) = [%.4f %.4f; %.4f %.4f],  rho = %.4f\n', ...
    report.B1(1,1), report.B1(1,2), report.B1(2,1), report.B1(2,2), report.rho_B1);
fprintf('  B^(2) = [%.4f %.4f; %.4f %.4f],  rho = %.4f\n', ...
    report.B2(1,1), report.B2(1,2), report.B2(2,1), report.B2(2,2), report.rho_B2);
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

function ylim_auto_from_fit(ax, fitM1, fitM2, dataM1, dataM2)
vals = [fitM1.lo(:); fitM1.hi(:); fitM1.mu(:); ...
        fitM2.lo(:); fitM2.hi(:); fitM2.mu(:); ...
        dataM1(:); dataM2(:)];
pad = 0.05 * max(range(vals), 1);
ylim(ax, [min(vals) - pad, max(vals) + pad]);
end
