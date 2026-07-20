clear; close all; clc

%% Data

% averages
newtime = [0, 1, 2, 3, 5, 7, 14];
datapointsM1 = [5, 27.5, 122.5, 139.8, 325, 445, 816.67];
newtimeM2 = [0, 1, 2, 3, 5, 7, 14];
datapointsM2 = [5, 78.33, 179.5, 126.4, 800, 319, 136.67];

% full data (unused for now; averaged-only ICM below)
timeM1 = [0, 1, 3, 5, 7, 14, ...   % Hu2012
          3, 7, ...                % Li2018
          2, ...                   % Ma2020
          14, ...                  % Wang2017
          3, ...                   % Xu2021
          2, ...                   % Li2023
          0, 1, 3, 7, 14, ...      % Suenaga2015
          3, 7];                   % Yang2017
dataM1 = [0, 5, 375, 325, 600, 750, ...
          62, 55, ...
          120, ...
          400, ...
          102, ...
          125, ...
          10, 50, 100, 900, 1300, ...
          60, 225];

timeM2 = [0, 1, 3, 5, 7, 14, ...   % Hu2012 (day 5 restored)
          1, 3, 7, ...             % Li2018
          2, ...                   % Ma2020
          14, ...                  % Wang2017
          3, ...                   % Xu2021
          2, ...                   % Li2023
          0, 1, 3, 7, 14, ...      % Suenaga2015
          3, 7];                   % Yang2017
dataM2 = [0, 170, 300, 800, 600, 200, ...
          15, 15, 6, ...
          90, ...
          110, ...
          57, ...
          269, ...
          10, 50, 100, 400, 100, ...
          160, 270];

%% Ratio R = M1/M2 from averaged data
% Pair at times where both averaged M1 and M2 exist (including day 5).

[t_avg, R_avg] = build_paired_ratio_avg(newtime, datapointsM1, newtimeM2, datapointsM2);

fprintf('Averaged R = M1/M2:\n');
for i = 1:numel(t_avg)
    fprintf('  t = %2g days: R = %.4f\n', t_avg(i), R_avg(i));
end

%% GPML setup (SE ICM)

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
tgrid = (0:0.1:14)';
k_plot = 1.96;
max_iters = -200;
n_out2 = 2;
n_out3 = 3;
sigma_R_factor = 4;   % downweight: sigma_R = 4 * sn (moderate soft regularizer)
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];

%% --- Previous SE GP + ridge plots (commented out) ---
%{
% (previous figures 1-3: SE GP on R, ridge, LOOCV ridge)
%}

%% Fits: naive, 2-out ICM (no R), 3-out ICM equal R, 3-out ICM downweighted R

fprintf('\n=== Naive independent SE GPs (averaged M1, M2) ===\n');
naive = fit_naive_gp(newtime(:), datapointsM1(:), newtimeM2(:), datapointsM2(:), ...
    tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot);
report_naive(naive.report);

fprintf('\n=== 2-output SE-ICM (averaged M1, M2; no R) ===\n');
icm2 = fit_icm2(newtime(:), datapointsM1(:), newtimeM2(:), datapointsM2(:), ...
    tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot, n_out2);
report_icm2(icm2.report);

fprintf('\n=== Homoscedastic 3-output SE-ICM (averaged M1, M2, R) ===\n');
icm_homo = fit_icm3(newtime(:), datapointsM1(:), newtimeM2(:), datapointsM2(:), ...
    t_avg, R_avg, tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot, n_out3);
report_icm3('Equal-weight R', icm_homo.report);

fprintf('\n=== Downweighted-R 3-output SE-ICM (sigma_R = %.0f * sn) ===\n', sigma_R_factor);
icm_down = fit_icm3_downweight_R(newtime(:), datapointsM1(:), newtimeM2(:), datapointsM2(:), ...
    t_avg, R_avg, icm_homo, tgrid, temporalKernel, max_iters, k_plot, n_out3, sigma_R_factor);
report_icm3(sprintf('Downweighted R (x%.0f sn)', sigma_R_factor), icm_down.report);

%% Figure 1: Naive vs 2-output ICM (M1+M2 only, no R)
figure(1)
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
ax = nexttile;
plot_m1_m2_panel(ax, tgrid, naive, newtime, datapointsM1, newtimeM2, datapointsM2, ...
    col_M1, col_M2, 'Naive independent SE GPs');
ax = nexttile;
plot_m1_m2_panel(ax, tgrid, icm2, newtime, datapointsM1, newtimeM2, datapointsM2, ...
    col_M1, col_M2, '2-output SE-ICM (no R)');
sgtitle('Averaged data: naive SE vs 2-output ICM (M1, M2 only)', 'fontsize', 16);

%% Figure 2: 3-output ICM equal-weight R (R | M1+M2)
figure(2)
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
ax = nexttile;
plot_r_panel(ax, tgrid, icm_homo, t_avg, R_avg, 'R (equal-weight)');
ax = nexttile;
plot_m1_m2_panel(ax, tgrid, icm_homo, newtime, datapointsM1, newtimeM2, datapointsM2, ...
    col_M1, col_M2, 'M1 & M2 (equal-weight R ICM)');
sgtitle('3-output SE-ICM with equal-weight R (averaged)', 'fontsize', 16);

%% Figure 3: 3-output ICM downweighted R (R | M1+M2)
figure(3)
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
ax = nexttile;
plot_r_panel(ax, tgrid, icm_down, t_avg, R_avg, sprintf('R (downweighted \\times%.0f sn)', sigma_R_factor));
ax = nexttile;
plot_m1_m2_panel(ax, tgrid, icm_down, newtime, datapointsM1, newtimeM2, datapointsM2, ...
    col_M1, col_M2, sprintf('M1 & M2 (R downweighted \\times%.0f sn)', sigma_R_factor));
sgtitle(sprintf('3-output SE-ICM with downweighted R (\\sigma_R = %.0f sn, averaged)', sigma_R_factor), 'fontsize', 16);

%% Local helpers

function [t, R] = build_paired_ratio_avg(timeM1, yM1, timeM2, yM2)
% Paired ratio R = M1/M2 at times where both averaged observations exist.
timeM1 = timeM1(:);
yM1 = yM1(:);
timeM2 = timeM2(:);
yM2 = yM2(:);
shared = intersect(timeM1, timeM2);
t = shared(:);
R = zeros(numel(t), 1);
for i = 1:numel(t)
    m1 = yM1(timeM1 == t(i));
    m2 = yM2(timeM2 == t(i));
    R(i) = m1(1) / m2(1);
end
end

function covICM = build_icm3_kernel(temporalKernel, n_out)
covICM = {@covProd, { ...
    {@covMask, {1, temporalKernel}}, ...
    {@covMask, {2, {@covDiscrete, n_out}}} }};
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

function nB = n_coreg_hypers(n_out)
nB = n_out * (n_out + 1) / 2;
end

function Lchol0 = init_Lchol(n_out)
% Diagonal B = 0.5 I  =>  L_ii = sqrt(0.5), off-diag = 0 (triu packing).
nB = n_coreg_hypers(n_out);
Lchol0 = zeros(nB, 1);
Ltmp = zeros(n_out);
mask = triu(true(n_out));
diag_mask = false(n_out);
diag_mask(1:(n_out + 1):end) = true;
vals = zeros(n_out);
vals(diag_mask) = log(sqrt(0.5));
Lchol0(:) = vals(mask);
end

function B = chol2cov(hyp, n_out)
L = zeros(n_out);
L(triu(true(n_out))) = hyp(:);
L(1:(n_out + 1):end) = exp(diag(L));
B = L' * L;
end

function rho = corr_from_B(B)
d = size(B, 1);
rho = zeros(d);
for i = 1:d
    for j = 1:d
        rho(i, j) = B(i, j) / sqrt(max(B(i, i) * B(j, j), eps));
    end
end
end

function [hyp0, inffunc] = init_icm3_hyp(temporalKernel, x_aug, y_aug, n_out)
meanfunc = @meanZero;
likfunc  = @likGauss;
covICM   = build_icm3_kernel(temporalKernel, n_out);

[nTemp, hasAlpha] = temporal_hyp_layout(temporalKernel);
t_all = x_aug(:, 1);
ell0 = max(std(t_all), 0.5);
Lchol0 = init_Lchol(n_out);
nB = numel(Lchol0);

hyp0.mean = [];
if hasAlpha
    hyp0.cov = [log(ell0); 0; log(1); Lchol0];
else
    hyp0.cov = [log(ell0); 0; Lchol0];
end
hyp0.lik = log(0.1);

prior.cov = cell(1, nTemp + nB);
prior.cov{1} = {@priorGauss, log(ell0), 0.5^2};
prior.cov{2} = @priorClamped;
if hasAlpha
    prior.cov{3} = {@priorGauss, 0, 0.5^2};
end
inffunc = {@infPrior, @infGaussLik, prior};

gp(hyp0, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);
end

function [x_aug, y_aug, mu1, sd1, mu2, sd2, muR, sdR, n1, n2, nR] = ...
    build_aug3(timeM1, dataM1, timeM2, dataM2, tR, yR)
LABEL_M1 = 1;
LABEL_M2 = 2;
LABEL_R  = 3;

y_M1 = max(dataM1(:), 0);
y_M2 = max(dataM2(:), 0);
y_R  = yR(:);

mu1 = mean(y_M1); sd1 = std(y_M1);
mu2 = mean(y_M2); sd2 = std(y_M2);
muR = mean(y_R);  sdR = std(y_R);
if sd1 < eps, sd1 = 1; end
if sd2 < eps, sd2 = 1; end
if sdR < eps, sdR = 1; end

n1 = numel(timeM1);
n2 = numel(timeM2);
nR = numel(tR);

x_aug = [timeM1(:), LABEL_M1 * ones(n1, 1); ...
         timeM2(:), LABEL_M2 * ones(n2, 1); ...
         tR(:),     LABEL_R  * ones(nR, 1)];
y_aug = [ (y_M1 - mu1) / sd1; (y_M2 - mu2) / sd2; (y_R - muR) / sdR ];
end

function pheno = pack_raw_fit(mu, sf, k_plot)
pheno.mu = mu(:);
pheno.sf = sf(:);
pheno.lo = mu - k_plot .* sf;
pheno.hi = mu + k_plot .* sf;
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

function out = fit_icm2(timeM1, dataM1, timeM2, dataM2, tgrid, ...
    temporalKernel, meanfunc, likfunc, max_iters, k_plot, n_out)
% Homoscedastic 2-output SE-ICM on (M1, M2) only — no ratio.
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
covICM = build_icm3_kernel(temporalKernel, n_out);
[hyp0, inffunc] = init_icm3_hyp(temporalKernel, x_aug, y_aug, n_out);
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
nB = n_coreg_hypers(n_out);
B = chol2cov(hyp.cov(nTemp + (1:nB)), n_out);
rho = corr_from_B(B);
out.report.nlml = nlml;
out.report.ell = exp(hyp.cov(1));
out.report.B = B;
out.report.rho = rho;
out.report.sn = exp(hyp.lik);
end

function report_naive(report)
fprintf('[Naive] NLML=%.4f, ell_M1=%.4f, ell_M2=%.4f, sn_M1=%.4f, sn_M2=%.4f\n', ...
    report.nlml, report.ell_M1, report.ell_M2, report.sn_M1, report.sn_M2);
end

function report_icm2(report)
fprintf('[2-out ICM] NLML=%.4f, ell=%.4f, sn=%.4f, rho12=%.4f\n', ...
    report.nlml, report.ell, report.sn, report.rho(1, 2));
fprintf('  B =\n');
disp(report.B);
end

function plot_m1_m2_panel(ax, tgrid, fit, tM1, yM1, tM2, yM2, col_M1, col_M2, title_str)
hold(ax, 'on')
fill(ax, [tgrid; flipud(tgrid)], [fit.M1.hi; flipud(fit.M1.lo)], ...
    col_M1, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M1 95% band');
plot(ax, tgrid, fit.M1.mu, ':', 'Color', col_M1, 'LineWidth', 2.0, 'DisplayName', 'M1 mean');
s1 = scatter(ax, tM1, yM1, 'filled', 'MarkerFaceColor', col_M1, 'DisplayName', 'M1 data');
s1.Marker = 'hexagram';
s1.SizeData = 120;
fill(ax, [tgrid; flipud(tgrid)], [fit.M2.hi; flipud(fit.M2.lo)], ...
    col_M2, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M2 95% band');
plot(ax, tgrid, fit.M2.mu, ':', 'Color', col_M2, 'LineWidth', 2.0, 'DisplayName', 'M2 mean');
s2 = scatter(ax, tM2, yM2, 'filled', 'MarkerFaceColor', col_M2, 'DisplayName', 'M2 data');
s2.Marker = 'hexagram';
s2.SizeData = 120;
hold(ax, 'off')
xlabel(ax, 'Time (Days)', 'fontsize', 14)
ylabel(ax, 'cells/mm^2', 'fontsize', 14)
title(ax, title_str)
legend(ax, 'Location', 'northwest')
xlim(ax, [0, 14])
set(ax, 'fontsize', 14)
grid(ax, 'on')
end

function plot_r_panel(ax, tgrid, fit, tR, yR, title_str)
col_R = [0.12, 0.35, 0.75];
hold(ax, 'on')
fill(ax, [tgrid; flipud(tgrid)], [fit.R.hi; flipud(fit.R.lo)], ...
    col_R, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'R 95% band');
plot(ax, tgrid, fit.R.mu, ':', 'Color', col_R, 'LineWidth', 2.0, 'DisplayName', 'R mean');
sR = scatter(ax, tR, yR, 'filled', 'MarkerFaceColor', col_R, 'DisplayName', 'R data');
sR.Marker = 'hexagram';
sR.SizeData = 120;
yline(ax, 1, 'k--', 'HandleVisibility', 'off');
hold(ax, 'off')
xlabel(ax, 'Time (Days)', 'fontsize', 14)
ylabel(ax, 'R = M1/M2', 'fontsize', 14)
title(ax, title_str)
legend(ax, 'Location', 'northwest')
xlim(ax, [0, 14])
set(ax, 'fontsize', 14)
grid(ax, 'on')
end

function out = fit_icm3(timeM1, dataM1, timeM2, dataM2, tR, yR, tgrid, ...
    temporalKernel, meanfunc, likfunc, max_iters, k_plot, n_out)
% Homoscedastic 3-output SE-ICM on (M1, M2, R).

LABEL_M1 = 1;
LABEL_M2 = 2;
LABEL_R  = 3;

[x_aug, y_aug, mu1, sd1, mu2, sd2, muR, sdR] = ...
    build_aug3(timeM1, dataM1, timeM2, dataM2, tR, yR);

covICM = build_icm3_kernel(temporalKernel, n_out);
[hyp0, inffunc] = init_icm3_hyp(temporalKernel, x_aug, y_aug, n_out);

hyp = minimize(hyp0, @gp, max_iters, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);
nlml = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);

x_te_M1 = [tgrid, LABEL_M1 * ones(size(tgrid))];
x_te_M2 = [tgrid, LABEL_M2 * ones(size(tgrid))];
x_te_R  = [tgrid, LABEL_R  * ones(size(tgrid))];
[~, ~, fmu1, fs21] = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug, x_te_M1);
[~, ~, fmu2, fs22] = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug, x_te_M2);
[~, ~, fmuR, fs2R] = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug, x_te_R);

mu_y1 = mu1 + sd1 * fmu1(:);
sf_y1 = sd1 * sqrt(max(fs21(:), 0));
mu_y2 = mu2 + sd2 * fmu2(:);
sf_y2 = sd2 * sqrt(max(fs22(:), 0));
mu_yR = muR + sdR * fmuR(:);
sf_yR = sdR * sqrt(max(fs2R(:), 0));

out.M1 = pack_raw_fit(mu_y1, sf_y1, k_plot);
out.M2 = pack_raw_fit(mu_y2, sf_y2, k_plot);
out.R  = pack_raw_fit(mu_yR, sf_yR, k_plot);
out.hyp = hyp;
out.nlml = nlml;
out.mu1 = mu1; out.sd1 = sd1;
out.mu2 = mu2; out.sd2 = sd2;
out.muR = muR; out.sdR = sdR;
out.x_aug = x_aug;
out.y_aug = y_aug;

[nTemp, ~] = temporal_hyp_layout(temporalKernel);
nB = n_coreg_hypers(n_out);
B = chol2cov(hyp.cov(nTemp + (1:nB)), n_out);
rho = corr_from_B(B);
out.report.nlml = nlml;
out.report.ell = exp(hyp.cov(1));
out.report.B = B;
out.report.rho = rho;
out.report.sn = exp(hyp.lik);
out.report.sigma_R = exp(hyp.lik);
end

function out = fit_icm3_downweight_R(timeM1, dataM1, timeM2, dataM2, tR, yR, ...
    icm_homo, tgrid, temporalKernel, max_iters, k_plot, n_out, sigma_R_factor)
% Heteroscedastic 3-output ICM: M1/M2 use sn from homo; R uses factor*sn.

LABEL_M1 = 1;
LABEL_M2 = 2;
LABEL_R  = 3;

[x_aug, y_aug, mu1, sd1, mu2, sd2, muR, sdR, n1, n2, nR] = ...
    build_aug3(timeM1, dataM1, timeM2, dataM2, tR, yR);

sn_homo = exp(icm_homo.hyp.lik);
noise_var = [sn_homo^2 * ones(n1 + n2, 1); ...
             (sigma_R_factor * sn_homo)^2 * ones(nR, 1)];

covICM = build_icm3_kernel(temporalKernel, n_out);
[nTemp, ~] = temporal_hyp_layout(temporalKernel);
nB = n_coreg_hypers(n_out);

hyp0 = icm_homo.hyp;
hyp0.lik = [];
hyp0.mean = [];

obj = @(h) gp_icm_hetero_noise('nlml', h, covICM, x_aug, y_aug, noise_var, nTemp, nB);
hyp = minimize(hyp0, obj, max_iters);
hyp.cov(2) = 0;
nlml = gp_icm_hetero_noise('nlml', hyp, covICM, x_aug, y_aug, noise_var, nTemp, nB);

x_te_M1 = [tgrid, LABEL_M1 * ones(size(tgrid))];
x_te_M2 = [tgrid, LABEL_M2 * ones(size(tgrid))];
x_te_R  = [tgrid, LABEL_R  * ones(size(tgrid))];
[~, ~, fmu1, fs21] = gp_icm_hetero_noise('pred', hyp, covICM, x_aug, y_aug, noise_var, nTemp, nB, x_te_M1);
[~, ~, fmu2, fs22] = gp_icm_hetero_noise('pred', hyp, covICM, x_aug, y_aug, noise_var, nTemp, nB, x_te_M2);
[~, ~, fmuR, fs2R] = gp_icm_hetero_noise('pred', hyp, covICM, x_aug, y_aug, noise_var, nTemp, nB, x_te_R);

mu_y1 = mu1 + sd1 * fmu1(:);
sf_y1 = sd1 * sqrt(max(fs21(:), 0));
mu_y2 = mu2 + sd2 * fmu2(:);
sf_y2 = sd2 * sqrt(max(fs22(:), 0));
mu_yR = muR + sdR * fmuR(:);
sf_yR = sdR * sqrt(max(fs2R(:), 0));

out.M1 = pack_raw_fit(mu_y1, sf_y1, k_plot);
out.M2 = pack_raw_fit(mu_y2, sf_y2, k_plot);
out.R  = pack_raw_fit(mu_yR, sf_yR, k_plot);
out.hyp = hyp;
out.nlml = nlml;
out.mu1 = mu1; out.sd1 = sd1;
out.mu2 = mu2; out.sd2 = sd2;
out.muR = muR; out.sdR = sdR;

B = chol2cov(hyp.cov(nTemp + (1:nB)), n_out);
rho = corr_from_B(B);
out.report.nlml = nlml;
out.report.ell = exp(hyp.cov(1));
out.report.B = B;
out.report.rho = rho;
out.report.sn = sn_homo;
out.report.sigma_R = sigma_R_factor * sn_homo;
end

function varargout = gp_icm_hetero_noise(mode, hyp, covICM, x, y, noise_var, nTemp, nB, xs)
% ICM GP with fixed per-row observation noise. Optimizes hyp.cov; hyp.lik unused.
switch lower(mode)
    case 'nlml'
        [varargout{1:nargout}] = icm_hetero_nlml(hyp, covICM, x, y, noise_var, nTemp, nB);
    case 'pred'
        [varargout{1}, varargout{2}, varargout{3}, varargout{4}] = ...
            icm_hetero_pred(hyp, covICM, x, y, noise_var, xs);
    otherwise
        error('gp_icm_hetero_noise:UnknownMode', 'Unknown mode: %s', mode);
end
end

function [nlml, dnlml] = icm_hetero_nlml(hyp, covICM, x, y, noise_var, nTemp, nB)
hyp.cov(2) = 0;
[Ky, z, nTot] = icm_build_Ky(hyp, covICM, x, y, noise_var);
L = chol(Ky, 'lower');
alpha = L' \ (L \ z);
nlml = 0.5 * (z' * alpha) + sum(log(diag(L))) + 0.5 * nTot * log(2 * pi);

if nargout > 1
    nCov = numel(hyp.cov);
    dnll_vec = zeros(nCov, 1);
    step = 1e-4;
    free_idx = [1, (nTemp + 1):(nTemp + nB)];  % ell + B; skip clamped sf
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

function report_icm3(label, report)
fprintf('[%s] NLML=%.4f, ell=%.4f, sn=%.4f, sigma_R=%.4f\n', ...
    label, report.nlml, report.ell, report.sn, report.sigma_R);
B = report.B;
rho = report.rho;
fprintf('  B =\n');
disp(B);
fprintf('  rho12=%.4f, rho13=%.4f, rho23=%.4f\n', rho(1,2), rho(1,3), rho(2,3));
end

function plot_icm3_panels(tgrid, icm, tM1, yM1, tM2, yM2, tR, yR, sgtitle_str)
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];
col_R  = [0.12, 0.35, 0.75];

tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

% M1
ax = nexttile;
hold(ax, 'on')
fill(ax, [tgrid; flipud(tgrid)], [icm.M1.hi; flipud(icm.M1.lo)], ...
    col_M1, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M1 95% band');
plot(ax, tgrid, icm.M1.mu, 'Color', col_M1, 'LineWidth', 2.0, 'DisplayName', 'M1 mean');
s1 = scatter(ax, tM1, yM1, 'filled', 'MarkerFaceColor', col_M1, 'DisplayName', 'M1 data');
s1.Marker = 'hexagram';
s1.SizeData = 120;
hold(ax, 'off')
xlabel(ax, 'Time (Days)', 'fontsize', 14)
ylabel(ax, 'cells/mm^2', 'fontsize', 14)
title(ax, 'M1')
legend(ax, 'Location', 'northwest')
xlim(ax, [0, 14])
set(ax, 'fontsize', 14)
grid(ax, 'on')

% M2
ax = nexttile;
hold(ax, 'on')
fill(ax, [tgrid; flipud(tgrid)], [icm.M2.hi; flipud(icm.M2.lo)], ...
    col_M2, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M2 95% band');
plot(ax, tgrid, icm.M2.mu, 'Color', col_M2, 'LineWidth', 2.0, 'DisplayName', 'M2 mean');
s2 = scatter(ax, tM2, yM2, 'filled', 'MarkerFaceColor', col_M2, 'DisplayName', 'M2 data');
s2.Marker = 'hexagram';
s2.SizeData = 120;
hold(ax, 'off')
xlabel(ax, 'Time (Days)', 'fontsize', 14)
ylabel(ax, 'cells/mm^2', 'fontsize', 14)
title(ax, 'M2')
legend(ax, 'Location', 'northwest')
xlim(ax, [0, 14])
set(ax, 'fontsize', 14)
grid(ax, 'on')

% R (+ implied mu1/mu2)
R_implied = icm.M1.mu ./ max(icm.M2.mu, eps);
ax = nexttile;
hold(ax, 'on')
fill(ax, [tgrid; flipud(tgrid)], [icm.R.hi; flipud(icm.R.lo)], ...
    col_R, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'R 95% band');
plot(ax, tgrid, icm.R.mu, 'Color', col_R, 'LineWidth', 2.0, 'DisplayName', 'R mean (ICM)');
plot(ax, tgrid, R_implied, 'k--', 'LineWidth', 1.5, 'DisplayName', '\mu_{M1}/\mu_{M2}');
sR = scatter(ax, tR, yR, 'filled', 'MarkerFaceColor', col_R, 'DisplayName', 'R data');
sR.Marker = 'hexagram';
sR.SizeData = 120;
yline(ax, 1, ':', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
hold(ax, 'off')
xlabel(ax, 'Time (Days)', 'fontsize', 14)
ylabel(ax, 'R = M1/M2', 'fontsize', 14)
title(ax, 'R')
legend(ax, 'Location', 'northwest')
xlim(ax, [0, 14])
set(ax, 'fontsize', 14)
grid(ax, 'on')

sgtitle(sgtitle_str, 'fontsize', 16);
end

% --- Unused helpers from previous SE / ridge experiments ---
%{
function [hyp, mu, s2] = fit_gp(x, y, xs, inffunc, meanfunc, covfunc, likfunc)
x = x(:); y = y(:); xs = xs(:);
ell0 = std(x);
sf0  = std(y);
sn0  = 0.1 * std(y);
if sn0 <= 0
    sn0 = 0.1;
end
hyp.mean = [];
hyp.cov  = log([ell0; sf0]);
hyp.lik  = log(sn0);
hyp = minimize(hyp, @gp, -100, inffunc, meanfunc, covfunc, likfunc, x, y);
[mu, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
mu = mu(:);
s2 = s2(:);
end

function [beta, yhat] = fit_poly_ridge(x, y, xs, deg, lambda, t_scale)
x = x(:); y = y(:); xs = xs(:);
u  = x / t_scale;
us = xs / t_scale;
X  = zeros(numel(u), deg + 1);
Xs = zeros(numel(us), deg + 1);
for k = 0:deg
    X(:, k + 1)  = u.^k;
    Xs(:, k + 1) = us.^k;
end
I_pen = eye(deg + 1);
I_pen(1, 1) = 0;
beta = (X' * X + lambda * I_pen) \ (X' * y);
yhat = Xs * beta;
end

function [lambda_best, mse, beta, yhat] = choose_lambda_loocv(x, y, xs, deg, lambda_grid, t_scale)
x = x(:); y = y(:);
n = numel(x);
n_lam = numel(lambda_grid);
mse = zeros(n_lam, 1);
for il = 1:n_lam
    lam = lambda_grid(il);
    sq_err = 0;
    for i = 1:n
        train = true(n, 1);
        train(i) = false;
        [~, y_pred] = fit_poly_ridge(x(train), y(train), x(i), deg, lam, t_scale);
        sq_err = sq_err + (y(i) - y_pred)^2;
    end
    mse(il) = sq_err / n;
end
[~, idx] = min(mse);
lambda_best = lambda_grid(idx);
[beta, yhat] = fit_poly_ridge(x, y, xs, deg, lambda_best, t_scale);
end
%}
