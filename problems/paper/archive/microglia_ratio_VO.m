% microglia_ratio_VO.m — Soft VO log-ratio paths on averaged microglia data.
% Path A: naive SE vs 2-out ICM (counts only).
% Path B: soft VO on r = log1p(M1)-log1p(M2); reconstruct M1/M2 from naive level.
% Path C: post-hoc coupling — ICM2 level + same VO-augmented r.
clear; close all; clc

%% Data (averaged, day 5 restored — same as microglia_ratio.m)

newtime = [0, 1, 2, 3, 5, 7, 14];
datapointsM1 = [5, 27.5, 122.5, 139.8, 325, 445, 816.67];
newtimeM2 = [0, 1, 2, 3, 5, 7, 14];
datapointsM2 = [5, 78.33, 179.5, 126.4, 800, 319, 136.67];

%% VO / GP config

t_cross = 5;
delta_r = 0.5;
n_virt_early = 6;
n_virt_late  = 6;
sigma_virt_soft = 0.4;
sigma_virt_cross = 0.05;
n_mc_coupled = 3000;
max_iters = -200;
k_plot = 1.96;
n_out2 = 2;

col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];
col_R  = [0.12, 0.35, 0.75];

%% GPML setup

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
covfunc  = @covSEiso;
inffunc  = @infGaussLik;
temporalKernel = @covSEiso;
tgrid = (0:0.1:14)';

%% Paired log-ratio from averaged data

[t_r, r_real] = build_paired_log_ratio_avg(newtime, datapointsM1, newtimeM2, datapointsM2);
fprintf('Averaged log-ratio r = log1p(M1)-log1p(M2):\n');
for i = 1:numel(t_r)
    fprintf('  t = %2g days: r = %.4f\n', t_r(i), r_real(i));
end

%% ===== Path A: Naive SE + 2-out ICM =====

fprintf('\n=== Path A: Naive independent SE GPs ===\n');
naive = fit_naive_gp(newtime(:), datapointsM1(:), newtimeM2(:), datapointsM2(:), ...
    tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot);
report_naive(naive.report);

fprintf('\n=== Path A: 2-output SE-ICM (no R) ===\n');
icm2 = fit_icm2(newtime(:), datapointsM1(:), newtimeM2(:), datapointsM2(:), ...
    tgrid, temporalKernel, meanfunc, likfunc, max_iters, k_plot, n_out2);
report_icm2(icm2.report);

%% ===== Path B: Soft VO on log-ratio + reconstruct from naive level =====

fprintf('\n=== Path B: Soft VO log-ratio GP ===\n');
[r_base, r_vo, virt_meta] = fit_log_ratio_vo( ...
    t_r, r_real, tgrid, t_cross, delta_r, n_virt_early, n_virt_late, ...
    sigma_virt_soft, sigma_virt_cross, inffunc, meanfunc, covfunc, likfunc);

recon_B = reconstruct_from_level_and_r( ...
    naive.M1.mu, naive.M1.sf, naive.M2.mu, naive.M2.sf, ...
    r_vo.mu, r_vo.sf, n_mc_coupled, k_plot);

%% ===== Path C: Post-hoc ICM2 level + VO r =====

fprintf('\n=== Path C: Post-hoc ICM2 level + VO r ===\n');
recon_C = reconstruct_from_level_and_r( ...
    icm2.M1.mu, icm2.M1.sf, icm2.M2.mu, icm2.M2.sf, ...
    r_vo.mu, r_vo.sf, n_mc_coupled, k_plot);

%% Figure 1 (A): Naive vs 2-out ICM

figure(1)
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
ax = nexttile;
plot_m1_m2_panel(ax, tgrid, naive, newtime, datapointsM1, newtimeM2, datapointsM2, ...
    col_M1, col_M2, 'Naive independent SE GPs');
ax = nexttile;
plot_m1_m2_panel(ax, tgrid, icm2, newtime, datapointsM1, newtimeM2, datapointsM2, ...
    col_M1, col_M2, '2-output SE-ICM (no R)');
sgtitle('Path A: averaged counts — naive SE vs 2-output ICM', 'fontsize', 16);

%% Figure 2 (B): r baseline vs VO | reconstructed M1+M2 from naive level

sd_base = r_base.sf;
sd_vo = r_vo.sf;
ylim_r = [min([r_real; r_base.mu - k_plot * sd_base; r_vo.mu - k_plot * sd_vo]) - 0.15, ...
          max([r_real; r_base.mu + k_plot * sd_base; r_vo.mu + k_plot * sd_vo]) + 0.15];

figure(2)
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
ax = nexttile;
plot_ratio_compare(ax, tgrid, r_base, r_vo, t_r, r_real, virt_meta, t_cross, k_plot, ylim_r, col_R);
ax = nexttile;
plot_m1_m2_recon(ax, tgrid, recon_B, newtime, datapointsM1, newtimeM2, datapointsM2, ...
    col_M1, col_M2, 'Reconstructed M1 & M2 (naive level + VO r)');
sgtitle('Path B: soft VO log-ratio (naive level for reconstruction)', 'fontsize', 16);

%% Figure 3 (C): ICM2 raw | ICM2 level + VO r reconstructed

figure(3)
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
ax = nexttile;
plot_m1_m2_panel(ax, tgrid, icm2, newtime, datapointsM1, newtimeM2, datapointsM2, ...
    col_M1, col_M2, '2-output ICM (raw counts)');
ax = nexttile;
plot_m1_m2_recon(ax, tgrid, recon_C, newtime, datapointsM1, newtimeM2, datapointsM2, ...
    col_M1, col_M2, 'Post-hoc: ICM2 level + VO r');
sgtitle('Path C: does soft VO ordering reshape ICM counts?', 'fontsize', 16);

%% Local helpers

function [t, r] = build_paired_log_ratio_avg(timeM1, yM1, timeM2, yM2)
timeM1 = timeM1(:); yM1 = yM1(:);
timeM2 = timeM2(:); yM2 = yM2(:);
shared = intersect(timeM1, timeM2);
t = shared(:);
r = zeros(numel(t), 1);
for i = 1:numel(t)
    m1 = yM1(timeM1 == t(i));
    m2 = yM2(timeM2 == t(i));
    r(i) = log1p(m1(1)) - log1p(m2(1));
end
end

function [r_base, r_vo, virt_meta] = fit_log_ratio_vo( ...
    t_real, r_real, tgrid, t_cross, delta_r, n_virt_early, n_virt_late, ...
    sigma_virt_soft, sigma_virt_cross, inffunc, meanfunc, covfunc, likfunc)

t_real = t_real(:);
r_real = r_real(:);
n_real = numel(t_real);

[hyp_base, mu_base, s2_base, nlml_base] = fit_ratio_baseline( ...
    t_real, r_real, tgrid, inffunc, meanfunc, covfunc, likfunc);
sigma_real_use = exp(hyp_base.lik);

if n_virt_early > 0
    t_virt_early = linspace(0, max(0, t_cross - 0.1), n_virt_early)';
else
    t_virt_early = zeros(0, 1);
end
if n_virt_late > 0
    t_virt_late = linspace(min(14, t_cross + 0.1), 14, n_virt_late)';
else
    t_virt_late = zeros(0, 1);
end
t_virt_cross = t_cross;
r_virt_early = -delta_r * ones(numel(t_virt_early), 1);
r_virt_cross = 0;
r_virt_late  =  delta_r * ones(numel(t_virt_late), 1);

x_aug = [t_real; t_virt_early; t_virt_cross; t_virt_late];
y_aug = [r_real; r_virt_early; r_virt_cross; r_virt_late];
noise_var_aug = [sigma_real_use^2 * ones(n_real, 1); ...
    sigma_virt_soft^2 * ones(numel(t_virt_early) + numel(t_virt_late), 1); ...
    sigma_virt_cross^2];

[hyp_aug, mu_aug, s2_aug, nlml_aug] = fit_ratio_virtual_obs( ...
    x_aug, y_aug, noise_var_aug, tgrid, hyp_base);

fprintf('  n_real=%d | virtual early=%d cross=1 late=%d\n', ...
    n_real, numel(t_virt_early), numel(t_virt_late));
fprintf('  t_cross=%.2f | delta_r=%.3g | sigma_real=%.4g | sigma_virt_soft=%.4g | sigma_virt_cross=%.4g\n', ...
    t_cross, delta_r, sigma_real_use, sigma_virt_soft, sigma_virt_cross);
fprintf('  Baseline r: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_base.cov(1)), exp(hyp_base.cov(2)), exp(hyp_base.lik), nlml_base);
fprintf('  VO-aug r:   ell=%.4f, sf=%.4f | NLML=%.4f (hetero, n_aug=%d)\n', ...
    exp(hyp_aug.cov(1)), exp(hyp_aug.cov(2)), nlml_aug, numel(y_aug));

r_base.mu = mu_base(:);
r_base.sf = sqrt(max(s2_base(:), 0));
r_base.hyp = hyp_base;
r_base.nlml = nlml_base;

r_vo.mu = mu_aug(:);
r_vo.sf = sqrt(max(s2_aug(:), 0));
r_vo.hyp = hyp_aug;
r_vo.nlml = nlml_aug;

virt_meta = struct( ...
    't_early', t_virt_early, 'r_early', r_virt_early, ...
    't_cross', t_virt_cross, 'r_cross', r_virt_cross, ...
    't_late', t_virt_late, 'r_late', r_virt_late);
end

function recon = reconstruct_from_level_and_r( ...
    mu_M1_level, sf_M1_level, mu_M2_level, sf_M2_level, mu_r, sf_r, n_mc, k_plot)
% Symmetric reconstruction: z_mu from level GPs; separation from r GP.
mu_M1_level = mu_M1_level(:);
mu_M2_level = mu_M2_level(:);
mu_r = mu_r(:);
sf_r = sf_r(:);

z_mu = 0.5 * (log1p(max(0, mu_M1_level)) + log1p(max(0, mu_M2_level)));
mu_M1 = expm1(z_mu + 0.5 * mu_r);
mu_M2 = expm1(z_mu - 0.5 * mu_r);

[lo_M1, hi_M1, lo_M2, hi_M2] = coupled_band_mc( ...
    mu_M1_level, sf_M1_level(:), mu_M2_level, sf_M2_level(:), mu_r, sf_r, n_mc);

recon.M1 = pack_raw_fit(mu_M1, (hi_M1 - lo_M1) / (2 * k_plot), k_plot);
recon.M1.lo = lo_M1;
recon.M1.hi = hi_M1;
recon.M2 = pack_raw_fit(mu_M2, (hi_M2 - lo_M2) / (2 * k_plot), k_plot);
recon.M2.lo = lo_M2;
recon.M2.hi = hi_M2;
end

function [lo_M1, hi_M1, lo_M2, hi_M2] = coupled_band_mc( ...
    mu_M1, sd_M1, mu_M2, sd_M2, mu_r, sd_r, n_mc)
mu_M1 = mu_M1(:); mu_M2 = mu_M2(:); mu_r = mu_r(:);
sd_M1 = sd_M1(:); sd_M2 = sd_M2(:); sd_r = sd_r(:);
n_t = numel(mu_M1);
s1 = mu_M1 + sd_M1 .* randn(n_t, n_mc);
s2 = mu_M2 + sd_M2 .* randn(n_t, n_mc);
sr = mu_r + sd_r .* randn(n_t, n_mc);
z_s = 0.5 * (log1p(max(0, s1)) + log1p(max(0, s2)));
m1_s = expm1(z_s + 0.5 * sr);
m2_s = expm1(z_s - 0.5 * sr);
lo_M1 = max(0, quantile(m1_s, 0.025, 2));
hi_M1 = quantile(m1_s, 0.975, 2);
lo_M2 = max(0, quantile(m2_s, 0.025, 2));
hi_M2 = quantile(m2_s, 0.975, 2);
lo_M1 = lo_M1(:); hi_M1 = hi_M1(:);
lo_M2 = lo_M2(:); hi_M2 = hi_M2(:);
end

function [hyp, mu, s2, nlml] = fit_ratio_baseline(x, y, xs, inffunc, meanfunc, covfunc, likfunc)
[hyp, mu, s2] = fit_gp(x, y, xs, inffunc, meanfunc, covfunc, likfunc);
nlml = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x(:), y(:));
end

function [hyp, mu, s2, nlml] = fit_ratio_virtual_obs(x, y, noise_var, xs, hyp0)
x = x(:); y = y(:); noise_var = noise_var(:); xs = xs(:);
hyp_tpl = struct('mean', [], 'cov', hyp0.cov(:), 'lik', []);
obj = @(h) gp_seiso_hetero_noise('nlml', h, x, y, noise_var);
hyp = minimize(hyp_tpl, obj, -100);
nlml = obj(hyp);
[~, ~, mu, s2] = gp_seiso_hetero_noise('pred', hyp, x, y, noise_var, xs);
mu = mu(:);
s2 = s2(:);
end

function [hyp, mu, s2] = fit_gp(x, y, xs, inffunc, meanfunc, covfunc, likfunc)
x = x(:); y = y(:); xs = xs(:);
ell0 = max(std(x), 1e-3);
sf0  = max(std(y), 1e-3);
sn0  = max(0.1 * std(y), 1e-3);
hyp.mean = [];
hyp.cov  = log([ell0; sf0]);
hyp.lik  = log(sn0);
hyp = minimize(hyp, @gp, -100, inffunc, meanfunc, covfunc, likfunc, x, y);
[mu, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
mu = mu(:);
s2 = s2(:);
end

function covICM = build_icm_kernel(temporalKernel, n_out)
covICM = {@covProd, { ...
    {@covMask, {1, temporalKernel}}, ...
    {@covMask, {2, {@covDiscrete, n_out}}} }};
end

function [nTemp, hasAlpha] = temporal_hyp_layout(covfunc)
if iscell(covfunc) && strcmp(func2str(covfunc{1}), 'covMaterniso')
    nTemp = 2; hasAlpha = false;
elseif isa(covfunc, 'function_handle') && strcmp(func2str(covfunc), 'covRQiso')
    nTemp = 3; hasAlpha = true;
else
    nTemp = 2; hasAlpha = false;
end
end

function nB = n_coreg_hypers(n_out)
nB = n_out * (n_out + 1) / 2;
end

function Lchol0 = init_Lchol(n_out)
nB = n_coreg_hypers(n_out);
Lchol0 = zeros(nB, 1);
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

function [hyp0, inffunc] = init_icm_hyp(temporalKernel, x_aug, y_aug, n_out)
meanfunc = @meanZero;
likfunc  = @likGauss;
covICM   = build_icm_kernel(temporalKernel, n_out);
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

function pheno = pack_raw_fit(mu, sf, k_plot)
pheno.mu = mu(:);
pheno.sf = sf(:);
pheno.lo = mu(:) - k_plot .* sf(:);
pheno.hi = mu(:) + k_plot .* sf(:);
end

function fit = fit_single_gp(x, y, tgrid, temporalKernel, meanfunc, likfunc, max_iters)
x = x(:); y = y(:);
inffunc = @infGaussLik;
ell0 = max(std(x), 0.5);
sf0 = max(std(y), 0.1);
sn0 = 0.1 * sf0;
[~, hasAlpha] = temporal_hyp_layout(temporalKernel);
if hasAlpha
    hyp.mean = []; hyp.cov = log([ell0; sf0; 1]);
else
    hyp.mean = []; hyp.cov = log([ell0; sf0]);
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
covICM = build_icm_kernel(temporalKernel, n_out);
[hyp0, inffunc] = init_icm_hyp(temporalKernel, x_aug, y_aug, n_out);
hyp = minimize(hyp0, @gp, max_iters, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);
nlml = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);
x_te_M1 = [tgrid, LABEL_M1 * ones(size(tgrid))];
x_te_M2 = [tgrid, LABEL_M2 * ones(size(tgrid))];
[~, ~, fmu1, fs21] = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug, x_te_M1);
[~, ~, fmu2, fs22] = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug, x_te_M2);
out.M1 = pack_raw_fit(mu1 + sd1 * fmu1(:), sd1 * sqrt(max(fs21(:), 0)), k_plot);
out.M2 = pack_raw_fit(mu2 + sd2 * fmu2(:), sd2 * sqrt(max(fs22(:), 0)), k_plot);
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
s1.Marker = 'hexagram'; s1.SizeData = 120;
fill(ax, [tgrid; flipud(tgrid)], [fit.M2.hi; flipud(fit.M2.lo)], ...
    col_M2, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M2 95% band');
plot(ax, tgrid, fit.M2.mu, ':', 'Color', col_M2, 'LineWidth', 2.0, 'DisplayName', 'M2 mean');
s2 = scatter(ax, tM2, yM2, 'filled', 'MarkerFaceColor', col_M2, 'DisplayName', 'M2 data');
s2.Marker = 'hexagram'; s2.SizeData = 120;
hold(ax, 'off')
xlabel(ax, 'Time (Days)', 'fontsize', 14)
ylabel(ax, 'cells/mm^2', 'fontsize', 14)
title(ax, title_str)
legend(ax, 'Location', 'northwest')
xlim(ax, [0, 14])
set(ax, 'fontsize', 14)
grid(ax, 'on')
end

function plot_m1_m2_recon(ax, tgrid, recon, tM1, yM1, tM2, yM2, col_M1, col_M2, title_str)
% Same overlay style; bands from MC coupled reconstruction.
hold(ax, 'on')
fill(ax, [tgrid; flipud(tgrid)], [recon.M1.hi; flipud(recon.M1.lo)], ...
    col_M1, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M1 95% band (MC)');
plot(ax, tgrid, recon.M1.mu, ':', 'Color', col_M1, 'LineWidth', 2.0, 'DisplayName', 'M1 coupled mean');
s1 = scatter(ax, tM1, yM1, 'filled', 'MarkerFaceColor', col_M1, 'DisplayName', 'M1 data');
s1.Marker = 'hexagram'; s1.SizeData = 120;
fill(ax, [tgrid; flipud(tgrid)], [recon.M2.hi; flipud(recon.M2.lo)], ...
    col_M2, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M2 95% band (MC)');
plot(ax, tgrid, recon.M2.mu, ':', 'Color', col_M2, 'LineWidth', 2.0, 'DisplayName', 'M2 coupled mean');
s2 = scatter(ax, tM2, yM2, 'filled', 'MarkerFaceColor', col_M2, 'DisplayName', 'M2 data');
s2.Marker = 'hexagram'; s2.SizeData = 120;
hold(ax, 'off')
xlabel(ax, 'Time (Days)', 'fontsize', 14)
ylabel(ax, 'cells/mm^2', 'fontsize', 14)
title(ax, title_str)
legend(ax, 'Location', 'northwest')
xlim(ax, [0, 14])
set(ax, 'fontsize', 14)
grid(ax, 'on')
end

function plot_ratio_compare(ax, tgrid, r_base, r_vo, t_real, r_real, virt, t_cross, k_plot, ylim_r, col_R)
hold(ax, 'on')
fill(ax, [tgrid; flipud(tgrid)], [r_vo.mu + k_plot*r_vo.sf; flipud(r_vo.mu - k_plot*r_vo.sf)], ...
    col_R, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'VO r 95% band');
plot(ax, tgrid, r_base.mu, '--', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.5, ...
    'DisplayName', 'Baseline r mean');
plot(ax, tgrid, r_vo.mu, ':', 'Color', col_R, 'LineWidth', 2.0, 'DisplayName', 'VO r mean');
scatter(ax, t_real, r_real, 70, 'filled', 'MarkerFaceColor', col_R, 'DisplayName', 'Paired r data');
if ~isempty(virt.t_early)
    scatter(ax, virt.t_early, virt.r_early, 70, 's', ...
        'MarkerFaceColor', [0.85 0.85 0.85], 'MarkerEdgeColor', 'k', ...
        'DisplayName', 'VO early (M2>M1)');
end
scatter(ax, virt.t_cross, virt.r_cross, 90, 'd', ...
    'MarkerFaceColor', [0.85 0.85 0.85], 'MarkerEdgeColor', 'k', ...
    'DisplayName', 'VO crossover');
if ~isempty(virt.t_late)
    scatter(ax, virt.t_late, virt.r_late, 70, 's', ...
        'MarkerFaceColor', [0.85 0.85 0.85], 'MarkerEdgeColor', 'k', ...
        'DisplayName', 'VO late (M1>M2)');
end
yline(ax, 0, 'k:', 'HandleVisibility', 'off');
xline(ax, t_cross, 'k:', 'HandleVisibility', 'off');
hold(ax, 'off')
xlabel(ax, 'Time (Days)', 'fontsize', 14)
ylabel(ax, 'r = log1p(M1)-log1p(M2)', 'fontsize', 14)
title(ax, 'Baseline vs soft-VO log-ratio GP')
legend(ax, 'Location', 'best')
xlim(ax, [0, 14])
ylim(ax, ylim_r)
set(ax, 'fontsize', 14)
grid(ax, 'on')
end
