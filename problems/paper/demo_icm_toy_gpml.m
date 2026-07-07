% Toy two-output multi-output GP (MOGP) using the intrinsic coregionalization
% model (ICM) in GPML. The demo shows how an ICM MOGP borrows strength between
% two sparse time series: output 2 has a deliberate observation gap on (4, 8),
% and the ICM model fills it using output-1 information through the shared
% latent process, whereas an independent GP cannot.
%
% Model A (baseline): two independent single-output SE GPs (one per output).
% Model B (ICM):      a single MOGP with covariance
%                        k((t,d),(t',d')) = B(d,d') * exp(-0.5 (t-t')^2/ell^2)
%                     where B = L*L' is a 2x2 PSD coregionalization matrix and
%                     the base SE signal variance is clamped to 1 (its scale is
%                     non-identifiable with the diagonal of B). See covICM_SE.m.
%
% Requires covICM_SE.m, sample_gp_se.m, gp_nlml_cov_only.m (problems/) and GPML.
clear; clc; close all;

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
addpath(fileparts(fileparts(mfilename('fullpath'))));  % problems/

meanfunc = @meanZero;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

%% ===== Synthetic data =====
% u(t) is the shared latent temporal process. Both outputs are scaled copies
% of the same u(t), so all their structure is common up to a per-output scale:
%   f1(t) = u(t),   f2(t) = 0.7 * u(t).
% That common origin is exactly what the ICM captures with one latent SE kernel
% and a 2x2 coregionalization matrix B. With f2 = 0.7*f1 the outputs are
% perfectly correlated, so the true correlation is rho = 1 and, with unit u(t)
% variance, the true B = [1, 0.7; 0.7, 0.49].
rng(42);

ell_true = 3;                       % true SE length scale of u(t)
sn_true  = 0.1;                     % observation noise std for both outputs

t_dense = linspace(0, 14, 400)';    % dense grid for truth and sampling
u       = sample_gp_se(t_dense, ell_true, 1, 1e-10);   % u(t) ~ GP(0, k_SE)
f1_true = u;
f2_true = 0.7 * u;

% Sparse noisy observations. The two outputs are sampled at DIFFERENT times:
% the ICM cross-covariance depends only on the (time, label) pairs, not on
% aligned sampling, so outputs need not share observation time points.
n1 = 10;                            % ~8-12 points per output
t1 = sort(14 * rand(n1, 1));        % output 1 spans the whole window

% Output 2: sample only from [0,4] and [8,14] -> deliberate gap on (4, 8).
n2_left  = 5;
n2_right = 5;
t2 = sort([4 * rand(n2_left, 1); 8 + 6 * rand(n2_right, 1)]);
n2 = numel(t2);

f1_at = interp1(t_dense, f1_true, t1, 'pchip');
f2_at = interp1(t_dense, f2_true, t2, 'pchip');
y1 = f1_at + sn_true * randn(n1, 1);
y2 = f2_at + sn_true * randn(n2, 1);

% Stacked ICM training set: X(:,1)=time, X(:,2)=output label (1 or 2).
X = [t1, ones(n1, 1); t2, 2 * ones(n2, 1)];
y = [y1; y2];

gap_lo = 4; gap_hi = 8;             % highlighted output-2 gap
fprintf('Synthetic ICM toy data on t in [0, 14]\n');
fprintf('  output 1: n=%d points across the window\n', n1);
fprintf('  output 2: n=%d points, none in (%g, %g)\n', n2, gap_lo, gap_hi);
assert(~any(t2 > gap_lo & t2 < gap_hi), 'output 2 must have no data in the gap');

t_test = linspace(0, 14, 200)';

%% ===== Model A: independent single-output GPs =====
% Baseline: fit output 1 and output 2 separately with a zero mean, SE kernel,
% and Gaussian likelihood. Noise std is fixed at the known simulation value so
% only ell and sf are optimized (matches the repo's gp_nlml_cov_only pattern).
fprintf('\n=== Model A: independent SE GPs ===\n');
covSE = @covSEiso;
[indep1.m, indep1.sd, indep1.hyp, indep1.nlml] = ...
    fit_se_indep(t1, y1, sn_true, t_test, inffunc, meanfunc, covSE, likfunc);
[indep2.m, indep2.sd, indep2.hyp, indep2.nlml] = ...
    fit_se_indep(t2, y2, sn_true, t_test, inffunc, meanfunc, covSE, likfunc);
fprintf('  output 1: ell=%.3f sf=%.3f | NLML=%.3f\n', ...
    exp(indep1.hyp.cov(1)), exp(indep1.hyp.cov(2)), indep1.nlml);
fprintf('  output 2: ell=%.3f sf=%.3f | NLML=%.3f\n', ...
    exp(indep2.hyp.cov(1)), exp(indep2.hyp.cov(2)), indep2.nlml);

%% ===== Model B: ICM MOGP =====
% One shared SE latent kernel over time times a 2x2 PSD matrix B. B is
% parameterized through its Cholesky factor L (see covICM_SE.m) so B = L*L' is
% always PSD. Optimized covariance hyperparameters: [log(ell); b1; b21; b2].
% A single Gaussian noise term (hyp.lik) is shared by both outputs and
% optimized jointly (see the note on output-specific noise at the end).
fprintf('\n=== Model B: ICM MOGP ===\n');
hyp_icm.mean = [];
hyp_icm.cov  = [log(3); 0; 0; 0];  % ell~3, L=I -> B=I, uncorrelated start
hyp_icm.lik  = log(0.1);           % single shared Gaussian noise, optimized

hyp_icm  = minimize(hyp_icm, @gp, -200, inffunc, meanfunc, @covICM_SE, likfunc, X, y);
nlml_icm = gp(hyp_icm, inffunc, meanfunc, @covICM_SE, likfunc, X, y);

Xtest1 = [t_test, ones(size(t_test))];
Xtest2 = [t_test, 2 * ones(size(t_test))];
[~, ~, fmu1, fs2_1] = gp(hyp_icm, inffunc, meanfunc, @covICM_SE, likfunc, X, y, Xtest1);
[~, ~, fmu2, fs2_2] = gp(hyp_icm, inffunc, meanfunc, @covICM_SE, likfunc, X, y, Xtest2);
icm1.m = fmu1; icm1.sd = sqrt(max(fs2_1, 0));
icm2.m = fmu2; icm2.sd = sqrt(max(fs2_2, 0));

% Learned quantities.
%   ell_hat   : shared temporal length scale of the latent process.
%   B_hat     : coregionalization matrix. B(1,1) and B(2,2) are the marginal
%               variances of outputs 1 and 2 (on the latent time scale); B(1,2)
%               is their covariance. Large positive B(1,2) means the outputs
%               move together, which is what lets output 1 inform output 2.
%   rho_hat   : output correlation, B(1,2)/sqrt(B(1,1)*B(2,2)); ~1 here since
%               f2 = 0.7*f1 makes the two outputs perfectly correlated.
%   noise_hat : shared observation noise std.
ell_hat   = exp(hyp_icm.cov(1));
B_hat     = icm_chol2cov(hyp_icm.cov(2:4));
rho_hat   = B_hat(1, 2) / sqrt(B_hat(1, 1) * B_hat(2, 2));
noise_hat = exp(hyp_icm.lik);
negative_log_marginal_likelihood = nlml_icm;

%% ===== Plots =====
col1 = [0.00, 0.45, 0.74];   % output 1 (blue)
col2 = [0.85, 0.16, 0.16];   % output 2 (red)
zc   = 1.96;                 % 95% band

% Shared y-limits from data + all 95% bands (keeps the signal readable).
y_hi = max([y1; y2; ...
    indep1.m + zc * indep1.sd; indep2.m + zc * indep2.sd; ...
    icm1.m + zc * icm1.sd;     icm2.m + zc * icm2.sd]);
y_lo = min([y1; y2; ...
    indep1.m - zc * indep1.sd; indep2.m - zc * indep2.sd; ...
    icm1.m - zc * icm1.sd;     icm2.m - zc * icm2.sd]);
pad  = 0.15 * (y_hi - y_lo);
ylim_shared = [y_lo - pad, y_hi + pad];

figure('Color', 'w', 'Position', [80, 80, 720, 900], ...
    'Name', 'ICM toy MOGP: independent GP vs ICM');
tl = tiledlayout(3, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

% Panel 1: truth + sparse noisy observations, with the output-2 gap shaded.
ax1 = nexttile; hold(ax1, 'on'); grid(ax1, 'on');
shade_gap(ax1, gap_lo, gap_hi);
plot(ax1, t_dense, f1_true, '-', 'Color', col1, 'LineWidth', 1.8, 'DisplayName', 'f_1(t) truth');
plot(ax1, t_dense, f2_true, '-', 'Color', col2, 'LineWidth', 1.8, 'DisplayName', 'f_2(t) truth');
plot(ax1, t1, y1, 'o', 'MarkerSize', 6, 'MarkerFaceColor', col1, 'MarkerEdgeColor', 'k', 'DisplayName', 'output 1 data');
plot(ax1, t2, y2, 's', 'MarkerSize', 7, 'MarkerFaceColor', col2, 'MarkerEdgeColor', 'k', 'DisplayName', 'output 2 data');
title(ax1, 'True latent outputs and sparse noisy observations');
xlabel(ax1, 't'); ylabel(ax1, 'y'); xlim(ax1, [0, 14]); ylim(ax1, ylim_shared);
legend(ax1, 'Location', 'northeast', 'NumColumns', 2, 'FontSize', 8);

% Panel 2: independent GP posterior mean + 95% bands.
ax2 = nexttile; hold(ax2, 'on'); grid(ax2, 'on');
shade_gap(ax2, gap_lo, gap_hi);
plot_fit(ax2, t_test, indep1, col1, zc, 'output 1', t1, y1, t_dense, f1_true);
plot_fit(ax2, t_test, indep2, col2, zc, 'output 2', t2, y2, t_dense, f2_true);
title(ax2, 'Model A: independent single-output GPs');
xlabel(ax2, 't'); ylabel(ax2, 'y'); xlim(ax2, [0, 14]); ylim(ax2, ylim_shared);
legend(ax2, 'Location', 'northeast', 'NumColumns', 2, 'FontSize', 8);

% Panel 3: ICM posterior mean + 95% bands (output 2 tightens across the gap).
ax3 = nexttile; hold(ax3, 'on'); grid(ax3, 'on');
shade_gap(ax3, gap_lo, gap_hi);
plot_fit(ax3, t_test, icm1, col1, zc, 'output 1', t1, y1, t_dense, f1_true);
plot_fit(ax3, t_test, icm2, col2, zc, 'output 2', t2, y2, t_dense, f2_true);
title(ax3, sprintf('Model B: ICM MOGP'));
xlabel(ax3, 't'); ylabel(ax3, 'y'); xlim(ax3, [0, 14]); ylim(ax3, ylim_shared);
legend(ax3, 'Location', 'northeast', 'NumColumns', 2, 'FontSize', 8);

% Quantify the borrowing: mean output-2 posterior std inside the gap.
in_gap = t_test > gap_lo & t_test < gap_hi;
fprintf('\nOutput-2 mean posterior std in the gap (%g, %g):\n', gap_lo, gap_hi);
fprintf('  independent GP : %.4f\n', mean(indep2.sd(in_gap)));
fprintf('  ICM MOGP       : %.4f  (smaller = borrowed strength from output 1)\n', ...
    mean(icm2.sd(in_gap)));

%% ===== Learned parameters =====
fprintf('\n--- Learned ICM parameters ---\n');
ell_hat
B_hat
rho_hat
noise_hat
negative_log_marginal_likelihood

%% ===== Note: output-specific noise (optional, not enabled) =====
% The main model above uses a single Gaussian noise term via @likGauss, which
% is homoscedastic: it applies the same sigma_n to every row of y regardless of
% output label. Genuinely separate noise terms sigma_{n,1} and sigma_{n,2} are
% not expressible with the standard scalar @likGauss, because that likelihood
% has one parameter shared across all observations.
%
% To get output-specific noise you would either:
%   (a) build the noise into the covariance -- add a diagonal term
%       diag(sigma_n(d_i)^2) inside a custom kernel/likelihood, in the spirit of
%       gp_seiso_hetero_noise.m (problems/), which assembles Ky by hand with a
%       per-row noise vector; or
%   (b) use a heteroscedastic-capable likelihood and feed it per-row noise.
% Both go beyond the stock @likGauss interface, so we keep the single-noise
% version as the main implementation and document the limitation here.

%% ===== local functions =====
function [m, sd, hyp, nlml] = fit_se_indep(t, y, sn, t_test, inffunc, meanfunc, covfunc, likfunc)
% Independent single-output SE GP with fixed noise (optimize ell, sf only).
t = t(:); y = y(:);
sn_fixed = log(sn);
hyp_cov0 = log([std(t); std(y)]);
hyp_cov  = minimize(hyp_cov0, @gp_nlml_cov_only, -100, sn_fixed, ...
    inffunc, meanfunc, covfunc, likfunc, t, y);
hyp  = struct('mean', [], 'cov', hyp_cov(:), 'lik', sn_fixed);
nlml = gp(hyp, inffunc, meanfunc, covfunc, likfunc, t, y);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, t, y, t_test(:));
m  = fmu(:);
sd = sqrt(max(fs2(:), 0));
end

function B = icm_chol2cov(hyp)
% Reconstruct B = L*L' from ICM Cholesky hyperparameters [b1; b21; b2] with
% L = [exp(b1) 0; b21 exp(b2)] (lower triangular; see covICM_SE.m).
L = [exp(hyp(1)), 0; hyp(2), exp(hyp(3))];
B = L * L';
end

function shade_gap(ax, lo, hi)
% Shade the output-2 data gap and give it a legend-hidden patch.
yl = [-10, 10];   % generous; clipped by axis limits
p = patch(ax, [lo hi hi lo], [yl(1) yl(1) yl(2) yl(2)], [0.9 0.9 0.9], ...
    'EdgeColor', 'none', 'FaceAlpha', 0.5);
set(get(get(p, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
uistack(p, 'bottom');
end

function plot_fit(ax, t_test, fit, col, zc, name, t_train, y_train, t_dense, f_true)
tt = t_test(:)';
fill(ax, [tt, fliplr(tt)], [fit.m + zc * fit.sd; flipud(fit.m - zc * fit.sd)]', ...
    col, 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', sprintf('%s 95%%', name));
plot(ax, t_dense, f_true, ':', 'Color', col, 'LineWidth', 1.0, ...
    'HandleVisibility', 'off');
plot(ax, t_test, fit.m, '-', 'Color', col, 'LineWidth', 1.8, ...
    'DisplayName', sprintf('%s mean', name));
plot(ax, t_train, y_train, 'o', 'MarkerSize', 5, ...
    'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
end
