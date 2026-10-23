% SINDy.m — ensemble GP-SINDy for M1/M2 microglia dynamics.
% States and derivatives come from independent GPs on the full dataset.
% Active GPs: the informed GPs of GP_for_Andrea.m (Figure 3), fit on the
%   raw M1 and M2 counts.
%   M1 is zero-mean SE. Its day-0 noise variance is the day-0 sample
%   variance, and every other row shares one learned noise level.
%   M2 is zero-mean Matern 5/2 whose noise is a second GP on the log
%   sample variance of the replicates (days with at least two points).
%   That noise GP has lengthscale k*ell with k = 1+exp(raw), a constant
%   mean fixed at the average observed log variance, and a fixed nugget
%   2/(n-1). Days with one replicate use exp(mu + s2/2) of that noise GP.
% Figure 198 shows the GP fits on days 0-14. Figure 203 extrapolates them to
%   day 50 and adds the held-out day 35 points (never used for fitting).
% Figure 202 shows the derivative GP posteriors (mean and 95% band of
%   dM1/dt and dM2/dt) with a separate legend figure.
%
% Library: 1, M1, M2, M1^2, M1*M2, M2^2,
%          M1^3, M1^2*M2, M1*M2^2, M2^3
%
% Ensemble GP-SINDy (inspired by Chung et al. 2019):
%   - Draws 500 joint [state; derivative] trajectories per species from the
%     informed GP posteriors, on 50 points over the data domain (days
%     0-14). It uses analytic kernel derivative blocks and seed 123.
%   - Rejects a whole M1-M2 draw if any state on the grid is negative, until
%     500 are accepted.
%   - Fits STLS (lambda = 0.01) to each draw separately.
%   - Reports selection frequencies, coefficient statistics, library
%     conditioning, and forward simulations (figures 301-307). Figures
%     308-309 solve one individual model from Figure 306 (idxFwd(2)) on
%     days 0-14 and 0-50.
%   - Section E keeps the terms selected in more than 20% of the fits,
%     refits every draw on those terms by least squares, and solves the
%     models built from the mean and the median of each refit coefficient
%     (figures 310-312).
%   - Saves everything to results/SINDy_ensemble/SINDy_ensemble_results.mat.

clear; close all; clc

%% Paths / GPML
gpml_folder_name = "C:\Users\chorc\OneDrive\Documents\Stroke Research\Gaussian Processes\Old\gpml-matlab-master\gpml-matlab-master";
if ~exist('minimize', 'file')
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

%% Data (full replicate series from microglia.m, no Day 35)
timeM1 = [0, 1, 3, 5, 7, 14, ...
          3, 7, ...
          2, ...
          14, ...
          3, ...
          3, ...
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
          3, ...
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

timeM1 = timeM1(:);
dataM1 = max(dataM1(:), 0);
timeM2 = timeM2(:);
dataM2 = max(dataM2(:), 0);

% Day 35 holdout: plotted in Figure 203 only, never used for fitting.
timeM1_hold = 35;
dataM1_hold = 550;
timeM2_hold = 35;
dataM2_hold = 50;

%% Shared settings
tgrid = (0:0.1:35)';
tgrid50 = (0:0.1:50)';
k_plot = 1.96;

%% Fit the informed GPs of GP_for_Andrea.m on the raw counts
fprintf('=== Fit informed GPs (M1 SE day-0 noise, M2 Matern 5/2 log-noise GP) ===\n');
max_iters  = -200;
meanfunc   = @meanZero;
covM1      = @covSEiso;
covM2      = {@covMaterniso, 5};
mean_noise = @meanConst;
likfunc    = @likGauss;
inffunc    = @infGaussLik;

s2_day0_m1 = day0_sample_var(timeM1, dataM1);
fitM1 = fit_m1_day0(timeM1, dataM1, s2_day0_m1, tgrid, meanfunc, covM1, max_iters);
fitM2 = fit_full_lognoise(timeM2, dataM2, tgrid, meanfunc, mean_noise, ...
    covM2, inffunc, likfunc, max_iters);
noiseM1 = fitM1.noise_var;
noiseM2 = fitM2.noise_var;
muM1_grid = fitM1.mu;
muM2_grid = fitM2.mu;
sfM1_grid = sqrt(max(fitM1.s2, 0));
sfM2_grid = sqrt(max(fitM2.s2, 0));

%% Derivative GP posteriors on the day 0-50 plotting grid (analytic)
[dmuM1_grid, ds2M1_grid] = deriv_fixed_noise(fitM1.hyp, 'se', meanfunc, covM1, ...
    timeM1, dataM1, noiseM1, tgrid50);
[dmuM2_grid, ds2M2_grid] = deriv_fixed_noise(fitM2.hyp, 'matern5', meanfunc, covM2, ...
    timeM2, dataM2, noiseM2, tgrid50);
dsfM1_grid = sqrt(ds2M1_grid);
dsfM2_grid = sqrt(ds2M2_grid);

% Plot fitted GPs vs full dataset
figure(198)
hold on
fill([tgrid; flipud(tgrid)], ...
    [muM1_grid(:) + k_plot * sfM1_grid; flipud(muM1_grid(:) - k_plot * sfM1_grid)], ...
    'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
fill([tgrid; flipud(tgrid)], ...
    [muM2_grid(:) + k_plot * sfM2_grid; flipud(muM2_grid(:) - k_plot * sfM2_grid)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(tgrid, muM1_grid, 'k', 'LineWidth', 2.0);
plot(tgrid, muM2_grid, 'r', 'LineWidth', 2.0);
scatter(timeM1, dataM1, 50, 'k', 'filled');
scatter(timeM2, dataM2, 50, 'r', 'filled');
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
xlim([0, 14])
set(gca, 'fontsize', 20)

% Same GPs extrapolated to day 50, with the held-out day 35 points
% (filled diamonds). The GPs are fit on days 0-14 only.
[muM1_50, s2M1_50] = pred_fixed_noise(fitM1.hyp, meanfunc, covM1, timeM1, dataM1, noiseM1, tgrid50);
[muM2_50, s2M2_50] = pred_fixed_noise(fitM2.hyp, meanfunc, covM2, timeM2, dataM2, noiseM2, tgrid50);
sfM1_50 = sqrt(s2M1_50);
sfM2_50 = sqrt(s2M2_50);
figure(203)
hold on
fill([tgrid50; flipud(tgrid50)], ...
    [muM1_50 + k_plot * sfM1_50; flipud(muM1_50 - k_plot * sfM1_50)], ...
    'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
fill([tgrid50; flipud(tgrid50)], ...
    [muM2_50 + k_plot * sfM2_50; flipud(muM2_50 - k_plot * sfM2_50)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(tgrid50, muM1_50, 'k', 'LineWidth', 2.0);
plot(tgrid50, muM2_50, 'r', 'LineWidth', 2.0);
scatter(timeM1, dataM1, 50, 'k', 'filled');
scatter(timeM2, dataM2, 50, 'r', 'filled');
scatter(timeM1_hold, dataM1_hold, 60, 'k', 'd', 'filled');
scatter(timeM2_hold, dataM2_hold, 60, 'r', 'd', 'filled');
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
xlim([0, 50])
set(gca, 'fontsize', 20)

% Separate legend figure (not overlaid on the GP plot)
figL = figure('Color', 'w', 'Position', [100, 100, 900, 80], ...
    'Name', 'Informed GP legend');
axL = axes('Parent', figL, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
    'Position', [0 0 1 1]);
hold(axL, 'on');
hL = gobjects(6, 1);
hL(1) = fill(axL, nan, nan, 'k', 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', 'M1 95% confidence region');
hL(2) = plot(axL, nan, nan, 'k', 'LineWidth', 2.0, 'DisplayName', 'M1 GP mean');
hL(3) = plot(axL, nan, nan, 'o', 'Color', 'k', 'MarkerFaceColor', 'k', ...
    'MarkerSize', 6, 'DisplayName', 'M1 data');
hL(4) = fill(axL, nan, nan, 'r', 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', 'M2 95% confidence region');
hL(5) = plot(axL, nan, nan, 'r', 'LineWidth', 2.0, 'DisplayName', 'M2 GP mean');
hL(6) = plot(axL, nan, nan, 'o', 'Color', 'r', 'MarkerFaceColor', 'r', ...
    'MarkerSize', 6, 'DisplayName', 'M2 data');
lgd = legend(axL, hL, 'Orientation', 'horizontal', 'NumColumns', 3);
lgd.FontSize = 16;
lgd.ItemTokenSize = [20, 12];
drawnow;
figL.Units = 'pixels';
lgd.Units = 'pixels';
lp = lgd.Position;
margin = 6;
figL.Position(3:4) = [lp(3) + 2 * margin, lp(4) + 2 * margin];
lgd.Position = [margin, margin, lp(3), lp(4)];

% Plot GP derivative posteriors (mean and 95% band of dM1/dt, dM2/dt)
figure(202)
hold on
fill([tgrid50; flipud(tgrid50)], ...
    [dmuM1_grid(:) + k_plot * dsfM1_grid; flipud(dmuM1_grid(:) - k_plot * dsfM1_grid)], ...
    'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
fill([tgrid50; flipud(tgrid50)], ...
    [dmuM2_grid(:) + k_plot * dsfM2_grid; flipud(dmuM2_grid(:) - k_plot * dsfM2_grid)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(tgrid50, dmuM1_grid, 'k', 'LineWidth', 2.0);
plot(tgrid50, dmuM2_grid, 'r', 'LineWidth', 2.0);
yline(0, ':', 'LineWidth', 1.5);
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2/day', 'fontsize', 20)
xlim([0, 50])
set(gca, 'fontsize', 20)

figLd = figure('Color', 'w', 'Position', [100, 100, 900, 80], ...
    'Name', 'Informed GP derivative legend');
axLd = axes('Parent', figLd, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
    'Position', [0 0 1 1]);
hold(axLd, 'on');
hLd = gobjects(4, 1);
hLd(1) = fill(axLd, nan, nan, 'k', 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', 'M1 95% confidence region');
hLd(2) = plot(axLd, nan, nan, 'k', 'LineWidth', 2.0, 'DisplayName', 'M1 GP derivative mean');
hLd(3) = fill(axLd, nan, nan, 'r', 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', 'M2 95% confidence region');
hLd(4) = plot(axLd, nan, nan, 'r', 'LineWidth', 2.0, 'DisplayName', 'M2 GP derivative mean');
lgdD = legend(axLd, hLd, 'Orientation', 'horizontal', 'NumColumns', 2);
lgdD.FontSize = 16;
lgdD.ItemTokenSize = [20, 12];
drawnow;
figLd.Units = 'pixels';
lgdD.Units = 'pixels';
lpD = lgdD.Position;
figLd.Position(3:4) = [lpD(3) + 2 * margin, lpD(4) + 2 * margin];
lgdD.Position = [margin, margin, lpD(3), lpD(4)];

%% Ensemble GP-SINDy: GP posterior trajectory sampling
% Inspired by Chung et al. (2019). Joint draws of each latent state and its
% analytic derivative come from the informed GPs fitted above, with the same
% hyperparameters and fixed noise and no refitting, on a 50-point grid over
% the data domain. Any M1-M2 draw with a negative state on the grid is
% rejected as a whole, so the ensemble samples the GP posterior truncated to
% nonnegative grid values. This is not positivity in continuous time.
% Each accepted draw gets its own STLS fit.
% Spread across the ensemble reflects uncertainty in the fitted GP posterior
% only. Ensemble agreement is not independent experimental confirmation.
fprintf('\n=== Ensemble GP-SINDy ===\n');
ens_cfg.seed         = 123;
ens_cfg.nSamples     = 500;
ens_cfg.nGrid        = 50;
ens_cfg.maxDraws     = 200000;
ens_cfg.lambda       = 0.001;
ens_cfg.rel_jitters  = [0, 1e-12, 1e-10, 1e-8, 1e-6];   % relative to each variance
ens_cfg.sel_tol      = 0;      % STLS sets unselected terms to exactly 0
ens_cfg.nShow        = 40;
ens_cfg.nFwdModels   = 6;
ens_cfg.blowup       = 1e5;    % cells/mm^2; |state| above this counts as divergence
ens_cfg.subset_pThresh = 0.2;  % section E keeps terms selected in more than this fraction of fits
ens_cfg.save_results = true;

tData = [timeM1; timeM2];
nGrid = ens_cfg.nGrid;
tGrid = linspace(min(tData), max(tData), nGrid)';
nSamples = ens_cfg.nSamples;
lib_names = sindy_library_names();
nLib = numel(lib_names);
xlE = [tGrid(1), tGrid(end)];

% Kernel derivative blocks vs finite differences of the GPML kernels
fd_pairs = [0.5, 0.5; 1.3, 4.1; 7.5, 3.2; 10.0, 10.7; 2.0, 13.0];
errKdM1 = check_kernel_derivs('se', fitM1.hyp.cov, covM1, fd_pairs);
errKdM2 = check_kernel_derivs('matern5', fitM2.hyp.cov, covM2, fd_pairs);
fprintf('Kernel derivative FD check (max scaled error): M1 SE %.2e | M2 Matern 5/2 %.2e\n', ...
    max(errKdM1(:)), max(errKdM2(:)));
if max([errKdM1(:); errKdM2(:)]) > 1e-5
    warning('SINDy:KernelDeriv', 'Analytic kernel derivatives disagree with finite differences.');
end

% Joint posteriors of [f(tGrid); df/dt(tGrid)]
postM1 = joint_state_deriv_posterior(fitM1.hyp, 'se', meanfunc, covM1, ...
    timeM1, dataM1, noiseM1, tGrid);
postM2 = joint_state_deriv_posterior(fitM2.hyp, 'matern5', meanfunc, covM2, ...
    timeM2, dataM2, noiseM2, tGrid);
SM1 = joint_sampler(postM1.Sigma, ens_cfg.rel_jitters);
SM2 = joint_sampler(postM2.Sigma, ens_cfg.rel_jitters);
[fmuE1, fs2E1] = pred_fixed_noise(fitM1.hyp, meanfunc, covM1, timeM1, dataM1, noiseM1, tGrid);
[fmuE2, fs2E2] = pred_fixed_noise(fitM2.hyp, meanfunc, covM2, timeM2, dataM2, noiseM2, tGrid);
[dmuE1, ds2E1] = deriv_fixed_noise(fitM1.hyp, 'se', meanfunc, covM1, timeM1, dataM1, noiseM1, tGrid);
[dmuE2, ds2E2] = deriv_fixed_noise(fitM2.hyp, 'matern5', meanfunc, covM2, timeM2, dataM2, noiseM2, tGrid);
report_joint_post('M1', postM1, fmuE1, fs2E1, dmuE1, ds2E1, SM1);
report_joint_post('M2', postM2, fmuE2, fs2E2, dmuE2, ds2E2, SM2);

% Rejection sampling: a draw is rejected as a whole if any M1 or M2 state on
% the grid is negative
rng(ens_cfg.seed);
[Xsamples, dXsamples, acc] = draw_nonneg_pairs(postM1, SM1, postM2, SM2, ...
    nSamples, ens_cfg.maxDraws);
fprintf(['Sampling: %d draws, %d accepted, %d rejected (acceptance %.3f). ' ...
    'Rejected for M1 only %d, M2 only %d, both %d.\n'], acc.draws, acc.accepted, ...
    acc.rejected, acc.rate, acc.rej_M1_only, acc.rej_M2_only, acc.rej_both);
if acc.rate < 0.05
    warning('SINDy:LowAcceptance', ['Acceptance rate %.3f: the truncated ensemble ' ...
        'differs strongly from the GP posterior.'], acc.rate);
end

% Pairing diagnostic: trapezoid integral of each derivative draw vs its own
% state draw, and vs the next draw's state (deliberately mismatched).
pair_err = zeros(nSamples, 2);
pair_err_mismatch = zeros(nSamples, 2);
for j = 1:nSamples
    jn = mod(j, nSamples) + 1;
    for s = 1:2
        x = Xsamples(:, s, j);
        recon = x(1) + cumtrapz(tGrid, dXsamples(:, s, j));
        pair_err(j, s) = max(abs(recon - x)) / max(max(x) - min(x), eps);
        xn = Xsamples(:, s, jn);
        recon_n = xn(1) + cumtrapz(tGrid, dXsamples(:, s, j));
        pair_err_mismatch(j, s) = max(abs(recon_n - xn)) / max(max(xn) - min(xn), eps);
    end
end
fprintf(['Pairing check (max |x(1) + int dx - x| / range x): matched median %.3f, ' ...
    'mismatched median %.3f\n'], median(pair_err(:)), median(pair_err_mismatch(:)));

%% Per-trajectory STLS fits
% No pooling or averaging across draws. A failed fit stays NaN.
XiSamples = nan(nLib, 2, nSamples);
fit_ok = false(nSamples, 1);
fit_msg = repmat({''}, nSamples, 1);
condRaw = nan(nSamples, 1);
condStd = nan(nSamples, 1);
rankRaw = nan(nSamples, 1);
rankStd = nan(nSamples, 1);
cosSum = zeros(nLib);
vminAbsSum = zeros(nLib, 1);
ws_fit = warning;
warning('off', 'MATLAB:rankDeficientMatrix');
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:singularMatrix');
for j = 1:nSamples
    Th = sindy_library(Xsamples(:, 1, j), Xsamples(:, 2, j));
    sv = svd(Th);
    rankRaw(j) = sum(sv > max(size(Th)) * eps(sv(1)));
    condRaw(j) = sv(1) / sv(end);
    Thn = Th ./ max(sqrt(sum(Th.^2, 1)), realmin);
    [~, Sn, Vn] = svd(Thn, 'econ');
    svn = diag(Sn);
    rankStd(j) = sum(svn > max(size(Thn)) * eps(svn(1)));
    condStd(j) = svn(1) / svn(end);
    cosSum = cosSum + abs(Thn' * Thn);
    vminAbsSum = vminAbsSum + abs(Vn(:, end));
    try
        Xi_j = stls(Th, dXsamples(:, :, j), ens_cfg.lambda);
        if any(~isfinite(Xi_j(:)))
            fit_msg{j} = 'non-finite coefficients';
        else
            XiSamples(:, :, j) = Xi_j;
            fit_ok(j) = true;
        end
    catch ME
        fit_msg{j} = ME.message;
    end
end
warning(ws_fit);
valid = find(fit_ok);
nValid = numel(valid);
fprintf('STLS fits: %d of %d succeeded (lambda = %g).\n', nValid, nSamples, ens_cfg.lambda);
if nValid < nSamples
    fprintf('  Failures: %s\n', strjoin(unique(fit_msg(~fit_ok))', '; '));
end
if nValid == 0
    error('SINDy:EnsembleFits', 'No ensemble STLS fit succeeded.');
end

%% A-B. Summary statistics of the coefficient ensemble
CvAll = XiSamples(:, :, valid);
selFreq = zeros(nLib, 2);
statsU = zeros(nLib, 5, 2);
statsC = zeros(nLib, 5, 2);
eqn_names = {'dM1/dt', 'dM2/dt'};
for s = 1:2
    C = reshape(CvAll(:, s, :), nLib, nValid);
    [statsU(:, :, s), statsC(:, :, s), selFreq(:, s)] = coefficient_stats(C, ens_cfg.sel_tol);
    print_coef_table(sprintf('%s: %d valid fits; unconditional (zeros included) | conditional on selection', ...
        eqn_names{s}, nValid), lib_names, selFreq(:, s), statsU(:, :, s), statsC(:, :, s));
end
XiMed = median(CvAll, 3);
XiMean = mean(CvAll, 3);
XiSd = std(CvAll, 0, 3);

% Most frequent sparsity patterns across both equations
selPat = reshape(abs(CvAll) > ens_cfg.sel_tol, 2 * nLib, nValid)';
[patU, ~, patIdx] = unique(selPat, 'rows');
patCount = accumarray(patIdx, 1);
[patCount, patOrder] = sort(patCount, 'descend');
patU = patU(patOrder, :);
fprintf('\n%d distinct sparsity patterns. Most frequent:\n', size(patU, 1));
for p = 1:min(5, size(patU, 1))
    selM1 = patU(p, 1:nLib);
    selM2 = patU(p, nLib+1:end);
    namesM1 = lib_names(selM1);
    namesM2 = lib_names(selM2);
    fprintf('  %5.1f%%  dM1/dt: {%s}  dM2/dt: {%s}\n', 100 * patCount(p) / nValid, ...
        strjoin(namesM1(:)', ', '), strjoin(namesM2(:)', ', '));
end

%% C. Library conditioning
lc = log10(condRaw);
lcs = log10(condStd);
fprintf('\nLibrary conditioning over %d draws:\n', nSamples);
fprintf('  log10 cond (raw):       median %.2f, 95%% range [%.2f, %.2f]\n', ...
    median(lc), pct_mid(lc, 2.5), pct_mid(lc, 97.5));
fprintf('  log10 cond (unit-norm): median %.2f, 95%% range [%.2f, %.2f]\n', ...
    median(lcs), pct_mid(lcs, 2.5), pct_mid(lcs, 97.5));
fprintf('  numerically rank deficient (raw): %d | (unit-norm): %d of %d (full rank = %d)\n', ...
    nnz(rankRaw < nLib), nnz(rankStd < nLib), nSamples, nLib);
cosMean = cosSum / nSamples;
[iP, jP] = find(triu(cosMean, 1) > 0.995);
if isempty(iP)
    fprintf('  No column pair with mean |cosine| > 0.995.\n');
else
    fprintf('  Nearly collinear column pairs (mean |cosine| > 0.995):\n');
    for p = 1:numel(iP)
        fprintf('    %-24s ~ %-24s  %.5f\n', lib_names{iP(p)}, lib_names{jP(p)}, cosMean(iP(p), jP(p)));
    end
end
vminMean = vminAbsSum / nSamples;
[vSorted, vOrder] = sort(vminMean, 'descend');
fprintf('  Terms loading on the weakest direction (mean |v_min|, unit-norm library):\n');
for p = 1:min(5, nLib)
    fprintf('    %-24s %.3f\n', lib_names{vOrder(p)}, vSorted(p));
end

% Does coefficient instability track conditioning? Deviation of each fit from
% the ensemble median in units of the ensemble sd, RMS over varying entries.
sdMask = XiSd > 0;
coefDev = nan(nSamples, 1);
for j = valid'
    z = (XiSamples(:, :, j) - XiMed) ./ XiSd;
    coefDev(j) = sqrt(mean(z(sdMask).^2));
end
if any(sdMask(:)) && nValid > 2
    rc = corrcoef(lcs(valid), log10(coefDev(valid) + eps));
    hiCond = lcs(valid) >= pct_mid(lcs(valid), 90);
    fprintf(['  corr(log10 cond, log10 coef deviation) = %.3f | median deviation: ' ...
        'top 10%% cond %.3f vs rest %.3f\n'], rc(1, 2), ...
        median(coefDev(valid(hiCond))), median(coefDev(valid(~hiCond))));
end

%% D. Forward simulation of ensemble models on the data domain
% Each model starts from its own sampled state at t = tGrid(1). The median
% model starts from the GP posterior mean there.
fwd_status = repmat({'not run'}, nSamples, 1);
fwd_neg = false(nSamples, 1);
fwd_sol = nan(nGrid, 2, nSamples);
ws_ode = warning;
warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');
for j = valid'
    simj = simulate_sindy(XiSamples(:, :, j), Xsamples(1, :, j)', tGrid, ens_cfg.blowup);
    fwd_status{j} = simj.status;
    fwd_neg(j) = simj.negative;
    if strcmp(simj.status, 'ok')
        fwd_sol(:, :, j) = simj.y;
    end
end
x0Med = [postM1.mu_f(1); postM2.mu_f(1)];
simMed = simulate_sindy(XiMed, x0Med, tGrid, ens_cfg.blowup);
warning(ws_ode);
fwdOK = strcmp(fwd_status, 'ok');
fwdFeas = fwdOK & ~fwd_neg;
fprintf(['\nForward simulations (%d valid models): ok %d, diverged %d, failed %d; ' ...
    'went negative %d; ok and nonnegative %d\n'], nValid, nnz(fwdOK), ...
    nnz(strcmp(fwd_status, 'diverged')), nnz(strcmp(fwd_status, 'failed')), ...
    nnz(fwd_neg), nnz(fwdFeas));
medMsg = '';
if ~isempty(simMed.msg)
    medMsg = sprintf(' (%s)', simMed.msg);
end
fprintf('Median model: status %s%s, negative %d\n', simMed.status, medMsg, simMed.negative);
fprintf('  dM1/dt = %s\n', format_sindy_eq(XiMed(:, 1), lib_names));
fprintf('  dM2/dt = %s\n', format_sindy_eq(XiMed(:, 2), lib_names));
fprintf(['  The coefficient-wise median model is not itself a fitted model ' ...
    'and may combine terms that never co-occur.\n']);

% Plot choices (drawn after sampling, so the seed fixes them too)
idxShow = sort(randperm(nSamples, min(ens_cfg.nShow, nSamples)));
% Individual models in Figure 306 come from the same ok and nonnegative set as
% the grey envelope.
feasIdx = find(fwdFeas);
idxFwd = feasIdx(sort(randperm(numel(feasIdx), min(ens_cfg.nFwdModels, numel(feasIdx)))));

%% Figures 301-306
% 301: GP trajectory samples (Figure A)
fig301 = figure(301);
set(fig301, 'Color', 'w', 'Name', 'Ensemble: GP state samples', 'WindowStyle', 'docked');
tl301 = tiledlayout(fig301, 1, 2, 'TileSpacing', 'compact');
plot_sample_panel(nexttile(tl301), tgrid, muM1_grid, sfM1_grid, k_plot, tGrid, ...
    reshape(Xsamples(:, 1, idxShow), nGrid, []), timeM1, dataM1, 'k', ...
    sprintf('M1: %d of %d draws', numel(idxShow), nSamples), 'cells/mm^2', xlE);
plot_sample_panel(nexttile(tl301), tgrid, muM2_grid, sfM2_grid, k_plot, tGrid, ...
    reshape(Xsamples(:, 2, idxShow), nGrid, []), timeM2, dataM2, 'r', ...
    sprintf('M2: %d of %d draws', numel(idxShow), nSamples), 'cells/mm^2', xlE);

% 302: paired derivative samples, same draws as 301 (Figure B)
fig302 = figure(302);
set(fig302, 'Color', 'w', 'Name', 'Ensemble: GP derivative samples', 'WindowStyle', 'docked');
tl302 = tiledlayout(fig302, 1, 2, 'TileSpacing', 'compact');
plot_sample_panel(nexttile(tl302), tgrid50, dmuM1_grid, dsfM1_grid, k_plot, tGrid, ...
    reshape(dXsamples(:, 1, idxShow), nGrid, []), [], [], 'k', ...
    'dM1/dt (paired with the draws in Figure 301)', 'cells/mm^2/day', xlE);
plot_sample_panel(nexttile(tl302), tgrid50, dmuM2_grid, dsfM2_grid, k_plot, tGrid, ...
    reshape(dXsamples(:, 2, idxShow), nGrid, []), [], [], 'r', ...
    'dM2/dt (paired with the draws in Figure 301)', 'cells/mm^2/day', xlE);

% 303: term selection frequency
fig303 = figure(303);
set(fig303, 'Color', 'w', 'Name', 'Ensemble: term selection frequency', 'WindowStyle', 'docked');
hb = bar(selFreq);
hb(1).FaceColor = [0.2 0.2 0.2];
hb(2).FaceColor = [0.85 0.2 0.2];
set(gca, 'XTick', 1:nLib, 'XTickLabel', lib_names, 'XTickLabelRotation', 45, ...
    'TickLabelInterpreter', 'none', 'FontSize', 14);
ylim([0 1]);
ylabel('Selection frequency');
legend({'dM1/dt', 'dM2/dt'}, 'Location', 'northeast');
title(sprintf('STLS \\lambda = %g, %d valid fits', ens_cfg.lambda, nValid));
box on

% 307: distribution of each coefficient over the valid fits that selected it.
% Unselected fits (exactly 0) are left out so they do not swamp the histogram;
% the title gives P(sel) = selection frequency. One tile per coefficient
% because the terms differ by orders of magnitude.
% Rows 1-2: dM1/dt, rows 3-4: dM2/dt. Dashed line: median when selected.
fig307 = figure(307);
set(fig307, 'Color', 'w', 'Name', 'Ensemble: coefficient distributions', 'WindowStyle', 'docked');
tl307 = tiledlayout(fig307, 4, ceil(nLib / 2), 'TileSpacing', 'compact', 'Padding', 'compact');
histCols = {[0.2 0.2 0.2], [0.85 0.2 0.2]};
for s = 1:2
    for k = 1:nLib
        ax = nexttile(tl307);
        c = reshape(CvAll(k, s, :), [], 1);
        c = c(abs(c) > ens_cfg.sel_tol);
        if isempty(c)
            text(ax, 0.5, 0.5, 'never selected', 'Units', 'normalized', ...
                'HorizontalAlignment', 'center', 'FontSize', 9);
            set(ax, 'XTick', [], 'YTick', []);
        else
            histogram(ax, c, 30, 'FaceColor', histCols{s}, 'EdgeColor', 'none', ...
                'FaceAlpha', 0.7);
            xline(ax, 0, ':');
            xline(ax, median(c), '--', 'Color', histCols{s}, 'LineWidth', 1.5);
            set(ax, 'FontSize', 9);
        end
        title(ax, sprintf('%s: %s (P = %.2f)', eqn_names{s}, lib_names{k}, selFreq(k, s)), ...
            'Interpreter', 'none', 'FontSize', 9, 'FontWeight', 'normal');
        box(ax, 'on');
    end
end
title(tl307, sprintf('Coefficients over %d valid fits when selected (STLS \\lambda = %g)', ...
    nValid, ens_cfg.lambda));

% 304: coefficient distributions conditional on selection
fig304 = figure(304);
set(fig304, 'Color', 'w', 'Name', 'Ensemble: coefficient distributions', 'Position', [60 60 1400 520]);
tl304 = tiledlayout(fig304, 1, 2, 'TileSpacing', 'compact');
cols = {'k', 'r'};
for s = 1:2
    ax = nexttile(tl304);
    hold(ax, 'on');
    for k = 1:nLib
        if selFreq(k, s) > 0
            plot(ax, [k k], statsC(k, 4:5, s), '-', 'Color', cols{s}, 'LineWidth', 2.5);
            plot(ax, k, statsC(k, 2, s), 'o', 'Color', cols{s}, 'MarkerFaceColor', cols{s});
        end
    end
    yline(ax, 0, ':');
    hold(ax, 'off');
    set(ax, 'XTick', 1:nLib, 'XTickLabel', lib_names, 'XTickLabelRotation', 45, ...
        'TickLabelInterpreter', 'none', 'FontSize', 12, 'YScale', 'linear');
    xlim(ax, [0.5, nLib + 0.5]);
    ylabel(ax, 'Coefficient');
    title(ax, sprintf('%s: median and 95%% range when selected', eqn_names{s}));
    box(ax, 'on');
end

% 305: conditioning (Figure E)
fig305 = figure(305);
set(fig305, 'Color', 'w', 'Name', 'Ensemble: library conditioning', 'Position', [60 60 1300 500]);
tl305 = tiledlayout(fig305, 1, 2, 'TileSpacing', 'compact');
ax = nexttile(tl305);
histogram(ax, lc, 30, 'FaceColor', [0.5 0.5 0.5]);
hold(ax, 'on');
histogram(ax, lcs, 30, 'FaceColor', [0 0.45 0.75]);
hold(ax, 'off');
legend(ax, {'raw library', 'unit-norm columns'});
xlabel(ax, 'log_{10} condition number');
ylabel(ax, 'Draws');
set(ax, 'FontSize', 14);
ax = nexttile(tl305);
scatter(ax, lcs(valid), coefDev(valid) + eps, 15, 'filled', 'MarkerFaceAlpha', 0.5);
set(ax, 'YScale', 'log', 'FontSize', 14);
xlabel(ax, 'log_{10} cond (unit-norm library)');
ylabel(ax, 'RMS coefficient z-deviation from median');
box(ax, 'on');

% 306: forward simulations (Figure F)
fig306 = figure(306);
set(fig306, 'Color', 'w', 'Name', 'Ensemble: forward simulations', 'WindowStyle', 'docked');
tl306 = tiledlayout(fig306, 1, 2, 'TileSpacing', 'compact');
fwdCols = lines(numel(idxFwd));
gpMu = {muM1_grid, muM2_grid};
gpSd = {sfM1_grid, sfM2_grid};
tObsC = {timeM1, timeM2};
yObsC = {dataM1, dataM2};
for s = 1:2
    ax = nexttile(tl306);
    hold(ax, 'on');
    fill(ax, [tgrid; flipud(tgrid)], [gpMu{s} + k_plot * gpSd{s}; flipud(gpMu{s} - k_plot * gpSd{s})], ...
        cols{s}, 'FaceAlpha', 0.12, 'EdgeColor', 'none');
    if nnz(fwdFeas) > 1
        env = zeros(nGrid, 2);
        for i = 1:nGrid
            env(i, :) = pct_mid(reshape(fwd_sol(i, s, fwdFeas), [], 1), [2.5, 97.5]);
        end
        fill(ax, [tGrid; flipud(tGrid)], [env(:, 2); flipud(env(:, 1))], [0.3 0.3 0.3], ...
            'FaceAlpha', 0.18, 'EdgeColor', 'none');
    end
    for p = 1:numel(idxFwd)
        j = idxFwd(p);
        if ~isempty(fwd_sol(:, s, j)) && all(isfinite(fwd_sol(:, s, j)))
            plot(ax, tGrid, fwd_sol(:, s, j), '-', 'Color', fwdCols(p, :), 'LineWidth', 1.2);
        end
    end
    if ~isempty(simMed.y)
        plot(ax, simMed.t, simMed.y(:, s), '--', 'Color', cols{s}, 'LineWidth', 2.5);
    end
    scatter(ax, tObsC{s}, yObsC{s}, 40, cols{s}, 'filled');
    hold(ax, 'off');
    xlim(ax, xlE);
    xlabel(ax, 'Time (Days)');
    ylabel(ax, 'cells/mm^2');
    set(ax, 'FontSize', 12);
    box(ax, 'on');
end

% 308/309: the individual model drawn as the second thin curve in Figure 306
% (idxFwd(2), the orange-red line), started from its own sampled day-0 state
% as in Figure 306.
pickFwd = 6;
pick = struct('pos', pickFwd, 'idx', [], 'Xi', [], 'x0', [], 'sim14', [], 'sim50', []);
if numel(idxFwd) < pickFwd
    warning('SINDy:PickModel', 'Fewer than %d individual models; figures 308-309 skipped.', pickFwd);
else
    pick.idx = idxFwd(pickFwd);
    pick.Xi = XiSamples(:, :, pick.idx);
    pick.x0 = Xsamples(1, :, pick.idx)';
    ws_pk = warning;
    warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');
    pick.sim14 = simulate_sindy(pick.Xi, pick.x0, (0:0.1:14)', ens_cfg.blowup);
    pick.sim50 = simulate_sindy(pick.Xi, pick.x0, tgrid50, ens_cfg.blowup);
    warning(ws_pk);
    fprintf(['\nFigure 306 individual model %d (draw %d), figures 308-309. ' ...
        'x0 = sampled day-0 state = [%.4g, %.4g]\n'], ...
        pickFwd, pick.idx, pick.x0);
    fprintf('  dM1/dt = %s\n', format_sindy_eq(pick.Xi(:, 1), lib_names));
    fprintf('  dM2/dt = %s\n', format_sindy_eq(pick.Xi(:, 2), lib_names));
    fprintf('  day 0-14: %s %s | day 0-50: %s %s | days 0-14 ok and nonnegative: %d\n', ...
        pick.sim14.status, pick.sim14.msg, pick.sim50.status, pick.sim50.msg, fwdFeas(pick.idx));

    fig308 = figure(308);
    set(fig308, 'Color', 'w', 'Name', sprintf('Ensemble: model from draw %d, days 0-14', pick.idx), ...
        'WindowStyle', 'docked');
    ax = axes(fig308);
    plot_ode_with_data(ax, pick.sim14, timeM1, dataM1, timeM2, dataM2, [0, 14]);

    % Same model to day 50, with the held-out day 35 points (filled diamonds).
    fig309 = figure(309);
    set(fig309, 'Color', 'w', 'Name', sprintf('Ensemble: model from draw %d, days 0-50', pick.idx), ...
        'WindowStyle', 'docked');
    ax = axes(fig309);
    plot_ode_with_data(ax, pick.sim50, timeM1, dataM1, timeM2, dataM2, [0, 50]);
    hold(ax, 'on');
    scatter(ax, timeM1_hold, dataM1_hold, 60, 'k', 'd', 'filled');
    scatter(ax, timeM2_hold, dataM2_hold, 60, 'r', 'd', 'filled');
    hold(ax, 'off');
end

%% E. Subset refit on the frequently selected terms
% Keep term k in equation s if selFreq(k, s) > subset_pThresh. Every valid
% draw is refit by ordinary least squares on the kept terms only, with no
% thresholding, so every kept term is present in every refit and the refit
% coefficient distributions contain no structural zeros. The mean and the
% median of each refit coefficient are plugged into the ODE.
% Coefficient-wise means and medians ignore the correlation between
% coefficients, so neither model is guaranteed to be close to any single
% refit; the refit ensemble (envelope and median trajectory) is the reference.
fprintf('\n=== Subset refit (terms selected in more than %.0f%% of fits) ===\n', ...
    100 * ens_cfg.subset_pThresh);
keep = selFreq > ens_cfg.subset_pThresh;
for s = 1:2
    fprintf('  %s terms: {%s}\n', eqn_names{s}, strjoin(lib_names(keep(:, s)), ', '));
end
if ~all(any(keep, 1))
    error('SINDy:Subset', 'No term passes the selection threshold in at least one equation.');
end

XiSub = nan(nLib, 2, nValid);
condSub = nan(nValid, 2);
ws_sub = warning;
warning('off', 'MATLAB:rankDeficientMatrix');
warning('off', 'MATLAB:nearlySingularMatrix');
warning('off', 'MATLAB:singularMatrix');
for i = 1:nValid
    j = valid(i);
    Th = sindy_library(Xsamples(:, 1, j), Xsamples(:, 2, j));
    for s = 1:2
        A = Th(:, keep(:, s));
        xi = zeros(nLib, 1);
        xi(keep(:, s)) = A \ dXsamples(:, s, j);
        XiSub(:, s, i) = xi;
        condSub(i, s) = cond(A ./ max(sqrt(sum(A.^2, 1)), realmin));
    end
end
warning(ws_sub);

XiSubMean = mean(XiSub, 3);
XiSubMed = median(XiSub, 3);
statsSub = nan(nLib, 5, 2);
corrSub = cell(1, 2);
for s = 1:2
    C = reshape(XiSub(:, s, :), nLib, nValid);
    statsSub(:, :, s) = coefficient_stats(C, ens_cfg.sel_tol);
    ks = find(keep(:, s))';
    fprintf(['\n--- %s: least-squares refit over %d draws ' ...
        '(log10 cond, unit-norm columns: median %.2f, max %.2f) ---\n'], ...
        eqn_names{s}, nValid, median(log10(condSub(:, s))), max(log10(condSub(:, s))));
    fprintf('%-12s %6s | %11s %11s %11s %11s %11s\n', ...
        'term', 'P(sel)', 'mean', 'median', 'std', 'p2.5', 'p97.5');
    for k = ks
        fprintf('%-12s %6.3f | %11.4g %11.4g %11.4g %11.4g %11.4g\n', ...
            lib_names{k}, selFreq(k, s), statsSub(k, :, s));
    end
    if numel(ks) > 1
        corrSub{s} = corrcoef(C(ks, :)');
        fprintf('  Correlation between refit coefficients:\n');
        fprintf('  %12s', ''); fprintf(' %9s', lib_names{ks}); fprintf('\n');
        for p = 1:numel(ks)
            fprintf('  %12s', lib_names{ks(p)}); fprintf(' %9.3f', corrSub{s}(p, :)); fprintf('\n');
        end
    end
end

% Mean and median models, started from the GP posterior mean at day 0
ws_ode = warning;
warning('off', 'MATLAB:ode15s:IntegrationTolNotMet');
t14 = (0:0.1:14)';
sub = struct();
sub.x0 = x0Med;
sub.simMean14 = simulate_sindy(XiSubMean, sub.x0, t14, ens_cfg.blowup);
sub.simMed14 = simulate_sindy(XiSubMed, sub.x0, t14, ens_cfg.blowup);
sub.simMean50 = simulate_sindy(XiSubMean, sub.x0, tgrid50, ens_cfg.blowup);
sub.simMed50 = simulate_sindy(XiSubMed, sub.x0, tgrid50, ens_cfg.blowup);

% Reference: every refit model from its own sampled day-0 state
x0Sub = reshape(Xsamples(1, :, valid), 2, nValid);
[sub.sol14, sub.status14, sub.neg14] = simulate_ensemble(XiSub, x0Sub, t14, ens_cfg.blowup);
[sub.sol50, sub.status50, sub.neg50] = simulate_ensemble(XiSub, x0Sub, tgrid50, ens_cfg.blowup);
warning(ws_ode);
sub.feas14 = strcmp(sub.status14, 'ok') & ~sub.neg14;
sub.feas50 = strcmp(sub.status50, 'ok') & ~sub.neg50;
sub.env14 = traj_envelope(sub.sol14, sub.feas14);
sub.env50 = traj_envelope(sub.sol50, sub.feas50);

% Errors: RMSE against the days 0-14 data, and model minus data at day 35
tHold = [timeM1_hold, timeM2_hold];
yHold = [dataM1_hold, dataM2_hold];
sub.errMean = sim_errors(sub.simMean50, tObsC, yObsC, tHold, yHold);
sub.errMed = sim_errors(sub.simMed50, tObsC, yObsC, tHold, yHold);
sub.rmseEns = nan(nValid, 2);
for i = find(sub.feas14)'
    for s = 1:2
        yhat = interp1(t14, sub.sol14(:, s, i), tObsC{s});
        sub.rmseEns(i, s) = sqrt(mean((yhat - yObsC{s}).^2));
    end
end

fprintf(['\nRefit ensemble forward simulations (%d models): days 0-14 ok and ' ...
    'nonnegative %d | days 0-50 ok and nonnegative %d\n'], nValid, nnz(sub.feas14), nnz(sub.feas50));
modelNames = {'Mean', 'Median'};
modelXi = {XiSubMean, XiSubMed};
modelSims = {sub.simMean50, sub.simMed50};
modelErr = {sub.errMean, sub.errMed};
for m = 1:2
    sm = modelSims{m};
    fprintf('%s-coefficient model: days 0-50 status %s %s, negative %d\n', ...
        modelNames{m}, sm.status, sm.msg, sm.negative);
    fprintf('  dM1/dt = %s\n', format_sindy_eq(modelXi{m}(:, 1), lib_names));
    fprintf('  dM2/dt = %s\n', format_sindy_eq(modelXi{m}(:, 2), lib_names));
    for s = 1:2
        r = sub.rmseEns(:, s);
        r = r(isfinite(r));
        fprintf(['  M%d: RMSE vs data %.4g (refit ensemble median %.4g; %.0f%% of ' ...
            'refits fit better) | day 35 error %+.4g\n'], s, modelErr{m}.rmse(s), ...
            median(r), 100 * mean(r < modelErr{m}.rmse(s)), modelErr{m}.hold(s));
    end
end

% 310/311: mean and median models against the refit ensemble, days 0-14 and 0-50
subFigs = {310, 311};
subT = {t14, tgrid50};
subEnv = {sub.env14, sub.env50};
subMean = {sub.simMean14, sub.simMean50};
subMed = {sub.simMed14, sub.simMed50};
subXl = {[0, 14], [0, 50]};
for f = 1:2
    figS = figure(subFigs{f});
    set(figS, 'Color', 'w', 'WindowStyle', 'docked', 'Name', ...
        sprintf('Subset refit: mean and median models, days %d-%d', subXl{f}));
    tlS = tiledlayout(figS, 1, 2, 'TileSpacing', 'compact');
    for s = 1:2
        ax = nexttile(tlS);
        plot_subset_fit(ax, s, subT{f}, subEnv{f}, subMean{f}, subMed{f}, ...
            tObsC{s}, yObsC{s}, cols{s}, subXl{f});
        if f == 2
            hold(ax, 'on');
            scatter(ax, tHold(s), yHold(s), 60, cols{s}, 'd', 'filled', ...
                'DisplayName', 'Day 35 (held out)');
            hold(ax, 'off');
        end
        title(ax, sprintf('M%d', s));
    end
end

% 312: distributions of the refit coefficients with their mean and median
fig312 = figure(312);
set(fig312, 'Color', 'w', 'Name', 'Subset refit: coefficient distributions', 'WindowStyle', 'docked');
nKeepMax = max(sum(keep, 1));
tl312 = tiledlayout(fig312, 2, nKeepMax, 'TileSpacing', 'compact', 'Padding', 'compact');
for s = 1:2
    ks = find(keep(:, s))';
    for p = 1:nKeepMax
        ax = nexttile(tl312);
        if p > numel(ks)
            axis(ax, 'off');
            continue
        end
        k = ks(p);
        c = reshape(XiSub(k, s, :), [], 1);
        histogram(ax, c, 30, 'FaceColor', histCols{s}, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
        xline(ax, 0, ':');
        xline(ax, XiSubMean(k, s), '-', 'Color', cols{s}, 'LineWidth', 1.5);
        xline(ax, XiSubMed(k, s), '--', 'Color', cols{s}, 'LineWidth', 1.5);
        set(ax, 'FontSize', 9);
        title(ax, sprintf('%s: %s', eqn_names{s}, lib_names{k}), ...
            'Interpreter', 'none', 'FontSize', 9, 'FontWeight', 'normal');
        box(ax, 'on');
    end
end
title(tl312, sprintf(['Least-squares refit on terms with P(sel) > %.2f, %d draws ' ...
    '(solid: mean, dashed: median)'], ens_cfg.subset_pThresh, nValid));

%% Validation
fprintf('\n--- Ensemble validation ---\n');
check_true(acc.accepted == nSamples && size(Xsamples, 3) == nSamples, ...
    sprintf('exactly %d accepted trajectory pairs', nSamples));
check_true(all(Xsamples(:) >= 0), 'all sampled states nonnegative on the grid');
check_true(isequal(size(Xsamples), [nGrid, 2, nSamples]) && ...
    isequal(size(dXsamples), [nGrid, 2, nSamples]), 'state and derivative array sizes');
check_true(isequal(size(XiSamples), [nLib, 2, nSamples]), 'coefficient array size');
check_true(median(pair_err(:)) < 0.5 * median(pair_err_mismatch(:)), ...
    'each derivative draw integrates to its own state draw');
check_true(nValid + nnz(~fit_ok) == nSamples && all(isnan(reshape(XiSamples(:, :, ~fit_ok), [], 1))), ...
    'one STLS fit per trajectory; failures stay NaN');
selCheck = squeeze(mean(abs(CvAll) > ens_cfg.sel_tol, 3));
check_true(max(abs(selCheck(:) - selFreq(:))) < 1e-12, 'selection frequencies');
medCheck = zeros(nLib, 2);
for s = 1:2
    medCheck(:, s) = statsU(:, 2, s);
end
check_true(max(abs(medCheck(:) - XiMed(:))) <= 1e-12 * max(1, max(abs(XiMed(:)))), ...
    'coefficient medians');
rng_after = rng;
rng(ens_cfg.seed);
[Xrep, dXrep] = draw_nonneg_pairs(postM1, SM1, postM2, SM2, 3, ens_cfg.maxDraws);
rng(rng_after);
check_true(isequal(Xrep, Xsamples(:, :, 1:3)) && isequal(dXrep, dXsamples(:, :, 1:3)), ...
    'reproducible with the fixed seed');
check_true(all(reshape(XiSub(repmat(~keep, 1, 1, nValid)), [], 1) == 0) && ...
    all(isfinite(XiSub(:))), 'subset refit: dropped terms exactly 0, kept terms finite');
check_true(all(XiSubMean(~keep) == 0) && all(XiSubMed(~keep) == 0), ...
    'subset mean and median models use only the kept terms');

%% Save ensemble results
ens = struct();
ens.cfg = ens_cfg;
ens.seed = ens_cfg.seed;
ens.tGrid = tGrid;
ens.Xsamples = Xsamples;
ens.dXsamples = dXsamples;
ens.XiSamples = XiSamples;
ens.lib_names = lib_names;
ens.lambda = ens_cfg.lambda;
ens.regression = sprintf('STLS (stls in SINDy.m), lambda = %g, 100 iterations, raw library', ens_cfg.lambda);
ens.hypM1 = fitM1.hyp;
ens.hypM2 = fitM2.hyp;
ens.noiseM1 = noiseM1;
ens.noiseM2 = noiseM2;
ens.kernels = {'covSEiso (M1)', 'covMaterniso 5 (M2)'};
ens.timeM1 = timeM1; ens.dataM1 = dataM1;
ens.timeM2 = timeM2; ens.dataM2 = dataM2;
ens.postM1 = rmfield(postM1, 'Sigma');
ens.postM2 = rmfield(postM2, 'Sigma');
ens.jitter = struct('train', 1e-6, 'rel_M1', SM1.eps_rel, 'rel_M2', SM2.eps_rel, ...
    'min_eig_corr_M1', SM1.min_eig_corr, 'min_eig_corr_M2', SM2.min_eig_corr);
ens.kernel_fd_err = struct('M1', errKdM1, 'M2', errKdM2, 'pairs', fd_pairs);
ens.acceptance = acc;
ens.pair_err = pair_err;
ens.pair_err_mismatch = pair_err_mismatch;
ens.fit_ok = fit_ok;
ens.fit_msg = fit_msg;
ens.condRaw = condRaw;
ens.condStd = condStd;
ens.rankRaw = rankRaw;
ens.rankStd = rankStd;
ens.cosMean = cosMean;
ens.selFreq = selFreq;
ens.stats_cols = {'mean', 'median', 'std', 'p2.5', 'p97.5'};
ens.statsU = statsU;
ens.statsC = statsC;
ens.XiMed = XiMed;
ens.XiMean = XiMean;
ens.XiSd = XiSd;
ens.coefDev = coefDev;
ens.patterns = patU;
ens.pattern_counts = patCount;
ens.fwd_status = fwd_status;
ens.fwd_neg = fwd_neg;
ens.fwd_sol = fwd_sol;
ens.simMed = simMed;
ens.pick = pick;
ens.idxShow = idxShow;
ens.idxFwd = idxFwd;
sub.pThresh = ens_cfg.subset_pThresh;
sub.keep = keep;
sub.XiSub = XiSub;
sub.XiSubMean = XiSubMean;
sub.XiSubMed = XiSubMed;
sub.stats = statsSub;
sub.corr = corrSub;
sub.condSub = condSub;
sub.drawIdx = valid;
sub.regression = 'ordinary least squares on the kept terms, no thresholding';
ens.subset = sub;
if ens_cfg.save_results
    repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    ens_dir = fullfile(repo_root, 'results', 'SINDy_ensemble');
    if ~exist(ens_dir, 'dir')
        mkdir(ens_dir);
    end
    ens_file = fullfile(ens_dir, 'SINDy_ensemble_results.mat');
    save(ens_file, 'ens', '-v7');
    fprintf('Saved %s\n', ens_file);
end

%% Local helpers
function Xi = stls(Theta, X_dot, lambda)
% Sequential thresholded least squares. Theta is n x p, X_dot is n x 2.
Xi = Theta \ X_dot;
for k = 1:100
    smallinds = abs(Xi) < lambda;
    Xi(smallinds) = 0;
    for ind = 1:size(X_dot, 2)
        biginds = ~smallinds(:, ind);
        if any(biginds)
            Xi(biginds, ind) = Theta(:, biginds) \ X_dot(:, ind);
        end
    end
end
end

function rhs = SINDyfwd(~, inits, epsguess)
theta = sindy_library(inits(1), inits(2));
dM1 = theta * epsguess(:, 1);
dM2 = theta * epsguess(:, 2);
rhs = [dM1; dM2];
end

function Theta = sindy_library(M1, M2)
% Columns: 1, M1, M2, M1^2, M1*M2, M2^2, M1^3, M1^2*M2, M1*M2^2, M2^3.
M1 = M1(:);
M2 = M2(:);
Theta = [ones(numel(M1), 1), M1, M2, ...
    M1.^2, M1 .* M2, M2.^2, ...
    M1.^3, M1.^2 .* M2, M1 .* M2.^2, M2.^3];
end

function names = sindy_library_names()
names = {'1', 'M1', 'M2', ...
         'M1^2', 'M1*M2', 'M2^2', ...
         'M1^3', 'M1^2*M2', 'M1*M2^2', 'M2^3'};
end

%% GP helpers (same models as GP_for_Andrea.m)
function fit = fit_m1_day0(x, y, s2_day0, xs, meanfunc, covfunc, max_iters)
is_day0 = abs(x(:)) < 1e-12;
[hyp, mu, s2, nlml] = fit_day0_noise(x, y, xs, is_day0, s2_day0, ...
    meanfunc, covfunc, max_iters);
fit.hyp = hyp;
fit.nlml = nlml;
fit.mu = mu;
fit.s2 = s2;
fit.noise_var = day0_noise_vec(hyp, is_day0, s2_day0);
fprintf('M1 SE: NLML=%.4f | ell=%.4f | sf=%.4f | sn=%.4f | sn_day0=%.4f\n', ...
    nlml, exp(hyp.cov(1)), exp(hyp.cov(2)), exp(hyp.lik), sqrt(s2_day0));
end

function s2 = day0_sample_var(t, y)
yi = y(abs(t) < 1e-12);
if numel(yi) < 2
    error('SINDy:Day0Var', 'Day 0 needs at least two full-data points.');
end
s2 = var(yi, 0);
end

function [hyp, mu, s2, nlml] = fit_day0_noise(x, y, xs, is_day0, s2_day0, meanfunc, covfunc, max_iters)
% Zero-mean GP. Day-0 noise variance is fixed. One sn for every other time.
x = x(:); y = y(:); xs = xs(:); is_day0 = logical(is_day0(:));
ell0 = max(std(x), 0.5);
sf0 = max(std(y), 0.1);
sn0 = 0.1 * max(std(y(~is_day0)), 0.1);
hyp.mean = [];
hyp.cov = log([ell0; sf0]);
hyp.lik = log(sn0);
obj = @(h) nlml_day0_fixed(h, meanfunc, covfunc, x, y, is_day0, s2_day0);
hyp = minimize(hyp, obj, max_iters);
nlml = obj(hyp);
noise_var = day0_noise_vec(hyp, is_day0, s2_day0);
[mu, s2] = pred_fixed_noise(hyp, meanfunc, covfunc, x, y, noise_var, xs);
end

function noise = day0_noise_vec(hyp, is_day0, s2_day0)
noise = exp(2 * hyp.lik) * ones(numel(is_day0), 1);
noise(is_day0) = s2_day0;
end

function [nlml, dnlml] = nlml_day0_fixed(hyp, meanfunc, covfunc, x, y, is_day0, s2_day0)
n = numel(x);
m = feval(meanfunc, hyp.mean, x);
[K, dK] = cov_eval(covfunc, hyp.cov, x);
sn2 = exp(2 * hyp.lik);
noise = day0_noise_vec(hyp, is_day0, s2_day0);
jitter = 1e-6;
L = chol(K + diag(noise + jitter), 'lower');
alpha = L' \ (L \ (y - m));
nlml = (y - m)' * alpha / 2 + sum(log(diag(L))) + n * log(2 * pi) / 2;
if nargout > 1
    Q = L' \ (L \ eye(n)) - alpha * alpha';
    dnlml = hyp;
    dnlml.mean = [];
    dnlml.cov = dK(Q) / 2;
    dnlml.lik = sum(diag(Q) .* sn2 .* ~is_day0);
end
end

function fit = fit_full_lognoise(x, y, xs, meanfunc, mean_noise, ...
    covfunc, inffunc, likfunc, max_iters)
x = x(:); y = y(:);
hyp_s = fit_homo_best(x, y, inffunc, meanfunc, covfunc, likfunc, max_iters);
[t_obs, log_s2, n_rep] = log_replicate_var(x, y);
noise_mean = mean(log_s2);
hyp_n = init_noise_hyp(log_s2);
for pass = 1:2
    ell = exp(hyp_s.cov(1));
    hyp_n = fit_noise_gp(t_obs, log_s2, n_rep, ell, noise_mean, mean_noise, covfunc, hyp_n, max_iters);
    noise_var = predict_noise_var(hyp_n, ell, noise_mean, mean_noise, covfunc, t_obs, log_s2, n_rep, x);
    [hyp_s, mu, s2, nlml] = fit_fixed_noise(x, y, xs, noise_var, ...
        meanfunc, covfunc, hyp_s, max_iters);
end
fit.hyp = hyp_s;
fit.nlml = nlml;
fit.mu = mu;
fit.s2 = s2;
fit.noise_var = noise_var;
fprintf('M2 Matern 5/2: NLML=%.4f | ell=%.4f | sf=%.4f | k=%.4f | log-noise mean=%.4f\n', ...
    nlml, exp(hyp_s.cov(1)), exp(hyp_s.cov(2)), 1 + exp(hyp_n.lik), noise_mean);
[tu, ~, ic] = unique(x);
for i = 1:numel(tu)
    nv = noise_var(find(ic == i, 1));
    fprintf('  t=%g (n=%d): sn=%.4f\n', tu(i), sum(ic == i), sqrt(nv));
end
end

function [t_obs, log_s2, n_rep] = log_replicate_var(t, y)
% Log sample variance at unique times with at least two replicates.
t = t(:); y = y(:);
t_all = unique(t);
t_obs = zeros(0, 1);
log_s2 = zeros(0, 1);
n_rep = zeros(0, 1);
for i = 1:numel(t_all)
    yi = y(abs(t - t_all(i)) < 1e-12);
    if numel(yi) >= 2
        s2 = max(var(yi, 0), 1e-8);
        t_obs(end+1, 1) = t_all(i); %#ok<AGROW>
        log_s2(end+1, 1) = log(s2); %#ok<AGROW>
        n_rep(end+1, 1) = numel(yi); %#ok<AGROW>
    end
end
end

function hyp = init_noise_hyp(log_s2)
% The constant mean is fixed outside this struct. Only amplitude and k are free.
hyp.mean = [];
hyp.cov = log(max(std(log_s2), 0.1));
hyp.lik = 0;   % k = 1 + exp(lik) starts at 2
end

function hyp = fit_noise_gp(t_obs, log_s2, n_rep, ell, noise_mean, mean_noise, covfunc, hyp, max_iters)
obj = @(h) nlml_noise_gp(h, ell, noise_mean, mean_noise, covfunc, t_obs, log_s2, n_rep);
hyp = minimize(hyp, obj, max_iters);
end

function noise_var = predict_noise_var(hyp, ell, noise_mean, mean_noise, covfunc, t_obs, log_s2, n_rep, t)
[mu_log, s2_log] = pred_noise_mean(hyp, ell, noise_mean, mean_noise, covfunc, t_obs, log_s2, n_rep, t);
% Replicate days use the posterior mean. Singletons use the lognormal mean.
is_obs = false(numel(t), 1);
for i = 1:numel(t_obs)
    is_obs = is_obs | abs(t(:) - t_obs(i)) < 1e-12;
end
log_var = mu_log;
log_var(~is_obs) = mu_log(~is_obs) + 0.5 * s2_log(~is_obs);
noise_var = max(exp(log_var), 1e-8);
end

function hyp = fit_homo_best(x, y, inffunc, meanfunc, covfunc, likfunc, max_iters)
% Several lengthscale starts. Matern 5/2 has a white-noise local minimum.
x = x(:); y = y(:);
starts = unique([max(std(x), 0.5); 0.5; 1; 2; 4]);
best = inf;
hyp = [];
for i = 1:numel(starts)
    hyp_i = optimize_homo(x, y, starts(i), inffunc, meanfunc, covfunc, likfunc, max_iters);
    nlml = gp(hyp_i, inffunc, meanfunc, covfunc, likfunc, x, y);
    if nlml < best
        best = nlml;
        hyp = hyp_i;
    end
end
end

function hyp = optimize_homo(x, y, ell0, inffunc, meanfunc, covfunc, likfunc, max_iters)
sf0 = max(std(y), 0.1);
sn0 = 0.1 * sf0;
if sn0 <= 0
    sn0 = 0.1;
end
hyp.mean = [];
hyp.cov = log([ell0; sf0]);
hyp.lik = log(sn0);
hyp = minimize(hyp, @gp, max_iters, inffunc, meanfunc, covfunc, likfunc, x, y);
end

function [hyp, mu, s2, nlml] = fit_fixed_noise(x, y, xs, noise_var, meanfunc, covfunc, hyp0, max_iters)
hyp.mean = [];
hyp.cov = hyp0.cov(:);
hyp.lik = [];
obj = @(h) nlml_fixed_noise(h, meanfunc, covfunc, x, y, noise_var);
hyp = minimize(hyp, obj, max_iters);
nlml = obj(hyp);
[mu, s2] = pred_fixed_noise(hyp, meanfunc, covfunc, x, y, noise_var, xs);
end

function [nlml, dnlml] = nlml_fixed_noise(hyp, meanfunc, covfunc, x, y, noise_var)
n = numel(x);
m = feval(meanfunc, hyp.mean, x);
[K, dK] = cov_eval(covfunc, hyp.cov, x);
jitter = 1e-6;
L = chol(K + diag(noise_var(:) + jitter), 'lower');
alpha = L' \ (L \ (y(:) - m));
nlml = (y(:) - m)' * alpha / 2 + sum(log(diag(L))) + n * log(2 * pi) / 2;
if nargout > 1
    Q = L' \ (L \ eye(n)) - alpha * alpha';
    dnlml = hyp;
    dnlml.mean = [];
    dnlml.cov = dK(Q) / 2;
    dnlml.lik = [];
end
end

function [fmu, fs2] = pred_fixed_noise(hyp, meanfunc, covfunc, x, y, noise_var, xs)
m = feval(meanfunc, hyp.mean, x);
K = cov_eval(covfunc, hyp.cov, x);
jitter = 1e-6;
L = chol(K + diag(noise_var(:) + jitter), 'lower');
alpha = L' \ (L \ (y(:) - m));
ms = feval(meanfunc, hyp.mean, xs(:));
Ks = cov_eval(covfunc, hyp.cov, x, xs(:));
kss = cov_eval(covfunc, hyp.cov, xs(:), 'diag');
fmu = ms(:) + Ks' * alpha;
V = L \ Ks;
fs2 = max(kss(:) - sum(V.^2, 1)', 0);
end

function [dmu, ds2] = deriv_fixed_noise(hyp, kernel, meanfunc, covfunc, x, y, noise_var, xs)
% Posterior of f'(xs) for a zero-mean GP:
%   dmu = K_{f',f} Ky^{-1} (y - m),  ds2 = diag(K_{f',f'} - K_{f',f} Ky^{-1} K_{f,f'}).
% kernel is 'se' (covSEiso) or 'matern5' (covMaterniso, 5); hyp.cov = log([ell; sf]).
x = x(:); y = y(:); xs = xs(:);
m = feval(meanfunc, hyp.mean, x);
K = cov_eval(covfunc, hyp.cov, x);
jitter = 1e-6;
L = chol(K + diag(noise_var(:) + jitter), 'lower');
alpha = L' \ (L \ (y - m));
ell = exp(hyp.cov(1));
sf2 = exp(2 * hyp.cov(2));
d = xs - x';
switch kernel
    case 'se'
        dKs = -sf2 * (d / ell^2) .* exp(-d.^2 / (2 * ell^2));
        kdd = sf2 / ell^2;
    case 'matern5'
        a = sqrt(5) * abs(d) / ell;
        dKs = -sf2 * (5 * d / (3 * ell^2)) .* (1 + a) .* exp(-a);
        kdd = 5 * sf2 / (3 * ell^2);
    otherwise
        error('SINDy:Kernel', 'Unknown kernel ''%s''.', kernel);
end
dmu = dKs * alpha;
if nargout > 1
    V = L \ dKs';
    ds2 = max(kdd - sum(V.^2, 1)', 0);
end
end

function v = noise_nugget(n_rep)
% Delta-method variance of log(sample variance) for n replicates.
v = 2 ./ (n_rep(:) - 1);
end

function [nlml, dnlml] = nlml_noise_gp(hyp, ell, noise_mean, mean_noise, covfunc, x, y, n_rep)
% ell_g = k*ell, k = 1+exp(hyp.lik). hyp.cov is log(sf) of the noise kernel.
% noise_mean is held fixed in meanConst; hyp.mean is empty.
k_mult = 1 + exp(hyp.lik);
hyp_cov = [log(ell * k_mult); hyp.cov(:)];
m = feval(mean_noise, noise_mean, x);
[K, dK] = cov_eval(covfunc, hyp_cov, x);
n = numel(x);
jitter = 1e-6;
L = chol(K + diag(noise_nugget(n_rep) + jitter), 'lower');
alpha = L' \ (L \ (y(:) - m));
nlml = (y(:) - m)' * alpha / 2 + sum(log(diag(L))) + n * log(2 * pi) / 2;
if nargout > 1
    Q = L' \ (L \ eye(n)) - alpha * alpha';
    dhyp = dK(Q) / 2;
    dnlml = hyp;
    dnlml.mean = [];
    dnlml.cov = dhyp(2) * ones(size(hyp.cov));
    dnlml.lik = dhyp(1) * (exp(hyp.lik) / k_mult) * ones(size(hyp.lik));
end
end

function [mu, s2] = pred_noise_mean(hyp, ell, noise_mean, mean_noise, covfunc, x, y, n_rep, xs)
k_mult = 1 + exp(hyp.lik);
hyp_cov = [log(ell * k_mult); hyp.cov(:)];
m = feval(mean_noise, noise_mean, x);
K = cov_eval(covfunc, hyp_cov, x);
n = numel(x);
jitter = 1e-6;
L = chol(K + diag(noise_nugget(n_rep) + jitter), 'lower');
alpha = L' \ (L \ (y(:) - m));
ms = feval(mean_noise, noise_mean, xs(:));
Ks = cov_eval(covfunc, hyp_cov, x, xs(:));
kss = cov_eval(covfunc, hyp_cov, xs(:), 'diag');
mu = ms(:) + Ks' * alpha;
V = L \ Ks;
s2 = max(kss(:) - sum(V.^2, 1)', 0);
end

function varargout = cov_eval(covfunc, hyp, varargin)
if iscell(covfunc)
    [varargout{1:nargout}] = feval(covfunc{:}, hyp, varargin{:});
else
    [varargout{1:nargout}] = feval(covfunc, hyp, varargin{:});
end
end

%% Joint state-derivative GP posterior and ensemble helpers
function [K, Kdf, Kfd, Kdd] = kernel_blocks(kernel, hyp_cov, ta, tb)
% Analytic covariance blocks of a stationary kernel k(t, t'), d = t - t'.
%   K   = k(ta, tb)
%   Kdf = dk/dt          = cov(f'(ta), f(tb))
%   Kfd = dk/dt'         = cov(f(ta), f'(tb))
%   Kdd = d2k/(dt dt')   = cov(f'(ta), f'(tb))
% kernel is 'se' (covSEiso) or 'matern5' (covMaterniso, 5); hyp_cov = log([ell; sf]),
% the GPML parameterization. Inputs are in days with no scaling.
ell = exp(hyp_cov(1));
sf2 = exp(2 * hyp_cov(2));
d = ta(:) - tb(:)';
switch kernel
    case 'se'
        e = exp(-d.^2 / (2 * ell^2));
        K = sf2 * e;
        Kdf = -sf2 * (d / ell^2) .* e;
        Kdd = sf2 * (1 / ell^2 - d.^2 / ell^4) .* e;
    case 'matern5'
        a = sqrt(5) * abs(d) / ell;
        e = exp(-a);
        K = sf2 * (1 + a + a.^2 / 3) .* e;
        Kdf = -sf2 * (5 * d / (3 * ell^2)) .* (1 + a) .* e;
        Kdd = sf2 * (5 / (3 * ell^2)) * (1 + a - a.^2) .* e;
    otherwise
        error('SINDy:Kernel', 'Unknown kernel ''%s''.', kernel);
end
Kfd = -Kdf;
end

function post = joint_state_deriv_posterior(hyp, kernel, meanfunc, covfunc, x, y, noise_var, xs)
% Posterior of [f(xs); f'(xs)] | y for a zero-mean GP with fixed per-row noise.
% Latent function only: no observation noise is added at xs.
% Time is in days, so f' is in cells/mm^2/day.
if ~strcmp(func2str(meanfunc), 'meanZero')
    error('SINDy:JointMean', 'The joint posterior assumes meanZero (zero mean derivative).');
end
x = x(:); y = y(:); xs = xs(:);
n = numel(xs);
Kxx = cov_eval(covfunc, hyp.cov, x);
jitter = 1e-6;   % same training jitter as pred_fixed_noise / deriv_fixed_noise
L = chol(Kxx + diag(noise_var(:) + jitter), 'lower');
alpha = L' \ (L \ y);
[Ksx, Kdx] = kernel_blocks(kernel, hyp.cov, xs, x);
[Kss, Kdf_ss, Kfd_ss, Kdd_ss] = kernel_blocks(kernel, hyp.cov, xs, xs);
C = [Ksx; Kdx];
P = [Kss, Kfd_ss; Kdf_ss, Kdd_ss];
mu = C * alpha;
V = L \ C';
Sigma = P - V' * V;
post.asym = norm(Sigma - Sigma', 'fro') / max(norm(Sigma, 'fro'), realmin);
Sigma = (Sigma + Sigma') / 2;
Kgpml = cov_eval(covfunc, hyp.cov, x, xs)';
post.kernel_gpml_err = max(abs(Ksx(:) - Kgpml(:))) / max(abs(Kgpml(:)));
post.t = xs;
post.mu = mu;
post.mu_f = mu(1:n);
post.mu_d = mu(n+1:end);
post.Sigma = Sigma;
end

function S = joint_sampler(Sigma, rel_jitters)
% Cholesky factor for drawing from N(0, Sigma). States and derivatives differ
% in scale, so the correlation matrix R = D^-1/2 Sigma D^-1/2 is factorized and
% any jitter is relative to each component's own variance:
%   Sigma + eps * diag(diag(Sigma)) = (D^1/2 Lr)(D^1/2 Lr)'.
% The smallest eps in rel_jitters that gives a positive-definite factor is used.
sd = sqrt(max(diag(Sigma), 0));
sd = max(sd, 1e-12 * max(sd));
R = Sigma ./ (sd * sd');
R = (R + R') / 2;
S.min_eig_corr = min(eig(R));
for eps_rel = rel_jitters(:)'
    [Lr, p] = chol(R + eps_rel * eye(size(R, 1)), 'lower');
    if p == 0
        S.L = sd .* Lr;
        S.eps_rel = eps_rel;
        S.sd = sd;
        return
    end
end
error('SINDy:JointChol', ['Joint posterior covariance is not positive definite even with ' ...
    'relative jitter %.1e (min correlation eigenvalue %.3e).'], max(rel_jitters), S.min_eig_corr);
end

function err = check_kernel_derivs(kernel, hyp_cov, covfunc, pairs)
% Analytic derivative blocks vs central differences of the GPML kernel.
% Errors are scaled by sf^2/ell (first derivatives) and sf^2/ell^2 (second).
ell = exp(hyp_cov(1));
sf2 = exp(2 * hyp_cov(2));
h = 1e-4 * ell;
k = @(a, b) cov_eval(covfunc, hyp_cov, a, b);
err = zeros(size(pairs, 1), 3);
for i = 1:size(pairs, 1)
    t = pairs(i, 1);
    tp = pairs(i, 2);
    [~, Kdf, Kfd, Kdd] = kernel_blocks(kernel, hyp_cov, t, tp);
    fd_df = (k(t + h, tp) - k(t - h, tp)) / (2 * h);
    fd_fd = (k(t, tp + h) - k(t, tp - h)) / (2 * h);
    fd_dd = (k(t + h, tp + h) - k(t + h, tp - h) - k(t - h, tp + h) + k(t - h, tp - h)) / (4 * h^2);
    err(i, :) = [abs(Kdf - fd_df) / (sf2 / ell), abs(Kfd - fd_fd) / (sf2 / ell), ...
        abs(Kdd - fd_dd) / (sf2 / ell^2)];
end
end

function report_joint_post(name, post, fmu, fs2, dmu, ds2, S)
n = numel(post.t);
v = diag(post.Sigma);
fprintf(['%s joint posterior: asymmetry %.2e | kernel vs GPML %.2e | ' ...
    'max |mean diff| state %.2e, deriv %.2e | max |var diff| state %.2e, deriv %.2e\n'], ...
    name, post.asym, post.kernel_gpml_err, ...
    max(abs(post.mu_f - fmu)), max(abs(post.mu_d - dmu)), ...
    max(abs(max(v(1:n), 0) - fs2)), max(abs(max(v(n+1:end), 0) - ds2)));
fprintf('  min eigenvalue of correlation matrix %.3e | relative jitter used %.1e\n', ...
    S.min_eig_corr, S.eps_rel);
end

function [X, dX, acc] = draw_nonneg_pairs(post1, S1, post2, S2, nWant, maxDraws)
% Joint draws [M1; dM1] and [M2; dM2] until nWant pairs have no negative state
% on the grid. Derivatives may be negative. The constraint is never relaxed.
n = numel(post1.t);
iS = 1:n;
iD = n+1:2*n;
X = zeros(n, 2, nWant);
dX = zeros(n, 2, nWant);
acc = struct('draws', 0, 'accepted', 0, 'rejected', 0, ...
    'rej_M1_only', 0, 'rej_M2_only', 0, 'rej_both', 0, 'rate', NaN);
while acc.accepted < nWant
    if acc.draws >= maxDraws
        error('SINDy:LowAcceptance', ['Accepted %d of %d trajectories after %d draws ' ...
            '(rate %.3g; rejected M1 only %d, M2 only %d, both %d). ' ...
            'Raise maxDraws; the positivity constraint is not relaxed.'], ...
            acc.accepted, nWant, acc.draws, acc.accepted / max(acc.draws, 1), ...
            acc.rej_M1_only, acc.rej_M2_only, acc.rej_both);
    end
    acc.draws = acc.draws + 1;
    z1 = post1.mu + S1.L * randn(2 * n, 1);
    z2 = post2.mu + S2.L * randn(2 * n, 1);
    neg1 = any(z1(iS) < 0);
    neg2 = any(z2(iS) < 0);
    if neg1 || neg2
        acc.rejected = acc.rejected + 1;
        if neg1 && neg2
            acc.rej_both = acc.rej_both + 1;
        elseif neg1
            acc.rej_M1_only = acc.rej_M1_only + 1;
        else
            acc.rej_M2_only = acc.rej_M2_only + 1;
        end
        continue
    end
    acc.accepted = acc.accepted + 1;
    X(:, :, acc.accepted) = [z1(iS), z2(iS)];
    dX(:, :, acc.accepted) = [z1(iD), z2(iD)];
end
acc.rate = acc.accepted / acc.draws;
end

function q = pct_mid(x, p)
% Percentiles with the midpoint rule used by prctile (no toolbox needed).
x = sort(x(:));
n = numel(x);
if n == 0
    q = nan(1, numel(p));
    return
end
if n == 1
    q = repmat(x, 1, numel(p));
    return
end
pos = min(max(n * p(:)' / 100 + 0.5, 1), n);
q = interp1(1:n, x, pos);
end

function [U, Cc, freq] = coefficient_stats(C, tol)
% C is nLib x nValid. Columns of U and Cc: mean, median, std, p2.5, p97.5.
% U includes zeros (term not selected); Cc uses only fits that selected the term.
nLib = size(C, 1);
U = nan(nLib, 5);
Cc = nan(nLib, 5);
freq = zeros(nLib, 1);
for k = 1:nLib
    c = C(k, :)';
    sel = abs(c) > tol;
    freq(k) = mean(sel);
    U(k, :) = [mean(c), median(c), std(c), pct_mid(c, [2.5, 97.5])];
    if any(sel)
        cs = c(sel);
        Cc(k, :) = [mean(cs), median(cs), std(cs), pct_mid(cs, [2.5, 97.5])];
    end
end
end

function print_coef_table(label, names, freq, U, Cc)
fprintf('\n--- %s ---\n', label);
fprintf('%-24s %6s | %11s %11s %11s %11s %11s | %11s %11s %11s %11s %11s\n', ...
    'term', 'P(sel)', 'mean', 'median', 'std', 'p2.5', 'p97.5', ...
    'sel.mean', 'sel.median', 'sel.std', 'sel.p2.5', 'sel.p97.5');
for k = 1:numel(names)
    fprintf('%-24s %6.3f | %11.4g %11.4g %11.4g %11.4g %11.4g | %11.4g %11.4g %11.4g %11.4g %11.4g\n', ...
        names{k}, freq(k), U(k, :), Cc(k, :));
end
end

function sim = simulate_sindy(Xi, x0, tspan, blowup)
% Forward-integrate one SINDy model. status: 'ok', 'diverged' (|state| reached
% blowup), or 'failed' (solver error or early stop). negative flags any state < 0.
sim = struct('status', 'ok', 'msg', '', 't', [], 'y', [], 'negative', false);
opts = odeset('Events', @(t, y) blowup_event(t, y, blowup));
try
    [t, y, te] = ode15s(@(t, y) SINDyfwd(t, y, Xi), tspan, x0, opts);
catch ME
    sim.status = 'failed';
    sim.msg = ME.message;
    return
end
sim.t = t;
sim.y = y;
sim.negative = any(y(:) < 0);
if ~isempty(te)
    sim.status = 'diverged';
    sim.msg = sprintf('|state| reached %g at t = %.3g', blowup, te(1));
elseif numel(t) < numel(tspan) || any(~isfinite(y(:)))
    sim.status = 'failed';
    sim.msg = sprintf('integration stopped at t = %.3g', t(end));
end
end

function [sol, status, neg] = simulate_ensemble(XiAll, x0All, tspan, blowup)
% Forward-integrate every model in XiAll (nLib x 2 x n) from x0All(:, i).
% sol stays NaN unless the solve finished ('ok').
n = size(XiAll, 3);
sol = nan(numel(tspan), 2, n);
status = repmat({'not run'}, n, 1);
neg = false(n, 1);
for i = 1:n
    sim = simulate_sindy(XiAll(:, :, i), x0All(:, i), tspan, blowup);
    status{i} = sim.status;
    neg(i) = sim.negative;
    if strcmp(sim.status, 'ok')
        sol(:, :, i) = sim.y;
    end
end
end

function env = traj_envelope(sol, use)
% Pointwise 2.5/50/97.5 percentiles (columns) per species (pages) over the
% models flagged in use. All NaN if fewer than two models qualify.
nT = size(sol, 1);
env = nan(nT, 3, 2);
if nnz(use) < 2
    return
end
for s = 1:2
    for i = 1:nT
        env(i, :, s) = pct_mid(reshape(sol(i, s, use), [], 1), [2.5, 50, 97.5]);
    end
end
end

function err = sim_errors(sim, tObs, yObs, tHold, yHold)
% Per species: RMSE against the data and model minus data at the holdout.
% NaN where the solution does not reach the time.
err = struct('rmse', nan(1, 2), 'hold', nan(1, 2));
if isempty(sim.y)
    return
end
for s = 1:2
    yhat = interp1(sim.t, sim.y(:, s), tObs{s});
    err.rmse(s) = sqrt(mean((yhat - yObs{s}).^2));
    err.hold(s) = interp1(sim.t, sim.y(:, s), tHold(s)) - yHold(s);
end
end

function plot_subset_fit(ax, s, tEnv, env, simMean, simMed, tObs, yObs, col, xl)
% Refit-ensemble envelope and median trajectory, the mean- and
% median-coefficient models, and the data, for species s.
hold(ax, 'on');
if all(isfinite(env(:, 1, s)))
    fill(ax, [tEnv; flipud(tEnv)], [env(:, 3, s); flipud(env(:, 1, s))], [0.3 0.3 0.3], ...
        'FaceAlpha', 0.18, 'EdgeColor', 'none', 'DisplayName', 'Refit ensemble 95% range');
    plot(ax, tEnv, env(:, 2, s), ':', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5, ...
        'DisplayName', 'Refit ensemble median trajectory');
end
if ~isempty(simMean.y)
    plot(ax, simMean.t, simMean.y(:, s), '-', 'Color', col, 'LineWidth', 2.0, ...
        'DisplayName', 'Mean-coefficient model');
end
if ~isempty(simMed.y)
    plot(ax, simMed.t, simMed.y(:, s), '--', 'Color', col, 'LineWidth', 2.0, ...
        'DisplayName', 'Median-coefficient model');
end
scatter(ax, tObs, yObs, 40, col, 'filled', 'DisplayName', 'Data');
hold(ax, 'off');
xlim(ax, xl);
xlabel(ax, 'Time (Days)');
ylabel(ax, 'cells/mm^2');
legend(ax, 'Location', 'best');
set(ax, 'FontSize', 12);
box(ax, 'on');
end

function [value, isterminal, direction] = blowup_event(~, y, limit)
value = limit - max(abs(y));
isterminal = 1;
direction = -1;
end

function plot_ode_with_data(ax, sim, tObs1, yObs1, tObs2, yObs2, xl)
% ODE solution (M1 black, M2 red) with the observed data, styled like Figure 198.
hold(ax, 'on');
if ~isempty(sim.y)
    plot(ax, sim.t, sim.y(:, 1), 'k', 'LineWidth', 2.0);
    plot(ax, sim.t, sim.y(:, 2), 'r', 'LineWidth', 2.0);
end
scatter(ax, tObs1, yObs1, 50, 'k', 'filled');
scatter(ax, tObs2, yObs2, 50, 'r', 'filled');
hold(ax, 'off');
xlim(ax, xl);
xlabel(ax, 'Time (Days)', 'fontsize', 20);
ylabel(ax, 'cells/mm^2', 'fontsize', 20);
set(ax, 'fontsize', 20);
box(ax, 'on');
end

function plot_sample_panel(ax, tg, mu, sd, kp, tS, S, tObs, yObs, col, ttl, ylab, xl)
hold(ax, 'on');
fill(ax, [tg; flipud(tg)], [mu + kp * sd; flipud(mu - kp * sd)], col, ...
    'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(ax, tS, S, 'Color', [0.45 0.45 0.45 0.35], 'LineWidth', 0.75);
plot(ax, tg, mu, 'Color', col, 'LineWidth', 2.0);
if ~isempty(tObs)
    scatter(ax, tObs, yObs, 40, col, 'filled');
end
hold(ax, 'off');
xlim(ax, xl);
xlabel(ax, 'Time (Days)');
ylabel(ax, ylab);
title(ax, ttl);
set(ax, 'FontSize', 16);
box(ax, 'on');
end

function check_true(ok, msg)
if ok
    fprintf('  PASS  %s\n', msg);
else
    error('SINDy:Validation', 'Validation failed: %s', msg);
end
end

function s = format_sindy_eq(xi, names)
terms = {};
for i = 1:numel(xi)
    if abs(xi(i)) > 0
        terms{end+1} = sprintf('%+.6g*%s', xi(i), names{i}); %#ok<AGROW>
    end
end
if isempty(terms)
    s = '0';
else
    s = strjoin(terms, ' ');
    if s(1) == '+'
        s = strtrim(s(2:end));
    end
end
end
