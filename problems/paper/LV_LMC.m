% Paper figure: Lotka-Volterra GP comparison (naive independent SE vs. LMC).
% Naive GP fits each state independently with an SE kernel (left panel).
% The Linear Model of Coregionalization (LMC) couples prey and predator
% through Q=2 shared latent processes (right panel):
%   k_1 = Matern 5/2 with a long length scale (smooth shared trend)
%   k_2 = periodic kernel with a short length scale (oscillation)
% Each latent process q gets a full-rank 2x2 coregionalization matrix B_q,
% built with GPML's covDiscrete (B_q = L_q'*L_q, PSD by construction). A long
% Matern length scale and a short periodic length scale are encouraged with
% Gaussian hyperpriors via infPrior; the base-kernel signal variances are
% clamped to 1 so the cross-output magnitude lives entirely in B_q.
% Requires gp_nlml_cov_only.m on the path (problems/) and the GPML toolbox.
clear; clc; close all;

%% Lotka-Volterra parameters
alpha = 1.1;   % prey growth rate
beta  = 0.4;   % predation rate
delta = 0.1;   % predator reproduction
gamma = 0.4;   % predator death rate
prey0 = 10;
pred0 = 5;

odefun = @(t, z) [ ...
    alpha * z(1) - beta * z(1) * z(2); ...
    delta * z(1) * z(2) - gamma * z(2)];
z0 = [prey0; pred0];

%% Ground truth (dense ode45 solve, ~3 cycles)
t_min = 0;
t_max = 30;
x_grid = linspace(t_min, t_max, 600)';
[~, z_grid] = ode45(odefun, x_grid, z0);
y_true_grid = max(z_grid, 0);   % [n_grid x 2] -> columns [prey, predator]

%% Training data (additive Gaussian noise, ~5% of each state's amplitude)
rng(100);
n_train = 10;
x_train = linspace(t_min, t_max, n_train)';   % shared sample times for both states

y_true_train = interp1(x_grid, y_true_grid, x_train, 'pchip');
y_true_train = max(y_true_train, 0);

noise_frac = 0.05;
amp_prey = max(y_true_grid(:, 1));
amp_pred = max(y_true_grid(:, 2));
sn_prey = noise_frac * amp_prey;
sn_pred = noise_frac * amp_pred;

y_train_prey = y_true_train(:, 1) + sn_prey * randn(n_train, 1);
y_train_pred = y_true_train(:, 2) + sn_pred * randn(n_train, 1);

fprintf('Synthetic LV data: n=%d per state on [%.0f, %.0f]\n', n_train, t_min, t_max);
fprintf('Additive noise: sigma_prey=%.4f (%.0f%% of %.2f), sigma_pred=%.4f (%.0f%% of %.2f)\n', ...
    sn_prey, 100 * noise_frac, amp_prey, sn_pred, 100 * noise_frac, amp_pred);

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

%% ===== Naive baseline: independent SE GP per state (left panel) =====
% sigma_n fixed at the known per-state noise (matches LV_Periodic baseline).
inffunc_se = @infGaussLik;
covSE      = @covSEiso;
fprintf('\n=== Naive (independent squared exponential) ===\n');
[se_prey.m, se_prey.sf, se_prey.hyp, se_prey.nlml] = fit_se_state( ...
    x_train, y_train_prey, sn_prey, x_grid, inffunc_se, meanfunc, covSE, likfunc);
[se_pred.m, se_pred.sf, se_pred.hyp, se_pred.nlml] = fit_se_state( ...
    x_train, y_train_pred, sn_pred, x_grid, inffunc_se, meanfunc, covSE, likfunc);

%% ===== LMC: Q=2 latent processes with full-rank coregionalization =====
fprintf('\n=== LMC (Q=2, Matern 5/2 + periodic, full-rank B_q) ===\n');

% Standardize each output (zero mean, unit std) so a single Gaussian noise is
% well-scaled across the two states; predictions are un-standardized below.
mu_prey = mean(y_train_prey);  sd_prey = std(y_train_prey);
mu_pred = mean(y_train_pred);  sd_pred = std(y_train_pred);

LABEL_PREY = 1; LABEL_PRED = 2;
x_aug = [ [x_train; x_train], [LABEL_PREY * ones(n_train, 1); LABEL_PRED * ones(n_train, 1)] ];
y_aug = [ (y_train_prey - mu_prey) / sd_prey; (y_train_pred - mu_pred) / sd_pred ];

% Composed LMC kernel: sum over q of [ base kernel over time ] .* [ B_q over label ]
covLMC = {@covSum, { ...
    {@covProd, { {@covMask, {1, {@covMaterniso, 5}}}, {@covMask, {2, {@covDiscrete, 2}}} }}, ...
    {@covProd, { {@covMask, {1, @covPeriodic}},        {@covMask, {2, {@covDiscrete, 2}}} }} }};

% Hyperprior targets (in log space): long Matern length scale, short periodic one.
ellLong  = (t_max - t_min) / 2;    % ~15: smooth shared trend over the window
ellShort = 0.5;                    % sub-period roughness for the oscillation
s2_ell   = 0.5^2;                  % hyperprior variance on log(ell)
p0       = 10;                     % period init (~30/3 cycles), optimized freely

% B_q Cholesky init: B_q = 0.5*I so each output gets unit variance summed over q.
% covDiscrete hyp = [log(L11); L21; log(L22)] with B = L'*L.
Lchol0 = [log(sqrt(0.5)); 0; log(sqrt(0.5))];

hyp_lmc.mean = [];
hyp_lmc.cov  = [ log(ellLong); 0;            Lchol0; ...   % Matern: ell1, sf1(clamped), B1
                 log(ellShort); log(p0); 0;  Lchol0 ];      % Periodic: ell2, p2, sf2(clamped), B2
hyp_lmc.lik  = log(0.1);           % single Gaussian noise (standardized units), optimized

% Priors: encourage long/short length scales, clamp base-kernel signal variances.
prior.cov = { {@priorGauss, log(ellLong), s2_ell}, @priorClamped, [], [], [], ...
              {@priorGauss, log(ellShort), s2_ell}, [], @priorClamped, [], [], [] };
inffunc_lmc = {@infPrior, @infGaussLik, prior};

fprintf('Optimizing LMC hyperparameters (infPrior + infGaussLik)...\n');
hyp_lmc = minimize(hyp_lmc, @gp, -300, inffunc_lmc, meanfunc, covLMC, likfunc, x_aug, y_aug);
nlml_lmc = gp(hyp_lmc, inffunc_lmc, meanfunc, covLMC, likfunc, x_aug, y_aug);

% Predict each state on the grid (carrying its label), then un-standardize.
x_te_prey = [x_grid, LABEL_PREY * ones(size(x_grid))];
x_te_pred = [x_grid, LABEL_PRED * ones(size(x_grid))];
[~, ~, fmu_p, fs2_p] = gp(hyp_lmc, inffunc_lmc, meanfunc, covLMC, likfunc, x_aug, y_aug, x_te_prey);
[~, ~, fmu_q, fs2_q] = gp(hyp_lmc, inffunc_lmc, meanfunc, covLMC, likfunc, x_aug, y_aug, x_te_pred);

lmc_prey.m  = mu_prey + sd_prey * fmu_p(:);
lmc_prey.sf = sd_prey * sqrt(max(fs2_p(:), 0));
lmc_pred.m  = mu_pred + sd_pred * fmu_q(:);
lmc_pred.sf = sd_pred * sqrt(max(fs2_q(:), 0));

%% Plot: naive (left) vs LMC (right), prey + predator overlaid
k_plot = 2;
band_label = sprintf('\\pm %g\\sigma_f', k_plot);
col_prey = [0.00, 0.45, 0.74];   % blue
col_pred = [0.85, 0.16, 0.16];   % red

ylim_shared = [0, max([ ...
    y_train_prey(:); y_train_pred(:); ...
    se_prey.m + k_plot * se_prey.sf; se_pred.m + k_plot * se_pred.sf; ...
    lmc_prey.m + k_plot * lmc_prey.sf; lmc_pred.m + k_plot * lmc_pred.sf]) * 1.05];

figure('Color', 'w', 'Position', [80, 80, 1200, 540], ...
    'Name', 'Lotka-Volterra GP: naive vs LMC');
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

panels(1) = struct('prey', se_prey,  'pred', se_pred,  'title', 'Naive GP (independent SE)');
panels(2) = struct('prey', lmc_prey, 'pred', lmc_pred, 'title', 'LMC (Q=2: Matern 5/2 + periodic)');

for pidx = 1:2
    nexttile;
    ax = gca; ax.Layer = 'top';
    hold on; grid on;
    plot_state(ax, x_grid, y_true_grid(:, 1), panels(pidx).prey, ...
        x_train, y_train_prey, col_prey, k_plot, 'Prey', band_label);
    plot_state(ax, x_grid, y_true_grid(:, 2), panels(pidx).pred, ...
        x_train, y_train_pred, col_pred, k_plot, 'Predator', band_label);
    xlabel('t');
    ylabel('Population');
    title(panels(pidx).title, 'Interpreter', 'none');
    xlim([t_min, t_max]);
    ylim(ylim_shared);
    legend('Location', 'northeast', 'NumColumns', 2, 'FontSize', 8);
end

%% Report
fprintf('\n--- Fitted hyperparameters ---\n');
fprintf('Naive  Prey: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(se_prey.hyp.cov(1)), exp(se_prey.hyp.cov(2)), exp(se_prey.hyp.lik), se_prey.nlml);
fprintf('Naive  Pred: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(se_pred.hyp.cov(1)), exp(se_pred.hyp.cov(2)), exp(se_pred.hyp.lik), se_pred.nlml);

ell1 = exp(hyp_lmc.cov(1));
ell2 = exp(hyp_lmc.cov(6));
per2 = exp(hyp_lmc.cov(7));
B1 = chol2cov(hyp_lmc.cov(3:5));
B2 = chol2cov(hyp_lmc.cov(9:11));
sn_lmc = exp(hyp_lmc.lik);
fprintf('LMC:   Matern 5/2 ell=%.4f | periodic ell=%.4f, p=%.4f | sigma_n(std)=%.4f | NLML=%.4f\n', ...
    ell1, ell2, per2, sn_lmc, nlml_lmc);
fprintf('LMC:   B_1 (Matern, shared trend)   = [%.4f %.4f; %.4f %.4f]\n', B1(1,1), B1(1,2), B1(2,1), B1(2,2));
fprintf('LMC:   B_2 (periodic, oscillation)  = [%.4f %.4f; %.4f %.4f]\n', B2(1,1), B2(1,2), B2(2,1), B2(2,2));

%% ----- local functions -----
function [m, sf, hyp, nlml] = fit_se_state(x, y, sn, x_grid, inffunc, meanfunc, covfunc, likfunc)
x = x(:); y = y(:);
ell0 = std(x);
sf0  = std(y);
sn_fixed = log(sn);
hyp_cov0 = log([ell0; sf0]);
hyp_cov = minimize(hyp_cov0, @gp_nlml_cov_only, -100, sn_fixed, ...
    inffunc, meanfunc, covfunc, likfunc, x, y);
hyp = struct('mean', [], 'cov', hyp_cov(:), 'lik', sn_fixed);
nlml = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, x_grid(:));
m = fmu(:);
sf = sqrt(max(fs2(:), 0));
end

function B = chol2cov(hyp)
% Reconstruct a 2x2 covDiscrete coregionalization matrix B = L'*L from its
% Cholesky hyperparameters [log(L11); L21; log(L22)].
L = zeros(2);
L(triu(true(2))) = hyp(:);
L(1:3:end) = exp(diag(L));
B = L' * L;
end

function plot_state(ax, x_grid, y_true, fit, x_train, y_train, col, k_plot, name, band_label)
xg = x_grid(:)';
m = fit.m; sf = fit.sf;
fill(ax, [xg, fliplr(xg)], [m + k_plot * sf; flipud(m - k_plot * sf)]', ...
    col, 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', sprintf('%s %s', name, band_label));
plot(ax, x_grid, y_true, '-', 'Color', col, 'LineWidth', 1.5, ...
    'DisplayName', sprintf('%s truth', name));
plot(ax, x_grid, m, '--', 'Color', col, 'LineWidth', 2, ...
    'DisplayName', sprintf('%s mean', name));
plot(ax, x_train, y_train, 'o', 'MarkerSize', 5, ...
    'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', ...
    'DisplayName', sprintf('%s data', name));
end
