% Paper figure: Lotka-Volterra GP comparison (independent periodic vs ICM).
% Left panel: prey and predator fit independently with periodic kernels.
% Right panel: ICM MOGP with shared periodic temporal kernel:
%   k((t,d),(t',d')) = B(d,d') * k_per(t,t').
% Base-kernel signal variance is clamped to 1 so magnitudes live in B.
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
% % --- Shared-grid sampling (original) ---
% rng(100);
% n_train = 8;
% x_train = linspace(t_min, t_max, n_train)';   % shared sample times for both states
%
% y_true_train = interp1(x_grid, y_true_grid, x_train, 'pchip');
% y_true_train = max(y_true_train, 0);
%
% noise_frac = 0.05;
% amp_prey = max(y_true_grid(:, 1));
% amp_pred = max(y_true_grid(:, 2));
% sn_prey = noise_frac * amp_prey;
% sn_pred = noise_frac * amp_pred;
%
% y_train_prey = y_true_train(:, 1) + sn_prey * randn(n_train, 1);
% y_train_pred = y_true_train(:, 2) + sn_pred * randn(n_train, 1);
%
% fprintf('Synthetic LV data: n=%d per state on [%.0f, %.0f]\n', n_train, t_min, t_max);
% fprintf('Additive noise: sigma_prey=%.4f (%.0f%% of %.2f), sigma_pred=%.4f (%.0f%% of %.2f)\n', ...
%     sn_prey, 100 * noise_frac, amp_prey, sn_pred, 100 * noise_frac, amp_pred);

% --- Dense predator, sparse prey with a full-period gap (matches LV_LMC) ---
rng(100);
noise_frac = 0.05;
amp_prey = max(y_true_grid(:, 1));
amp_pred = max(y_true_grid(:, 2));
sn_prey = noise_frac * amp_prey;
sn_pred = noise_frac * amp_pred;

T_period = 10;   % ~one LV cycle (~30/3; matches periodic init p0)
gap_lo = (t_min + t_max) / 2 - T_period / 2;   % gap starts at 10
gap_hi = gap_lo + T_period;                     % gap ends at 20

n_pred = 25;                                    % dense predator over full window
x_train_pred = linspace(t_min, t_max, n_pred)';

n_prey_side = 4;                                % sparse prey on each side of the gap (8 total)
x_train_prey = [linspace(t_min, gap_lo, n_prey_side)'; ...
                linspace(gap_hi, t_max, n_prey_side)'];
n_prey = numel(x_train_prey);

y_true_prey = max(interp1(x_grid, y_true_grid(:, 1), x_train_prey, 'pchip'), 0);
y_true_pred = max(interp1(x_grid, y_true_grid(:, 2), x_train_pred, 'pchip'), 0);
y_train_prey = y_true_prey + sn_prey * randn(n_prey, 1);
y_train_pred = y_true_pred + sn_pred * randn(n_pred, 1);

fprintf('Synthetic LV data: n_prey=%d (gap [%.1f, %.1f]), n_pred=%d on [%.0f, %.0f]\n', ...
    n_prey, gap_lo, gap_hi, n_pred, t_min, t_max);
fprintf('Additive noise: sigma_prey=%.4f (%.0f%% of %.2f), sigma_pred=%.4f (%.0f%% of %.2f)\n', ...
    sn_prey, 100 * noise_frac, amp_prey, sn_pred, 100 * noise_frac, amp_pred);

%% Period initialization (manual)
% p0 rounds the analytic small-oscillation period 2*pi/sqrt(alpha*gamma) ~ 9.47
% and matches the observed ~3 cycles over [0, 30] (period ~ 30/3 = 10).
p0 = 10;
fprintf('Periodic kernel: manual period init p0 = %.3f\n', p0);

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
covPer   = @covPeriodic;
likfunc  = @likGauss;
inffunc  = @infGaussLik;
max_iters = -200;

%% ===== Baseline: independent periodic GP per state (left panel) =====
% sigma_n fixed at the known per-state noise.
fprintf('\n=== Baseline (independent periodic) ===\n');
[base_prey.m, base_prey.sf, base_prey.hyp, base_prey.nlml] = fit_periodic_state( ...
    x_train_prey, y_train_prey, sn_prey, p0, x_grid, inffunc, meanfunc, covPer, likfunc);
[base_pred.m, base_pred.sf, base_pred.hyp, base_pred.nlml] = fit_periodic_state( ...
    x_train_pred, y_train_pred, sn_pred, p0, x_grid, inffunc, meanfunc, covPer, likfunc);

%% ===== ICM: shared periodic temporal kernel (right panel) =====
fprintf('\n=== ICM (shared periodic temporal kernel) ===\n');
[icm_prey, icm_pred, hyp_icm, nlml_icm, B_icm, rho_icm] = fit_icm_periodic( ...
    x_train_prey, y_train_prey, x_train_pred, y_train_pred, x_grid, covPer, ...
    meanfunc, likfunc, p0, max_iters);

%% Plot: independent periodic (left) vs ICM (right)
k_plot = 2;
band_label = sprintf('\\pm %g\\sigma_f', k_plot);
col_prey = [0.00, 0.45, 0.74];   % blue
col_pred = [0.85, 0.16, 0.16];   % red

ylo = min([ ...
    y_train_prey(:); y_train_pred(:); ...
    base_prey.m - k_plot * base_prey.sf; base_pred.m - k_plot * base_pred.sf; ...
    icm_prey.m - k_plot * icm_prey.sf; icm_pred.m - k_plot * icm_pred.sf]);
yhi = max([ ...
    y_train_prey(:); y_train_pred(:); ...
    base_prey.m + k_plot * base_prey.sf; base_pred.m + k_plot * base_pred.sf; ...
    icm_prey.m + k_plot * icm_prey.sf; icm_pred.m + k_plot * icm_pred.sf]);
pad = 0.05 * (yhi - ylo);
ylim_shared = [ylo - pad, yhi + pad];

figure('Color', 'w', 'Position', [80, 80, 1200, 540], ...
    'Name', 'Lotka-Volterra GP: independent periodic vs ICM');
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

panels(1) = struct('prey', base_prey, 'pred', base_pred, ...
    'title', 'Independent periodic GPs');
panels(2) = struct('prey', icm_prey, 'pred', icm_pred, ...
    'title', 'ICM (shared periodic kernel)');

for pidx = 1:2
    nexttile;
    ax = gca; ax.Layer = 'top';
    hold on; grid on;
    plot_state(ax, x_grid, y_true_grid(:, 1), panels(pidx).prey, ...
        x_train_prey, y_train_prey, col_prey, k_plot, 'Prey', band_label);
    plot_state(ax, x_grid, y_true_grid(:, 2), panels(pidx).pred, ...
        x_train_pred, y_train_pred, col_pred, k_plot, 'Predator', band_label);
    xlabel('t');
    ylabel('Population');
    title(panels(pidx).title, 'Interpreter', 'none');
    xlim([t_min, t_max]);
    ylim(ylim_shared);
end

%% Standalone shared legend (for LaTeX / Inkscape)
figL = figure('Color', 'w', 'Position', [100, 100, 900, 80], ...
    'Name', 'LV ICM shared legend');
axL = axes('Parent', figL, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
    'Position', [0 0 1 1]);
hold(axL, 'on');
hL = gobjects(8, 1);
hL(1) = fill(axL, nan, nan, col_prey, 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', 'Prey 95% CI');
hL(2) = plot(axL, nan, nan, '-', 'Color', col_prey, 'LineWidth', 1.5, ...
    'DisplayName', 'Prey True Model');
hL(3) = plot(axL, nan, nan, '--', 'Color', col_prey, 'LineWidth', 2, ...
    'DisplayName', 'Prey GP Mean');
hL(4) = plot(axL, nan, nan, 'o', 'Color', col_prey, 'MarkerFaceColor', col_prey, ...
    'MarkerEdgeColor', 'k', 'MarkerSize', 5, 'DisplayName', 'Prey Obs Data');
hL(5) = fill(axL, nan, nan, col_pred, 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', 'Predator 95% CI');
hL(6) = plot(axL, nan, nan, '-', 'Color', col_pred, 'LineWidth', 1.5, ...
    'DisplayName', 'Predator True Model');
hL(7) = plot(axL, nan, nan, '--', 'Color', col_pred, 'LineWidth', 2, ...
    'DisplayName', 'Predator GP Mean');
hL(8) = plot(axL, nan, nan, 'o', 'Color', col_pred, 'MarkerFaceColor', col_pred, ...
    'MarkerEdgeColor', 'k', 'MarkerSize', 5, 'DisplayName', 'Predator Obs Data');
lgd = legend(axL, hL, 'Orientation', 'horizontal', 'NumColumns', 4);
lgd.FontSize = 10;
lgd.ItemTokenSize = [20, 12];
drawnow;
figL.Units = 'pixels';
lgd.Units = 'pixels';
lp = lgd.Position;
margin = 6;
figL.Position(3:4) = [lp(3) + 2 * margin, lp(4) + 2 * margin];
lgd.Position = [margin, margin, lp(3), lp(4)];

%% Report
fprintf('\n--- Fitted hyperparameters ---\n');
report_periodic('Baseline Prey', base_prey);
report_periodic('Baseline Pred', base_pred);
fprintf('ICM: ell=%.4f, p=%.4f, sn(std)=%.4f, rho=%.4f | NLML=%.4f\n', ...
    exp(hyp_icm.cov(1)), exp(hyp_icm.cov(2)), exp(hyp_icm.lik), rho_icm, nlml_icm);
fprintf('ICM: B = [%.4f %.4f; %.4f %.4f]\n', ...
    B_icm(1,1), B_icm(1,2), B_icm(2,1), B_icm(2,2));
fprintf('Baseline total NLML = %.4f\n', base_prey.nlml + base_pred.nlml);

fprintf('\nDone.\n');

%% ----- local functions -----
function [m, sf, hyp, nlml] = fit_periodic_state(x, y, sn, p0, x_grid, inffunc, meanfunc, covfunc, likfunc)
% Period p is optimized jointly with ell and sf from the manual init p0.
x = x(:); y = y(:);
ell0 = 1;          % dimensionless roughness within one period
sf0  = std(y);
sn_fixed = log(sn);
hyp_cov0 = log([ell0; p0; sf0]);
hyp_cov = minimize(hyp_cov0, @gp_nlml_cov_only, -100, sn_fixed, ...
    inffunc, meanfunc, covfunc, likfunc, x, y);
hyp = struct('mean', [], 'cov', hyp_cov(:), 'lik', sn_fixed);
nlml = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, x_grid(:));
m = fmu(:);
sf = sqrt(max(fs2(:), 0));
end

function [prey, pred, hyp, nlml, B, rho] = fit_icm_periodic( ...
    t_prey, y_prey, t_pred, y_pred, tgrid, temporalKernel, ...
    meanfunc, likfunc, p0, max_iters)
% ICM with shared periodic temporal kernel; sf clamped to 1.

LABEL_PREY = 1;
LABEL_PRED = 2;

y_prey = y_prey(:); y_pred = y_pred(:);
mu_prey = mean(y_prey); sd_prey = std(y_prey);
mu_pred = mean(y_pred); sd_pred = std(y_pred);
if sd_prey < eps, sd_prey = 1; end
if sd_pred < eps, sd_pred = 1; end

x_aug = [t_prey(:), LABEL_PREY * ones(numel(t_prey), 1); ...
         t_pred(:), LABEL_PRED * ones(numel(t_pred), 1)];
y_aug = [ (y_prey - mu_prey) / sd_prey; (y_pred - mu_pred) / sd_pred ];

covICM = build_icm_kernel(temporalKernel);
[hyp0, ~, inffunc] = init_icm_periodic_hyp(temporalKernel, p0, x_aug, y_aug);

fprintf('Optimizing ICM hyperparameters (infPrior + infGaussLik)...\n');
hyp = minimize(hyp0, @gp, max_iters, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);
nlml = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);

x_te_prey = [tgrid(:), LABEL_PREY * ones(numel(tgrid), 1)];
x_te_pred = [tgrid(:), LABEL_PRED * ones(numel(tgrid), 1)];
[~, ~, fmu_p, fs2_p] = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug, x_te_prey);
[~, ~, fmu_q, fs2_q] = gp(hyp, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug, x_te_pred);

prey.m  = mu_prey + sd_prey * fmu_p(:);
prey.sf = sd_prey * sqrt(max(fs2_p(:), 0));
pred.m  = mu_pred + sd_pred * fmu_q(:);
pred.sf = sd_pred * sqrt(max(fs2_q(:), 0));

nTemp = temporal_hyp_count(temporalKernel);
B = chol2cov(hyp.cov(nTemp + (1:3)));
rho = corr_from_B(B);
end

function covICM = build_icm_kernel(temporalKernel)
covICM = {@covProd, { ...
    {@covMask, {1, temporalKernel}}, ...
    {@covMask, {2, {@covDiscrete, 2}}} }};
end

function [hyp0, prior, inffunc] = init_icm_periodic_hyp(temporalKernel, p0, x_aug, y_aug)
% Periodic ICM hyps: [log(ell); log(p); log(sf); Lchol(3)].
% Clamp sf to 1 so output magnitudes live entirely in B.
meanfunc = @meanZero;
likfunc  = @likGauss;
covICM   = build_icm_kernel(temporalKernel);

nTemp = temporal_hyp_count(temporalKernel);   % 3 for covPeriodic
ell0 = 1;
Lchol0 = [log(sqrt(0.5)); 0; log(sqrt(0.5))];

hyp0.mean = [];
hyp0.cov  = [log(ell0); log(p0); 0; Lchol0];   % sf = exp(0) = 1
hyp0.lik  = log(0.1);

prior.cov = cell(1, nTemp + 3);
prior.cov{3} = @priorClamped;                  % clamp periodic sf
inffunc = {@infPrior, @infGaussLik, prior};

gp(hyp0, inffunc, meanfunc, covICM, likfunc, x_aug, y_aug);
end

function nTemp = temporal_hyp_count(covfunc)
if isa(covfunc, 'function_handle') && strcmp(func2str(covfunc), 'covPeriodic')
    nTemp = 3;   % ell, p, sf
elseif isa(covfunc, 'function_handle') && strcmp(func2str(covfunc), 'covRQiso')
    nTemp = 3;
elseif iscell(covfunc) && strcmp(func2str(covfunc{1}), 'covMaterniso')
    nTemp = 2;
else
    nTemp = 2;   % SE default: ell, sf
end
end

function B = chol2cov(hyp)
% Reconstruct 2x2 covDiscrete coregionalization B = L'*L from
% [log(L11); L21; log(L22)].
L = zeros(2);
L(triu(true(2))) = hyp(:);
L(1:3:end) = exp(diag(L));
B = L' * L;
end

function rho = corr_from_B(B)
rho = B(1, 2) / sqrt(max(B(1, 1) * B(2, 2), eps));
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

function report_periodic(label, fit)
fprintf('%s: ell=%.4f, p=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    label, exp(fit.hyp.cov(1)), exp(fit.hyp.cov(2)), exp(fit.hyp.cov(3)), ...
    exp(fit.hyp.lik), fit.nlml);
end
