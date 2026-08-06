% Paper figure: Lotka-Volterra GP comparison (naive independent periodic vs. LMC).
% Naive GP fits each state independently with a periodic kernel. The Linear Model
% of Coregionalization (LMC) couples prey and predator through Q=2 shared
% latent processes:
%   k_1 = periodic kernel with a short length scale (oscillation)
%   k_2 = SE with a long length scale (smooth shared trend)
% Each latent process q gets a full-rank 2x2 coregionalization matrix B_q,
% built with GPML's covDiscrete (B_q = L_q'*L_q, PSD by construction). A short
% periodic length scale and a long SE length scale are encouraged with
% Gaussian hyperpriors via infPrior; the base-kernel signal variances are
% clamped to 1 so the cross-output magnitude lives entirely in B_q.
% Naive and LMC fits are written as separate figures per state (prey / predator),
% with a standalone shared legend for LaTeX / Inkscape.
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
% n_train = 10;
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

% --- Dense predator, sparse prey with a full-period gap ---
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

n_prey_side = 3;                                % sparse prey on each side of the gap (6 total)
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

%% ===== Naive baseline: independent periodic GP per state =====
% sigma_n fixed at the known per-state noise (matches LV_Periodic).
inffunc_base = @infGaussLik;
covPer       = @covPeriodic;
p0_base      = 10;   % period init (~30/3 cycles), optimized jointly with ell, sf
fprintf('\n=== Naive (independent periodic) ===\n');
[per_prey.m, per_prey.sf, per_prey.hyp, per_prey.nlml] = fit_periodic_state( ...
    x_train_prey, y_train_prey, sn_prey, p0_base, x_grid, inffunc_base, meanfunc, covPer, likfunc);
[per_pred.m, per_pred.sf, per_pred.hyp, per_pred.nlml] = fit_periodic_state( ...
    x_train_pred, y_train_pred, sn_pred, p0_base, x_grid, inffunc_base, meanfunc, covPer, likfunc);

%% ===== LMC: Q=2 latent processes with full-rank coregionalization =====
fprintf('\n=== LMC (Q=2, periodic + SE, full-rank B_q) ===\n');

% Standardize each output (zero mean, unit std) so a single Gaussian noise is
% well-scaled across the two states; predictions are un-standardized below.
mu_prey = mean(y_train_prey);  sd_prey = std(y_train_prey);
mu_pred = mean(y_train_pred);  sd_pred = std(y_train_pred);

LABEL_PREY = 1; LABEL_PRED = 2;
x_aug = [ [x_train_prey; x_train_pred], ...
          [LABEL_PREY * ones(n_prey, 1); LABEL_PRED * ones(n_pred, 1)] ];
y_aug = [ (y_train_prey - mu_prey) / sd_prey; (y_train_pred - mu_pred) / sd_pred ];

% Composed LMC kernel: sum over q of [ base kernel over time ] .* [ B_q over label ]
covLMC = {@covSum, { ...
    {@covProd, { {@covMask, {1, @covPeriodic}}, {@covMask, {2, {@covDiscrete, 2}}} }}, ...
    {@covProd, { {@covMask, {1, @covSEiso}},    {@covMask, {2, {@covDiscrete, 2}}} }} }};

% Hyperprior targets (in log space): short periodic length scale, long SE one.
ellLong  = (t_max - t_min) / 2;    % ~15: smooth shared trend over the window
ellShort = 0.5;                    % sub-period roughness for the oscillation
s2_ell   = 0.5^2;                  % hyperprior variance on log(ell)
p0       = 10;                     % period init (~30/3 cycles), optimized freely

% B_q Cholesky init: B_q = 0.5*I so each output gets unit variance summed over q.
% covDiscrete hyp = [log(L11); L21; log(L22)] with B = L'*L.
Lchol0 = [log(sqrt(0.5)); 0; log(sqrt(0.5))];

hyp_lmc.mean = [];
hyp_lmc.cov  = [ log(ellShort); log(p0); 0;  Lchol0; ...   % Periodic: ell1, p1, sf1(clamped), B1
                 log(ellLong); 0;            Lchol0 ];      % SE: ell2, sf2(clamped), B2
hyp_lmc.lik  = log(0.1);           % single Gaussian noise (standardized units), optimized

% Priors: encourage short/long length scales, clamp base-kernel signal variances.
prior.cov = { {@priorGauss, log(ellShort), s2_ell}, [], @priorClamped, [], [], [], ...
              {@priorGauss, log(ellLong), s2_ell}, @priorClamped, [], [], [] };
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

%% Plot: separate figures per method × state (shared y-limits within each state)
k_plot = 2;
band_label = sprintf('\\pm %g\\sigma_f', k_plot);
col_prey = [0.00, 0.45, 0.74];   % blue
col_pred = [0.85, 0.16, 0.16];   % red

ylo_prey = min([y_train_prey(:); ...
    per_prey.m - k_plot * per_prey.sf; lmc_prey.m - k_plot * lmc_prey.sf]);
yhi_prey = max([y_train_prey(:); ...
    per_prey.m + k_plot * per_prey.sf; lmc_prey.m + k_plot * lmc_prey.sf]);
pad_prey = 0.05 * (yhi_prey - ylo_prey);
ylim_prey = [ylo_prey - pad_prey, yhi_prey + pad_prey];

ylo_pred = min([y_train_pred(:); ...
    per_pred.m - k_plot * per_pred.sf; lmc_pred.m - k_plot * lmc_pred.sf]);
yhi_pred = max([y_train_pred(:); ...
    per_pred.m + k_plot * per_pred.sf; lmc_pred.m + k_plot * lmc_pred.sf]);
pad_pred = 0.05 * (yhi_pred - ylo_pred);
ylim_pred = [ylo_pred - pad_pred, yhi_pred + pad_pred];

panels(1) = struct('fit', per_prey, 'y_true', y_true_grid(:, 1), ...
    'x_obs', x_train_prey, 'y_obs', y_train_prey, 'col', col_prey, ...
    'ylim', ylim_prey, 'state', 'Prey', ...
    'title', 'Baseline GP (Independent Periodic) — Prey', ...
    'fname', 'LV_LMC_Naive_Prey.eps');
panels(2) = struct('fit', per_pred, 'y_true', y_true_grid(:, 2), ...
    'x_obs', x_train_pred, 'y_obs', y_train_pred, 'col', col_pred, ...
    'ylim', ylim_pred, 'state', 'Predator', ...
    'title', 'Baseline GP (Independent Periodic) — Predator', ...
    'fname', 'LV_LMC_Naive_Predator.eps');
panels(3) = struct('fit', lmc_prey, 'y_true', y_true_grid(:, 1), ...
    'x_obs', x_train_prey, 'y_obs', y_train_prey, 'col', col_prey, ...
    'ylim', ylim_prey, 'state', 'Prey', ...
    'title', 'LMC GP (Q=2: Periodic + SE) — Prey', ...
    'fname', 'LV_LMC_LMC_Prey.eps');
panels(4) = struct('fit', lmc_pred, 'y_true', y_true_grid(:, 2), ...
    'x_obs', x_train_pred, 'y_obs', y_train_pred, 'col', col_pred, ...
    'ylim', ylim_pred, 'state', 'Predator', ...
    'title', 'LMC GP (Q=2: Periodic + SE) — Predator', ...
    'fname', 'LV_LMC_LMC_Predator.eps');

ax_list = gobjects(4, 1);
fig_list = gobjects(4, 1);
for pidx = 1:4
    fig_list(pidx) = figure('Color', 'w', 'Position', [60 + 30*(pidx-1), 60 + 20*(pidx-1), 640, 480], ...
        'Name', panels(pidx).title);
    ax = axes('Parent', fig_list(pidx));
    ax.Layer = 'top';
    hold(ax, 'on'); grid(ax, 'on');
    plot_state(ax, x_grid, panels(pidx).y_true, panels(pidx).fit, ...
        panels(pidx).x_obs, panels(pidx).y_obs, panels(pidx).col, ...
        k_plot, panels(pidx).state, band_label);
    xlabel(ax, 't');
    ylabel(ax, 'Population');
    title(ax, panels(pidx).title, 'Interpreter', 'none');
    xlim(ax, [t_min, t_max]);
    ylim(ax, panels(pidx).ylim);
    ax_list(pidx) = ax;
end

%% Standalone shared legend (for LaTeX / Inkscape)
figL = figure('Color', 'w', 'Position', [100, 100, 900, 80], ...
    'Name', 'LV LMC shared legend');
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

%% Save each panel and the shared legend as EPS
plot_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
    'results', 'plots', 'Paper Draft 2', 'Interacting Species');
if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end
for i = 1:numel(ax_list)
    ax_list(i).Toolbar.Visible = 'off';
    disableDefaultInteractivity(ax_list(i));
    drawnow;
    out_path = fullfile(plot_dir, panels(i).fname);
    exportgraphics(ax_list(i), out_path, 'ContentType', 'vector');
    fprintf('Saved %s\n', out_path);
end
legend_path = fullfile(plot_dir, 'LV_LMC_legend.eps');
exportgraphics(figL, legend_path, 'ContentType', 'vector', 'BackgroundColor', 'white');
fprintf('Saved %s\n', legend_path);

%% Report
fprintf('\n--- Fitted hyperparameters ---\n');
fprintf('Naive  Prey: ell=%.4f, p=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(per_prey.hyp.cov(1)), exp(per_prey.hyp.cov(2)), exp(per_prey.hyp.cov(3)), ...
    exp(per_prey.hyp.lik), per_prey.nlml);
fprintf('Naive  Pred: ell=%.4f, p=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(per_pred.hyp.cov(1)), exp(per_pred.hyp.cov(2)), exp(per_pred.hyp.cov(3)), ...
    exp(per_pred.hyp.lik), per_pred.nlml);

ell_per = exp(hyp_lmc.cov(1));
per1    = exp(hyp_lmc.cov(2));
ell_se  = exp(hyp_lmc.cov(7));
B1 = chol2cov(hyp_lmc.cov(4:6));
B2 = chol2cov(hyp_lmc.cov(9:11));
sn_lmc = exp(hyp_lmc.lik);
fprintf('LMC:   periodic ell=%.4f, p=%.4f | SE ell=%.4f | sigma_n(std)=%.4f | NLML=%.4f\n', ...
    ell_per, per1, ell_se, sn_lmc, nlml_lmc);
fprintf('LMC:   B_1 (periodic, oscillation)  = [%.4f %.4f; %.4f %.4f]\n', B1(1,1), B1(1,2), B1(2,1), B1(2,2));
fprintf('LMC:   B_2 (SE, shared trend)       = [%.4f %.4f; %.4f %.4f]\n', B2(1,1), B2(1,2), B2(2,1), B2(2,2));

%% ----- local functions -----
function [m, sf, hyp, nlml] = fit_periodic_state(x, y, sn, p0, x_grid, inffunc, meanfunc, covfunc, likfunc)
% Period p is optimized jointly with ell and sf from the manual init p0.
x = x(:); y = y(:);
ell0 = 1;   % dimensionless roughness within one period
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
