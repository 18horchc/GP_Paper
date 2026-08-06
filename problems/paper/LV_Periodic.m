% Paper figure: Lotka-Volterra GP comparison (baseline SE vs. periodic kernel).
% Prey and predator are fit independently with each kernel. Baseline GP and
% periodic-kernel GP are written as separate figures (both states overlaid),
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
rng(100);
n_train = 8;
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

%% Period initialization (manual)
% p0 rounds the analytic small-oscillation period 2*pi/sqrt(alpha*gamma) ~ 9.47
% and matches the observed ~3 cycles over [0, 30] (period ~ 30/3 = 10).
% Same init for both states (shared orbit; periodic kernel is stationary so the
% predator's phase lag is irrelevant). Optimized jointly with ell, sf below.
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
covSE    = @covSEiso;
covPer   = @covPeriodic;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

%% Fit baseline SE and periodic GPs per state (sigma_n fixed at known noise)
fprintf('\n=== Baseline (squared exponential) ===\n');
[se_prey.m, se_prey.sf, se_prey.hyp, se_prey.nlml] = fit_se_state( ...
    x_train, y_train_prey, sn_prey, x_grid, inffunc, meanfunc, covSE, likfunc);
[se_pred.m, se_pred.sf, se_pred.hyp, se_pred.nlml] = fit_se_state( ...
    x_train, y_train_pred, sn_pred, x_grid, inffunc, meanfunc, covSE, likfunc);

fprintf('\n=== Periodic kernel ===\n');
[per_prey.m, per_prey.sf, per_prey.hyp, per_prey.nlml] = fit_periodic_state( ...
    x_train, y_train_prey, sn_prey, p0, x_grid, inffunc, meanfunc, covPer, likfunc);
[per_pred.m, per_pred.sf, per_pred.hyp, per_pred.nlml] = fit_periodic_state( ...
    x_train, y_train_pred, sn_pred, p0, x_grid, inffunc, meanfunc, covPer, likfunc);

%% Plot: separate figures for baseline and periodic (shared axes limits)
k_plot = 2;
band_label = sprintf('\\pm %g\\sigma_f', k_plot);
col_prey = [0.00, 0.45, 0.74];   % blue
col_pred = [0.85, 0.16, 0.16];   % red

ylo = min([ ...
    y_train_prey(:); y_train_pred(:); ...
    se_prey.m - k_plot * se_prey.sf; se_pred.m - k_plot * se_pred.sf; ...
    per_prey.m - k_plot * per_prey.sf; per_pred.m - k_plot * per_pred.sf]);
yhi = max([ ...
    y_train_prey(:); y_train_pred(:); ...
    se_prey.m + k_plot * se_prey.sf; se_pred.m + k_plot * se_pred.sf; ...
    per_prey.m + k_plot * per_prey.sf; per_pred.m + k_plot * per_pred.sf]);
pad = 0.05 * (yhi - ylo);
ylim_shared = [ylo - pad, yhi + pad];

panels(1) = struct('prey', se_prey,  'pred', se_pred,  ...
    'title', 'Baseline GP (SE Kernel)', 'fname', 'LV_Periodic_Baseline_GP.eps');
panels(2) = struct('prey', per_prey, 'pred', per_pred, ...
    'title', 'Periodic Kernel GP', 'fname', 'LV_Periodic_Periodic_GP.eps');

ax_list = gobjects(2, 1);
fig_list = gobjects(2, 1);
for pidx = 1:2
    fig_list(pidx) = figure('Color', 'w', 'Position', [80 + 40*(pidx-1), 80, 640, 480], ...
        'Name', panels(pidx).title);
    ax = axes('Parent', fig_list(pidx));
    ax.Layer = 'top';
    hold(ax, 'on'); grid(ax, 'on');
    plot_state(ax, x_grid, y_true_grid(:, 1), panels(pidx).prey, ...
        x_train, y_train_prey, col_prey, k_plot, 'Prey', band_label);
    plot_state(ax, x_grid, y_true_grid(:, 2), panels(pidx).pred, ...
        x_train, y_train_pred, col_pred, k_plot, 'Predator', band_label);
    xlabel(ax, 't');
    ylabel(ax, 'Population');
    title(ax, panels(pidx).title, 'Interpreter', 'none');
    xlim(ax, [t_min, t_max]);
    ylim(ax, ylim_shared);
    ax_list(pidx) = ax;
end

%% Standalone shared legend (for LaTeX / Inkscape)
figL = figure('Color', 'w', 'Position', [100, 100, 900, 80], ...
    'Name', 'LV Periodic shared legend');
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

% %% Save each panel and the shared legend as EPS
% plot_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
%     'results', 'plots', 'Paper Draft 2', 'Interacting Species');
% if ~exist(plot_dir, 'dir')
%     mkdir(plot_dir);
% end
% for i = 1:numel(ax_list)
%     ax_list(i).Toolbar.Visible = 'off';
%     disableDefaultInteractivity(ax_list(i));
%     drawnow;
%     out_path = fullfile(plot_dir, panels(i).fname);
%     exportgraphics(ax_list(i), out_path, 'ContentType', 'vector');
%     fprintf('Saved %s\n', out_path);
% end
% legend_path = fullfile(plot_dir, 'LV_Periodic_legend.eps');
% exportgraphics(figL, legend_path, 'ContentType', 'vector', 'BackgroundColor', 'white');
% fprintf('Saved %s\n', legend_path);

%% Report
fprintf('\n--- Fitted hyperparameters ---\n');
report('Baseline  Prey', se_prey, false);
report('Baseline  Pred', se_pred, false);
report('Periodic  Prey', per_prey, true);
report('Periodic  Pred', per_pred, true);

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

function [m, sf, hyp, nlml] = fit_periodic_state(x, y, sn, p0, x_grid, inffunc, meanfunc, covfunc, likfunc)
% Period p is optimized jointly with ell and sf (treated exactly like ell),
% in a single minimize run from the manual init p0.
x = x(:); y = y(:);
ell0 = 1;  %or std(x), intializing to std(x) gives slightly higher ell and sf   % dimensionless roughness within one period
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

function report(label, fit, is_periodic)
if is_periodic
    fprintf('%s: ell=%.4f, p=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
        label, exp(fit.hyp.cov(1)), exp(fit.hyp.cov(2)), exp(fit.hyp.cov(3)), ...
        exp(fit.hyp.lik), fit.nlml);
else
    fprintf('%s: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
        label, exp(fit.hyp.cov(1)), exp(fit.hyp.cov(2)), exp(fit.hyp.lik), fit.nlml);
end
end
