% Paper figure: Lotka-Volterra GP comparison (SE vs periodic vs quasi-periodic).
% Prey and predator are fit independently with each kernel, each with a
% constant mean (meanConst). Quasi-periodic kernel is the product
% covPeriodic * covSEiso, so oscillations can slowly decohere.
% Mean and covariance hyps are optimized jointly; sn is held at known noise.
% Tabbed figure (10 tabs: SE / periodic / quasi-periodic x prey / predator
% on [0, 30], then SE and QP extrapolation to t=60). Pensoneault lower-bound
% tabs are commented out.
% Requires the GPML toolbox.
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

%% Ground truth (dense ode45 solve, ~3 cycles), then shifted off zero
t_min = 0;
t_max = 30;
x_grid = linspace(t_min, t_max, 600)';
[~, z_grid] = ode45(odefun, x_grid, z0);
z_nonneg = max(z_grid, 0);   % [n_grid x 2] -> columns [prey, predator]
pop_offset = 8;              % additive shift so troughs sit well above 0
y_true_grid = z_nonneg + pop_offset;
fprintf('Population offset: +%.3g on both states (troughs lifted off 0)\n', pop_offset);

%% Training data (3 replicates per time; homoscedastic Gaussian noise)
rng(100);
n_times = 7;
nRep = 3;
%t_obs = linspace(t_min, t_max, n_times)';   % shared sample times for both states
t_obs = [0;5;10;12;16;28;30]';
%x_train = [10; 30; 60; 90; 200];


y_true_obs = interp1(x_grid, y_true_grid, t_obs, 'pchip');
y_true_obs = max(y_true_obs, 0);

noise_frac = 0.2;   % homoscedastic: sigma_n = noise_frac * SD of true curve
sd_true_prey = std(y_true_grid(:, 1));
sd_true_pred = std(y_true_grid(:, 2));
sn_prey = noise_frac * sd_true_prey;   % y ~ N(y_true, sigma_n^2)
sn_pred = noise_frac * sd_true_pred;

% Each unique t appears nRep times; independent draws, same sigma per state.
% Truncated-normal: redraw if y < 0. GP likelihood is unchanged.
n_train = n_times * nRep;
x_train = repelem(t_obs, nRep);
y_true_train = repelem(y_true_obs, nRep, 1);
y_train_prey = y_true_train(:, 1) + sn_prey * randn(n_train, 1);
y_train_pred = y_true_train(:, 2) + sn_pred * randn(n_train, 1);
n_reject = 0;
bad_prey = y_train_prey < 0;
bad_pred = y_train_pred < 0;
while any(bad_prey) || any(bad_pred)
    n_reject = n_reject + sum(bad_prey) + sum(bad_pred);
    if any(bad_prey)
        y_train_prey(bad_prey) = y_true_train(bad_prey, 1) + sn_prey * randn(sum(bad_prey), 1);
    end
    if any(bad_pred)
        y_train_pred(bad_pred) = y_true_train(bad_pred, 2) + sn_pred * randn(sum(bad_pred), 1);
    end
    bad_prey = y_train_prey < 0;
    bad_pred = y_train_pred < 0;
end

fprintf('Synthetic LV data: nTimes=%d, nRep=%d (%d obs per state) on [%.0f, %.0f]\n', ...
    n_times, nRep, n_train, t_min, t_max);
fprintf(['Homoscedastic noise: sigma_prey=%.4f (%.0f%% of SD of true prey curve = %.4f), ', ...
    'sigma_pred=%.4f (%.0f%% of SD of true pred curve = %.4f)\n'], ...
    sn_prey, 100 * noise_frac, sd_true_prey, sn_pred, 100 * noise_frac, sd_true_pred);
if n_reject > 0
    fprintf('Redrawn %d Gaussian draw(s) with y < 0 (truncated at population 0).\n', n_reject);
end

%% Period initialization (manual)
% p0 rounds the analytic small-oscillation period 2*pi/sqrt(alpha*gamma) ~ 9.47
% and matches the observed ~3 cycles over [0, 30] (period ~ 30/3 = 10).
% Same init for both states (shared orbit). Optimized jointly with the
% quasi-periodic hyps (periodic ell/sf and SE decoherence lengthscale) below.
p0 = 10;
fprintf('Periodic / quasi-periodic kernels: manual period init p0 = %.3f\n', p0);

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

meanfunc = @meanConst;   % constant offset per state (optimized with cov hyps)
covSE    = @covSEiso;
covPer   = @covPeriodic;
covQP    = {@covProd, {@covPeriodic, @covSEiso}};   % quasi-periodic: Per * SE
likfunc  = @likGauss;
inffunc  = @infGaussLik;

%% Fit SE, periodic, and quasi-periodic GPs per state (sn fixed at known noise)
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

fprintf('\n=== Quasi-periodic kernel (Periodic x SE) ===\n');
[qp_prey.m, qp_prey.sf, qp_prey.hyp, qp_prey.nlml] = fit_qp_state( ...
    x_train, y_train_prey, sn_prey, p0, x_grid, inffunc, meanfunc, covQP, likfunc);
[qp_pred.m, qp_pred.sf, qp_pred.hyp, qp_pred.nlml] = fit_qp_state( ...
    x_train, y_train_pred, sn_pred, p0, x_grid, inffunc, meanfunc, covQP, likfunc);

%% Extrapolation grid (same fitted hyps, predict to t=60)
t_max_ext = 60;
x_grid_ext = linspace(t_min, t_max_ext, 1200)';
[~, z_grid_ext] = ode45(odefun, x_grid_ext, z0);
y_true_ext = max(z_grid_ext, 0) + pop_offset;
fprintf('\n=== Extrapolation to t = %.0f (SE and QP, fitted hyps unchanged) ===\n', t_max_ext);
se_prey_ext = pred_from_hyp(se_prey, x_train, y_train_prey, x_grid_ext, ...
    inffunc, meanfunc, covSE, likfunc);
se_pred_ext = pred_from_hyp(se_pred, x_train, y_train_pred, x_grid_ext, ...
    inffunc, meanfunc, covSE, likfunc);
qp_prey_ext = pred_from_hyp(qp_prey, x_train, y_train_prey, x_grid_ext, ...
    inffunc, meanfunc, covQP, likfunc);
qp_pred_ext = pred_from_hyp(qp_pred, x_train, y_train_pred, x_grid_ext, ...
    inffunc, meanfunc, covQP, likfunc);

%% Pensoneault lower bound + data-fidelity tube on periodic kernel only
% eta = 0.022;   % 2.2% tail probability
% k_pens = -sqrt(2) * erfinv(2 * eta - 1);
% n_constraint = 120;
% X_c = linspace(t_min, t_max, n_constraint)';
% eps_prey = 2 * sn_prey;   % |y - y*(x)| <= epsilon at training points
% eps_pred = 2 * sn_pred;
% nTry = 2000;
% nMultistart = 10;
% opts_pens = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
%     'EnableFeasibilityMode', true, 'Display', 'off', ...
%     'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
%     'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);
%
% fprintf('\n=== Pensoneault lower bound + data tube, periodic kernel ===\n');
% fprintf('eta = %.3g%% | k = %.4f | X_c: %d points on [%.0f, %.0f]\n', ...
%     100 * eta, k_pens, n_constraint, t_min, t_max);
% fprintf('epsilon_prey = %.4f (2*sn) | epsilon_pred = %.4f (2*sn)\n', eps_prey, eps_pred);
%
% [lb_per_prey, ub_per_prey] = bound_box_per(y_train_prey, t_min, t_max, p0);
% [lb_per_pred, ub_per_pred] = bound_box_per(y_train_pred, t_min, t_max, p0);
%
% fprintf('\n--- Periodic + lower bound + data tube ---\n');
% per_prey_b = fit_bound_state(x_train, y_train_prey, per_prey.hyp, X_c, k_pens, eps_prey, x_grid, ...
%     inffunc, meanfunc, covPer, likfunc, lb_per_prey, ub_per_prey, opts_pens, nTry, nMultistart, 46);
% per_pred_b = fit_bound_state(x_train, y_train_pred, per_pred.hyp, X_c, k_pens, eps_pred, x_grid, ...
%     inffunc, meanfunc, covPer, likfunc, lb_per_pred, ub_per_pred, opts_pens, nTry, nMultistart, 47);

%% Tabbed figure (one tab per method × state)
k_plot = 2;
band_label = sprintf('\\pm %g\\sigma_f', k_plot);
col_prey = [0.00, 0.45, 0.74];   % blue
col_pred = [0.85, 0.16, 0.16];   % red

ylo_prey = min([0; y_train_prey(:); ...
    se_prey.m - k_plot * se_prey.sf; per_prey.m - k_plot * per_prey.sf; ...
    qp_prey.m - k_plot * qp_prey.sf]);
yhi_prey = max([y_train_prey(:); ...
    se_prey.m + k_plot * se_prey.sf; per_prey.m + k_plot * per_prey.sf; ...
    qp_prey.m + k_plot * qp_prey.sf]);
pad_prey = 0.05 * (yhi_prey - ylo_prey);
ylim_prey = [ylo_prey - pad_prey, yhi_prey + pad_prey];

ylo_pred = min([0; y_train_pred(:); ...
    se_pred.m - k_plot * se_pred.sf; per_pred.m - k_plot * per_pred.sf; ...
    qp_pred.m - k_plot * qp_pred.sf]);
yhi_pred = max([y_train_pred(:); ...
    se_pred.m + k_plot * se_pred.sf; per_pred.m + k_plot * per_pred.sf; ...
    qp_pred.m + k_plot * qp_pred.sf]);
pad_pred = 0.05 * (yhi_pred - ylo_pred);
ylim_pred = [ylo_pred - pad_pred, yhi_pred + pad_pred];

ylo_prey_ext = min([0; y_train_prey(:); y_true_ext(:, 1); ...
    se_prey_ext.m - k_plot * se_prey_ext.sf; qp_prey_ext.m - k_plot * qp_prey_ext.sf]);
yhi_prey_ext = max([y_train_prey(:); y_true_ext(:, 1); ...
    se_prey_ext.m + k_plot * se_prey_ext.sf; qp_prey_ext.m + k_plot * qp_prey_ext.sf]);
pad_prey_ext = 0.05 * (yhi_prey_ext - ylo_prey_ext);
ylim_prey_ext = [ylo_prey_ext - pad_prey_ext, yhi_prey_ext + pad_prey_ext];

ylo_pred_ext = min([0; y_train_pred(:); y_true_ext(:, 2); ...
    se_pred_ext.m - k_plot * se_pred_ext.sf; qp_pred_ext.m - k_plot * qp_pred_ext.sf]);
yhi_pred_ext = max([y_train_pred(:); y_true_ext(:, 2); ...
    se_pred_ext.m + k_plot * se_pred_ext.sf; qp_pred_ext.m + k_plot * qp_pred_ext.sf]);
pad_pred_ext = 0.05 * (yhi_pred_ext - ylo_pred_ext);
ylim_pred_ext = [ylo_pred_ext - pad_pred_ext, yhi_pred_ext + pad_pred_ext];

panels(1) = struct('fit', se_prey, 'x', x_grid, 'y_true', y_true_grid(:, 1), ...
    'x_obs', x_train, 'y_obs', y_train_prey, 'col', col_prey, ...
    'xlim', [t_min, t_max], 'ylim', ylim_prey, 'state', 'Prey', ...
    'show_zero', false, 'mark_train_end', false, ...
    'tab', 'SE Prey', 'title', 'Baseline GP (SE Kernel) — Prey', ...
    'fname', 'LV_Periodic_Baseline_Prey.eps');
panels(2) = struct('fit', se_pred, 'x', x_grid, 'y_true', y_true_grid(:, 2), ...
    'x_obs', x_train, 'y_obs', y_train_pred, 'col', col_pred, ...
    'xlim', [t_min, t_max], 'ylim', ylim_pred, 'state', 'Predator', ...
    'show_zero', false, 'mark_train_end', false, ...
    'tab', 'SE Pred', 'title', 'Baseline GP (SE Kernel) — Predator', ...
    'fname', 'LV_Periodic_Baseline_Predator.eps');
panels(3) = struct('fit', per_prey, 'x', x_grid, 'y_true', y_true_grid(:, 1), ...
    'x_obs', x_train, 'y_obs', y_train_prey, 'col', col_prey, ...
    'xlim', [t_min, t_max], 'ylim', ylim_prey, 'state', 'Prey', ...
    'show_zero', false, 'mark_train_end', false, ...
    'tab', 'Per Prey', 'title', 'Periodic Kernel GP — Prey', ...
    'fname', 'LV_Periodic_Periodic_Prey.eps');
panels(4) = struct('fit', per_pred, 'x', x_grid, 'y_true', y_true_grid(:, 2), ...
    'x_obs', x_train, 'y_obs', y_train_pred, 'col', col_pred, ...
    'xlim', [t_min, t_max], 'ylim', ylim_pred, 'state', 'Predator', ...
    'show_zero', false, 'mark_train_end', false, ...
    'tab', 'Per Pred', 'title', 'Periodic Kernel GP — Predator', ...
    'fname', 'LV_Periodic_Periodic_Predator.eps');
panels(5) = struct('fit', qp_prey, 'x', x_grid, 'y_true', y_true_grid(:, 1), ...
    'x_obs', x_train, 'y_obs', y_train_prey, 'col', col_prey, ...
    'xlim', [t_min, t_max], 'ylim', ylim_prey, 'state', 'Prey', ...
    'show_zero', false, 'mark_train_end', false, ...
    'tab', 'QP Prey', 'title', 'Quasi-Periodic Kernel GP — Prey', ...
    'fname', 'LV_Periodic_QuasiPeriodic_Prey.eps');
panels(6) = struct('fit', qp_pred, 'x', x_grid, 'y_true', y_true_grid(:, 2), ...
    'x_obs', x_train, 'y_obs', y_train_pred, 'col', col_pred, ...
    'xlim', [t_min, t_max], 'ylim', ylim_pred, 'state', 'Predator', ...
    'show_zero', false, 'mark_train_end', false, ...
    'tab', 'QP Pred', 'title', 'Quasi-Periodic Kernel GP — Predator', ...
    'fname', 'LV_Periodic_QuasiPeriodic_Predator.eps');
% panels(7) = struct('fit', per_prey_b, 'x', x_grid, 'y_true', y_true_grid(:, 1), ...
%     'x_obs', x_train, 'y_obs', y_train_prey, 'col', col_prey, ...
%     'xlim', [t_min, t_max], 'ylim', ylim_prey, 'state', 'Prey', ...
%     'show_zero', true, 'mark_train_end', false, ...
%     'tab', 'Per Bound Prey', 'title', 'Periodic Kernel + Lower Bound + Tube — Prey', ...
%     'fname', 'LV_Periodic_Periodic_Bound_Prey.eps');
% panels(8) = struct('fit', per_pred_b, 'x', x_grid, 'y_true', y_true_grid(:, 2), ...
%     'x_obs', x_train, 'y_obs', y_train_pred, 'col', col_pred, ...
%     'xlim', [t_min, t_max], 'ylim', ylim_pred, 'state', 'Predator', ...
%     'show_zero', true, 'mark_train_end', false, ...
%     'tab', 'Per Bound Pred', 'title', 'Periodic Kernel + Lower Bound + Tube — Predator', ...
%     'fname', 'LV_Periodic_Periodic_Bound_Predator.eps');
panels(7) = struct('fit', se_prey_ext, 'x', x_grid_ext, 'y_true', y_true_ext(:, 1), ...
    'x_obs', x_train, 'y_obs', y_train_prey, 'col', col_prey, ...
    'xlim', [t_min, t_max_ext], 'ylim', ylim_prey_ext, 'state', 'Prey', ...
    'show_zero', false, 'mark_train_end', true, ...
    'tab', 'SE Prey extra', 'title', 'Baseline GP (SE Kernel) — Prey, t=60', ...
    'fname', 'LV_Periodic_Baseline_Prey_t60.eps');
panels(8) = struct('fit', se_pred_ext, 'x', x_grid_ext, 'y_true', y_true_ext(:, 2), ...
    'x_obs', x_train, 'y_obs', y_train_pred, 'col', col_pred, ...
    'xlim', [t_min, t_max_ext], 'ylim', ylim_pred_ext, 'state', 'Predator', ...
    'show_zero', false, 'mark_train_end', true, ...
    'tab', 'SE Pred extra', 'title', 'Baseline GP (SE Kernel) — Predator, t=60', ...
    'fname', 'LV_Periodic_Baseline_Predator_t60.eps');
panels(9) = struct('fit', qp_prey_ext, 'x', x_grid_ext, 'y_true', y_true_ext(:, 1), ...
    'x_obs', x_train, 'y_obs', y_train_prey, 'col', col_prey, ...
    'xlim', [t_min, t_max_ext], 'ylim', ylim_prey_ext, 'state', 'Prey', ...
    'show_zero', false, 'mark_train_end', true, ...
    'tab', 'QP Prey extra', 'title', 'Quasi-Periodic Kernel GP — Prey, t=60', ...
    'fname', 'LV_Periodic_QuasiPeriodic_Prey_t60.eps');
panels(10) = struct('fit', qp_pred_ext, 'x', x_grid_ext, 'y_true', y_true_ext(:, 2), ...
    'x_obs', x_train, 'y_obs', y_train_pred, 'col', col_pred, ...
    'xlim', [t_min, t_max_ext], 'ylim', ylim_pred_ext, 'state', 'Predator', ...
    'show_zero', false, 'mark_train_end', true, ...
    'tab', 'QP Pred extra', 'title', 'Quasi-Periodic Kernel GP — Predator, t=60', ...
    'fname', 'LV_Periodic_QuasiPeriodic_Predator_t60.eps');

fig = figure('Color', 'w', 'Position', [80, 80, 900, 560], ...
    'Name', 'Lotka-Volterra GP: SE vs periodic vs quasi-periodic');
tg = uitabgroup(fig);
ax_list = gobjects(numel(panels), 1);
tab_list = gobjects(numel(panels), 1);
for pidx = 1:numel(panels)
    tab_list(pidx) = uitab(tg, 'Title', panels(pidx).tab);
    ax = axes('Parent', tab_list(pidx));
    ax.Layer = 'top';
    ax.FontSize = 16;
    hold(ax, 'on'); grid(ax, 'on');
    plot_state(ax, panels(pidx).x, panels(pidx).y_true, panels(pidx).fit, ...
        panels(pidx).x_obs, panels(pidx).y_obs, panels(pidx).col, ...
        k_plot, panels(pidx).state, band_label);
    if panels(pidx).show_zero
        yline(ax, 0, 'k:', 'HandleVisibility', 'off');
    end
    if panels(pidx).mark_train_end
        xline(ax, t_max, 'k--', 'HandleVisibility', 'off');
    end
    xlabel(ax, 't', 'FontSize', 16);
    ylabel(ax, 'Population', 'FontSize', 16);
    title(ax, panels(pidx).title, 'Interpreter', 'none', 'FontSize', 16);
    xlim(ax, panels(pidx).xlim);
    ylim(ax, panels(pidx).ylim);
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
lgd.FontSize = 16;
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
%     exportgraphics(ax_list(i), out_path, 'ContentType', 'image'); %maybe 'vector' instead of 'image? Ask Andrea
%     fprintf('Saved %s\n', out_path);
% end
% legend_path = fullfile(plot_dir, 'LV_Periodic_legend.eps');
% exportgraphics(figL, legend_path, 'ContentType', 'image', 'BackgroundColor', 'white');
% fprintf('Saved %s\n', legend_path);

%% Console report
fprintf('\n--- Fitted hyperparameters ---\n');
report('Baseline   Prey', se_prey);
report('Baseline   Pred', se_pred);
report('Periodic   Prey', per_prey);
report('Periodic   Pred', per_pred);
report('Quasi-per  Prey', qp_prey);
report('Quasi-per  Pred', qp_pred);
% report('Per+bound  Prey', per_prey_b);
% report('Per+bound  Pred', per_pred_b);

%% ----- local functions -----
function fit = pred_from_hyp(fit_in, x, y, xstar, inffunc, meanfunc, covfunc, likfunc)
% Posterior mean / sd of an already-fitted GP on a new grid (hyps unchanged).
[~, ~, fmu, fs2] = gp(fit_in.hyp, inffunc, meanfunc, covfunc, likfunc, x(:), y(:), xstar(:));
fit = fit_in;
fit.m = fmu(:);
fit.sf = sqrt(max(fs2(:), 0));
end

function [m, sf, hyp, nlml] = fit_se_state(x, y, sn, x_grid, inffunc, meanfunc, covfunc, likfunc)
x = x(:); y = y(:);
c0   = mean(y);   % meanConst hyp is the offset itself (not log)
ell0 = std(x);
sf0  = std(y);
sn_fixed = log(sn);
n_mean = 1;
hyp_mc0 = [c0; log([ell0; sf0])];
hyp_mc = minimize(hyp_mc0, @gp_nlml_mean_cov_only, -100, sn_fixed, n_mean, ...
    inffunc, meanfunc, covfunc, likfunc, x, y);
hyp = struct('mean', hyp_mc(1:n_mean), 'cov', hyp_mc(n_mean+1:end), 'lik', sn_fixed);
nlml = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, x_grid(:));
m = fmu(:);
sf = sqrt(max(fs2(:), 0));
end

function [m, sf, hyp, nlml] = fit_periodic_state(x, y, sn, p0, x_grid, inffunc, meanfunc, covfunc, likfunc)
% Pure periodic: covPeriodic, plus a constant mean.
% hyp.cov = [log(ell); log(p); log(sf)].
x = x(:); y = y(:);
c0   = mean(y);
ell0 = 1;                 % dimensionless roughness within one period
sf0  = std(y);
sn_fixed = log(sn);
n_mean = 1;
hyp_mc0 = [c0; log([ell0; p0; sf0])];
hyp_mc = minimize(hyp_mc0, @gp_nlml_mean_cov_only, -100, sn_fixed, n_mean, ...
    inffunc, meanfunc, covfunc, likfunc, x, y);
hyp = struct('mean', hyp_mc(1:n_mean), 'cov', hyp_mc(n_mean+1:end), 'lik', sn_fixed);
nlml = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, x_grid(:));
m = fmu(:);
sf = sqrt(max(fs2(:), 0));
end

function [m, sf, hyp, nlml] = fit_qp_state(x, y, sn, p0, x_grid, inffunc, meanfunc, covfunc, likfunc)
% Quasi-periodic: covPeriodic * covSEiso, plus a constant mean.
% hyp.cov = [log(ell_per); log(p); log(sf_per); log(ell_se); log(sf_se)].
% ell_se is the decoherence timescale (long => nearly purely periodic).
% Constant c, period p, both lengthscales, and both signal variances are
% optimized jointly (sn fixed) from inits c0 = mean(y) and the manual p0.
x = x(:); y = y(:);
c0       = mean(y);
ell_per0 = 1;                 % dimensionless roughness within one period
sf_per0  = std(y);
ell_se0  = max(x) - min(x);   % coherence over the observation window
if ell_se0 < eps, ell_se0 = p0; end
sf_se0   = 1;                 % overall amplitude lives mainly in sf_per
sn_fixed = log(sn);
n_mean = 1;
hyp_mc0 = [c0; log([ell_per0; p0; sf_per0; ell_se0; sf_se0])];
hyp_mc = minimize(hyp_mc0, @gp_nlml_mean_cov_only, -100, sn_fixed, n_mean, ...
    inffunc, meanfunc, covfunc, likfunc, x, y);
hyp = struct('mean', hyp_mc(1:n_mean), 'cov', hyp_mc(n_mean+1:end), 'lik', sn_fixed);
nlml = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, x_grid(:));
m = fmu(:);
sf = sqrt(max(fs2(:), 0));
end

function [nlml, dnlml] = gp_nlml_mean_cov_only(hyp_mc, sn_fixed, n_mean, varargin)
% GPML NLML with fixed likGauss noise; optimize meanConst offset + cov hyps.
hyp = struct('mean', hyp_mc(1:n_mean), 'cov', hyp_mc(n_mean+1:end), 'lik', sn_fixed);
[nlml, dnlml_s] = gp(hyp, varargin{:});
if nargout > 1
    dnlml = [dnlml_s.mean(:); dnlml_s.cov(:)];
end
end

function fit = fit_bound_state(x, y, hyp_unc, X_c, k, epsilon, x_grid, inffunc, meanfunc, covfunc, likfunc, ...
    hyp_lb, hyp_ub, opts, nTry, nMultistart, rng_seed)
% Pensoneault GP: min NLML s.t. mu_f - k*sigma_f >= 0 at X_c and
% |y - y*(x)| <= epsilon at training points. sn held at unconstrained hyp.lik.
x = x(:); y = y(:);
sn_fixed = hyp_unc.lik;
hyp_tpl = struct('mean', hyp_unc.mean(:), 'cov', hyp_unc.cov(:), 'lik', sn_fixed);
theta_unc = [hyp_unc.mean(:); hyp_unc.cov(:)];
n_theta = numel(theta_unc);
if numel(hyp_lb) ~= n_theta || numel(hyp_ub) ~= n_theta
    error('bound_box length (%d / %d) must match theta (%d).', ...
        numel(hyp_lb), numel(hyp_ub), n_theta);
end

objfun = @(theta) gp(theta_to_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x, y);
nonlcon = @(theta) pens_constraints_lower(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k, epsilon);

theta_unc_box = min(max(theta_unc, hyp_lb), hyp_ub);
fprintf('  Multistart: %d random starts\n', nTry);
feasible_starts = zeros(n_theta, 0);
best_feas_nlml = inf;
best_feas_theta = nan(n_theta, 1);
rng(rng_seed);
for t = 1:nTry
    theta_try = hyp_lb + rand(n_theta, 1) .* (hyp_ub - hyp_lb);
    try
        [c_try, ~] = nonlcon(theta_try);
        is_feas = all(isfinite(c_try)) && max(c_try) <= 0;
    catch
        is_feas = false;
    end
    if is_feas
        feasible_starts = [feasible_starts, theta_try]; %#ok<AGROW>
        nlml_try = objfun(theta_try);
        if nlml_try < best_feas_nlml
            best_feas_nlml = nlml_try;
            best_feas_theta = theta_try;
        end
    end
end
nFeas = size(feasible_starts, 2);
fprintf('  Feasible random starts: %d / %d\n', nFeas, nTry);
if nFeas > 0
    nlml_feas = arrayfun(@(j) objfun(feasible_starts(:, j)), 1:nFeas);
    [~, ord] = sort(nlml_feas, 'ascend');
    starts_for_fmincon = feasible_starts(:, ord(1:min(nMultistart, nFeas)));
else
    fprintf('  No feasible random start; using projected unconstrained theta.\n');
    starts_for_fmincon = theta_unc_box;
end
starts_for_fmincon = [theta_unc_box, starts_for_fmincon];
starts_for_fmincon = starts_for_fmincon(:, 1:min(nMultistart + 1, size(starts_for_fmincon, 2)));

best_nlml = inf;
theta_opt = nan(n_theta, 1);
nlml = nan;
exitflag = -99;
nStarts = size(starts_for_fmincon, 2);
for j = 1:nStarts
    theta0_j = starts_for_fmincon(:, j);
    [theta_j, nlml_j, ef_j] = fmincon(objfun, theta0_j, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts);
    if isfinite(nlml_j) && nlml_j < best_nlml
        best_nlml = nlml_j;
        theta_opt = theta_j;
        nlml = nlml_j;
        exitflag = ef_j;
    end
end
if ~isfinite(best_nlml)
    if nFeas > 0
        theta_opt = best_feas_theta;
    else
        theta_opt = theta_unc_box;
    end
    nlml = objfun(theta_opt);
    exitflag = -99;
    fprintf('  Warning: no successful fmincon run; using fallback theta.\n');
end

hyp = theta_to_hyp(theta_opt, hyp_tpl);
[c_final, ~] = nonlcon(theta_opt);
nC = numel(X_c);
max_c = max(c_final);
max_c_lower = max(c_final(1:nC));
max_c_data = max(c_final(nC+1:end));
fprintf('  exitflag=%d | max(c)=%.6g (feasible if <= 0)\n', exitflag, max_c);
fprintf('    lower max(c)=%.6g | data-tube max(c)=%.6g (epsilon=%.4g)\n', ...
    max_c_lower, max_c_data, epsilon);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, x_grid(:));
fit.m = fmu(:);
fit.sf = sqrt(max(fs2(:), 0));
fit.hyp = hyp;
fit.nlml = nlml;
fit.exitflag = exitflag;
fit.max_c = max_c;
fit.max_c_lower = max_c_lower;
fit.max_c_data = max_c_data;
end

function hyp = theta_to_hyp(theta, hyp_tpl)
hyp = hyp_tpl;
n_mean = numel(hyp_tpl.mean);
hyp.mean = theta(1:n_mean);
hyp.cov = theta(n_mean+1:end);
hyp.lik = hyp_tpl.lik;
end

function [c, ceq] = pens_constraints_lower(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k, epsilon)
% mu_f - k*sigma_f >= 0 at X_c, and |y - y*(x)| <= epsilon at training x.
% c <= 0 is feasible.
hyp = theta_to_hyp(theta, hyp_tpl);
nC = numel(X_c);
xstar = [X_c(:); x(:)];
[ymu, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xstar);
m_xc = fmu(1:nC);
s_xc = sqrt(max(fs2(1:nC), 0));
c_lower = k .* s_xc - m_xc;
y_star = ymu(nC+1:end);
c_data = abs(y(:) - y_star) - epsilon;
c = [c_lower(:); c_data(:)];
ceq = [];
end

function [lb, ub] = bound_box_se(y, t_min, t_max)
c_hi = max(2 * max(abs(y(:))), 1);
sf_hi = max(15, 2 * std(y(:)));
ell_hi = max(t_max - t_min, 1);
lb = [0; log(0.05); log(0.05)];
ub = [c_hi; log(ell_hi); log(sf_hi)];
end

function [lb, ub] = bound_box_per(y, t_min, t_max, p0)
% theta = [c; log(ell); log(p); log(sf)] for covPeriodic + meanConst.
c_hi = max(2 * max(abs(y(:))), 1);
sf_hi = max(15, 2 * std(y(:)));
p_lo = max(p0 / 4, 2);
lb = [0; log(0.05); log(p_lo); log(0.05)];
ub = [c_hi; log(10); log(max(t_max, p0)); log(sf_hi)];
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

function report(label, fit)
c = fit.hyp.mean(1);
n_cov = numel(fit.hyp.cov);
if n_cov == 5
    fprintf(['%s: c=%.4f, ell_per=%.4f, p=%.4f, sf_per=%.4f, ', ...
        'ell_se=%.4f, sf_se=%.4f, sn=%.4f | NLML=%.4f'], ...
        label, c, exp(fit.hyp.cov(1)), exp(fit.hyp.cov(2)), exp(fit.hyp.cov(3)), ...
        exp(fit.hyp.cov(4)), exp(fit.hyp.cov(5)), exp(fit.hyp.lik), fit.nlml);
elseif n_cov == 3
    fprintf('%s: c=%.4f, ell=%.4f, p=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f', ...
        label, c, exp(fit.hyp.cov(1)), exp(fit.hyp.cov(2)), exp(fit.hyp.cov(3)), ...
        exp(fit.hyp.lik), fit.nlml);
else
    fprintf('%s: c=%.4f, ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f', ...
        label, c, exp(fit.hyp.cov(1)), exp(fit.hyp.cov(2)), exp(fit.hyp.lik), fit.nlml);
end
if isfield(fit, 'max_c')
    fprintf(' | exitflag=%d | max(c)=%.4g', fit.exitflag, fit.max_c);
    if isfield(fit, 'max_c_data')
        fprintf(' | lower=%.4g | tube=%.4g', fit.max_c_lower, fit.max_c_data);
    end
end
fprintf('\n');
end
