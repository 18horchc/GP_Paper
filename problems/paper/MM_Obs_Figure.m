% Paper figure: Michaelis-Menten GP — baseline SE vs virtual anchors / deriv obs.
% Dataset: training [S] with synthetic homoscedastic noise (noise_sd_true = data-generating).
% Noise best practice: fix sigma_data at noise_sd_true; VO use fixed heteroscedastic
% sigma_VO; Solak soft Gaussian deriv targets use separate fixed sigma_deriv.
% Only ell, sf are optimized (cov hyperparameters).
% Tab 1: unconstrained SE-GP.
% Tab 2: virtual function-value anchors (hetero VO noise).
% Tab 3: Solak virtual derivative obs (fixed sn_deriv).
% Tab 4: VO + Solak deriv obs (hetero VO + fixed sn_deriv).
clear; clc; close all;

%% MM parameters
Vmax = 6;
Km   = .15;
mm_static = @(S) (Vmax .* S) ./ (Km + S);

%% Training data ([S] in mM, v_0 in μM/s)
x_max = 2;
noise_frac = 0.05;   % data-generating only: sigma = noise_frac * max v on [0, x_max]
x_train = [0.1; 0.3; 0.6; 0.9; 2];
n_train = numel(x_train);

rng(100);
v_true_at_train = mm_static(x_train);
y_domain_max = mm_static(x_max);
noise_sd_true = noise_frac * y_domain_max;
y_train = v_true_at_train + noise_sd_true * randn(size(v_true_at_train));
fprintf('Synthetic data: n=%d, data-generating sigma_n = %.4f (%.0f%% of v(x_max))\n', ...
    n_train, noise_sd_true, 100 * noise_frac);

%% Fixed observation noises (editable; not optimized)
sigma_data = noise_sd_true;                 % known assay noise
sigma_VO_zero = 0.05; %0.2 * noise_sd_true;       % tight soft-hard anchor at v(0)=0
sigma_VO_sat = 0.4; %0.20 * Vmax;                 % soft saturation target (units of v)
sn_deriv = 0.4; %0.2;                             % Solak soft Gaussian derivative noise

%% Virtual function-value observations (heteroscedastic VO)
x_obs = x_train(:);
y_obs = y_train(:);

x_virt_zero = 0;
y_virt_zero = 0;
x_virt_sat = 1.4;
y_virt_sat = 5.3;

x_aug = [x_obs; x_virt_zero; x_virt_sat];
y_aug = [y_obs; y_virt_zero; y_virt_sat];
noise_var_aug = [sigma_data^2 * ones(numel(y_obs), 1); ...
    sigma_VO_zero^2; sigma_VO_sat^2];

fprintf('Virtual obs: v(0)=0 (sigma=%.4g) | v(%.1f)=%.1f (sigma=%.4g)\n', ...
    sigma_VO_zero, x_virt_sat, y_virt_sat, sigma_VO_sat);

%% Virtual derivative observations (Solak; fixed separate sn_deriv)
%x_deriv = linspace(0, 2, 10)';          % 0, 2/9, ..., 2
%y_deriv = (27:-3:0)';                   % 27, ..., 0

x_deriv = [1; 1.4; 1.8];
y_deriv = 0.3 * ones(numel(x_deriv), 1);

fprintf('Virtual deriv obs: %d sites | y_deriv=0.3 | sn_deriv=%.4g (fixed)\n', ...
    numel(x_deriv), sn_deriv);

%% Ground truth curve
x_grid = linspace(0, x_max, 500);
y_true = mm_static(x_grid);

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

ell0 = std(x_train);
sf0  = std(y_train);
sn_fixed = log(sigma_data);
hyp0 = struct('mean', [], 'cov', log([ell0; sf0]), 'lik', sn_fixed);

meanfunc = @meanZero;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

x_col = x_train(:);
y_col = y_train(:);

%% Baseline GP (optimize ell, sf; sigma_n fixed)
fprintf('Optimizing baseline (ell, sf; sn fixed at %.4f)...\n', sigma_data);
obj_unc = @(hyp_cov) gp_nlml_cov_only(hyp_cov, sn_fixed, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
hyp_cov_unc = minimize(hyp0.cov, obj_unc, -100);
hyp_unc = struct('mean', [], 'cov', hyp_cov_unc(:), 'lik', sn_fixed);
nlml_unc = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);

%% Augmented GP with virtual function-value observations (heteroscedastic)
fprintf('\nOptimizing augmented / VO (ell, sf; heteroscedastic VO)...\n');
obj_aug = @(h) gp_seiso_hetero_noise('nlml', h, x_aug, y_aug, noise_var_aug);
hyp_aug = struct('mean', [], 'cov', hyp0.cov, 'lik', []);
hyp_aug = minimize(hyp_aug, obj_aug, -100);
nlml_aug = obj_aug(hyp_aug);

%% Solak derivative-observation GP on unaugmented data (fixed sn_deriv)
fprintf('\nOptimizing deriv-obs GP (ell, sf; sn=%.4f, sn_deriv=%.4g)...\n', ...
    sigma_data, sn_deriv);
obj_deriv = @(hyp_cov) gp_seiso_deriv_obs_nlml_cov_only(hyp_cov, sn_fixed, ...
    x_col, y_col, x_deriv, y_deriv, sn_deriv);
hyp_cov_deriv = minimize(hyp0.cov, obj_deriv, -100);
hyp_deriv = struct('mean', [], 'cov', hyp_cov_deriv(:), 'lik', sn_fixed);
nlml_deriv = obj_deriv(hyp_cov_deriv);

%% Combined: virtual function-value anchors + Solak deriv obs
fprintf('\nOptimizing VO+deriv GP (ell, sf; hetero VO + sn_deriv=%.4g)...\n', sn_deriv);
obj_both = @(hyp_cov) gp_seiso_deriv_obs_nlml_cov_only(hyp_cov, sn_fixed, ...
    x_aug, y_aug, x_deriv, y_deriv, sn_deriv, noise_var_aug);
hyp_cov_both = minimize(hyp0.cov, obj_both, -100);
hyp_both = struct('mean', [], 'cov', hyp_cov_both(:), 'lik', sn_fixed);
nlml_both = obj_both(hyp_cov_both);

%% Predict for plots
k_plot = 2;
[~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_unc = fmu_unc(:);
sf_unc = sqrt(max(fs2_unc(:), 0));

[~, ~, fmu_aug, fs2_aug] = gp_seiso_hetero_noise('pred', hyp_aug, x_aug, y_aug, noise_var_aug, x_grid(:));
m_aug = fmu_aug(:);
sf_aug = sqrt(max(fs2_aug(:), 0));

[~, ~, fmu_d, fs2_d] = gp_seiso_deriv_obs('pred', hyp_deriv, ...
    x_col, y_col, x_deriv, y_deriv, x_grid(:), sn_deriv);
m_deriv = fmu_d(:);
sf_deriv = sqrt(max(fs2_d(:), 0));

[~, ~, fmu_b, fs2_b] = gp_seiso_deriv_obs('pred', hyp_both, ...
    x_aug, y_aug, x_deriv, y_deriv, x_grid(:), sn_deriv, true, noise_var_aug);
m_both = fmu_b(:);
sf_both = sqrt(max(fs2_b(:), 0));

[m_deriv_at_xd, s2_deriv_at_xd] = gp_seiso_deriv_obs('deriv', hyp_deriv, ...
    x_col, y_col, x_deriv, y_deriv, x_deriv, sn_deriv);
mm_deriv_true = Vmax * Km ./ (Km + x_deriv).^2;

band_label = sprintf('\\mu_f \\pm %g\\sigma_f (latent)', k_plot);
ylim_shared = [0, max([y_train(:); Vmax; ...
    m_unc + k_plot * sf_unc; m_aug + k_plot * sf_aug; ...
    m_deriv + k_plot * sf_deriv; m_both + k_plot * sf_both]) * 1.02];

%% Tabbed figure
fig = figure('Color', 'w', 'Position', [80, 80, 900, 560], ...
    'Name', 'Michaelis-Menten GP: baseline, VO, deriv, VO+deriv');
tg = uitabgroup(fig);

tab_unc = uitab(tg, 'Title', 'Baseline GP');
ax1 = axes('Parent', tab_unc);
plot_mm_gp_panel(ax1, m_unc, sf_unc, false, [], ...
    x_grid, y_true, x_obs, y_obs, k_plot, band_label, ylim_shared, x_max, Vmax, ...
    x_virt_zero, y_virt_zero, x_virt_sat, y_virt_sat);
title(ax1, 'Baseline GP', 'Interpreter', 'none', 'FontSize', 16);

tab_aug = uitab(tg, 'Title', 'Virtual Obs GP');
ax2 = axes('Parent', tab_aug);
plot_mm_gp_panel(ax2, m_aug, sf_aug, true, [], ...
    x_grid, y_true, x_obs, y_obs, k_plot, band_label, ylim_shared, x_max, Vmax, ...
    x_virt_zero, y_virt_zero, x_virt_sat, y_virt_sat);
title(ax2, 'Virtual Obs GP', 'Interpreter', 'none', 'FontSize', 16);

tab_deriv = uitab(tg, 'Title', 'Virtual Deriv Obs GP');
ax3 = axes('Parent', tab_deriv);
plot_mm_gp_panel(ax3, m_deriv, sf_deriv, false, x_deriv, ...
    x_grid, y_true, x_obs, y_obs, k_plot, band_label, ylim_shared, x_max, Vmax, ...
    x_virt_zero, y_virt_zero, x_virt_sat, y_virt_sat);
title(ax3, 'Virtual Deriv Obs GP', 'Interpreter', 'none', 'FontSize', 16);

tab_both = uitab(tg, 'Title', 'Virtual + Deriv Obs GP');
ax4 = axes('Parent', tab_both);
plot_mm_gp_panel(ax4, m_both, sf_both, true, x_deriv, ...
    x_grid, y_true, x_obs, y_obs, k_plot, band_label, ylim_shared, x_max, Vmax, ...
    x_virt_zero, y_virt_zero, x_virt_sat, y_virt_sat);
title(ax4, 'Virtual + Deriv Obs GP', 'Interpreter', 'none', 'FontSize', 16);

%% Standalone legend (for LaTeX / Inkscape)
figL = figure('Color', 'w', 'Position', [100, 100, 900, 80], ...
    'Name', 'MM Obs shared legend');
axL = axes('Parent', figL, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
    'Position', [0 0 1 1]);
hold(axL, 'on');
hL = gobjects(6, 1);
hL(1) = fill(axL, nan, nan, [0.72, 0.72, 0.78], 'EdgeColor', 'none', ...
    'FaceAlpha', 0.5, 'DisplayName', '95% CI');
hL(2) = plot(axL, nan, nan, 'k--', 'LineWidth', 2, ...
    'DisplayName', 'GP Mean'); %Posterior mean
hL(3) = plot(axL, nan, nan, 'b-', 'LineWidth', 1.5, ...
    'DisplayName', 'True Model');
hL(4) = plot(axL, nan, nan, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', 'Observed Data');
hL(5) = scatter(axL, nan, nan, 90, 'd', ...
    'MarkerFaceColor', [0.85, 0.85, 0.85], 'MarkerEdgeColor', 'k', ...
    'LineWidth', 1.5, 'DisplayName', 'Virtual Obs');
hL(6) = plot(axL, nan, nan, '^', 'LineStyle', 'none', 'MarkerSize', 9, ...
    'LineWidth', 0.8, 'MarkerFaceColor', [0.55, 0.25, 0.65], ...
    'MarkerEdgeColor', 'k', 'DisplayName', 'Virtual Deriv Obs');
lgd = legend(axL, hL, 'Orientation', 'horizontal');
lgd.FontSize = 12;
lgd.ItemTokenSize = [20, 12];
lgd.Box = 'on';
drawnow;
% Shrink figure to the legend's natural size (tight box, no side padding)
figL.Units = 'pixels';
lgd.Units = 'pixels';
lp = lgd.Position;
margin = 4;
figL.Position(3:4) = [lp(3) + 2 * margin, lp(4) + 2 * margin];
lgd.Position = [margin, margin, lp(3), lp(4)];
axL.Position = [0 0 1 1];
drawnow;

%% Save each tab and the shared legend as EPS
plot_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
    'results', 'plots', 'Paper Draft 2', 'Enzyme Kinetics');
if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end
tab_list = {tab_unc, tab_aug, tab_deriv, tab_both};
ax_list  = {ax1, ax2, ax3, ax4};
name_list = {'MM_Obs_Baseline_GP.eps', 'MM_Obs_Virtual_Obs_GP.eps', ...
    'MM_Obs_Virtual_Deriv_Obs_GP.eps', 'MM_Obs_VO_plus_Deriv_Obs_GP.eps'};
for i = 1:numel(tab_list)
    tg.SelectedTab = tab_list{i};
    ax_list{i}.Toolbar.Visible = 'off';
    disableDefaultInteractivity(ax_list{i});
    drawnow;
    out_path = fullfile(plot_dir, name_list{i});
    exportgraphics(ax_list{i}, out_path, 'ContentType', 'vector');
    fprintf('Saved %s\n', out_path);
end
legend_path = fullfile(plot_dir, 'MM_Obs_legend.eps');
exportgraphics(figL, legend_path, 'ContentType', 'vector', 'BackgroundColor', 'white');
fprintf('Saved %s\n', legend_path);

fprintf('\nFixed noises: sigma_data=%.4f | sigma_VO_zero=%.4g | sigma_VO_sat=%.4g | sn_deriv=%.4g\n', ...
    sigma_data, sigma_VO_zero, sigma_VO_sat, sn_deriv);
fprintf('Baseline:      ell=%.4f, sf=%.4f, sn=%.4f (fixed) | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), sigma_data, nlml_unc);
fprintf('Augmented:     ell=%.4f, sf=%.4f | NLML=%.4f (heteroscedastic VO, n_aug=%d)\n', ...
    exp(hyp_aug.cov(1)), exp(hyp_aug.cov(2)), nlml_aug, numel(y_aug));
fprintf('Deriv obs:     ell=%.4f, sf=%.4f, sn=%.4f (fixed), sn_deriv=%.4g | NLML=%.4f\n', ...
    exp(hyp_deriv.cov(1)), exp(hyp_deriv.cov(2)), sigma_data, sn_deriv, nlml_deriv);
fprintf('VO+deriv:      ell=%.4f, sf=%.4f, sn_deriv=%.4g | NLML=%.4f (hetero VO + Solak, n_aug=%d)\n', ...
    exp(hyp_both.cov(1)), exp(hyp_both.cov(2)), sn_deriv, nlml_both, numel(y_aug));
fprintf('\nPosterior f'' at Solak derivative observation points (sn_deriv = %.3g):\n', sn_deriv);
fprintf('  [S]    target    post mean    post sd    MM analytic\n');
for j = 1:numel(x_deriv)
    fprintf('  %4.2f   %6.3f    %8.4f    %8.4f    %8.4f\n', ...
        x_deriv(j), y_deriv(j), m_deriv_at_xd(j), sqrt(s2_deriv_at_xd(j)), mm_deriv_true(j));
end

function plot_mm_gp_panel(ax, m, sf, show_virt, x_deriv, ...
    x_grid, y_true, x_obs, y_obs, k_plot, band_label, ylim_shared, x_max, Vmax, ...
    x_virt_zero, y_virt_zero, x_virt_sat, y_virt_sat)
ax.Layer = 'top';
hold(ax, 'on');
grid(ax, 'on');
fill(ax, [x_grid, fliplr(x_grid)], [m + k_plot * sf; flipud(m - k_plot * sf)]', ...
    [0.72, 0.72, 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
plot(ax, x_grid, m, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
plot(ax, x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
plot(ax, x_obs, y_obs, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', 'Observed data');
if show_virt
    scatter(ax, [x_virt_zero(:); x_virt_sat(:)], [y_virt_zero(:); y_virt_sat(:)], 90, 'd', ...
        'MarkerFaceColor', [0.85, 0.85, 0.85], 'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.5, 'DisplayName', 'Virtual observations');
end
yline(ax, Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
xlim(ax, [0, x_max]);
ylim(ax, ylim_shared);
if ~isempty(x_deriv)
    x_mark = x_deriv(isfinite(x_deriv) & x_deriv >= 0 & x_deriv <= x_max);
    if ~isempty(x_mark)
        y_mark = ylim_shared(1);
        h_deriv = plot(ax, x_mark, repmat(y_mark, numel(x_mark), 1), '^', ...
            'LineStyle', 'none', 'MarkerSize', 9, 'LineWidth', 0.8, ...
            'MarkerFaceColor', [0.55, 0.25, 0.65], 'MarkerEdgeColor', 'k', ...
            'Clipping', 'off', 'DisplayName', 'Solak deriv obs locations');
        uistack(h_deriv, 'top');
    end
end
ax.FontSize = 13;
xlabel(ax, '[S] (mM)', 'FontSize', 14);
ylabel(ax, 'v_0 (\muM/s)', 'FontSize', 14);
end
