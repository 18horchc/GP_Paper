% Paper figure: Michaelis-Menten GP — baseline SE vs virtual anchors / deriv obs.
% Dataset: [S] = [0.08, 0.2, 0.3, 0.5, 0.8, 1.8] with 5% homoscedastic noise.
% Tab 1: unconstrained SE-GP.
% Tab 2: virtual function-value anchors at S=0 and S=2 (same sn as data).
% Tab 3: Solak virtual derivative obs on unaugmented data (sn_deriv = sn).
% Tab 4: virtual function-value anchors + Solak deriv obs together.
clear; clc; close all;

%% MM parameters
Vmax = 6;
Km   = .15;
mm_static = @(S) (Vmax .* S) ./ (Km + S);

%% Training data ([S] in mM, v_0 in μM/s)
x_max = 2;
noise_frac = 0.1;   % homoscedastic: sigma = noise_frac * max v on [0, x_max]
x_train = [0.08; 0.2; 0.3; 0.5; 0.8; 2];
n_train = numel(x_train);

rng(100);
v_true_at_train = mm_static(x_train);
y_domain_max = mm_static(x_max);
noise_sd_true = noise_frac * y_domain_max;
y_train = v_true_at_train + noise_sd_true * randn(size(v_true_at_train));
fprintf('Synthetic data: n=%d at [0.08,0.2,0.3,0.5,0.8,1.8], homoscedastic noise sigma_n = %.4f (%.0f%% of v(x_max))\n', ...
    n_train, noise_sd_true, 100 * noise_frac);

%% Virtual function-value observations (homoscedastic; same sn as data)
x_obs = x_train(:);
y_obs = y_train(:);

S_high = x_max;
x_virt_zero = 0;
y_virt_zero = 0;
x_virt_sat = 1.5;
y_virt_sat = 5.5;

x_aug = [x_obs; x_virt_zero; x_virt_sat];
y_aug = [y_obs; y_virt_zero; y_virt_sat];

fprintf('Virtual obs: v(0)=0 | v(%.1f)=%.1f | shared sn=%.4f (homoscedastic)\n', ...
    S_high, Vmax, noise_sd_true);

%% Virtual derivative observations (Solak; sn_deriv = noise_sd_true)
%x_deriv = linspace(0, 2, 10)';          % 0, 2/9, ..., 2
%y_deriv = (27:-3:0)';                   % 27, ..., 0

x_deriv = [1; 1.2; 1.4; 1.6; 1.8];
y_deriv = 0.3 * ones(numel(x_deriv), 1);

sn_deriv = noise_sd_true;
fprintf('Virtual deriv obs: %d sites on [0, 2] | y=30:-3:3 | sn_deriv=%.4f\n', ...
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
sn0  = max(1e-3, noise_sd_true);
hyp = struct('mean', [], 'cov', log([ell0; sf0]), 'lik', log(sn0));

meanfunc = @meanZero;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

x_col = x_train(:);
y_col = y_train(:);

%% Baseline GP (sigma_n fixed at noise_sd_true; optimize ell, sf only)
sn_fixed = log(noise_sd_true);
fprintf('Optimizing baseline (ell, sf; sigma_n fixed at %.4f)...\n', noise_sd_true);
obj_unc = @(hyp_cov) gp_nlml_cov_only(hyp_cov, sn_fixed, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
hyp_cov_unc = minimize(hyp.cov, obj_unc, -100);
hyp_unc = struct('mean', [], 'cov', hyp_cov_unc(:), 'lik', sn_fixed);
nlml_unc = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);

%% Augmented GP with virtual function-value observations (homoscedastic)
fprintf('\nOptimizing augmented (ell, sf; sigma_n fixed at %.4f)...\n', noise_sd_true);
obj_aug = @(hyp_cov) gp_nlml_cov_only(hyp_cov, sn_fixed, inffunc, meanfunc, covfunc, likfunc, x_aug, y_aug);
hyp_cov_aug = minimize(hyp.cov, obj_aug, -100);
hyp_aug = struct('mean', [], 'cov', hyp_cov_aug(:), 'lik', sn_fixed);
nlml_aug = gp(hyp_aug, inffunc, meanfunc, covfunc, likfunc, x_aug, y_aug);

%% Solak derivative-observation GP on unaugmented data
fprintf('\nOptimizing deriv-obs GP (ell, sf; sigma_n fixed at %.4f, sn_deriv=%.4f)...\n', ...
    noise_sd_true, sn_deriv);
obj_deriv = @(hyp_cov) gp_seiso_deriv_obs_nlml_cov_only( ...
    hyp_cov, sn_fixed, x_col, y_col, x_deriv, y_deriv, sn_deriv);
hyp_cov_deriv = minimize(hyp.cov, obj_deriv, -100);
hyp_deriv = struct('mean', [], 'cov', hyp_cov_deriv(:), 'lik', sn_fixed);
nlml_deriv = obj_deriv(hyp_cov_deriv);

%% Combined: virtual function-value anchors + Solak deriv obs
fprintf('\nOptimizing VO+deriv GP (ell, sf; sigma_n fixed at %.4f, sn_deriv=%.4f)...\n', ...
    noise_sd_true, sn_deriv);
obj_both = @(hyp_cov) gp_seiso_deriv_obs_nlml_cov_only( ...
    hyp_cov, sn_fixed, x_aug, y_aug, x_deriv, y_deriv, sn_deriv);
hyp_cov_both = minimize(hyp.cov, obj_both, -100);
hyp_both = struct('mean', [], 'cov', hyp_cov_both(:), 'lik', sn_fixed);
nlml_both = obj_both(hyp_cov_both);

%% Predict for plots
k_plot = 2;
[~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_unc = fmu_unc(:);
sf_unc = sqrt(max(fs2_unc(:), 0));

[~, ~, fmu_aug, fs2_aug] = gp(hyp_aug, inffunc, meanfunc, covfunc, likfunc, x_aug, y_aug, x_grid(:));
m_aug = fmu_aug(:);
sf_aug = sqrt(max(fs2_aug(:), 0));

[~, ~, fmu_d, fs2_d] = gp_seiso_deriv_obs('pred', hyp_deriv, ...
    x_col, y_col, x_deriv, y_deriv, x_grid(:), sn_deriv);
m_deriv = fmu_d(:);
sf_deriv = sqrt(max(fs2_d(:), 0));

[~, ~, fmu_b, fs2_b] = gp_seiso_deriv_obs('pred', hyp_both, ...
    x_aug, y_aug, x_deriv, y_deriv, x_grid(:), sn_deriv);
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
title(ax1, 'Baseline GP', 'Interpreter', 'none');

tab_aug = uitab(tg, 'Title', 'Virtual Obs GP');
ax2 = axes('Parent', tab_aug);
plot_mm_gp_panel(ax2, m_aug, sf_aug, true, [], ...
    x_grid, y_true, x_obs, y_obs, k_plot, band_label, ylim_shared, x_max, Vmax, ...
    x_virt_zero, y_virt_zero, x_virt_sat, y_virt_sat);
title(ax2, 'Virtual Obs GP', 'Interpreter', 'none');

tab_deriv = uitab(tg, 'Title', 'Virtual Deriv Obs GP');
ax3 = axes('Parent', tab_deriv);
plot_mm_gp_panel(ax3, m_deriv, sf_deriv, false, x_deriv, ...
    x_grid, y_true, x_obs, y_obs, k_plot, band_label, ylim_shared, x_max, Vmax, ...
    x_virt_zero, y_virt_zero, x_virt_sat, y_virt_sat);
title(ax3, 'Virtual Deriv Obs GP', 'Interpreter', 'none');

tab_both = uitab(tg, 'Title', 'VO + Deriv Obs GP');
ax4 = axes('Parent', tab_both);
plot_mm_gp_panel(ax4, m_both, sf_both, true, x_deriv, ...
    x_grid, y_true, x_obs, y_obs, k_plot, band_label, ylim_shared, x_max, Vmax, ...
    x_virt_zero, y_virt_zero, x_virt_sat, y_virt_sat);
title(ax4, 'VO + Deriv Obs GP', 'Interpreter', 'none');

fprintf('\nBaseline:      ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);
fprintf('Augmented:     ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f (homoscedastic, n_aug=%d)\n', ...
    exp(hyp_aug.cov(1)), exp(hyp_aug.cov(2)), exp(hyp_aug.lik), nlml_aug, numel(y_aug));
fprintf('Deriv obs:     ell=%.4f, sf=%.4f, sn=%.4f, sn_deriv=%.4f | NLML=%.4f\n', ...
    exp(hyp_deriv.cov(1)), exp(hyp_deriv.cov(2)), exp(hyp_deriv.lik), sn_deriv, nlml_deriv);
fprintf('VO+deriv:      ell=%.4f, sf=%.4f, sn=%.4f, sn_deriv=%.4f | NLML=%.4f (n_aug=%d)\n', ...
    exp(hyp_both.cov(1)), exp(hyp_both.cov(2)), exp(hyp_both.lik), sn_deriv, nlml_both, numel(y_aug));
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
    scatter(ax, x_virt_zero, y_virt_zero, 90, 's', ...
        'MarkerFaceColor', [0.85, 0.85, 0.85], 'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.5, 'DisplayName', 'Virtual anchor: v(0)=0');
    scatter(ax, x_virt_sat, y_virt_sat, 90, 'd', ...
        'MarkerFaceColor', [0.85, 0.85, 0.85], 'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.5, 'DisplayName', 'Virtual soft saturation target');
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
xlabel(ax, '[S] (mM)');
ylabel(ax, 'v_0 (\muM/s)');
legend(ax, 'Location', 'southeast');
end
