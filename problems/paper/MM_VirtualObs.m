% Paper figure: Michaelis-Menten GP with virtual function-value anchors.
% Virtual anchors: (S=0, v=0) with sigma = 0.01*sigma_obs; (S=S_high, v=Vmax) with sigma = 0.20*Vmax.
% Requires gp_seiso_hetero_noise.m on the path (problems/).
clear; clc; close all;

%% MM parameters
Vmax = 6;
Km   = .15;
mm_static = @(S) (Vmax .* S) ./ (Km + S);

%% Training data ([S] in mM, v_0 in μM/s)
x_max = 2;
n_train = 3;
noise_frac = 0.05;   % homoscedastic: sigma = noise_frac * max v on [0, x_max]
x_train = [.5; 1; 1.5];

rng(100);
v_true_at_train = mm_static(x_train);
y_domain_max = mm_static(x_max);
noise_sd_true = noise_frac * y_domain_max;
y_train = v_true_at_train + noise_sd_true * randn(size(v_true_at_train));
fprintf('Synthetic data: n=%d equally spaced on [0, %.1f], homoscedastic noise sigma_n = %.4f (%.0f%% of v(x_max))\n', ...
    n_train, x_max, noise_sd_true, 100 * noise_frac);

%% Virtual observations (heteroscedastic augmentation)
x_obs = x_train(:);
y_obs = y_train(:);
n_obs = numel(y_obs);

Vmax_prior = Vmax;
S_high     = x_max;
sigma_obs  = noise_sd_true;

x_virt_zero = 0;
y_virt_zero = 0;
sigma_virt_zero = 0.01 * sigma_obs;

x_virt_sat = S_high;
y_virt_sat = Vmax_prior;
sigma_virt_sat = 0.20 * Vmax_prior;

x_aug = [x_obs; x_virt_zero; x_virt_sat];
y_aug = [y_obs; y_virt_zero; y_virt_sat];
sigma_aug = [sigma_obs * ones(n_obs, 1); sigma_virt_zero; sigma_virt_sat];
noise_var_aug = sigma_aug.^2;

fprintf('Virtual obs: v(0)=0 (sigma=%.4g) | v(%.1f)=%.1f (sigma=%.2f)\n', ...
    sigma_virt_zero, S_high, Vmax_prior, sigma_virt_sat);

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

ell_bounds_lo = 0.05;
ell_ub = 5;   % cap length scale at domain width
sf_bounds = [0.05, 35];
sn_bounds = [noise_sd_true, 2];   % floor sigma_n at known measurement noise

meanfunc = @meanZero;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

x_col = x_train(:);
y_col = y_train(:);

%% Unconstrained GP (bounded NLML: sigma_n floor, ell cap)
hyp_lb = log([ell_bounds_lo; sf_bounds(1); sn_bounds(1)]);
hyp_ub = log([ell_ub; sf_bounds(2); sn_bounds(2)]);
to_hyp = @(theta) struct('mean', [], 'cov', theta(1:2), 'lik', theta(3));
obj_unc = @(theta) gp(to_hyp(theta), inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
theta0 = min(max([hyp.cov(:); hyp.lik], hyp_lb), hyp_ub);
opts_unc = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'off');
fprintf('Optimizing hyperparameters (bounded NLML: ell in [%.2g, %.2g], sn >= %.4f)...\n', ...
    ell_bounds_lo, ell_ub, sn_bounds(1));
[theta_unc, nlml_unc, exitflag_unc] = fmincon(obj_unc, theta0, [], [], [], [], hyp_lb, hyp_ub, [], opts_unc);
hyp_unc = to_hyp(theta_unc);

%% Augmented GP with virtual observations (heteroscedastic NLML)
obj_aug = @(h) gp_seiso_hetero_noise('nlml', h, x_aug, y_aug, noise_var_aug);
hyp_aug = struct('mean', [], 'cov', hyp.cov, 'lik', []);
fprintf('\nOptimizing hyperparameters (augmented heteroscedastic NLML)...\n');
hyp_aug = minimize(hyp_aug, obj_aug, -100);
nlml_aug = obj_aug(hyp_aug);

%% Plot baseline vs augmented GP
k_plot = 2;
[~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_unc = fmu_unc(:);
sf_unc = sqrt(max(fs2_unc(:), 0));

[~, ~, fmu_aug, fs2_aug] = gp_seiso_hetero_noise('pred', hyp_aug, x_aug, y_aug, noise_var_aug, x_grid(:));
m_aug = fmu_aug(:);
sf_aug = sqrt(max(fs2_aug(:), 0));

band_label = sprintf('\\mu_f \\pm %g\\sigma_f (latent)', k_plot);
ylim_shared = [0, max([y_train(:); Vmax; m_unc + k_plot * sf_unc; m_aug + k_plot * sf_aug]) * 1.02];

figure('Color', 'w', 'Position', [80, 80, 1100, 520], ...
    'Name', 'Michaelis-Menten GP: virtual anchors');
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

panels(1) = struct('m', m_unc, 'sf', sf_unc, 'title', 'Baseline GP', ...
    'show_virt', false);
panels(2) = struct('m', m_aug, 'sf', sf_aug, 'title', 'Augmented GP', ...
    'show_virt', true);

for p = 1:2
    nexttile;
    ax = gca;
    ax.Layer = 'top';
    hold on; grid on;
    m = panels(p).m;
    sf = panels(p).sf;
    fill([x_grid, fliplr(x_grid)], [m + k_plot * sf; flipud(m - k_plot * sf)]', ...
        [0.72, 0.72, 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
    plot(x_grid, m, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
    plot(x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
    plot(x_obs, y_obs, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
        'DisplayName', 'Observed data');
    if panels(p).show_virt
        scatter(x_virt_zero, y_virt_zero, 90, 's', ...
            'MarkerFaceColor', [0.85, 0.85, 0.85], 'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.5, 'DisplayName', 'Virtual anchor: v(0)=0');
        scatter(x_virt_sat, y_virt_sat, 90, 'd', ...
            'MarkerFaceColor', [0.85, 0.85, 0.85], 'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.5, 'DisplayName', 'Virtual soft saturation target');
    end
    yline(Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
    xlabel('[S] (mM)');
    ylabel('v_0 (\muM/s)');
    title(panels(p).title, 'Interpreter', 'none');
    xlim([0, x_max]);
    ylim(ylim_shared);
    legend('Location', 'southeast');
end

fprintf('\nBaseline:      ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc, exitflag_unc);
fprintf('Augmented:     ell=%.4f, sf=%.4f | NLML=%.4f (heteroscedastic noise, n_aug=%d)\n', ...
    exp(hyp_aug.cov(1)), exp(hyp_aug.cov(2)), nlml_aug, numel(y_aug));
