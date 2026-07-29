% Paper figure: Michaelis-Menten GP with virtual derivative observations (Solak).
% Deriv obs: x_deriv = [1; 1.2; 1.4; 1.6; 1.8] mM, y_deriv = 0.3, sn_deriv = 0.2.
% Requires gp_seiso_deriv_obs.m on the path (problems/).
clear; clc; close all;

%% MM parameters
Vmax = 6;
Km   = 0.15;
mm_static = @(S) (Vmax .* S) ./ (Km + S);

%% Training data ([S] in mM, v_0 in μM/s)
x_max = 2;
n_train = 7;
noise_frac = 0.05;   % homoscedastic: sigma = noise_frac * max v on [0, x_max]
x_train = [0.01; 0.08; 0.2; 0.3; 0.5; 0.8; 1.8];

rng(100);
v_true_at_train = mm_static(x_train);
y_domain_max = mm_static(x_max);
noise_sd_true = noise_frac * y_domain_max;
y_train = v_true_at_train + noise_sd_true * randn(size(v_true_at_train));
fprintf('Synthetic data: n=%d equally spaced on [0, %.1f], homoscedastic noise sigma_n = %.4f (%.0f%% of v(x_max))\n', ...
    n_train, x_max, noise_sd_true, 100 * noise_frac);

%% Derivative observations (Solak et al. 2002): df/d[S] at selected [S]
x_deriv = [1; 1.2; 1.4; 1.6; 1.8];
y_deriv = 0.3 * ones(numel(x_deriv), 1);
sn_deriv = 0.2;   % soft: larger => weaker derivative pull (Solak Gaussian noise on dY/dx)

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

%% Solak derivative-observation GP (augmented NLML; Pensoneault disabled)
hyp_tpl = hyp_unc;
fprintf('\n=== Solak deriv obs only (no Pensoneault) ===\n');
fprintf('deriv obs: %d at [S] in [%.1f, %.1f] | y_deriv = %.3g | sn_deriv = %.4g\n', ...
    numel(x_deriv), min(x_deriv), max(x_deriv), y_deriv(1), sn_deriv);

obj_deriv = @(hyp_cov) gp_seiso_deriv_obs_nlml_cov_only(hyp_cov, sn_fixed, x_col, y_col, ...
    x_deriv, y_deriv, sn_deriv);
fprintf('Optimizing augmented GP (ell, sf; sigma_n fixed at %.4f)...\n', noise_sd_true);
hyp_cov_con = minimize(hyp_unc.cov, obj_deriv, -100);
hyp_con = struct('mean', [], 'cov', hyp_cov_con(:), 'lik', sn_fixed);
nlml_opt = obj_deriv(hyp_cov_con);
exitflag = 1;

%% Plot baseline vs Solak-augmented GP
k_plot = 2;
[~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_unc = fmu_unc(:);
sf_unc = sqrt(max(fs2_unc(:), 0));

[~, ~, fmu_con, fs2_con] = gp_seiso_deriv_obs('pred', hyp_con, x_col, y_col, ...
    x_deriv, y_deriv, x_grid(:), sn_deriv);
m_con = fmu_con(:);
sf_con = sqrt(max(fs2_con(:), 0));

[m_deriv_at_xd, s2_deriv_at_xd] = gp_seiso_deriv_obs('deriv', hyp_con, x_col, y_col, ...
    x_deriv, y_deriv, x_deriv, sn_deriv);
mm_deriv_true = Vmax * Km ./ (Km + x_deriv).^2;

band_label = sprintf('\\mu_f \\pm %g\\sigma_f (latent)', k_plot);
ylim_shared = [0, max([y_train(:); Vmax; m_unc + k_plot * sf_unc; m_con + k_plot * sf_con]) * 1.02];

figure('Color', 'w', 'Position', [80, 80, 1100, 520], ...
    'Name', sprintf('Michaelis-Menten GP: derivative obs (n=%d, %.0f%% noise)', n_train, 100 * noise_frac));
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

panels(1) = struct('m', m_unc, 'sf', sf_unc, 'title', 'Baseline GP', 'show_deriv', false);
panels(2) = struct('m', m_con, 'sf', sf_con, 'title', 'Virtual Deriv Obs GP', 'show_deriv', true);

for p = 1:2
    ax = nexttile;
    ax.Layer = 'top';
    hold(ax, 'on');
    grid(ax, 'on');
    m = panels(p).m;
    sf = panels(p).sf;
    fill(ax, [x_grid, fliplr(x_grid)], [m + k_plot * sf; flipud(m - k_plot * sf)]', ...
        [0.72, 0.72, 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
    plot(ax, x_grid, m, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
    plot(ax, x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
    plot(ax, x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
        'DisplayName', 'Observed data');
    yline(ax, Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
    xlim(ax, [0, x_max]);
    ylim(ax, ylim_shared);
    if panels(p).show_deriv
        x_mark = x_deriv(isfinite(x_deriv) & x_deriv >= 0 & x_deriv <= x_max);
        if ~isempty(x_mark)
            y_mark = ylim_shared(1);   % on x-axis (v_0 = 0)
            h_deriv = plot(ax, x_mark, repmat(y_mark, numel(x_mark), 1), '^', ...
                'LineStyle', 'none', 'MarkerSize', 9, 'LineWidth', 0.8, ...
                'MarkerFaceColor', [0.55, 0.25, 0.65], 'MarkerEdgeColor', 'k', ...
                'Clipping', 'off', 'DisplayName', 'Solak deriv obs locations');
            uistack(h_deriv, 'top');
        end
    end
    xlabel(ax, '[S] (mM)');
    ylabel(ax, 'v_0 (\muM/s)');
    title(ax, panels(p).title, 'Interpreter', 'none');
    legend(ax, 'Location', 'southeast');
end

fprintf('\nBaseline:      ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);
fprintf('Augmented:     ell=%.4f, sf=%.4f, sn=%.4f, sn_deriv=%.4f | NLML=%.4f | exitflag=%d\n', ...
    exp(hyp_con.cov(1)), exp(hyp_con.cov(2)), exp(hyp_con.lik), sn_deriv, nlml_opt, exitflag);
fprintf('\nPosterior f'' at Solak derivative observation points (sn_deriv = %.3g):\n', sn_deriv);
fprintf('  [S]    target    post mean    post sd    MM analytic\n');
for j = 1:numel(x_deriv)
    fprintf('  %4.2f   %6.3f    %8.4f    %8.4f    %8.4f\n', ...
        x_deriv(j), y_deriv(j), m_deriv_at_xd(j), sqrt(s2_deriv_at_xd(j)), mm_deriv_true(j));
end
