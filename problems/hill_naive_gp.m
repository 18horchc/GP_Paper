% GP Research Project: Hill Equation -- Naive GP vs derivative-observation GP
clear; clc; close all;

%% 1. Define Realistic Parameters (Dose-Response)
% Parameters for a standard dose-response curve
E_min = 5.0;       % Minimum effect (baseline)
E_max = 10.0;      % Maximum effect (plateau)
EC50  = 12.5;       % Half-maximal effective concentration
n_hill = 3;      % Hill coefficient (slope / cooperativity)

x_max = 35;        % Upper end of the concentration domain

% Hill equation: E(C) = E_min + (E_max - E_min) / (1 + (EC50 / C)^n)
hill_func = @(C) E_min + (E_max - E_min) ./ (1 + (EC50 ./ C).^n_hill);
hill_deriv = @(C) (E_max - E_min) .* n_hill .* EC50.^n_hill ./ ...
    (C.^(n_hill + 1) .* (1 + (EC50 ./ C).^n_hill).^2);

%% 2. Ground truth curve on a concentration grid
x_grid = linspace(1e-6, x_max, 500);
y_true = hill_func(x_grid);

%% 3. Sampling options (select ONE; comment the other two)
% Gaussian observation noise scale (same units as the response)
noise_level = 0.2;

% --- Option 1: Uniform sampling of n_samples points across the domain ---
%n_samples = 7;
%x_train = linspace(0, x_max, n_samples);

% --- Option 2: Randomly select n_samples points from a half-step grid ---
% n_samples = 12;
% half_step_grid = 0:0.5:x_max;             % candidate locations at 0.5 spacing
% idx_sel  = randperm(numel(half_step_grid), n_samples);
% x_train  = sort(half_step_grid(idx_sel));

% --- Option 3: Set specific sample locations ---
x_train   = [0.5, 1.0, 2.0, 4.0, 6.0, 7.5, 9.0, 12.0, 18.0, 28.0];
n_samples = numel(x_train);

% Noisy observations at the chosen training locations
rng(100);
y_clean = hill_func(x_train);
y_train = y_clean + noise_level * randn(size(y_clean));

% Initial GP observation noise scale (used to seed sigma_n below)
noise_level_gp = noise_level;

%% Derivative observations (Solak et al. 2002): dE/dC at selected C
x_deriv = [24; 32];
y_deriv = [0.06; 0.02];
sn_deriv = max(0.005, 0.5 * noise_level);

%% 4. Fit the Naive GP
% Make sure GPML is on the path (looks for a core function like gp.m)
gpml_folder_name = "C:\Users\chorc\OneDrive\Documents\Stroke Research\Gaussian Processes\Old\gpml-matlab-master\gpml-matlab-master";
addpath(fileparts(mfilename('fullpath')));

if ~exist('gp', 'file')
    fprintf('GPML not found. Searching in local directory: %s...\n', gpml_folder_name);
    if exist(gpml_folder_name, 'dir')
        addpath(genpath(gpml_folder_name));
        fprintf('GPML successfully added to path.\n');
    else
        error(['GPML toolbox missing! Please download it and ensure ' ...
               'the folder "%s" is in your project directory.'], gpml_folder_name);
    end
else
    fprintf('GPML toolbox is already loaded and ready.\n');
end

try
    startup;
catch
    % Some GPML versions ship without startup.m; safe to ignore.
end

% Configuration & Initial Hyperparameters
ell0 = std(x_train);
sf0  = std(y_train);
sn0  = max(1e-3, noise_level_gp);

hyp = struct();
hyp.mean = [];                  % matches @meanZero
hyp.cov  = log([ell0; sf0]);
hyp.lik  = log(sn0);

meanfunc = @meanZero;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

x_col = x_train(:);
y_col = y_train(:);

% Hyperparameter Optimization (unconstrained NLML, naive GP)
fprintf('Optimizing hyperparameters (unconstrained NLML)...\n');
hyp_unc = minimize(hyp, @gp, -100, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
nlml_unc = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);

%% Derivative-observation GP (Solak-style augmented NLML)
obj_deriv = @(h) gp_seiso_deriv_obs('nlml', h, x_col, y_col, x_deriv, y_deriv, [], sn_deriv);
fprintf('\nOptimizing hyperparameters (derivative-observation NLML)...\n');
hyp_deriv = minimize(hyp, obj_deriv, -100);
nlml_deriv = obj_deriv(hyp_deriv);

%% 5. Predictions and visualization
k_plot = 2;
[~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_unc = fmu_unc(:);
sf_unc = sqrt(max(fs2_unc(:), 0));

[~, ~, fmu_deriv, fs2_deriv] = gp_seiso_deriv_obs('pred', hyp_deriv, x_col, y_col, ...
    x_deriv, y_deriv, x_grid(:), sn_deriv);
m_deriv = fmu_deriv(:);
sf_deriv = sqrt(max(fs2_deriv(:), 0));

[m_deriv_at_xd, s2_deriv_at_xd] = gp_seiso_deriv_obs('deriv', hyp_deriv, x_col, y_col, ...
    x_deriv, y_deriv, x_deriv, sn_deriv);
hill_deriv_true = hill_deriv(x_deriv);

band_label = sprintf('\\mu_f \\pm %g\\sigma_f (latent)', k_plot);
ylim_plot = [0, 11];

fig = figure('Color', 'w', 'Position', [80, 80, 920, 620], ...
    'Name', sprintf('Hill GP (n=%d, noise=%.2f)', n_samples, noise_level));
tg = uitabgroup(fig);

tab_unc = uitab(tg, 'Title', 'Naive GP');
plot_hill_gp_tab(tab_unc, x_grid, x_max, m_unc, sf_unc, k_plot, band_label, ...
    y_true, x_train, y_train, noise_level, E_min, E_max, 'Naive GPML', ylim_plot);

tab_deriv = uitab(tg, 'Title', 'Deriv-obs');
plot_hill_gp_tab(tab_deriv, x_grid, x_max, m_deriv, sf_deriv, k_plot, band_label, ...
    y_true, x_train, y_train, noise_level, E_min, E_max, ...
    sprintf('Deriv-obs GP (\\ell=%.2f, NLML=%.2f)', exp(hyp_deriv.cov(1)), nlml_deriv), ...
    ylim_plot, x_deriv);

fprintf('\nGP optimization complete.\n');
fprintf('Naive GP:    ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);
fprintf('Deriv-obs:   ell=%.4f, sf=%.4f, sn=%.4f, sn_deriv=%.4f | NLML=%.4f\n', ...
    exp(hyp_deriv.cov(1)), exp(hyp_deriv.cov(2)), exp(hyp_deriv.lik), sn_deriv, nlml_deriv);
fprintf('\nPosterior dE/dC at derivative observation points (sn_deriv = %.3g):\n', sn_deriv);
fprintf('  C      target    post mean    post sd    Hill analytic\n');
for j = 1:numel(x_deriv)
    fprintf('  %4.1f   %6.3f    %8.4f    %8.4f    %8.4f\n', ...
        x_deriv(j), y_deriv(j), m_deriv_at_xd(j), sqrt(s2_deriv_at_xd(j)), hill_deriv_true(j));
end

function plot_hill_gp_tab(tab, x_grid, x_max, m, sf, k_plot, band_label, ...
    y_true, x_train, y_train, noise_level, E_min, E_max, panel_title, ylim_fixed, x_deriv)
if nargin < 16
    x_deriv = [];
end
ax = axes('Parent', tab, 'Position', [0.09, 0.11, 0.865, 0.815]);
ax.Layer = 'top';
hold(ax, 'on'); grid(ax, 'on');
fill(ax, [x_grid, fliplr(x_grid)], [m + k_plot * sf; flipud(m - k_plot * sf)]', ...
    [0.72, 0.72, 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
plot(ax, x_grid, m, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
plot(ax, x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (Hill)');
plot(ax, x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', sprintf('Data (\\sigma=%.2g)', noise_level));
yline(ax, E_max, 'k:', 'E_{max}', 'Alpha', 0.5);
yline(ax, E_min, 'k:', 'E_{min}', 'Alpha', 0.5);
xlabel(ax, 'Drug Concentration (C)');
ylabel(ax, 'Biological Effect (E)');
title(ax, panel_title, 'Interpreter', 'none');
xlim(ax, [0, x_max]);
ylim(ax, ylim_fixed);
if ~isempty(x_deriv)
    x_mark = x_deriv(:);
    x_mark = x_mark(x_mark >= 0 & x_mark <= x_max);
    if ~isempty(x_mark)
        y_axis = ax.YLim(1);
        h_deriv = plot(ax, x_mark, repmat(y_axis, numel(x_mark), 1), '^', ...
            'MarkerSize', 9, 'LineWidth', 0.8, ...
            'MarkerFaceColor', [0.55, 0.25, 0.65], 'MarkerEdgeColor', 'k', ...
            'Clipping', 'off', 'DisplayName', 'Derivative obs locations');
        uistack(h_deriv, 'top');
    end
end
legend(ax, 'Location', 'southeast');
set(ax, 'FontSize', 11);
end
