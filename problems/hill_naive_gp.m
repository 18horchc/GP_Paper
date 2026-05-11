% GP Research Project: Hill Equation -- Naive GP Surrogate
clear; clc; close all;

%% 1. Define Realistic Parameters (Dose-Response)
% Parameters for a standard dose-response curve
E_min = 5.0;       % Minimum effect (baseline)
E_max = 10.0;      % Maximum effect (plateau)
EC50  = 7.5;       % Half-maximal effective concentration
n_hill = 3;      % Hill coefficient (slope / cooperativity)

x_max = 35;        % Upper end of the concentration domain

% Hill equation: E(C) = E_min + (E_max - E_min) / (1 + (EC50 / C)^n)
hill_func = @(C) E_min + (E_max - E_min) ./ (1 + (EC50 ./ C).^n_hill);

%% 2. Ground truth curve on a concentration grid
x_grid = linspace(1e-6, x_max, 500);
y_true = hill_func(x_grid);

%% 3. Sampling options (select ONE; comment the other two)
% Gaussian observation noise scale (same units as the response)
noise_level = 0.1;

% --- Option 1: Uniform sampling of n_samples points across the domain ---
n_samples = 8;
x_train = linspace(0, x_max, n_samples);

% --- Option 2: Randomly select n_samples points from a half-step grid ---
% n_samples = 12;
% half_step_grid = 0:0.5:x_max;             % candidate locations at 0.5 spacing
% idx_sel  = randperm(numel(half_step_grid), n_samples);
% x_train  = sort(half_step_grid(idx_sel));

% --- Option 3: Set specific sample locations ---
% x_train   = [0.5, 1.0, 2.0, 4.0, 6.0, 7.5, 9.0, 12.0, 18.0, 28.0];
% n_samples = numel(x_train);

% Noisy observations at the chosen training locations
y_clean = hill_func(x_train);
y_train = y_clean + noise_level * randn(size(y_clean));

% Initial GP observation noise scale (used to seed sigma_n below)
noise_level_gp = noise_level;

%% 4. Fit the Naive GP
% Make sure GPML is on the path (looks for a core function like gp.m)
gpml_folder_name = "C:\Users\chorc\OneDrive\Documents\Stroke Research\Gaussian Processes\Old\gpml-matlab-master\gpml-matlab-master";

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

[m_unc, s2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
f_upper_unc = m_unc + 2 * sqrt(max(s2_unc, 0));
f_lower_unc = m_unc - 2 * sqrt(max(s2_unc, 0));

%% 5. Visualization
figure('Color', 'w', 'Position', [100, 100, 800, 500]);
hold on; grid on;

fill([x_grid, fliplr(x_grid)], [f_upper_unc', fliplr(f_lower_unc')], ...
    [0.75, 0.75, 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.5, ...
    'DisplayName', '95% CI');
plot(x_grid, m_unc, 'k--', 'LineWidth', 2, 'DisplayName', 'GP mean');
plot(x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (Hill)');
plot(x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', sprintf('Sampled data (n=%d, noise=%.2f)', n_samples, noise_level));
yline(E_max, 'k:', 'E_{max}', 'Alpha', 0.5);
yline(E_min, 'k:', 'E_{min}', 'Alpha', 0.5);

xlabel('Drug Concentration (C)');
ylabel('Biological Effect (E)');
title('Hill Equation GP: unconstrained (naive) GPML fit');
legend('Location', 'southeast');
set(gca, 'FontSize', 11);

fprintf('GP optimization complete.\n');
fprintf('Unconstrained: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), ...
    gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col));
