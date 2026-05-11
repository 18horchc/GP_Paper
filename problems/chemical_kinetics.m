% GP Research Project: Chemical Kinetics Equation Sampling
clear; clc; close all;

%% 1. Define Realistic Parameters 
% Reaction: A + B <-> C
k_f = 0.3;      % Forward rate constant (L / mol*s)
k_r = 0.05;     % Reverse rate constant (1 / s)
A0  = 1.0;      % Initial concentration of Reactant A (mol/L)
B0  = 1.0;      % Initial concentration of Reactant B (mol/L)
C_offset = 0.4;

x_max = 20;

% Equilibrium Product Concentration (Asymptote)
% For A0 = B0, the steady state C_eq solves: k_f*(A0-C)^2 = k_r*C
% Beta represents the characteristic rate cluster for the approach to equilibrium

beta = sqrt(k_r^2 + 4*k_f*k_r*A0); 
C_eq = ((2*k_f*A0 + k_r) - sqrt((2*k_f*A0 + k_r)^2 - 4*k_f^2*A0^2)) / (2*k_f);

% Analytical Solution for Product Concentration [C](t)
rxn_func = @(t) C_offset + C_eq * (1 - exp(-beta * t)) ./ (1 - (C_eq * k_f / beta) * exp(-beta * t));
%% 2. Generate Ground Truth Curve
x_grid = linspace(0.01, x_max, 500); % Continuous concentration range
y_true = rxn_func(x_grid);

%% 3. Random Sampling with Noise 
n_samples = 12;                   
noise_level = 0.05;               % Gaussian Noise

% Uncomment the line below to sample concentration points randomly across the domain 
%x_train = sort(x_max * rand(1, n_samples)); 

% Uncomment the line below to sample point uniformly acros domain
%x_train = linspace(0, x_max, n_samples);

% Uncommnet the two lines below to sample points at specific locations across domain
x_train = [0.5, 1.2, 2.0, 3.5, 12.0, 15.0, 18.0];
n_samples = length(x_train);    % Automatically update sample count

% Calculate true response at sampled points and add relative noise
y_clean = rxn_func(x_train);
%y_train = y_clean + noise_level * randn(size(y_clean)); % Absolute noise
y_train = y_clean + noise_level * max(y_true) * randn(size(y_clean));

%% 4. Visualization
figure('Color', 'w', 'Position', [100, 100, 800, 500]);
hold on; grid on;

% Plot continuous solution curve 
plot(x_grid, y_true, 'b-', 'LineWidth', 2, 'DisplayName', 'Ground Truth');

% Plot sampled noisy points
plot(x_train, y_train, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', sprintf('Sampled Data (n=%d, %g%%  Noise)', n_samples, noise_level*100));

% Labels and Formatting
xlabel('X');
ylabel('Y');
title('Chemical Kinetics');
legend('Location', 'southeast');
set(gca, 'FontSize', 12);

%fprintf('Simulation complete. n=%d samples generated.\n', n_samples);






