% GP Research Project: Acute Transient Equation Sampling
clear; clc; close all;

%% 1. Define Realistic Parameters 
% Parameters 
p.b = 5;
p.A = 20;
p.c = 0.6;

x_max = 35;

% Define function
f = @(x) p.b + p.A *x.* exp(-p.c .* x);


%% 2. Generate Ground Truth Curve
x_grid = linspace(0.01, x_max, 500); % Continuous concentration range
y_true = f(x_grid);

%% 3. Random Sampling with Noise 
n_samples = 12;                   
noise_level = 0.5;               % Gaussian Noise

% Uncomment the line below to sample concentration points randomly across the domain 
%x_train = sort(x_max * rand(1, n_samples)); 

% Uncomment the line below to sample point uniformly acros domain
x_train = linspace(0, x_max, n_samples);

% Uncommnet the two lines below to sample points at specific locations across domain
%x_train = [0.1, 0.5, 1.0, 1.5, 2.0, 6.0, 7.5, 9.0, 10.0];
%n_samples = length(x_train);    % Automatically update sample count

% Calculate true response at sampled points and add noise
y_clean = f(x_train);
y_train = y_clean + noise_level * randn(size(y_clean));

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
title('Acute Transient');
legend('Location', 'southeast');
set(gca, 'FontSize', 12);

%fprintf('Simulation complete. n=%d samples generated.\n', n_samples);






