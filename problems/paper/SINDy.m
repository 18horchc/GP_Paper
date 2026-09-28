% SINDy.m — GP-based SINDy for M1/M2 microglia dynamics.
% Structure mirrors Arnold et al. Data-Driven-Modeling-Microglia/SINDy.m,
% but states/derivatives come from independent GPs on the full dataset.
% Active GP: NN (covNNone) + empirical heteroscedastic noise.
% (Naive SE-iso + homoscedastic path is commented out below.)
%
% Library: 1, M1, M2, Hill-1/2/3, cross Hill-1/2/3
%          with Ki = median(datapointsMi)
% Sampled once per day on days 0..35. STLS with lambda = 0.01.

clear; close all; clc

%% Paths / GPML
addpath(fileparts(fileparts(mfilename('fullpath'))));  % problems/ for gp_nn_hetero_noise
gpml_folder_name = "C:\Users\chorc\OneDrive\Documents\Stroke Research\Gaussian Processes\Old\gpml-matlab-master\gpml-matlab-master";
if ~exist('minimize', 'file')
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
if ~exist('gp_nn_hetero_noise', 'file')
    error('SINDy:MissingHelper', 'gp_nn_hetero_noise.m not found on path.');
end

%% Data (full replicate series from microglia.m)
% Day 35 points from Suenaga et al. (2015), as plotted in Arnold & Amato.
timeM1 = [0, 1, 3, 5, 7, 14, ...
          3, 7, ...
          2, ...
          14, ...
          3, ...
          3, ...
          0, 1, 3, 7, 14, ...
          3, 7, ...
          35];
dataM1 = [0, 5, 375, 325, 600, 750, ...
          62, 55, ...
          120, ...
          400, ...
          102, ...
          125, ...
          10, 50, 100, 900, 1300, ...
          60, 225, ...
          550];

timeM2 = [0, 1, 3, 5, 7, 14, ...
          1, 3, 7, ...
          2, ...
          14, ...
          3, ...
          3, ...
          0, 1, 3, 7, 14, ...
          3, 7, ...
          35];
dataM2 = [0, 170, 300, 800, 600, 200, ...
          15, 15, 6, ...
          90, ...
          110, ...
          57, ...
          269, ...
          10, 50, 100, 400, 100, ...
          160, 270, ...
          50];

timeM1 = timeM1(:);
dataM1 = max(dataM1(:), 0);
timeM2 = timeM2(:);
dataM2 = max(dataM2(:), 0);

%% Shared settings
max_iters = -200;
t_sindy = (0:1:35)';   % once per day, day 0 through day 35
tgrid = (0:0.1:35)';   % extend through Day 35 Suenaga point
k_plot = 1.96;

%% Fit NN GPs with empirical heteroscedastic noise (full data)
fprintf('=== Fit NN GP (empirical hetero noise) on full data ===\n');
[noise_var_M1, ~, ~, ~, ~] = empirical_time_noise(timeM1, dataM1);
[noise_var_M2, ~, ~, ~, ~] = empirical_time_noise(timeM2, dataM2);

ell0_M1 = max(std(timeM1), 0.5);
sf0_M1  = max(std(dataM1), 0.1);
hyp0_M1 = struct('mean', [], 'cov', log([ell0_M1; sf0_M1]), 'lik', []);
obj_M1 = @(h) gp_nn_hetero_noise('nlml', h, timeM1, dataM1, noise_var_M1);
hyp_M1 = minimize(hyp0_M1, obj_M1, max_iters);
nlml_M1 = obj_M1(hyp_M1);

ell0_M2 = max(std(timeM2), 0.5);
sf0_M2  = max(std(dataM2), 0.1);
hyp0_M2 = struct('mean', [], 'cov', log([ell0_M2; sf0_M2]), 'lik', []);
obj_M2 = @(h) gp_nn_hetero_noise('nlml', h, timeM2, dataM2, noise_var_M2);
hyp_M2 = minimize(hyp0_M2, obj_M2, max_iters);
nlml_M2 = obj_M2(hyp_M2);

fprintf('M1: NLML=%.4f | ell=%.4f | sf=%.4f\n', nlml_M1, exp(hyp_M1.cov(1)), exp(hyp_M1.cov(2)));
fprintf('M2: NLML=%.4f | ell=%.4f | sf=%.4f\n', nlml_M2, exp(hyp_M2.cov(1)), exp(hyp_M2.cov(2)));

%% Obtain Derivative Information (analytic GP mean derivative)
[~, ~, muM1_grid, fs2_M1] = gp_nn_hetero_noise('pred', hyp_M1, timeM1, dataM1, noise_var_M1, tgrid);
[~, ~, muM2_grid, fs2_M2] = gp_nn_hetero_noise('pred', hyp_M2, timeM2, dataM2, noise_var_M2, tgrid);
sfM1_grid = sqrt(max(fs2_M1(:), 0));
sfM2_grid = sqrt(max(fs2_M2(:), 0));

[~, ~, muM1, ~] = gp_nn_hetero_noise('pred', hyp_M1, timeM1, dataM1, noise_var_M1, t_sindy);
[~, ~, muM2, ~] = gp_nn_hetero_noise('pred', hyp_M2, timeM2, dataM2, noise_var_M2, t_sindy);
derivM1 = gp_nn_hetero_noise('deriv', hyp_M1, timeM1, dataM1, noise_var_M1, t_sindy);
derivM2 = gp_nn_hetero_noise('deriv', hyp_M2, timeM2, dataM2, noise_var_M2, t_sindy);

datapointsM1 = muM1(:);   % GP mean at daily sample times
datapointsM2 = muM2(:);
derivM1 = derivM1(:);
derivM2 = derivM2(:);
newtime = t_sindy(:);

% ----- Alternative GP: naive SE-iso + homoscedastic noise (kept for reference) -----
% meanfunc = @meanZero;
% covfunc  = @covSEiso;
% likfunc  = @likGauss;
% inffunc  = @infGaussLik;
%
% fprintf('=== Fit naive SE GP (homoscedastic noise) on full data ===\n');
% [hyp_M1, muM1_grid, fs2_M1] = fit_naive_se_gp(timeM1, dataM1, tgrid, ...
%     inffunc, meanfunc, covfunc, likfunc, max_iters);
% [hyp_M2, muM2_grid, fs2_M2] = fit_naive_se_gp(timeM2, dataM2, tgrid, ...
%     inffunc, meanfunc, covfunc, likfunc, max_iters);
% sfM1_grid = sqrt(max(fs2_M1(:), 0));
% sfM2_grid = sqrt(max(fs2_M2(:), 0));
%
% nlml_M1 = gp(hyp_M1, inffunc, meanfunc, covfunc, likfunc, timeM1, dataM1);
% nlml_M2 = gp(hyp_M2, inffunc, meanfunc, covfunc, likfunc, timeM2, dataM2);
% fprintf('M1: NLML=%.4f | ell=%.4f | sf=%.4f | sn=%.4f\n', ...
%     nlml_M1, exp(hyp_M1.cov(1)), exp(hyp_M1.cov(2)), exp(hyp_M1.lik));
% fprintf('M2: NLML=%.4f | ell=%.4f | sf=%.4f | sn=%.4f\n', ...
%     nlml_M2, exp(hyp_M2.cov(1)), exp(hyp_M2.cov(2)), exp(hyp_M2.lik));
%
% [~, ~, muM1, ~] = gp(hyp_M1, inffunc, meanfunc, covfunc, likfunc, ...
%     timeM1, dataM1, t_sindy);
% [~, ~, muM2, ~] = gp(hyp_M2, inffunc, meanfunc, covfunc, likfunc, ...
%     timeM2, dataM2, t_sindy);
% derivM1 = gp_seiso_mean_deriv(hyp_M1, timeM1, dataM1, t_sindy);
% derivM2 = gp_seiso_mean_deriv(hyp_M2, timeM2, dataM2, t_sindy);
%
% datapointsM1 = muM1(:);
% datapointsM2 = muM2(:);
% derivM1 = derivM1(:);
% derivM2 = derivM2(:);
% newtime = t_sindy(:);
% ----- end naive SE + homoscedastic GP -----

% Plot fitted GPs vs full dataset (before SINDy forward check)
figure(198)
hold on
fill([tgrid; flipud(tgrid)], ...
    [muM1_grid(:) + k_plot * sfM1_grid; flipud(muM1_grid(:) - k_plot * sfM1_grid)], ...
    'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
fill([tgrid; flipud(tgrid)], ...
    [muM2_grid(:) + k_plot * sfM2_grid; flipud(muM2_grid(:) - k_plot * sfM2_grid)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(tgrid, muM1_grid, 'k', 'LineWidth', 2.0);
plot(tgrid, muM2_grid, 'r', 'LineWidth', 2.0);
scatter(timeM1, dataM1, 50, 'k', 'filled');
scatter(timeM2, dataM2, 50, 'r', 'filled');
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
xlim([0, 35])
set(gca, 'fontsize', 20)
title('NN GP (Full Data + Day 35)')

% Separate legend figure (not overlaid on the GP plot)
figL = figure('Color', 'w', 'Position', [100, 100, 900, 80], ...
    'Name', 'NN GP legend');
axL = axes('Parent', figL, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
    'Position', [0 0 1 1]);
hold(axL, 'on');
hL = gobjects(6, 1);
hL(1) = fill(axL, nan, nan, 'k', 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', 'M1 95% confidence region');
hL(2) = plot(axL, nan, nan, 'k', 'LineWidth', 2.0, 'DisplayName', 'M1 GP mean');
hL(3) = plot(axL, nan, nan, 'o', 'Color', 'k', 'MarkerFaceColor', 'k', ...
    'MarkerSize', 6, 'DisplayName', 'M1 data');
hL(4) = fill(axL, nan, nan, 'r', 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', 'M2 95% confidence region');
hL(5) = plot(axL, nan, nan, 'r', 'LineWidth', 2.0, 'DisplayName', 'M2 GP mean');
hL(6) = plot(axL, nan, nan, 'o', 'Color', 'r', 'MarkerFaceColor', 'r', ...
    'MarkerSize', 6, 'DisplayName', 'M2 data');
lgd = legend(axL, hL, 'Orientation', 'horizontal', 'NumColumns', 3);
lgd.FontSize = 16;
lgd.ItemTokenSize = [20, 12];
drawnow;
figL.Units = 'pixels';
lgd.Units = 'pixels';
lp = lgd.Position;
margin = 6;
figL.Position(3:4) = [lp(3) + 2 * margin, lp(4) + 2 * margin];
lgd.Position = [margin, margin, lp(3), lp(4)];

%% SINDy
X_dot = [derivM1(:), derivM2(:)];   % n x 2  (rhs of x_dot = Theta * Xi)

M1v = datapointsM1(:);
M2v = datapointsM2(:);
nT = numel(M1v);
K1 = median(datapointsM1);
K2 = median(datapointsM2);
Theta = [ones(nT, 1), M1v, M2v, ...
    M1v ./ (K1 + M1v), M2v ./ (K2 + M2v), ...
    M1v.^2 ./ (K1^2 + M1v.^2), M2v.^2 ./ (K2^2 + M2v.^2), ...
    M1v.^3 ./ (K1^3 + M1v.^3), M2v.^3 ./ (K2^3 + M2v.^3), ...
    (M1v .* M2v) ./ (K1 + M1v), (M1v .* M2v) ./ (K2 + M2v), ...
    (M1v .* M2v) ./ (K1^2 + M1v.^2), (M1v .* M2v) ./ (K2^2 + M2v.^2), ...
    (M1v .* M2v) ./ (K1^3 + M1v.^3), (M1v .* M2v) ./ (K2^3 + M2v.^3)];   % n x 15

lambda = 0.4; % threshold value

% STLS (Brunton orientation: Theta n x p, X_dot n x 2, Xi p x 2)
epsguess = Theta \ X_dot; % initial guess: Least-squares

for k = 1:100
    smallinds = (abs(epsguess) < lambda); % find small coefficients
    epsguess(smallinds) = 0;              % and threshold
    for ind = 1:2 % n is state dimension (M1 and M2)
        biginds = ~smallinds(:, ind);
        % Regress dynamics onto remaining terms to find sparse Xi
        epsguess(biginds, ind) = Theta(:, biginds) \ X_dot(:, ind);
    end
end

[time, sol] = ode15s(@(t, y) SINDyfwd(t, y, epsguess, K1, K2), 0:0.1:35, ...
    [datapointsM1(1); datapointsM2(1)]);

% Plot (Arnold-style forward check)
figure(199)
plot(time, sol(:, 1), 'k', 'LineWidth', 2.0)
hold on
s = scatter(newtime, datapointsM1, 'k', 'filled');
s.Marker = 'hexagram';
s.SizeData = 150;
hold on
plot(time, sol(:, 2), 'r', 'LineWidth', 2.0)
hold on
s = scatter(newtime, datapointsM2, 'r', 'filled');
s.Marker = 'hexagram';
s.SizeData = 150;
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
legend({'SINDy M1', 'GP mean M1', 'SINDy M2', 'GP mean M2'}, 'Location', 'northwest')
set(gca, 'fontsize', 20)
title('GP-SINDy forward simulation')

% Plot: original data + ODE forecast to day 200
[time200, sol200] = ode15s(@(t, y) SINDyfwd(t, y, epsguess, K1, K2), 0:0.1:200, ...
    [datapointsM1(1); datapointsM2(1)]);

figure(200)
hold on
plot(time200, sol200(:, 1), 'k', 'LineWidth', 2.0, 'DisplayName', 'SINDy M1');
plot(time200, sol200(:, 2), 'r', 'LineWidth', 2.0, 'DisplayName', 'SINDy M2');
scatter(timeM1, dataM1, 50, 'k', 'filled', 'DisplayName', 'M1 data');
scatter(timeM2, dataM2, 50, 'r', 'filled', 'DisplayName', 'M2 data');
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
xlim([0, 200])
legend('Location', 'northwest')
set(gca, 'fontsize', 20)
title('SINDy ODE to Day 200')

%% Return / display Xi and equations
Xi = epsguess;
fprintf('Xi (rows = library terms, cols = [dM1/dt, dM2/dt]):\n');
for i = 1:size(Xi, 1)
    fprintf('%16.8f  %16.8f\n', Xi(i, 1), Xi(i, 2));
end

lib_names = {'1', 'M1', 'M2', ...
             'M1/(K1+M1)', 'M2/(K2+M2)', ...
             'M1^2/(K1^2+M1^2)', 'M2^2/(K2^2+M2^2)', ...
             'M1^3/(K1^3+M1^3)', 'M2^3/(K2^3+M2^3)', ...
             'M1*M2/(K1+M1)', 'M1*M2/(K2+M2)', ...
             'M1*M2/(K1^2+M1^2)', 'M1*M2/(K2^2+M2^2)', ...
             'M1*M2/(K1^3+M1^3)', 'M1*M2/(K2^3+M2^3)'};
fprintf('\ndM1/dt = %s\n', format_sindy_eq(Xi(:, 1), lib_names));
fprintf('dM2/dt = %s\n', format_sindy_eq(Xi(:, 2), lib_names));

%% Local helpers
function rhs = SINDyfwd(~, inits, epsguess, K1, K2)
M1 = inits(1); % initial condition for M1
M2 = inits(2); % initial condition for M2

% library: 1, M1, M2, Hill-1/2/3, cross Hill-1/2/3
theta = [1; ...
         M1; ...
         M2; ...
         M1 / (K1 + M1); ...
         M2 / (K2 + M2); ...
         M1^2 / (K1^2 + M1^2); ...
         M2^2 / (K2^2 + M2^2); ...
         M1^3 / (K1^3 + M1^3); ...
         M2^3 / (K2^3 + M2^3); ...
         (M1 * M2) / (K1 + M1); ...
         (M1 * M2) / (K2 + M2); ...
         (M1 * M2) / (K1^2 + M1^2); ...
         (M1 * M2) / (K2^2 + M2^2); ...
         (M1 * M2) / (K1^3 + M1^3); ...
         (M1 * M2) / (K2^3 + M2^3)];

dM1 = theta' * epsguess(:, 1);
dM2 = theta' * epsguess(:, 2);

rhs = [dM1; dM2];
end

function s = format_sindy_eq(xi, names)
terms = {};
for i = 1:numel(xi)
    if abs(xi(i)) > 0
        terms{end+1} = sprintf('%+.6g*%s', xi(i), names{i}); %#ok<AGROW>
    end
end
if isempty(terms)
    s = '0';
else
    s = strjoin(terms, ' ');
    if s(1) == '+'
        s = strtrim(s(2:end));
    end
end
end

function [noise_var, t_unique, s2_unique, n_per_t, used_fallback] = empirical_time_noise(x, y)
% Sample variance of y at each unique x. n=1 times use pooled variance
% from times with n>=2 (else var(y)), floored away from zero.
x = x(:); y = y(:);
t_unique = unique(x);
nU = numel(t_unique);
s2_unique = zeros(nU, 1);
n_per_t = zeros(nU, 1);
eps_floor = 1e-6 * max(var(y), 1);
for i = 1:nU
    yi = y(abs(x - t_unique(i)) < 1e-12);
    n_per_t(i) = numel(yi);
    if n_per_t(i) >= 2
        s2_unique(i) = var(yi, 0);
    else
        s2_unique(i) = NaN;
    end
end
used_fallback = isnan(s2_unique);
if any(~used_fallback)
    pooled = mean(s2_unique(~used_fallback));
else
    pooled = max(var(y, 0), eps_floor);
end
s2_unique(used_fallback) = pooled;
s2_unique = max(s2_unique, eps_floor);
noise_var = zeros(size(x));
for i = 1:nU
    noise_var(abs(x - t_unique(i)) < 1e-12) = s2_unique(i);
end
end

% ----- SE-only helpers (used only by the commented naive SE path) -----
% function [hyp, mu, s2] = fit_naive_se_gp(x, y, xs, inffunc, meanfunc, covfunc, likfunc, max_iters)
% % Naive SE-iso GP with single learned sn (homoscedastic), matching microglia.m fit_gp.
% x = x(:); y = y(:); xs = xs(:);
% ell0 = max(std(x), 0.5);
% sf0  = max(std(y), 0.1);
% sn0  = 0.1 * sf0;
% if sn0 <= 0
%     sn0 = 0.1;
% end
% hyp.mean = [];
% hyp.cov  = log([ell0; sf0]);
% hyp.lik  = log(sn0);
% hyp = minimize(hyp, @gp, max_iters, inffunc, meanfunc, covfunc, likfunc, x, y);
% [~, ~, mu, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
% mu = mu(:);
% s2 = s2(:);
% end
%
% function dmu = gp_seiso_mean_deriv(hyp, x, y, xs)
% % Analytic posterior mean derivative for SE-iso + homoscedastic noise:
% %   mu'(xs) = K_df(xs, x) * alpha,  alpha = (K_f + sn^2 I)^{-1} y
% x = x(:); y = y(:); xs = xs(:);
% ell = exp(hyp.cov(1));
% sf2 = exp(2 * hyp.cov(2));
% sn2 = exp(2 * hyp.lik(1));
%
% Rxx = x - x.';
% K_f = sf2 * exp(-0.5 * (Rxx ./ ell).^2);
% jitter = 1e-8 * mean(diag(K_f));
% Ky = K_f + (sn2 + jitter) * eye(numel(x));
% L = chol(Ky, 'lower');
% alpha = L' \ (L \ y);
%
% R = xs - x.';
% Kxc = sf2 * exp(-0.5 * (R ./ ell).^2);
% K_df = -Kxc .* (R ./ ell^2);   % cov(df/dx(xs), f(x)) = dk/d xs
% dmu = K_df * alpha;
% dmu = dmu(:);
% end
