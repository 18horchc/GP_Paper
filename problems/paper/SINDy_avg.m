% SINDy_avg.m — GP-based SINDy for M1/M2 on the averaged dataset.
% Same GP as SINDy.m:
%   - independent zero-mean NN GPs (covNNone / gp_nn_hetero_noise)
%   - observations on the raw count scale (not z-scored); only ell and sf are fit
%   - fixed empirical heteroscedastic noise = replicate sigma^2(t) from the
%     full dataset (n=1 times, including Day 35, use pooled variance)
% The means being fit are the averaged series (+ Day 35 Suenaga points).
%
% Library: 1, M1, M2, M1^2, M1*M2, M2^2, M1^3, M1^2*M2, M1*M2^2, M2^3,
%          M1/(K1+M1), M2/(K2+M2),
%          M1^0.5/(K1^0.5+M1^0.5), M2^0.5/(K2^0.5+M2^0.5),
%          M1^2/(K1^2+M1^2), M2^2/(K2^2+M2^2)
%          with Ki = median(datapointsMi) (GP mean at the daily sample times).
% Sampled once per day on days 0..35. STLS with lambda = 0.01.

clear; close all; clc

%% Paths / GPML
addpath(fileparts(fileparts(mfilename('fullpath'))));  % problems/ for gp_nn_hetero_noise
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
if ~exist('gp_nn_hetero_noise', 'file')
    error('SINDy_avg:MissingHelper', 'gp_nn_hetero_noise.m not found on path.');
end

%% Data (averaged series from microglia.m + Day 35)
% Day 35 points from Suenaga et al. (2015), as plotted in Arnold & Amato.
timeM1 = [0, 1, 2, 3, 5, 7, 14, 35]';
dataM1 = [5, 27.5, 122.5, 139.8, 325, 445, 816.67, 550]';

timeM2 = [0, 1, 2, 3, 5, 7, 14, 35]';
dataM2 = [5, 78.33, 179.5, 126.4, 800, 319, 136.67, 50]';

timeM1 = timeM1(:);
dataM1 = max(dataM1(:), 0);
timeM2 = timeM2(:);
dataM2 = max(dataM2(:), 0);

% Full replicate series (for empirical sigma^2(t) only; GP is fit to averages)
timeM1_full = [0, 1, 3, 5, 7, 14, ...
               3, 7, ...
               2, ...
               14, ...
               3, ...
               3, ...
               0, 1, 3, 7, 14, ...
               3, 7, ...
               35]';
dataM1_full = [0, 5, 375, 325, 600, 750, ...
               62, 55, ...
               120, ...
               400, ...
               102, ...
               125, ...
               10, 50, 100, 900, 1300, ...
               60, 225, ...
               550]';

timeM2_full = [0, 1, 3, 5, 7, 14, ...
               1, 3, 7, ...
               2, ...
               14, ...
               3, ...
               3, ...
               0, 1, 3, 7, 14, ...
               3, 7, ...
               35]';
dataM2_full = [0, 170, 300, 800, 600, 200, ...
               15, 15, 6, ...
               90, ...
               110, ...
               57, ...
               269, ...
               10, 50, 100, 400, 100, ...
               160, 270, ...
               50]';

timeM1_full = timeM1_full(:);
dataM1_full = max(dataM1_full(:), 0);
timeM2_full = timeM2_full(:);
dataM2_full = max(dataM2_full(:), 0);

%% Shared settings
max_iters = -200;
t_sindy = (0:1:35)';   % once per day, day 0 through day 35
tgrid = (0:0.1:35)';
k_plot = 1.96;

%% Fit zero-mean NN GPs with empirical heteroscedastic noise (raw averaged counts)
% Same model as SINDy.m: gp_nn_hetero_noise, mean fixed at 0, noise fixed.
fprintf('=== Fit NN GP (empirical sigma^2(t)) on averaged data + Day 35 ===\n');
[~, t_u_M1, s2_u_M1, n_M1, fb_M1] = empirical_time_noise(timeM1_full, dataM1_full);
[~, t_u_M2, s2_u_M2, n_M2, fb_M2] = empirical_time_noise(timeM2_full, dataM2_full);
noise_var_M1 = map_emp_noise_to_times(timeM1, t_u_M1, s2_u_M1);
noise_var_M2 = map_emp_noise_to_times(timeM2, t_u_M2, s2_u_M2);

[hyp_M1, muM1_grid, fs2_M1, nlml_M1] = fit_nn_emp_noise( ...
    timeM1, dataM1, tgrid, noise_var_M1, max_iters);
[hyp_M2, muM2_grid, fs2_M2, nlml_M2] = fit_nn_emp_noise( ...
    timeM2, dataM2, tgrid, noise_var_M2, max_iters);
sfM1_grid = sqrt(max(fs2_M1(:), 0));
sfM2_grid = sqrt(max(fs2_M2(:), 0));

fprintf('M1: NLML=%.4f | ell=%.4f | sf=%.4f\n', ...
    nlml_M1, exp(hyp_M1.cov(1)), exp(hyp_M1.cov(2)));
fprintf('M2: NLML=%.4f | ell=%.4f | sf=%.4f\n', ...
    nlml_M2, exp(hyp_M2.cov(1)), exp(hyp_M2.cov(2)));
fprintf('Empirical sn(t)=sqrt(s2) at averaged times (*=pooled):\n');
print_emp_schedule('M1', timeM1, noise_var_M1, t_u_M1, fb_M1);
print_emp_schedule('M2', timeM2, noise_var_M2, t_u_M2, fb_M2);

%% Obtain Derivative Information (analytic GP mean derivative)
[~, ~, muM1, ~] = gp_nn_hetero_noise('pred', hyp_M1, timeM1, dataM1, noise_var_M1, t_sindy);
[~, ~, muM2, ~] = gp_nn_hetero_noise('pred', hyp_M2, timeM2, dataM2, noise_var_M2, t_sindy);
derivM1 = gp_nn_hetero_noise('deriv', hyp_M1, timeM1, dataM1, noise_var_M1, t_sindy);
derivM2 = gp_nn_hetero_noise('deriv', hyp_M2, timeM2, dataM2, noise_var_M2, t_sindy);

datapointsM1 = muM1(:);   % GP mean at daily sample times
datapointsM2 = muM2(:);
derivM1 = derivM1(:);
derivM2 = derivM2(:);
newtime = t_sindy(:);

% Plot fitted GPs vs averaged dataset (before SINDy forward check)
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
title('NN GP (Averaged + Day 35)')

% Separate legend figure (not overlaid on the GP plot)
figL = figure('Color', 'w', 'Position', [100, 100, 900, 80], ...
    'Name', 'NN GP legend (averaged)');
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
Theta = sindy_library(M1v, M2v, K1, K2);   % n x 16

lambda = 0.01; % threshold value

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

% Plot: averaged data + ODE forecast to day 200
[time200, sol200] = ode15s(@(t, y) SINDyfwd(t, y, epsguess, K1, K2), 0:0.1:200, ...
    [datapointsM1(1); datapointsM2(1)]);

figure(200)
hold on
plot(time200, sol200(:, 1), 'k', 'LineWidth', 2.0, 'DisplayName', 'SINDy M1');
plot(time200, sol200(:, 2), 'r', 'LineWidth', 2.0, 'DisplayName', 'SINDy M2');
scatter(timeM1, dataM1, 50, 'k', 'filled', 'DisplayName', 'M1 data');
scatter(timeM2, dataM2, 50, 'r', 'filled', 'DisplayName', 'M2 data');
yline(0, 'k:', 'LineWidth', 1.0, 'HandleVisibility', 'off');
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
xlim([0, 200])
ylim([-20, 800])
legend('Location', 'northwest')
set(gca, 'fontsize', 20)
title('SINDy ODE to Day 200')

%% Return / display Xi and equations
Xi = epsguess;
fprintf('Xi (rows = library terms, cols = [dM1/dt, dM2/dt]):\n');
for i = 1:size(Xi, 1)
    fprintf('%16.8f  %16.8f\n', Xi(i, 1), Xi(i, 2));
end

lib_names = sindy_library_names();
fprintf('\ndM1/dt = %s\n', format_sindy_eq(Xi(:, 1), lib_names));
fprintf('dM2/dt = %s\n', format_sindy_eq(Xi(:, 2), lib_names));

%% Local helpers
function [hyp, mu, s2, nlml] = fit_nn_emp_noise(x, y, xs, noise_var, max_iters)
% Zero-mean NN GP on raw counts. Only ell and sf are fit; noise_var is fixed.
x = x(:); y = y(:); xs = xs(:); noise_var = noise_var(:);
ell0 = max(std(x), 0.5);
sf0  = max(std(y), 0.1);
hyp0 = struct('mean', [], 'cov', log([ell0; sf0]), 'lik', []);
obj = @(h) gp_nn_hetero_noise('nlml', h, x, y, noise_var);
hyp = minimize(hyp0, obj, max_iters);
nlml = obj(hyp);
[~, ~, mu, s2] = gp_nn_hetero_noise('pred', hyp, x, y, noise_var, xs);
mu = mu(:);
s2 = s2(:);
end

function noise_var = map_emp_noise_to_times(t, t_unique, s2_unique)
% Assign empirical s2(t) to each training time; unknown times use pooled mean.
t = t(:);
noise_var = zeros(size(t));
pooled = mean(s2_unique);
for i = 1:numel(t)
    j = find(abs(t_unique - t(i)) < 1e-12, 1);
    if isempty(j)
        noise_var(i) = pooled;
    else
        noise_var(i) = s2_unique(j);
    end
end
end

function print_emp_schedule(name, t, noise_var, t_unique, used_fallback)
for i = 1:numel(t)
    j = find(abs(t_unique - t(i)) < 1e-12, 1);
    mark = ' ';
    if isempty(j) || used_fallback(j)
        mark = '*';
    end
    fprintf('  %s t=%g: sn=%.4f%s\n', name, t(i), sqrt(noise_var(i)), mark);
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

function rhs = SINDyfwd(~, inits, epsguess, K1, K2)
theta = sindy_library(inits(1), inits(2), K1, K2);
dM1 = theta * epsguess(:, 1);
dM2 = theta * epsguess(:, 2);
rhs = [dM1; dM2];
end

function Theta = sindy_library(M1, M2, K1, K2)
% Columns: 1, M1, M2, M1^2, M1*M2, M2^2, M1^3, M1^2*M2, M1*M2^2, M2^3,
%          M1/(K1+M1), M2/(K2+M2),
%          M1^0.5/(K1^0.5+M1^0.5), M2^0.5/(K2^0.5+M2^0.5),
%          M1^2/(K1^2+M1^2), M2^2/(K2^2+M2^2).
% Half-powers use max(.,0) so a negative state stays real.
M1 = M1(:);
M2 = M2(:);
sM1 = sqrt(max(M1, 0));
sM2 = sqrt(max(M2, 0));
sK1 = sqrt(max(K1, 0));
sK2 = sqrt(max(K2, 0));
Theta = [ones(numel(M1), 1), M1, M2, ...
    M1.^2, M1 .* M2, M2.^2, ...
    M1.^3, (M1.^2) .* M2, M1 .* (M2.^2), M2.^3, ...
    M1 ./ (K1 + M1), M2 ./ (K2 + M2), ...
    sM1 ./ (sK1 + sM1), sM2 ./ (sK2 + sM2), ...
    M1.^2 ./ (K1^2 + M1.^2), M2.^2 ./ (K2^2 + M2.^2)];
end

function names = sindy_library_names()
names = {'1', 'M1', 'M2', ...
         'M1^2', 'M1*M2', 'M2^2', ...
         'M1^3', 'M1^2*M2', 'M1*M2^2', 'M2^3', ...
         'M1/(K1+M1)', 'M2/(K2+M2)', ...
         'M1^0.5/(K1^0.5+M1^0.5)', 'M2^0.5/(K2^0.5+M2^0.5)', ...
         'M1^2/(K1^2+M1^2)', 'M2^2/(K2^2+M2^2)'};
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
