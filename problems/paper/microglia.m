clear; close all; clc

%% Data 

%averages 
newtime = [0, 1, 2, 3, 7, 14]; %These are the points where we observe experimental data
datapointsM2 = [5, 78.33, 179.5, 126.4, 319, 136.67]; %Experimental measurements of M2 (averaged per time point)
datapointsM1 = [5, 27.5, 122.5, 139.8, 445, 816.67]; %Experimental measurements of M1 (averaged per time point)

% full data
% Individual measurements compiled from the experimental studies (Fig. 1 / Table 1).
% Each study reports at its own time points, so several measurements
% share the same time. Columns are grouped by study to match the tables.
timeM1 = [0, 1, 3, 5, 7, 14, ...   % Hu2012
          3, 7, ...                % Li2018
          2, ...                   % Ma2020
          14, ...                  % Wang2017
          3, ...                   % Xu2021
          2, ...                   % Li2023
          0, 1, 3, 7, 14, ...      % Suenega2015
          3, 7];                   % Yang2017
dataM1 = [0, 5, 375, 325, 600, 750, ...
          62, 55, ...
          120, ...
          400, ...
          102, ...
          125, ...
          10, 50, 100, 900, 1300, ...
          60, 225];

timeM2 = [0, 1, 3, 5, 7, 14, ...   % Hu2012
          1, 3, 7, ...             % Li2018
          2, ...                   % Ma2020
          14, ...                  % Wang2017
          3, ...                   % Xu2021
          2, ...                   % Li2023
          0, 1, 3, 7, 14, ...      % Suenega2015
          3, 7];                   % Yang2017
dataM2 = [0, 170, 300, 800, 600, 200, ...
          15, 15, 6, ...
          90, ...
          110, ...
          57, ...
          269, ...
          10, 50, 100, 400, 100, ...
          160, 270];


%% GP
% Fit Gaussian Process to the data using the GPML toolbox with a zero mean
% function and squared exponential (covSEiso) covariance function.

% --- GPML setup ---
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

meanfunc = @meanZero;       % zero mean
covfunc  = @covSEiso;       % squared exponential kernel
likfunc  = @likGauss;       % Gaussian likelihood
inffunc  = @infGaussLik;    % exact inference

% Fit each state independently (hyperparameters optimized by maximizing the
% marginal likelihood). Returns the fitted hyperparameters along with the
% predictive mean and variance on a dense time grid.
tgrid = (0:0.1:14)';
[gpM1.hyp, gpM1.mu, gpM1.s2] = fit_gp(timeM1', dataM1', tgrid, ...
    inffunc, meanfunc, covfunc, likfunc);
[gpM2.hyp, gpM2.mu, gpM2.s2] = fit_gp(timeM2', dataM2', tgrid, ...
    inffunc, meanfunc, covfunc, likfunc);

% --- Plot GP fits with 95% uncertainty bands ---
sdM1 = sqrt(max(gpM1.s2, 0));   % predictive standard deviation
sdM2 = sqrt(max(gpM2.s2, 0));
k = 1.96;                       % ~95% confidence band

figure(99)
hold on

% M1 (black): uncertainty band, mean, and data
fill([tgrid; flipud(tgrid)], [gpM1.mu + k*sdM1; flipud(gpM1.mu - k*sdM1)], ...
    'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(tgrid, gpM1.mu, 'k', 'LineWidth', 2.0)
sM1 = scatter(timeM1, dataM1, 'k', 'filled');
sM1.Marker = 'hexagram';
sM1.SizeData = 150;

% M2 (red): uncertainty band, mean, and data
fill([tgrid; flipud(tgrid)], [gpM2.mu + k*sdM2; flipud(gpM2.mu - k*sdM2)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(tgrid, gpM2.mu, 'r', 'LineWidth', 2.0)
sM2 = scatter(timeM2, dataM2, 'r', 'filled');
sM2.Marker = 'hexagram';
sM2.SizeData = 150;
hold off

xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
legend({'M1 GP mean', 'M1 data', 'M2 GP mean', 'M2 data'}, 'Location', 'northwest')
title('GP fits with 95% uncertainty bands')
set(gca, 'fontsize', 20)

% --- Separate subplots for M1 and M2 ---
figure(100)
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% M1 subplot (left)
nexttile;
hold on
fill([tgrid; flipud(tgrid)], [gpM1.mu + k*sdM1; flipud(gpM1.mu - k*sdM1)], ...
    'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M1 95% band');
plot(tgrid, gpM1.mu, 'k', 'LineWidth', 2.0, 'DisplayName', 'M1 GP mean')
sM1_sub = scatter(timeM1, dataM1, 'k', 'filled', 'DisplayName', 'M1 data');
sM1_sub.Marker = 'hexagram';
sM1_sub.SizeData = 150;
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
title('M1 GP fit with 95% uncertainty band')
legend('Location', 'northwest')
set(gca, 'fontsize', 20)

% M2 subplot (right)
nexttile;
hold on
fill([tgrid; flipud(tgrid)], [gpM2.mu + k*sdM2; flipud(gpM2.mu - k*sdM2)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M2 95% band');
plot(tgrid, gpM2.mu, 'r', 'LineWidth', 2.0, 'DisplayName', 'M2 GP mean')
sM2_sub = scatter(timeM2, dataM2, 'r', 'filled', 'DisplayName', 'M2 data');
sM2_sub.Marker = 'hexagram';
sM2_sub.SizeData = 150;
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
title('M2 GP fit with 95% uncertainty band')
legend('Location', 'northwest')
set(gca, 'fontsize', 20)


%% Bounded GP (Pensoneault lower bound at 0)
% Probabilistic lower bound mu_f(x) - k*sigma_f(x) >= 0 on 41 equispaced points.
% SE kernel, zero mean; hyperparameters (ell, sf) optimized via fmincon with sigma_n
% fixed from the unconstrained fits above.

x_min = 0;
x_max = 14;
eta = 0.022;   % 2.2% tail probability (Pensoneault et al.)
k_pens = -sqrt(2) * erfinv(2 * eta - 1);
n_constraint = 41;
X_c = linspace(x_min, x_max, n_constraint)';

ell_bounds_lo = 0.05;
ell_ub = x_max;
sf_bounds_M1 = [0.05, max(15, 1.5 * std(dataM1))];
sf_bounds_M2 = [0.05, max(15, 1.5 * std(dataM2))];
hyp_lb_M1 = log([ell_bounds_lo; sf_bounds_M1(1)]);
hyp_ub_M1 = log([ell_ub; sf_bounds_M1(2)]);
hyp_lb_M2 = log([ell_bounds_lo; sf_bounds_M2(1)]);
hyp_ub_M2 = log([ell_ub; sf_bounds_M2(2)]);

opts_pens = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, 'Display', 'off', ...
    'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);
nTry = 2000;
nMultistart = 10;

fprintf('\n=== Pensoneault GP (lower bound at 0) ===\n');
fprintf('eta = %.3g%% | k = %.4f | X_c: %d points on [%.0f, %.0f]\n', ...
    100 * eta, k_pens, n_constraint, x_min, x_max);

[gpM1_bound.hyp, gpM1_bound.mu, gpM1_bound.s2, gpM1_bound.nlml, gpM1_bound.exitflag, gpM1_bound.max_c] = ...
    fit_gp_lower_bound(timeM1', dataM1', gpM1.hyp, X_c, k_pens, tgrid, ...
    inffunc, meanfunc, covfunc, likfunc, hyp_lb_M1, hyp_ub_M1, opts_pens, nTry, nMultistart, 42);
[gpM2_bound.hyp, gpM2_bound.mu, gpM2_bound.s2, gpM2_bound.nlml, gpM2_bound.exitflag, gpM2_bound.max_c] = ...
    fit_gp_lower_bound(timeM2', dataM2', gpM2.hyp, X_c, k_pens, tgrid, ...
    inffunc, meanfunc, covfunc, likfunc, hyp_lb_M2, hyp_ub_M2, opts_pens, nTry, nMultistart, 43);

fprintf('M1 bounded: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(gpM1_bound.hyp.cov(1)), exp(gpM1_bound.hyp.cov(2)), exp(gpM1_bound.hyp.lik), ...
    gpM1_bound.nlml, gpM1_bound.exitflag, gpM1_bound.max_c);
fprintf('M2 bounded: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(gpM2_bound.hyp.cov(1)), exp(gpM2_bound.hyp.cov(2)), exp(gpM2_bound.hyp.lik), ...
    gpM2_bound.nlml, gpM2_bound.exitflag, gpM2_bound.max_c);

% --- Plot bounded GP fits with 95% uncertainty bands ---
sdM1_bound = sqrt(max(gpM1_bound.s2, 0));
sdM2_bound = sqrt(max(gpM2_bound.s2, 0));
k_plot = 1.96;
ylim_bound = [0, max([dataM1(:); dataM2(:); ...
    gpM1_bound.mu + k_plot * sdM1_bound; gpM2_bound.mu + k_plot * sdM2_bound]) * 1.05];

figure(101)
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% M1 subplot (left)
nexttile;
hold on
fill([tgrid; flipud(tgrid)], [gpM1_bound.mu + k_plot*sdM1_bound; flipud(gpM1_bound.mu - k_plot*sdM1_bound)], ...
    'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M1 95% band');
plot(tgrid, gpM1_bound.mu, 'k', 'LineWidth', 2.0, 'DisplayName', 'M1 bounded GP mean')
sM1_b = scatter(timeM1, dataM1, 'k', 'filled', 'DisplayName', 'M1 data');
sM1_b.Marker = 'hexagram';
sM1_b.SizeData = 150;
yline(0, 'k:', 'HandleVisibility', 'off');
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
title('M1 Pensoneault lower-bound GP')
legend('Location', 'northwest')
ylim(ylim_bound)
set(gca, 'fontsize', 20)

% M2 subplot (right)
nexttile;
hold on
fill([tgrid; flipud(tgrid)], [gpM2_bound.mu + k_plot*sdM2_bound; flipud(gpM2_bound.mu - k_plot*sdM2_bound)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M2 95% band');
plot(tgrid, gpM2_bound.mu, 'r', 'LineWidth', 2.0, 'DisplayName', 'M2 bounded GP mean')
sM2_b = scatter(timeM2, dataM2, 'r', 'filled', 'DisplayName', 'M2 data');
sM2_b.Marker = 'hexagram';
sM2_b.SizeData = 150;
yline(0, 'k:', 'HandleVisibility', 'off');
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
title('M2 Pensoneault lower-bound GP')
legend('Location', 'northwest')
ylim(ylim_bound)
set(gca, 'fontsize', 20)


%% GP on averaged data
% Unconstrained SE-kernel GP fits on per-timepoint averages (newtime, datapointsM1/M2).

[gpM1_avg.hyp, gpM1_avg.mu, gpM1_avg.s2] = fit_gp(newtime', datapointsM1', tgrid, ...
    inffunc, meanfunc, covfunc, likfunc);
[gpM2_avg.hyp, gpM2_avg.mu, gpM2_avg.s2] = fit_gp(newtime', datapointsM2', tgrid, ...
    inffunc, meanfunc, covfunc, likfunc);

sdM1_avg = sqrt(max(gpM1_avg.s2, 0));
sdM2_avg = sqrt(max(gpM2_avg.s2, 0));

figure(102)
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% M1 subplot (left)
nexttile;
hold on
fill([tgrid; flipud(tgrid)], [gpM1_avg.mu + k_plot*sdM1_avg; flipud(gpM1_avg.mu - k_plot*sdM1_avg)], ...
    'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M1 95% band');
plot(tgrid, gpM1_avg.mu, 'k', 'LineWidth', 2.0, 'DisplayName', 'M1 GP mean')
sM1_avg = scatter(newtime, datapointsM1, 'k', 'filled', 'DisplayName', 'M1 averaged data');
sM1_avg.Marker = 'hexagram';
sM1_avg.SizeData = 150;
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
title('M1 GP fit (averaged data)')
legend('Location', 'northwest')
set(gca, 'fontsize', 20)

% M2 subplot (right)
nexttile;
hold on
fill([tgrid; flipud(tgrid)], [gpM2_avg.mu + k_plot*sdM2_avg; flipud(gpM2_avg.mu - k_plot*sdM2_avg)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M2 95% band');
plot(tgrid, gpM2_avg.mu, 'r', 'LineWidth', 2.0, 'DisplayName', 'M2 GP mean')
sM2_avg = scatter(newtime, datapointsM2, 'r', 'filled', 'DisplayName', 'M2 averaged data');
sM2_avg.Marker = 'hexagram';
sM2_avg.SizeData = 150;
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
title('M2 GP fit (averaged data)')
legend('Location', 'northwest')
set(gca, 'fontsize', 20)


%% Bounded GP on averaged data
% Pensoneault lower bound at 0 on averaged data; reuses X_c, k_pens, opts_pens from above.

sf_bounds_avg_M1 = [0.05, max(15, 1.5 * std(datapointsM1))];
sf_bounds_avg_M2 = [0.05, max(15, 1.5 * std(datapointsM2))];
hyp_lb_avg_M1 = log([ell_bounds_lo; sf_bounds_avg_M1(1)]);
hyp_ub_avg_M1 = log([ell_ub; sf_bounds_avg_M1(2)]);
hyp_lb_avg_M2 = log([ell_bounds_lo; sf_bounds_avg_M2(1)]);
hyp_ub_avg_M2 = log([ell_ub; sf_bounds_avg_M2(2)]);

fprintf('\n=== Pensoneault GP on averaged data (lower bound at 0) ===\n');

[gpM1_avg_bound.hyp, gpM1_avg_bound.mu, gpM1_avg_bound.s2, gpM1_avg_bound.nlml, gpM1_avg_bound.exitflag, gpM1_avg_bound.max_c] = ...
    fit_gp_lower_bound(newtime', datapointsM1', gpM1_avg.hyp, X_c, k_pens, tgrid, ...
    inffunc, meanfunc, covfunc, likfunc, hyp_lb_avg_M1, hyp_ub_avg_M1, opts_pens, nTry, nMultistart, 44);
[gpM2_avg_bound.hyp, gpM2_avg_bound.mu, gpM2_avg_bound.s2, gpM2_avg_bound.nlml, gpM2_avg_bound.exitflag, gpM2_avg_bound.max_c] = ...
    fit_gp_lower_bound(newtime', datapointsM2', gpM2_avg.hyp, X_c, k_pens, tgrid, ...
    inffunc, meanfunc, covfunc, likfunc, hyp_lb_avg_M2, hyp_ub_avg_M2, opts_pens, nTry, nMultistart, 45);

fprintf('M1 avg bounded: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(gpM1_avg_bound.hyp.cov(1)), exp(gpM1_avg_bound.hyp.cov(2)), exp(gpM1_avg_bound.hyp.lik), ...
    gpM1_avg_bound.nlml, gpM1_avg_bound.exitflag, gpM1_avg_bound.max_c);
fprintf('M2 avg bounded: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(gpM2_avg_bound.hyp.cov(1)), exp(gpM2_avg_bound.hyp.cov(2)), exp(gpM2_avg_bound.hyp.lik), ...
    gpM2_avg_bound.nlml, gpM2_avg_bound.exitflag, gpM2_avg_bound.max_c);

sdM1_avg_bound = sqrt(max(gpM1_avg_bound.s2, 0));
sdM2_avg_bound = sqrt(max(gpM2_avg_bound.s2, 0));
ylim_avg_bound = [0, max([datapointsM1(:); datapointsM2(:); ...
    gpM1_avg_bound.mu + k_plot * sdM1_avg_bound; gpM2_avg_bound.mu + k_plot * sdM2_avg_bound]) * 1.05];

figure(103)
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% M1 subplot (left)
nexttile;
hold on
fill([tgrid; flipud(tgrid)], [gpM1_avg_bound.mu + k_plot*sdM1_avg_bound; flipud(gpM1_avg_bound.mu - k_plot*sdM1_avg_bound)], ...
    'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M1 95% band');
plot(tgrid, gpM1_avg_bound.mu, 'k', 'LineWidth', 2.0, 'DisplayName', 'M1 bounded GP mean')
sM1_avg_b = scatter(newtime, datapointsM1, 'k', 'filled', 'DisplayName', 'M1 averaged data');
sM1_avg_b.Marker = 'hexagram';
sM1_avg_b.SizeData = 150;
yline(0, 'k:', 'HandleVisibility', 'off');
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
title('M1 Pensoneault lower-bound GP (averaged data)')
legend('Location', 'northwest')
ylim(ylim_avg_bound)
set(gca, 'fontsize', 20)

% M2 subplot (right)
nexttile;
hold on
fill([tgrid; flipud(tgrid)], [gpM2_avg_bound.mu + k_plot*sdM2_avg_bound; flipud(gpM2_avg_bound.mu - k_plot*sdM2_avg_bound)], ...
    'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'M2 95% band');
plot(tgrid, gpM2_avg_bound.mu, 'r', 'LineWidth', 2.0, 'DisplayName', 'M2 bounded GP mean')
sM2_avg_b = scatter(newtime, datapointsM2, 'r', 'filled', 'DisplayName', 'M2 averaged data');
sM2_avg_b.Marker = 'hexagram';
sM2_avg_b.SizeData = 150;
yline(0, 'k:', 'HandleVisibility', 'off');
hold off
xlabel('Time (Days)', 'fontsize', 20)
ylabel('cells/mm^2', 'fontsize', 20)
title('M2 Pensoneault lower-bound GP (averaged data)')
legend('Location', 'northwest')
ylim(ylim_avg_bound)
set(gca, 'fontsize', 20)


%% Obtain Derivative Information 
%  next stepp will be to wire the GP-derived derivatives into the SINDy step (replacing the polynomial derivatives)
% t = [0:0.1:14]; %time that we want to fit the polynomial over
% 
% pM1four = polyfit(newtime, datapointsM1, 4); %obtain coefficients for a fourth order polynomial fit to M1 data
% polyM1four = polyval(pM1four, t); %Compute y values of polynomial for t
% 
% %analytically compute derivative for all values of t
% for i = 1:length(t)
% M1estderiv(i) = 0 + pM1four(4) + 2*pM1four(3)*t(i) + 3*pM1four(2)*t(i)^2 + 4*pM1four(1)*t(i)^3; %
% end
% 
% %Extract derivative information for time points where experimental data is
% %present
% derivM1 = [M1estderiv(1), M1estderiv(11), M1estderiv(21), M1estderiv(31), M1estderiv(71), M1estderiv(141)];
% 
% pM2quad = polyfit(newtime, datapointsM2, 2); %obtain coefficients for a second order polynomial fit to M2 data
% polyM2quad = polyval(pM2quad, t); %Compute y values of polynomial for t
% 
% %analytically compute derivative for all values of t
% for i = 1:length(t)
% M2estderiv(i) = 0 + pM2quad(2) + 2*pM2quad(1)*t(i);%
% end
% 
% %Extract derivative information for time points where experimental data is
% %present
% derivM2 = [M2estderiv(1), M2estderiv(11), M2estderiv(21), M2estderiv(31), M2estderiv(71), M2estderiv(141)];



%% SINDy

% X_dot = [derivM1; derivM2]'; %Derivative information; rhs of 2.1
% 
% Theta = [ones(length(datapointsM1),1)'; datapointsM1(1:end); datapointsM2(1:end); datapointsM1(1:end).^2;...
%    datapointsM2(1:end).^2; datapointsM1(1:end).*datapointsM2(1:end)]'; %candidate function matrix
% 
% lambda = 0.01; %threshhold value
% 
% %STLS
% epsguess = Theta\X_dot; % initial guess: Least-squares
% 
% for k=1:100
% smallinds = (abs(epsguess)<lambda); % find small coefficients
% epsguess(smallinds)=0; % and threshold (set small coeff = 0)
% for ind = 1:2 % n is state dimension (in this case we have M1 and M2)
% biginds = ~smallinds(:,ind);
% % Regress dynamics onto remaining terms to find sparse Xi
% epsguess(biginds,ind) = Theta(:,biginds)\X_dot(:,ind); %compute LS for non-small coeff
% end
% end
% 
% [time, sol] = ode15s(@SINDyfwd, [0:0.1:14], [datapointsM1(1), datapointsM2(1)], [], epsguess); %Run fwd model
% 
% %Plot (used to obtain figure 6)
% figure(199)
% plot(time, sol(:,1), 'k', 'LineWidth',2.0)
% hold on
% s = scatter(newtime, datapointsM1, 'k', 'filled');
% s.Marker = 'hexagram';
% s.SizeData = 150;
% hold on
% plot(time, sol(:,2), 'r', 'LineWidth',2.0)
% hold on
% s = scatter(newtime, datapointsM2, 'r', 'filled');
% s.Marker = 'hexagram';
% s.SizeData = 150;
% hold off
% xlabel('Time (Days)', 'fontsize',20) 
% ylim([0 1500])
% ylabel('cells/mm^2', 'fontsize',20)
% set(gca, 'fontsize', 20)
% 
% epsguess % display resulting coefficients


%% Function for fwd model
% function rhs = SINDyfwd(t, inits, epsguess)
% M1 = inits(1); %initial condition for M1
% M2 = inits(2); %initial condition for M2
% 
% %coefficient values
% theta1 = epsguess(1,1);
% theta2 = epsguess(2,1);
% theta3 = epsguess(3,1);
% theta4 = epsguess(4,1);
% theta5 = epsguess(5,1);
% theta6 = epsguess(6,1);
% theta7 = epsguess(1, 2);
% theta8 = epsguess(2, 2);
% theta9 = epsguess(3, 2);
% theta10 = epsguess(4, 2);
% theta11 = epsguess(5,2); 
% theta12 = epsguess(6,2);
% 
% %Equations
% dM1 = theta1 + theta2*M1 + theta3*M2 + theta4*M1^2 + theta5*M2^2 +theta6*M1*M2;
% dM2 = theta7 + theta8*M1 + theta9*M2 + theta10*M1^2 + theta11*M2^2 + theta12*M1*M2;
% 
% rhs = [dM1; dM2];
% end

%% Function to fit a GP with GPML (zero mean, squared exponential kernel)
function [hyp, mu, s2] = fit_gp(x, y, xs, inffunc, meanfunc, covfunc, likfunc)
x = x(:); y = y(:); xs = xs(:);

% Initialize hyperparameters from the data scale (log-transformed for GPML)
ell0 = std(x);              % characteristic length scale
sf0  = std(y);              % signal standard deviation
sn0  = 0.1 * std(y);        % observation noise standard deviation

hyp.mean = [];                      % meanZero has no parameters
hyp.cov  = log([ell0; sf0]);        % covSEiso: [log(ell); log(sf)]
hyp.lik  = log(sn0);                % likGauss: log(sn)

% Optimize hyperparameters by minimizing the negative log marginal likelihood
hyp = minimize(hyp, @gp, -100, inffunc, meanfunc, covfunc, likfunc, x, y);

% Predictive mean and variance on the query points
[mu, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
end

function [hyp, mu, s2, nlml, exitflag, max_c] = fit_gp_lower_bound( ...
    x, y, hyp_unc, X_c, k, xs, inffunc, meanfunc, covfunc, likfunc, ...
    hyp_lb, hyp_ub, opts, nTry, nMultistart, rng_seed)
% Pensoneault lower-bound GP: minimize NLML subject to mu_f - k*sigma_f >= 0 at X_c.
x = x(:); y = y(:); xs = xs(:);
sn_fixed = hyp_unc.lik;
hyp_tpl = struct('mean', [], 'cov', hyp_unc.cov(:), 'lik', sn_fixed);
theta_unc = hyp_unc.cov(:);

objfun = @(theta) gp(theta_to_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x, y);
nonlcon = @(theta) pens_constraints_lower(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k);

theta_unc_box = min(max(theta_unc, hyp_lb), hyp_ub);
fprintf('Multistart: %d random starts\n', nTry);
feasible_starts = zeros(2, 0);
best_feas_nlml = inf;
best_feas_theta = nan(2, 1);
rng(rng_seed);
for t = 1:nTry
    theta_try = hyp_lb + rand(2, 1) .* (hyp_ub - hyp_lb);
    [c_try, ~] = nonlcon(theta_try);
    if max(c_try) <= 0
        feasible_starts = [feasible_starts, theta_try];
        nlml_try = objfun(theta_try);
        if nlml_try < best_feas_nlml
            best_feas_nlml = nlml_try;
            best_feas_theta = theta_try;
        end
    end
end
nFeas = size(feasible_starts, 2);
fprintf('Feasible random starts: %d / %d\n', nFeas, nTry);
if nFeas > 0
    nlml_feas = arrayfun(@(j) objfun(feasible_starts(:, j)), 1:nFeas);
    [~, ord] = sort(nlml_feas, 'ascend');
    starts_for_fmincon = feasible_starts(:, ord(1:min(nMultistart, nFeas)));
else
    fprintf('No feasible random start; using projected baseline theta.\n');
    starts_for_fmincon = theta_unc_box;
end
starts_for_fmincon = [theta_unc_box, starts_for_fmincon];
starts_for_fmincon = starts_for_fmincon(:, 1:min(nMultistart + 1, size(starts_for_fmincon, 2)));

best_nlml = inf;
theta_opt = nan(2, 1);
nlml = nan;
exitflag = -99;
nStarts = size(starts_for_fmincon, 2);
for j = 1:nStarts
    theta0_j = starts_for_fmincon(:, j);
    [theta_j, nlml_j, ef_j] = fmincon(objfun, theta0_j, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts);
    if isfinite(nlml_j) && nlml_j < best_nlml
        best_nlml = nlml_j;
        theta_opt = theta_j;
        nlml = nlml_j;
        exitflag = ef_j;
    end
end
if ~isfinite(best_nlml)
    if nFeas > 0
        theta_opt = best_feas_theta;
    else
        theta_opt = theta_unc_box;
    end
    nlml = objfun(theta_opt);
    exitflag = -99;
    fprintf('Warning: no successful fmincon run; using fallback theta.\n');
end

hyp = struct('mean', [], 'cov', theta_opt(:), 'lik', sn_fixed);
[c_final, ~] = nonlcon(theta_opt);
max_c = max(c_final);
[~, ~, mu, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
mu = mu(:);
s2 = s2(:);
end

function hyp = theta_to_hyp(theta, hyp_tpl)
hyp = hyp_tpl;
hyp.cov = theta(1:2);
hyp.mean = [];
end

function [c, ceq] = pens_constraints_lower(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k)
% Pensoneault lower bound at 0 on latent f: mu_f - k*sigma_f >= 0  <=>  c <= 0.
hyp = theta_to_hyp(theta, hyp_tpl);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, X_c(:));
m_xc = fmu(:);
s_xc = sqrt(max(fs2(:), 0));
c = k .* s_xc - m_xc;
ceq = [];
end
