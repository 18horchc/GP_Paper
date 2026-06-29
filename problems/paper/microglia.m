clear; close all; clc

%% Data 

%averages 
newtime = [0, 1, 2, 3, 7, 14]; %These are the points where we observe experimental data
datapointsM2 = [5, 78.33, 179.5, 126.4, 319, 136.67]; %Experimental measurements of M2 (averaged per time point)
datapointsM1 = [5, 27.5, 122.5, 139.8, 445, 816.67]; %Experimental measurements of M1 (averaged per time point)

% full data
% Individual measurements compiled from the experimental studies (Fig. 1 / Table 1).
% Each study (color) reports at its own time points, so several measurements
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
