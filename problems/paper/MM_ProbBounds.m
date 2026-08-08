% Paper figure: Michaelis-Menten GP with Pensoneault probabilistic bounds.
% Combines naive SE-GP, lower-bound, upper-bound, and both-bound fits into
% one 2x2 figure:
%   top-left: naive GP | top-right: lower bound (mu - k*sigma >= 0)
%   bottom-left: upper bound (mu + k*sigma <= Vmax)
%   bottom-right: both bounds
% Constraints evaluated at 41 grid points. No data-fidelity tube.
% eta = 2.2%% => k from erfinv.
clear; clc; close all;

%% MM parameters
Vmax = 6;
Km   = .15;
mm_static = @(S) (Vmax .* S) ./ (Km + S);

%% Training data ([S] in mM, v_0 in μM/s)
x_max = 2;
noise_frac = 0.1;   % homoscedastic: sigma = noise_frac * max v on [0, x_max]
x_train = [0.0; 0.2; 0.4; 0.6; 0.8; 2.0];
n_train = numel(x_train);

rng(100);
v_true_at_train = mm_static(x_train);
y_domain_max = mm_static(x_max);
noise_sd_true = noise_frac * y_domain_max;
y_train = v_true_at_train + noise_sd_true * randn(size(v_true_at_train));
fprintf('Synthetic data: n=%d on [0, %.1f], homoscedastic noise sigma_n = %.4f (%.0f%% of v(x_max))\n', ...
    n_train, x_max, noise_sd_true, 100 * noise_frac);

x_obs = x_train(:);
y_obs = y_train(:);

%% Ground truth curve
x_grid = linspace(0, x_max, 500);
y_true = mm_static(x_grid);

%% Pensoneault constraint grid at X_c
eta = 0.022;   % 2.2% tail probability
k   = -sqrt(2) * erfinv(2 * eta - 1);
n_constraint = 41;
X_c = linspace(0, x_max, n_constraint)';
y_max = Vmax;

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
ell_ub = 3;   % cap length scale at domain width
sf_bounds = [0.05, 15];
meanfunc = @meanZero;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

x_col = x_train(:);
y_col = y_train(:);

%% Baseline / naive GP (sigma_n fixed at noise_sd_true; optimize ell, sf only)
sn_fixed = log(noise_sd_true);
fprintf('Optimizing baseline (ell, sf; sigma_n fixed at %.4f)...\n', noise_sd_true);
obj_unc = @(hyp_cov) gp_nlml_cov_only(hyp_cov, sn_fixed, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
hyp_cov_unc = minimize(hyp.cov, obj_unc, -100);
hyp_unc = struct('mean', [], 'cov', hyp_cov_unc(:), 'lik', sn_fixed);
nlml_unc = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
theta_unc = hyp_unc.cov(:);

hyp_lb = log([ell_bounds_lo; sf_bounds(1)]);
hyp_ub = log([ell_ub; sf_bounds(2)]);
hyp_tpl = hyp_unc;
objfun = @(theta) gp(theta_to_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
opts_pens = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, 'Display', 'off', ...
    'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);
nTry = 2000;
nMultistart = 10;
theta_unc_box = min(max(theta_unc, hyp_lb), hyp_ub);

fprintf('\neta = %.3g%% | k = %.4f | X_c: %d points | random starts: %d\n', ...
    100 * eta, k, numel(X_c), nTry);

%% Lower-bound GP
fprintf('\n=== Pensoneault GP (lower bound at 0) ===\n');
nonlcon_lo = @(theta) pens_constraints_lower(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x_col, y_col, X_c, k);
[hyp_lo, nlml_lo, exitflag_lo, c_lo] = fit_pens_constrained( ...
    objfun, nonlcon_lo, hyp_tpl, hyp_lb, hyp_ub, theta_unc_box, opts_pens, nTry, nMultistart, 42);
fprintf('Final max(c) = %.6g (feasible if <= 0)\n', max(c_lo));

%% Upper-bound GP
fprintf('\n=== Pensoneault GP (upper bound at Vmax) ===\n');
nonlcon_up = @(theta) pens_constraints_upper(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x_col, y_col, X_c, k, y_max);
[hyp_up, nlml_up, exitflag_up, c_up] = fit_pens_constrained( ...
    objfun, nonlcon_up, hyp_tpl, hyp_lb, hyp_ub, theta_unc_box, opts_pens, nTry, nMultistart, 42);
fprintf('Final max(c) = %.6g (feasible if <= 0)\n', max(c_up));

%% Both-bounds GP
fprintf('\n=== Pensoneault GP (lower + upper bounds) ===\n');
nonlcon_both = @(theta) pens_constraints_both(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x_col, y_col, X_c, k, y_max);
[hyp_both, nlml_both, exitflag_both, c_both] = fit_pens_constrained( ...
    objfun, nonlcon_both, hyp_tpl, hyp_lb, hyp_ub, theta_unc_box, opts_pens, nTry, nMultistart, 42);
nC = numel(X_c);
fprintf('Final max(c) = %.6g (feasible if <= 0)\n', max(c_both));
fprintf('  lower max(c) = %.6g | upper max(c) = %.6g\n', ...
    max(c_both(1:nC)), max(c_both(nC+1:end)));

%% Predictions
k_plot = 2;
[~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_unc = fmu_unc(:);
sf_unc = sqrt(max(fs2_unc(:), 0));

[~, ~, fmu_lo, fs2_lo] = gp(hyp_lo, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_lo = fmu_lo(:);
sf_lo = sqrt(max(fs2_lo(:), 0));

[~, ~, fmu_up, fs2_up] = gp(hyp_up, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_up = fmu_up(:);
sf_up = sqrt(max(fs2_up(:), 0));

[~, ~, fmu_both, fs2_both] = gp(hyp_both, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_both = fmu_both(:);
sf_both = sqrt(max(fs2_both(:), 0));

band_label = sprintf('\\mu_f \\pm %g\\sigma_f (latent)', k_plot);
ylim_shared = [-1, max([y_train(:); Vmax; ...
    m_unc + k_plot * sf_unc; m_lo + k_plot * sf_lo; ...
    m_up + k_plot * sf_up; m_both + k_plot * sf_both]) * 1.02];

%% Tabbed figure: naive | lower / upper | both
fig = figure('Color', 'w', 'Position', [80, 80, 900, 560], ...
    'Name', 'Michaelis-Menten GP: probabilistic bounds');
tg = uitabgroup(fig);

panels(1) = struct('m', m_unc,  'sf', sf_unc,  'title', 'Baseline GP');
panels(2) = struct('m', m_lo,   'sf', sf_lo,   'title', 'Lower-bound GP');
panels(3) = struct('m', m_up,   'sf', sf_up,   'title', 'Upper-bound GP');
panels(4) = struct('m', m_both, 'sf', sf_both, 'title', 'Both-bounds GP');

ax_list = gobjects(4, 1);
tab_list = gobjects(4, 1);
for p = 1:4
    tab_list(p) = uitab(tg, 'Title', panels(p).title);
    ax_list(p) = axes('Parent', tab_list(p));
    plot_mm_bounds_panel(ax_list(p), panels(p).m, panels(p).sf, ...
        x_grid, y_true, x_obs, y_obs, k_plot, band_label, ylim_shared, x_max, Vmax);
    title(ax_list(p), panels(p).title, 'Interpreter', 'none', 'FontSize', 18);
end

%% Standalone legend (for LaTeX / Inkscape)
figL = figure('Color', 'w', 'Position', [100, 100, 700, 80], ...
    'Name', 'MM ProbBounds shared legend');
axL = axes('Parent', figL, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
    'Position', [0 0 1 1]);
hold(axL, 'on');
hL = gobjects(4, 1);
hL(1) = fill(axL, nan, nan, [0.72, 0.72, 0.78], 'EdgeColor', 'none', ...
    'FaceAlpha', 0.5, 'DisplayName', '95% CI');
hL(2) = plot(axL, nan, nan, 'k--', 'LineWidth', 2, ...
    'DisplayName', 'GP Mean');
hL(3) = plot(axL, nan, nan, 'b-', 'LineWidth', 1.5, ...
    'DisplayName', 'True Model');
hL(4) = plot(axL, nan, nan, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', 'Observed Data');
lgd = legend(axL, hL, 'Orientation', 'horizontal');
lgd.FontSize = 14;
lgd.ItemTokenSize = [20, 12];
drawnow;
figL.Units = 'pixels';
lgd.Units = 'pixels';
lp = lgd.Position;
margin = 6;
figL.Position(3:4) = [lp(3) + 2 * margin, lp(4) + 2 * margin];
lgd.Position = [margin, margin, lp(3), lp(4)];

% %% Save each tab and the shared legend as EPS
% plot_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
%     'results', 'plots', 'Paper Draft 2', 'Enzyme Kinetics');
% if ~exist(plot_dir, 'dir')
%     mkdir(plot_dir);
% end
% name_list = {'MM_ProbBounds_Baseline_GP.eps', 'MM_ProbBounds_Lower_bound_GP.eps', ...
%     'MM_ProbBounds_Upper_bound_GP.eps', 'MM_ProbBounds_Both_bounds_GP.eps'};
% for i = 1:numel(tab_list)
%     tg.SelectedTab = tab_list(i);
%     ax_list(i).Toolbar.Visible = 'off';
%     disableDefaultInteractivity(ax_list(i));
%     drawnow;
%     out_path = fullfile(plot_dir, name_list{i});
%     exportgraphics(ax_list(i), out_path, 'ContentType', 'image');
%     fprintf('Saved %s\n', out_path);
% end
% legend_path = fullfile(plot_dir, 'MM_ProbBounds_legend.eps');
% exportgraphics(figL, legend_path, 'ContentType', 'image', 'BackgroundColor', 'white');
% fprintf('Saved %s\n', legend_path);

%% Console report
fprintf('\nNaive / baseline: ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);
fprintf('Lower bound:      ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(hyp_lo.cov(1)), exp(hyp_lo.cov(2)), exp(hyp_lo.lik), nlml_lo, exitflag_lo, max(c_lo));
fprintf('Upper bound:      ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(hyp_up.cov(1)), exp(hyp_up.cov(2)), exp(hyp_up.lik), nlml_up, exitflag_up, max(c_up));
fprintf('Both bounds:      ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(hyp_both.cov(1)), exp(hyp_both.cov(2)), exp(hyp_both.lik), nlml_both, exitflag_both, max(c_both));

%% ----- local functions -----
function [hyp_con, nlml_con, exitflag_con, c_final] = fit_pens_constrained( ...
    objfun, nonlcon, hyp_tpl, hyp_lb, hyp_ub, theta_unc_box, opts_pens, nTry, nMultistart, rng_seed)

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
nlml_con = nan;
exitflag_con = -99;
nStarts = size(starts_for_fmincon, 2);
for j = 1:nStarts
    theta0_j = starts_for_fmincon(:, j);
    [theta_j, nlml_j, ef_j] = fmincon(objfun, theta0_j, [], [], [], [], hyp_lb, hyp_ub, nonlcon, opts_pens);
    if isfinite(nlml_j) && nlml_j < best_nlml
        best_nlml = nlml_j;
        theta_opt = theta_j;
        nlml_con = nlml_j;
        exitflag_con = ef_j;
    end
end

if ~isfinite(best_nlml)
    if nFeas > 0
        theta_opt = best_feas_theta;
    else
        theta_opt = theta_unc_box;
    end
    nlml_con = objfun(theta_opt);
    exitflag_con = -99;
    fprintf('Warning: no successful fmincon run; using fallback theta.\n');
end

hyp_con = theta_to_hyp(theta_opt, hyp_tpl);
[c_final, ~] = nonlcon(theta_opt);
end

function hyp = theta_to_hyp(theta, hyp_tpl)
hyp = hyp_tpl;
hyp.cov = theta(1:2);
hyp.mean = [];
end

function [c, ceq] = pens_constraints_lower(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k)
% mu_f - k*sigma_f >= 0  <=>  c <= 0
hyp = theta_to_hyp(theta, hyp_tpl);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, X_c(:));
m_xc = fmu(:);
s_xc = sqrt(max(fs2(:), 0));
c = k .* s_xc - m_xc;
ceq = [];
end

function [c, ceq] = pens_constraints_upper(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k, y_max)
% mu_f + k*sigma_f <= y_max  <=>  c <= 0
hyp = theta_to_hyp(theta, hyp_tpl);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, X_c(:));
m_xc = fmu(:);
s_xc = sqrt(max(fs2(:), 0));
c = m_xc + k .* s_xc - y_max;
ceq = [];
end

function [c, ceq] = pens_constraints_both(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k, y_max)
% lower: mu - k*sigma >= 0; upper: mu + k*sigma <= y_max
hyp = theta_to_hyp(theta, hyp_tpl);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, X_c(:));
m_xc = fmu(:);
s_xc = sqrt(max(fs2(:), 0));
c_lower = k .* s_xc - m_xc;
c_upper = m_xc + k .* s_xc - y_max;
c = [c_lower; c_upper];
ceq = [];
end

function plot_mm_bounds_panel(ax, m, sf, ...
    x_grid, y_true, x_obs, y_obs, k_plot, band_label, ylim_shared, x_max, Vmax)
ax.Layer = 'top';
hold(ax, 'on');
grid(ax, 'on');
fill(ax, [x_grid, fliplr(x_grid)], [m + k_plot * sf; flipud(m - k_plot * sf)]', ...
    [0.72, 0.72, 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
plot(ax, x_grid, m, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
plot(ax, x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
plot(ax, x_obs, y_obs, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', 'Observed data');
yh0 = yline(ax, 0, 'k:', 'v_0 = 0', 'Alpha', 0.5);
yh0.Annotation.LegendInformation.IconDisplayStyle = 'off';
yhV = yline(ax, Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
yhV.Annotation.LegendInformation.IconDisplayStyle = 'off';
ax.FontSize = 18;
xlabel(ax, '[S] (mM)', 'FontSize', 18);
ylabel(ax, 'v_0 (\muM/s)', 'FontSize', 18);
xlim(ax, [0, x_max]);
ylim(ax, ylim_shared);
end
