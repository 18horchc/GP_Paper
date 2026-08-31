% Paper figure: Michaelis-Menten GP with virtual observations and Pensoneault
% upper bound, monotonicity (f' >= 0), and data-fidelity tube.
% Tabs:
%   1. Baseline SE-GP
%   2. Virtual function-value observations only
%   3. Solak deriv + data-fidelity tube
%   4. Function VOs + Solak deriv + tube
%   5. Upper bound + data-fidelity tube
%   6. Function VOs + upper bound + tube
%   7. Solak deriv + upper bound + tube
%   8. S=0 VO + Solak deriv + upper bound + tube
%   9. Function VOs + Solak deriv + upper bound + tube
% Currently disabled (commented out): saturation interpolant VO; Pensoneault f'>=0.
% eta from erfinv.
clear; clc; close all;

%% MM parameters
Vmax = 100;
Km   = 18;
mm_static = @(S) (Vmax .* S) ./ (Km + S);

%% Training data ([S] in mM, v_0 in μM/s)
x_max = 200;
noise_frac = 0.05;   % homoscedastic: sigma = noise_frac * max v on [0, x_max]
x_sites = [10; 30; 60; 90; 200];
n_rep = 3;   % independent assays at each S, same N(0, sigma_n^2)
x_train = repelem(x_sites, n_rep);
n_train = numel(x_train);

rng(100);
v_true_at_train = mm_static(x_train);
y_domain_max = mm_static(x_max);
noise_sd_true = noise_frac * y_domain_max;
% Truncated-normal assays: redraw if y > Vmax. GP likelihood is unchanged.
y_train = v_true_at_train + noise_sd_true * randn(size(v_true_at_train));
n_reject = 0;
bad = y_train > Vmax;
while any(bad)
    n_reject = n_reject + sum(bad);
    y_train(bad) = v_true_at_train(bad) + noise_sd_true * randn(sum(bad), 1);
    bad = y_train > Vmax;
end
fprintf(['Synthetic data: %d sites x %d replicates (n=%d) on [0, %.1f], ', ...
    'homoscedastic noise sigma_n = %.4f (%.0f%% of v(x_max))\n'], ...
    numel(x_sites), n_rep, n_train, x_max, noise_sd_true, 100 * noise_frac);
if n_reject > 0
    fprintf('Redrawn %d Gaussian draw(s) with y > Vmax (truncated at Vmax).\n', n_reject);
end

x_obs = x_train(:);
y_obs = y_train(:);

%% Fixed observation noises (not optimized)
sigma_data = noise_sd_true;                 % known assay noise
sigma_VO_zero = 0.8 * noise_sd_true;        % tight soft-hard anchor at v(0)=0

%% Virtual function-value observations (heteroscedastic VO)
x_virt_zero = 0;
y_virt_zero = 0;
% Saturation VO (noisy linear interpolant at midpoint of S=90 and S=200):
% x_virt_mid = 0.5 * (90 + 200);
% y_virt_mid = interp1([90; 200], [mean(y_train(x_train == 90)); mean(y_train(x_train == 200))], ...
%     x_virt_mid, 'linear');

x_virt = x_virt_zero;              % [x_virt_zero; x_virt_mid]
y_virt = y_virt_zero;              % [y_virt_zero; y_virt_mid]

x_aug = [x_obs; x_virt];
y_aug = [y_obs; y_virt];
noise_var_aug = [sigma_data^2 * ones(numel(y_obs), 1); ...
    sigma_VO_zero^2 * ones(numel(y_virt), 1)];

x_aug0 = [x_obs; x_virt_zero];
y_aug0 = [y_obs; y_virt_zero];
noise_var_aug0 = [sigma_data^2 * ones(numel(y_obs), 1); sigma_VO_zero^2];

fprintf('Virtual obs: v(0)=0 (sigma=%.4g); saturation interpolant VO commented out\n', ...
    sigma_VO_zero);

%% Solak virtual derivative observations
x_deriv = [150; 175; 200];
y_deriv = zeros(size(x_deriv));
sn_deriv = 0.1; %0.2 * noise_sd_true;  
fprintf('Solak deriv obs: %d sites at S=[%s], v''=0 | sn_deriv=%.4g\n', ...
    numel(x_deriv), strjoin(compose('%.0f', x_deriv), ', '), sn_deriv);

%% Ground truth curve
x_grid = linspace(0, x_max, 500);
y_true = mm_static(x_grid);

%% Pensoneault constraint grid at X_c
eta = 0.022;   % 2.2% tail probability
k   = -sqrt(2) * erfinv(2 * eta - 1);
n_constraint = 800;
X_c = linspace(0, x_max, n_constraint)';
y_max = Vmax;
epsilon = 2*noise_sd_true;   % data fidelity: |y - y*(x)| <= epsilon at training pts
use_mono = false;            % Pensoneault f'(S)>=0 bound (set true to restore)

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

ell0 = std(x_sites);
sf0  = std(y_train);
sn0  = max(1e-3, noise_sd_true);
hyp = struct('mean', [], 'cov', log([ell0; sf0]), 'lik', log(sn0));

ell_bounds_lo = 0.05;
ell_ub = x_max;   % cap length scale at domain width
sf_bounds = [0.05, max(15, Vmax)];
meanfunc = @meanZero;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

x_col = x_train(:);
y_col = y_train(:);

hyp_lb = log([ell_bounds_lo; sf_bounds(1)]);
hyp_ub = log([ell_ub; sf_bounds(2)]);
sn_fixed = log(noise_sd_true);
hyp_tpl = struct('mean', [], 'cov', hyp.cov(:), 'lik', sn_fixed);
nC = numel(X_c);

opts_pens = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, 'Display', 'off', ...
    'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);
nTry = 2000;
nMultistart = 10;

fprintf('\neta = %.3g%% | k = %.4f | epsilon = %.4g | X_c: %d points | random starts: %d\n', ...
    100 * eta, k, epsilon, numel(X_c), nTry);

%% Baseline / naive GP (sigma_n fixed; 10 bounded NLML multistarts)
fprintf('\nOptimizing baseline (ell, sf; sigma_n fixed at %.4f; %d multistarts)...\n', ...
    noise_sd_true, nMultistart);
obj_unc = @(theta) gp_nlml_cov_only(theta, sn_fixed, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
[hyp_unc, nlml_unc] = fit_nlml_multistart(obj_unc, hyp_tpl, hyp.cov(:), ...
    hyp_lb, hyp_ub, nMultistart, opts_pens, 40);
theta_unc_box = min(max(hyp_unc.cov(:), hyp_lb), hyp_ub);
objfun = @(theta) gp(theta_to_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x_col, y_col);

%% Virtual function-value observations only (heteroscedastic)
fprintf('\nOptimizing Virtual Obs GP (ell, sf; heteroscedastic VO; %d multistarts)...\n', nMultistart);
hyp_tpl_vo = struct('mean', [], 'cov', hyp.cov(:), 'lik', sn_fixed);
obj_vo = @(theta) gp_seiso_hetero_noise('nlml', theta_to_hyp(theta, hyp_tpl_vo), ...
    x_aug, y_aug, noise_var_aug);
[hyp_vo, nlml_vo] = fit_nlml_multistart(obj_vo, hyp_tpl_vo, hyp.cov(:), ...
    hyp_lb, hyp_ub, nMultistart, opts_pens, 41);
theta_vo_box = min(max(hyp_vo.cov(:), hyp_lb), hyp_ub);

%% v(0) VO template (tab 8); unconstrained v(0)+Solak warm start is below
hyp_tpl_vo0 = struct('mean', [], 'cov', hyp.cov(:), 'lik', sn_fixed);

%% Solak deriv unconstrained warm starts
hyp_tpl_d = struct('mean', [], 'cov', hyp.cov(:), 'lik', sn_fixed);
fprintf('\nOptimizing Solak deriv GP (ell, sf; v''=0 at S=[%s], sn_deriv=%.4g; %d multistarts)...\n', ...
    strjoin(compose('%.0f', x_deriv), ', '), sn_deriv, nMultistart);
obj_d = @(theta) gp_seiso_deriv_obs_nlml_cov_only(theta, sn_fixed, ...
    x_col, y_col, x_deriv, y_deriv, sn_deriv);
[hyp_d_unc, nlml_d_unc] = fit_nlml_multistart(obj_d, hyp_tpl_d, hyp.cov(:), ...
    hyp_lb, hyp_ub, nMultistart, opts_pens, 50);
theta_d_box = min(max(hyp_d_unc.cov(:), hyp_lb), hyp_ub);

fprintf('\nOptimizing VO+Solak deriv GP (ell, sf; %d multistarts)...\n', nMultistart);
obj_vd = @(theta) gp_seiso_deriv_obs_nlml_cov_only(theta, sn_fixed, ...
    x_aug, y_aug, x_deriv, y_deriv, sn_deriv, noise_var_aug);
[hyp_vd_unc, nlml_vd_unc] = fit_nlml_multistart(obj_vd, hyp_tpl_vo, hyp.cov(:), ...
    hyp_lb, hyp_ub, nMultistart, opts_pens, 51);
theta_vd_box = min(max(hyp_vd_unc.cov(:), hyp_lb), hyp_ub);

fprintf('\nOptimizing v(0)+Solak deriv GP (ell, sf; %d multistarts)...\n', nMultistart);
obj_d0 = @(theta) gp_seiso_deriv_obs_nlml_cov_only(theta, sn_fixed, ...
    x_aug0, y_aug0, x_deriv, y_deriv, sn_deriv, noise_var_aug0);
[hyp_d0_unc, nlml_d0_unc] = fit_nlml_multistart(obj_d0, hyp_tpl_vo0, hyp.cov(:), ...
    hyp_lb, hyp_ub, nMultistart, opts_pens, 52);
theta_d0_box = min(max(hyp_d0_unc.cov(:), hyp_lb), hyp_ub);

%% Solak deriv + data fidelity (no virtual obs)
fprintf('\n=== Pensoneault GP (Solak deriv + data fidelity) ===\n');
objfun_d = @(theta) gp_seiso_deriv_obs('nlml', theta_to_hyp(theta, hyp_tpl_d), ...
    x_col, y_col, x_deriv, y_deriv, [], sn_deriv, true, []);
nonlcon_d = @(theta) pens_constraints_upper_mono(theta, hyp_tpl_d, x_col, y_col, [], ...
    X_c, k, y_max, epsilon, x_obs, y_obs, false, use_mono, x_deriv, y_deriv, sn_deriv);
[hyp_d, nlml_d, exitflag_d, c_d] = fit_pens_constrained( ...
    objfun_d, nonlcon_d, hyp_tpl_d, hyp_lb, hyp_ub, theta_d_box, opts_pens, nTry, nMultistart, 49);
fprintf_c_blocks(c_d, nC, false, use_mono);

%% Function VOs + Solak deriv + data fidelity
fprintf('\n=== Pensoneault GP (VO + Solak deriv + data fidelity) ===\n');
objfun_vo = @(theta) gp_seiso_hetero_noise('nlml', theta_to_hyp(theta, hyp_tpl_vo), ...
    x_aug, y_aug, noise_var_aug);
objfun_vd = @(theta) gp_seiso_deriv_obs('nlml', theta_to_hyp(theta, hyp_tpl_vo), ...
    x_aug, y_aug, x_deriv, y_deriv, [], sn_deriv, true, noise_var_aug);
nonlcon_vo_d = @(theta) pens_constraints_upper_mono(theta, hyp_tpl_vo, ...
    x_aug, y_aug, noise_var_aug, X_c, k, y_max, epsilon, x_obs, y_obs, false, use_mono, ...
    x_deriv, y_deriv, sn_deriv);
[hyp_vo_d, nlml_vo_d, exitflag_vo_d, c_vo_d] = fit_pens_constrained( ...
    objfun_vd, nonlcon_vo_d, hyp_tpl_vo, hyp_lb, hyp_ub, theta_vd_box, ...
    opts_pens, nTry, nMultistart, 48);
fprintf_c_blocks(c_vo_d, nC, false, use_mono);

%% Upper bound + data fidelity (no virtual obs)
fprintf('\n=== Pensoneault GP (upper bound at Vmax + data fidelity) ===\n');
nonlcon_up = @(theta) pens_constraints_upper_mono(theta, hyp_tpl, x_col, y_col, [], ...
    X_c, k, y_max, epsilon, x_obs, y_obs, true, false);
[hyp_up, nlml_up, exitflag_up, c_up] = fit_pens_constrained( ...
    objfun, nonlcon_up, hyp_tpl, hyp_lb, hyp_ub, theta_unc_box, opts_pens, nTry, nMultistart, 42);
fprintf_c_blocks(c_up, nC, true, false);

%% Function VOs + upper bound + data fidelity
fprintf('\n=== Pensoneault GP (VO + upper bound + data fidelity) ===\n');
nonlcon_vo_ub = @(theta) pens_constraints_upper_mono(theta, hyp_tpl_vo, ...
    x_aug, y_aug, noise_var_aug, X_c, k, y_max, epsilon, x_obs, y_obs, true, false);
[hyp_vo_ub, nlml_vo_ub, exitflag_vo_ub, c_vo_ub] = fit_pens_constrained( ...
    objfun_vo, nonlcon_vo_ub, hyp_tpl_vo, hyp_lb, hyp_ub, theta_vo_box, ...
    opts_pens, nTry, nMultistart, 43);
fprintf_c_blocks(c_vo_ub, nC, true, false);

%% Solak deriv + upper bound + data fidelity
fprintf('\n=== Pensoneault GP (Solak deriv + upper bound + data fidelity) ===\n');
nonlcon_d_ub = @(theta) pens_constraints_upper_mono(theta, hyp_tpl_d, x_col, y_col, [], ...
    X_c, k, y_max, epsilon, x_obs, y_obs, true, use_mono, x_deriv, y_deriv, sn_deriv);
[hyp_d_ub, nlml_d_ub, exitflag_d_ub, c_d_ub] = fit_pens_constrained( ...
    objfun_d, nonlcon_d_ub, hyp_tpl_d, hyp_lb, hyp_ub, theta_d_box, ...
    opts_pens, nTry, nMultistart, 44);
fprintf_c_blocks(c_d_ub, nC, true, use_mono);

%% v(0)=0 VO + Solak deriv + upper bound + data fidelity
fprintf('\n=== Pensoneault GP (v(0)=0 VO + Solak deriv + upper bound + data fidelity) ===\n');
objfun_d0 = @(theta) gp_seiso_deriv_obs('nlml', theta_to_hyp(theta, hyp_tpl_vo0), ...
    x_aug0, y_aug0, x_deriv, y_deriv, [], sn_deriv, true, noise_var_aug0);
nonlcon_d0_ub = @(theta) pens_constraints_upper_mono(theta, hyp_tpl_vo0, ...
    x_aug0, y_aug0, noise_var_aug0, X_c, k, y_max, epsilon, x_obs, y_obs, true, use_mono, ...
    x_deriv, y_deriv, sn_deriv);
[hyp_d0_ub, nlml_d0_ub, exitflag_d0_ub, c_d0_ub] = fit_pens_constrained( ...
    objfun_d0, nonlcon_d0_ub, hyp_tpl_vo0, hyp_lb, hyp_ub, theta_d0_box, ...
    opts_pens, nTry, nMultistart, 46);
fprintf_c_blocks(c_d0_ub, nC, true, use_mono);

%% Function VOs + Solak deriv + upper bound + data fidelity
fprintf('\n=== Pensoneault GP (VO + Solak deriv + upper bound + data fidelity) ===\n');
nonlcon_both_ub = @(theta) pens_constraints_upper_mono(theta, hyp_tpl_vo, ...
    x_aug, y_aug, noise_var_aug, X_c, k, y_max, epsilon, x_obs, y_obs, true, use_mono, ...
    x_deriv, y_deriv, sn_deriv);
[hyp_both_ub, nlml_both_ub, exitflag_both_ub, c_both_ub] = fit_pens_constrained( ...
    objfun_vd, nonlcon_both_ub, hyp_tpl_vo, hyp_lb, hyp_ub, theta_vd_box, ...
    opts_pens, nTry, nMultistart, 45);
fprintf_c_blocks(c_both_ub, nC, true, use_mono);

%% Predictions (latent f)
k_plot = 2;
[~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_unc = fmu_unc(:);
sf_unc = sqrt(max(fs2_unc(:), 0));

[~, ~, fmu_vo, fs2_vo] = gp_seiso_hetero_noise('pred', hyp_vo, x_aug, y_aug, noise_var_aug, x_grid(:));
m_vo = fmu_vo(:);
sf_vo = sqrt(max(fs2_vo(:), 0));

[~, ~, fmu_d, fs2_d] = gp_seiso_deriv_obs('pred', hyp_d, ...
    x_col, y_col, x_deriv, y_deriv, x_grid(:), sn_deriv, true, []);
m_d = fmu_d(:);
sf_d = sqrt(max(fs2_d(:), 0));

[~, ~, fmu_vo_d, fs2_vo_d] = gp_seiso_deriv_obs('pred', hyp_vo_d, ...
    x_aug, y_aug, x_deriv, y_deriv, x_grid(:), sn_deriv, true, noise_var_aug);
m_vo_d = fmu_vo_d(:);
sf_vo_d = sqrt(max(fs2_vo_d(:), 0));

[~, ~, fmu_up, fs2_up] = gp(hyp_up, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_up = fmu_up(:);
sf_up = sqrt(max(fs2_up(:), 0));

[~, ~, fmu_vo_ub, fs2_vo_ub] = gp_seiso_hetero_noise('pred', hyp_vo_ub, ...
    x_aug, y_aug, noise_var_aug, x_grid(:));
m_vo_ub = fmu_vo_ub(:);
sf_vo_ub = sqrt(max(fs2_vo_ub(:), 0));

[~, ~, fmu_d_ub, fs2_d_ub] = gp_seiso_deriv_obs('pred', hyp_d_ub, ...
    x_col, y_col, x_deriv, y_deriv, x_grid(:), sn_deriv, true, []);
m_d_ub = fmu_d_ub(:);
sf_d_ub = sqrt(max(fs2_d_ub(:), 0));

[~, ~, fmu_d0_ub, fs2_d0_ub] = gp_seiso_deriv_obs('pred', hyp_d0_ub, ...
    x_aug0, y_aug0, x_deriv, y_deriv, x_grid(:), sn_deriv, true, noise_var_aug0);
m_d0_ub = fmu_d0_ub(:);
sf_d0_ub = sqrt(max(fs2_d0_ub(:), 0));

[~, ~, fmu_both_ub, fs2_both_ub] = gp_seiso_deriv_obs('pred', hyp_both_ub, ...
    x_aug, y_aug, x_deriv, y_deriv, x_grid(:), sn_deriv, true, noise_var_aug);
m_both_ub = fmu_both_ub(:);
sf_both_ub = sqrt(max(fs2_both_ub(:), 0));

band_label = sprintf('\\mu_f \\pm %g\\sigma_f (latent)', k_plot);
ylim_shared = [0, max([y_train(:); Vmax; ...
    m_unc + k_plot * sf_unc; m_vo + k_plot * sf_vo; ...
    m_d + k_plot * sf_d; m_vo_d + k_plot * sf_vo_d; ...
    m_up + k_plot * sf_up; m_vo_ub + k_plot * sf_vo_ub; ...
    m_d_ub + k_plot * sf_d_ub; m_d0_ub + k_plot * sf_d0_ub; ...
    m_both_ub + k_plot * sf_both_ub]) * 1.02];

%% Tabbed figure
fig = figure('Color', 'w', 'Position', [80, 80, 900, 560], ...
    'Name', 'Michaelis-Menten GP: VO, Solak deriv, and upper bound');
tg = uitabgroup(fig);

panels = struct( ...
    'm', {m_unc, m_vo, m_d, m_vo_d, m_up, m_vo_ub, m_d_ub, m_d0_ub, m_both_ub}, ...
    'sf', {sf_unc, sf_vo, sf_d, sf_vo_d, sf_up, sf_vo_ub, sf_d_ub, sf_d0_ub, sf_both_ub}, ...
    'title', {'Baseline', 'Boundary VO', 'Deriv VO', ...
              'Boundary VO + Deriv VO', 'Upper-bound', 'Boundary VO + Upper-bound', ...
              'Deriv VO + Upper-bound', 'Boundary VO + Deriv VO + Upper-bound', ...
              'Boundary VO + Deriv VO + Upper-bound'}, ...
    'x_v', {[], x_virt, [], x_virt, [], x_virt, [], x_virt_zero, x_virt}, ...
    'y_v', {[], y_virt, [], y_virt, [], y_virt, [], y_virt_zero, y_virt}, ...
    'show_deriv', {false, false, true, true, false, false, true, true, true});

ax_list = gobjects(numel(panels), 1);
tab_list = gobjects(numel(panels), 1);
for p = 1:numel(panels)
    tab_list(p) = uitab(tg, 'Title', panels(p).title);
    ax_list(p) = axes('Parent', tab_list(p));
    x_d_plot = [];
    if panels(p).show_deriv
        x_d_plot = x_deriv;
    end
    plot_mm_bounds_panel(ax_list(p), panels(p).m, panels(p).sf, ...
        x_grid, y_true, x_obs, y_obs, k_plot, band_label, ylim_shared, x_max, Vmax, ...
        panels(p).x_v, panels(p).y_v, x_d_plot);
    title(ax_list(p), panels(p).title, 'Interpreter', 'none', 'FontSize', 18);
end

%% Standalone legend (for LaTeX / Inkscape)
figL = figure('Color', 'w', 'Position', [100, 100, 900, 80], ...
    'Name', 'MM Figure shared legend');
axL = axes('Parent', figL, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
    'Position', [0 0 1 1]);
hold(axL, 'on');
hL = gobjects(6, 1);
hL(1) = fill(axL, nan, nan, [0.72, 0.72, 0.78], 'EdgeColor', 'none', ...
    'FaceAlpha', 0.5, 'DisplayName', '95% CI');
hL(2) = plot(axL, nan, nan, 'k--', 'LineWidth', 2, ...
    'DisplayName', 'GP Mean');
hL(3) = plot(axL, nan, nan, 'b-', 'LineWidth', 1.5, ...
    'DisplayName', 'True Model');
hL(4) = plot(axL, nan, nan, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', 'Observed Data');
hL(5) = scatter(axL, nan, nan, 90, 'd', ...
    'MarkerFaceColor', [0.85, 0.85, 0.85], 'MarkerEdgeColor', 'k', ...
    'LineWidth', 1.5, 'DisplayName', 'Virtual Obs');
hL(6) = plot(axL, nan, nan, '^', 'LineStyle', 'none', 'MarkerSize', 9, ...
    'LineWidth', 0.8, 'MarkerFaceColor', [0.55, 0.25, 0.65], ...
    'MarkerEdgeColor', 'k', 'DisplayName', 'Virtual Deriv Obs');
lgd = legend(axL, hL, 'Orientation', 'horizontal');
lgd.FontSize = 14;
lgd.ItemTokenSize = [20, 12];
lgd.Box = 'on';
drawnow;
figL.Units = 'pixels';
lgd.Units = 'pixels';
lp = lgd.Position;
margin = 4;
figL.Position(3:4) = [lp(3) + 2 * margin, lp(4) + 2 * margin];
lgd.Position = [margin, margin, lp(3), lp(4)];
axL.Position = [0 0 1 1];
drawnow;

% %% Save each tab and the shared legend as EPS
% plot_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
%     'results', 'plots', 'Paper Draft 2', 'Enzyme Kinetics');
% if ~exist(plot_dir, 'dir')
%     mkdir(plot_dir);
% end
% name_list = {'MM_Figure_Baseline_GP.eps', 'MM_Figure_Lower_bound_GP.eps', ...
%     'MM_Figure_Upper_bound_GP.eps', 'MM_Figure_Both_bounds_GP.eps'};
% for i = 1:numel(tab_list)
%     tg.SelectedTab = tab_list(i);
%     ax_list(i).Toolbar.Visible = 'off';
%     disableDefaultInteractivity(ax_list(i));
%     drawnow;
%     out_path = fullfile(plot_dir, name_list{i});
%     exportgraphics(ax_list(i), out_path, 'ContentType', 'image');
%     fprintf('Saved %s\n', out_path);
% end
% legend_path = fullfile(plot_dir, 'MM_Figure_legend.eps');
% exportgraphics(figL, legend_path, 'ContentType', 'image', 'BackgroundColor', 'white');
% fprintf('Saved %s\n', legend_path);

%% Console report
fprintf('\nFixed noises: sigma_data=%.4f | sigma_VO_zero=%.4g | sn_deriv=%.4g | epsilon=%.4g\n', ...
    sigma_data, sigma_VO_zero, sn_deriv, epsilon);
% fprintf('Interior VO at S=%.0f from noisy interp: y=%.4f\n', x_virt_mid, y_virt_mid);
fprintf('Solak deriv:        v''=0 at S=[%s]\n', strjoin(compose('%.0f', x_deriv), ', '));
fprintf('Baseline:           ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);
fprintf('Virtual Obs:        ell=%.4f, sf=%.4f | NLML=%.4f (n_aug=%d)\n', ...
    exp(hyp_vo.cov(1)), exp(hyp_vo.cov(2)), nlml_vo, numel(y_aug));
fprintf('Solak deriv unc:    ell=%.4f, sf=%.4f | NLML=%.4f\n', ...
    exp(hyp_d_unc.cov(1)), exp(hyp_d_unc.cov(2)), nlml_d_unc);
fprintf('Deriv+tube:         ell=%.4f, sf=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(hyp_d.cov(1)), exp(hyp_d.cov(2)), nlml_d, exitflag_d, max(c_d));
fprintf('VO+deriv:           ell=%.4f, sf=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(hyp_vo_d.cov(1)), exp(hyp_vo_d.cov(2)), nlml_vo_d, exitflag_vo_d, max(c_vo_d));
fprintf('Upper+tube:         ell=%.4f, sf=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(hyp_up.cov(1)), exp(hyp_up.cov(2)), nlml_up, exitflag_up, max(c_up));
fprintf('VO+upper:           ell=%.4f, sf=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(hyp_vo_ub.cov(1)), exp(hyp_vo_ub.cov(2)), nlml_vo_ub, exitflag_vo_ub, max(c_vo_ub));
fprintf('Deriv+upper:        ell=%.4f, sf=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(hyp_d_ub.cov(1)), exp(hyp_d_ub.cov(2)), nlml_d_ub, exitflag_d_ub, max(c_d_ub));
fprintf('v(0)+deriv+upper:   ell=%.4f, sf=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(hyp_d0_ub.cov(1)), exp(hyp_d0_ub.cov(2)), nlml_d0_ub, exitflag_d0_ub, max(c_d0_ub));
fprintf('VO+deriv+upper:     ell=%.4f, sf=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g\n', ...
    exp(hyp_both_ub.cov(1)), exp(hyp_both_ub.cov(2)), nlml_both_ub, exitflag_both_ub, max(c_both_ub));

%% ----- local functions -----
function [hyp_opt, nlml_opt] = fit_nlml_multistart(objfun, hyp_tpl, theta0, ...
    hyp_lb, hyp_ub, nMultistart, opts, rng_seed)
% Box-constrained NLML: heuristic start plus random starts in [hyp_lb, hyp_ub].
theta0 = min(max(theta0(:), hyp_lb), hyp_ub);
rng(rng_seed);
starts = theta0;
for i = 2:nMultistart
    starts = [starts, hyp_lb + rand(2, 1) .* (hyp_ub - hyp_lb)]; %#ok<AGROW>
end

best_nlml = inf;
theta_opt = theta0;
for j = 1:size(starts, 2)
    [theta_j, nlml_j] = fmincon(objfun, starts(:, j), [], [], [], [], ...
        hyp_lb, hyp_ub, [], opts);
    if isfinite(nlml_j) && nlml_j < best_nlml
        best_nlml = nlml_j;
        theta_opt = theta_j;
    end
end
if ~isfinite(best_nlml)
    theta_opt = theta0;
    best_nlml = objfun(theta_opt);
    fprintf('Warning: no successful unconstrained fmincon run; using heuristic theta.\n');
end
hyp_opt = theta_to_hyp(theta_opt, hyp_tpl);
nlml_opt = best_nlml;
end

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
        feasible_starts = [feasible_starts, theta_try]; %#ok<AGROW>
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

function [c, ceq] = pens_constraints_upper_mono(theta, hyp_tpl, x, y, noise_var, ...
    X_c, k, y_max, epsilon, x_data, y_data, do_upper, do_mono, x_d, y_d, sn_d)
% Pensoneault blocks on the function-value GP, optionally with Solak deriv obs.
%   c_upper:  mu_f + k*sigma_f <= y_max at X_c
%   c_mono:   mu_f' - k*sigma_f' >= 0  <=>  k*sigma_f' - mu_f' <= 0 at X_c
%             (currently disabled: callers pass use_mono=false)
%   c_data:   |y - mu_y(x)| <= epsilon at real assay points
if nargin < 15
    x_d = [];
    y_d = [];
    sn_d = 0;
end
hyp = theta_to_hyp(theta, hyp_tpl);
nC = numel(X_c);
c = zeros(0, 1);

need_f = do_upper || ~isempty(x_data);
if need_f
    xstar = [X_c(:); x_data(:)];
    [ymu, ~, fmu, fs2] = gp_seiso_deriv_obs('pred', hyp, x, y, x_d, y_d, xstar, ...
        sn_d, true, noise_var);
    if do_upper
        m_xc = fmu(1:nC);
        s_xc = sqrt(max(fs2(1:nC), 0));
        c = [c; m_xc + k .* s_xc - y_max];
    end
end
% Pensoneault monotonicity f'(S) >= 0 (commented out):
% if do_mono
%     [m_d, s2_d] = gp_seiso_deriv_obs('deriv', hyp, x, y, x_d, y_d, X_c(:), ...
%         sn_d, true, noise_var);
%     s_d = sqrt(max(s2_d(:), 0));
%     c = [c; k .* s_d - m_d(:)];
% end
if ~isempty(x_data)
    y_star = ymu(nC+1:end);
    c = [c; abs(y_data(:) - y_star) - epsilon];
end
ceq = [];
end

function fprintf_c_blocks(c, nC, do_upper, do_mono)
i = 0;
parts = {sprintf('max(c)=%.6g', max(c))};
if do_upper
    parts{end+1} = sprintf('upper=%.6g', max(c(i+1:i+nC))); %#ok<AGROW>
    i = i + nC;
end
if do_mono
    parts{end+1} = sprintf('mono=%.6g', max(c(i+1:i+nC))); %#ok<AGROW>
    i = i + nC;
end
parts{end+1} = sprintf('data=%.6g', max(c(i+1:end)));
fprintf('Final %s\n', strjoin(parts, ' | '));
end

function plot_mm_bounds_panel(ax, m, sf, ...
    x_grid, y_true, x_obs, y_obs, k_plot, band_label, ylim_shared, x_max, Vmax, ...
    x_virt, y_virt, x_deriv)
if nargin < 13
    x_virt = [];
    y_virt = [];
end
if nargin < 15
    x_deriv = [];
end
ax.Layer = 'top';
hold(ax, 'on');
grid(ax, 'on');
fill(ax, [x_grid, fliplr(x_grid)], [m + k_plot * sf; flipud(m - k_plot * sf)]', ...
    [0.72, 0.72, 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', band_label);
plot(ax, x_grid, m, 'k--', 'LineWidth', 2, 'DisplayName', 'Posterior mean \mu_f');
plot(ax, x_grid, y_true, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Ground truth (MM)');
plot(ax, x_obs, y_obs, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, ...
    'DisplayName', 'Observed data');
if ~isempty(x_virt)
    scatter(ax, x_virt(:), y_virt(:), 90, 'd', ...
        'MarkerFaceColor', [0.85, 0.85, 0.85], 'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.5, 'DisplayName', 'Virtual observations');
end
yh0 = yline(ax, 0, 'k:', 'Alpha', 0.5);
yh0.Annotation.LegendInformation.IconDisplayStyle = 'off';
yhV = yline(ax, Vmax, 'k:', 'V_{max}', 'Alpha', 0.5, 'FontSize', 18);
yhV.Annotation.LegendInformation.IconDisplayStyle = 'off';
xlim(ax, [0, x_max]);
ylim(ax, ylim_shared);
if ~isempty(x_deriv)
    x_mark = x_deriv(isfinite(x_deriv) & x_deriv >= 0 & x_deriv <= x_max);
    if ~isempty(x_mark)
        y_mark = ylim_shared(1);
        h_deriv = plot(ax, x_mark, repmat(y_mark, numel(x_mark), 1), '^', ...
            'LineStyle', 'none', 'MarkerSize', 9, 'LineWidth', 0.8, ...
            'MarkerFaceColor', [0.55, 0.25, 0.65], 'MarkerEdgeColor', 'k', ...
            'Clipping', 'off', 'DisplayName', 'Solak deriv obs locations');
        uistack(h_deriv, 'top');
    end
end
ax.FontSize = 18;
yhV.FontSize = 18;
xlabel(ax, '[S] (mM)', 'FontSize', 18);
ylabel(ax, 'v_0 (\muM/s)', 'FontSize', 18);
end
