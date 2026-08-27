% Paper figure: Michaelis-Menten GP with virtual observations and Pensoneault
% upper bound + data-fidelity tube.
% Tabs:
%   1. Baseline SE-GP
%   2. Virtual function-value observations only
%   3. Virtual derivative observations only
%   4. Function VOs + derivative VOs (no bound)
%   5. Upper bound + data-fidelity tube
%   6. Function VOs + upper bound + tube
%   7. Derivative VOs + upper bound + tube
%   8. Derivative VOs + S=0 VO + upper bound + tube
%   9. Function VOs + derivative VOs + upper bound + tube
% eta = 2.2% => k from erfinv.
clear; clc; close all;

%% MM parameters
Vmax = 100;
Km   = 18;
mm_static = @(S) (Vmax .* S) ./ (Km + S);

%% Training data ([S] in mM, v_0 in μM/s)
x_max = 200;
noise_frac = 0.05;   % homoscedastic: sigma = noise_frac * max v on [0, x_max]
x_train = [10; 30; 60; 90; 200];
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

%% Fixed observation noises (same as MM_Obs_Figure; not optimized)
sigma_data = noise_sd_true;                 % known assay noise
sigma_VO_zero = 0.8 * noise_sd_true;        % tight soft-hard anchor at v(0)=0
sn_deriv = 0.1; %0.2 * noise_sd_true;             % Solak soft Gaussian derivative noise

%% Virtual function-value observations (heteroscedastic VO)
x_virt_zero = 0;
y_virt_zero = 0;
x_virt_150 = 150;
y_virt_150 = 86.8813;

x_virt = [x_virt_zero; x_virt_150];
y_virt = [y_virt_zero; y_virt_150];

x_aug = [x_obs; x_virt];
y_aug = [y_obs; y_virt];
noise_var_aug = [sigma_data^2 * ones(numel(y_obs), 1); ...
    sigma_VO_zero^2 * ones(numel(y_virt), 1)];

x_aug0 = [x_obs; x_virt_zero];
y_aug0 = [y_obs; y_virt_zero];
noise_var_aug0 = [sigma_data^2 * ones(numel(y_obs), 1); sigma_VO_zero^2];

fprintf('Virtual obs: v(0)=0 | v(%.0f)=%.4f (sigma=%.4g each)\n', ...
    x_virt_150, y_virt_150, sigma_VO_zero);

%% Virtual derivative observations (Solak; fixed separate sn_deriv)
x_deriv = linspace(100, 200, 5)';
y_deriv = 0.02 * ones(numel(x_deriv), 1);

fprintf('Virtual deriv obs: %d sites on [%.0f, %.0f] | y_deriv=%.2g | sn_deriv=%.4g (fixed)\n', ...
    numel(x_deriv), x_deriv(1), x_deriv(end), y_deriv(1), sn_deriv);

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
ell_ub = x_max;   % cap length scale at domain width
sf_bounds = [0.05, max(15, Vmax)];
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

fprintf('\neta = %.3g%% | k = %.4f | epsilon = %.4g | X_c: %d points | random starts: %d\n', ...
    100 * eta, k, epsilon, numel(X_c), nTry);

%% Virtual function-value observations only (heteroscedastic)
fprintf('\nOptimizing Virtual Obs GP (ell, sf; heteroscedastic VO)...\n');
obj_vo = @(h) gp_seiso_hetero_noise('nlml', h, x_aug, y_aug, noise_var_aug);
hyp_vo = struct('mean', [], 'cov', hyp.cov, 'lik', []);
hyp_vo = minimize(hyp_vo, obj_vo, -100);
nlml_vo = obj_vo(hyp_vo);
theta_vo_box = min(max(hyp_vo.cov(:), hyp_lb), hyp_ub);

%% Virtual derivative observations only
fprintf('\nOptimizing Virtual Deriv Obs GP (ell, sf; sn=%.4f, sn_deriv=%.4g)...\n', ...
    sigma_data, sn_deriv);
obj_deriv = @(hyp_cov) gp_seiso_deriv_obs_nlml_cov_only(hyp_cov, sn_fixed, ...
    x_col, y_col, x_deriv, y_deriv, sn_deriv);
hyp_cov_deriv = minimize(hyp.cov, obj_deriv, -100);
hyp_deriv = struct('mean', [], 'cov', hyp_cov_deriv(:), 'lik', sn_fixed);
nlml_deriv = obj_deriv(hyp_cov_deriv);
theta_deriv_box = min(max(hyp_cov_deriv(:), hyp_lb), hyp_ub);

%% VO + Solak deriv GP (unconstrained; warm start for combined bound fit)
fprintf('\nOptimizing VO+deriv GP (ell, sf; hetero VO + sn_deriv=%.4g)...\n', sn_deriv);
obj_voderiv = @(hyp_cov) gp_seiso_deriv_obs_nlml_cov_only(hyp_cov, sn_fixed, ...
    x_aug, y_aug, x_deriv, y_deriv, sn_deriv, noise_var_aug);
hyp_cov_voderiv = minimize(hyp.cov, obj_voderiv, -100);
hyp_voderiv = struct('mean', [], 'cov', hyp_cov_voderiv(:), 'lik', sn_fixed);
nlml_voderiv = obj_voderiv(hyp_cov_voderiv);
theta_voderiv_box = min(max(hyp_cov_voderiv(:), hyp_lb), hyp_ub);

%% Upper bound + data fidelity (no virtual obs)
fprintf('\n=== Pensoneault GP (upper bound at Vmax + data fidelity) ===\n');
nonlcon_up = @(theta) pens_constraints_upper_fid(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x_col, y_col, X_c, k, y_max, epsilon);
[hyp_up, nlml_up, exitflag_up, c_up] = fit_pens_constrained( ...
    objfun, nonlcon_up, hyp_tpl, hyp_lb, hyp_ub, theta_unc_box, opts_pens, nTry, nMultistart, 42);
nC = numel(X_c);
fprintf('Final max(c) = %.6g | upper = %.6g | data = %.6g\n', ...
    max(c_up), max(c_up(1:nC)), max(c_up(nC+1:end)));

%% Function VOs + upper bound + data fidelity
fprintf('\n=== Pensoneault GP (VO + upper bound + data fidelity) ===\n');
hyp_tpl_vo = hyp_vo;
objfun_vo_ub = @(theta) gp_seiso_hetero_noise('nlml', theta_to_hyp(theta, hyp_tpl_vo), ...
    x_aug, y_aug, noise_var_aug);
nonlcon_vo_ub = @(theta) pens_constraints_upper_hetero(theta, hyp_tpl_vo, ...
    x_aug, y_aug, noise_var_aug, X_c, k, y_max, epsilon, x_obs, y_obs);
[hyp_vo_ub, nlml_vo_ub, exitflag_vo_ub, c_vo_ub] = fit_pens_constrained( ...
    objfun_vo_ub, nonlcon_vo_ub, hyp_tpl_vo, hyp_lb, hyp_ub, theta_vo_box, ...
    opts_pens, nTry, nMultistart, 43);
fprintf('Final max(c) = %.6g | upper = %.6g | data = %.6g\n', ...
    max(c_vo_ub), max(c_vo_ub(1:nC)), max(c_vo_ub(nC+1:end)));

%% Derivative VOs + upper bound + data fidelity
fprintf('\n=== Pensoneault GP (deriv + upper bound + data fidelity) ===\n');
hyp_tpl_d = hyp_deriv;
objfun_d_ub = @(theta) gp_seiso_deriv_obs('nlml', theta_to_hyp(theta, hyp_tpl_d), ...
    x_col, y_col, x_deriv, y_deriv, [], sn_deriv, true, []);
nonlcon_d_ub = @(theta) pens_constraints_upper_deriv(theta, hyp_tpl_d, ...
    x_col, y_col, x_deriv, y_deriv, sn_deriv, [], X_c, k, y_max, ...
    epsilon, x_obs, y_obs);
[hyp_d_ub, nlml_d_ub, exitflag_d_ub, c_d_ub] = fit_pens_constrained( ...
    objfun_d_ub, nonlcon_d_ub, hyp_tpl_d, hyp_lb, hyp_ub, theta_deriv_box, ...
    opts_pens, nTry, nMultistart, 44);
fprintf('Final max(c) = %.6g | upper = %.6g | data = %.6g\n', ...
    max(c_d_ub), max(c_d_ub(1:nC)), max(c_d_ub(nC+1:end)));

%% Deriv + S=0 VO + upper bound + data fidelity
fprintf('\nOptimizing deriv + v(0)=0 VO (ell, sf; hetero VO + sn_deriv=%.4g)...\n', sn_deriv);
obj_d0 = @(hyp_cov) gp_seiso_deriv_obs_nlml_cov_only(hyp_cov, sn_fixed, ...
    x_aug0, y_aug0, x_deriv, y_deriv, sn_deriv, noise_var_aug0);
hyp_cov_d0 = minimize(hyp.cov, obj_d0, -100);
hyp_d0 = struct('mean', [], 'cov', hyp_cov_d0(:), 'lik', sn_fixed);
theta_d0_box = min(max(hyp_cov_d0(:), hyp_lb), hyp_ub);

fprintf('\n=== Pensoneault GP (deriv + v(0)=0 VO + upper bound + data fidelity) ===\n');
hyp_tpl_d0 = hyp_d0;
objfun_d0_ub = @(theta) gp_seiso_deriv_obs('nlml', theta_to_hyp(theta, hyp_tpl_d0), ...
    x_aug0, y_aug0, x_deriv, y_deriv, [], sn_deriv, true, noise_var_aug0);
nonlcon_d0_ub = @(theta) pens_constraints_upper_deriv(theta, hyp_tpl_d0, ...
    x_aug0, y_aug0, x_deriv, y_deriv, sn_deriv, noise_var_aug0, X_c, k, y_max, ...
    epsilon, x_obs, y_obs);
[hyp_d0_ub, nlml_d0_ub, exitflag_d0_ub, c_d0_ub] = fit_pens_constrained( ...
    objfun_d0_ub, nonlcon_d0_ub, hyp_tpl_d0, hyp_lb, hyp_ub, theta_d0_box, ...
    opts_pens, nTry, nMultistart, 46);
fprintf('Final max(c) = %.6g | upper = %.6g | data = %.6g\n', ...
    max(c_d0_ub), max(c_d0_ub(1:nC)), max(c_d0_ub(nC+1:end)));

%% Function VOs + deriv VOs + upper bound + data fidelity
fprintf('\n=== Pensoneault GP (VO + deriv + upper bound + data fidelity) ===\n');
hyp_tpl_both = hyp_voderiv;
objfun_both_ub = @(theta) gp_seiso_deriv_obs('nlml', theta_to_hyp(theta, hyp_tpl_both), ...
    x_aug, y_aug, x_deriv, y_deriv, [], sn_deriv, true, noise_var_aug);
nonlcon_both_ub = @(theta) pens_constraints_upper_deriv(theta, hyp_tpl_both, ...
    x_aug, y_aug, x_deriv, y_deriv, sn_deriv, noise_var_aug, X_c, k, y_max, ...
    epsilon, x_obs, y_obs);
[hyp_both_ub, nlml_both_ub, exitflag_both_ub, c_both_ub] = fit_pens_constrained( ...
    objfun_both_ub, nonlcon_both_ub, hyp_tpl_both, hyp_lb, hyp_ub, theta_voderiv_box, ...
    opts_pens, nTry, nMultistart, 45);
fprintf('Final max(c) = %.6g | upper = %.6g | data = %.6g\n', ...
    max(c_both_ub), max(c_both_ub(1:nC)), max(c_both_ub(nC+1:end)));

%% Predictions
k_plot = 2;
[~, ~, fmu_unc, fs2_unc] = gp(hyp_unc, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_unc = fmu_unc(:);
sf_unc = sqrt(max(fs2_unc(:), 0));

[~, ~, fmu_vo, fs2_vo] = gp_seiso_hetero_noise('pred', hyp_vo, x_aug, y_aug, noise_var_aug, x_grid(:));
m_vo = fmu_vo(:);
sf_vo = sqrt(max(fs2_vo(:), 0));

[~, ~, fmu_d, fs2_d] = gp_seiso_deriv_obs('pred', hyp_deriv, ...
    x_col, y_col, x_deriv, y_deriv, x_grid(:), sn_deriv);
m_deriv = fmu_d(:);
sf_deriv = sqrt(max(fs2_d(:), 0));

[~, ~, fmu_voderiv, fs2_voderiv] = gp_seiso_deriv_obs('pred', hyp_voderiv, ...
    x_aug, y_aug, x_deriv, y_deriv, x_grid(:), sn_deriv, true, noise_var_aug);
m_voderiv = fmu_voderiv(:);
sf_voderiv = sqrt(max(fs2_voderiv(:), 0));

[~, ~, fmu_up, fs2_up] = gp(hyp_up, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, x_grid(:));
m_up = fmu_up(:);
sf_up = sqrt(max(fs2_up(:), 0));

[~, ~, fmu_vo_ub, fs2_vo_ub] = gp_seiso_hetero_noise('pred', hyp_vo_ub, ...
    x_aug, y_aug, noise_var_aug, x_grid(:));
m_vo_ub = fmu_vo_ub(:);
sf_vo_ub = sqrt(max(fs2_vo_ub(:), 0));

[~, ~, fmu_d_ub, fs2_d_ub] = gp_seiso_deriv_obs('pred', hyp_d_ub, ...
    x_col, y_col, x_deriv, y_deriv, x_grid(:), sn_deriv);
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
    m_deriv + k_plot * sf_deriv; m_voderiv + k_plot * sf_voderiv; ...
    m_up + k_plot * sf_up; m_vo_ub + k_plot * sf_vo_ub; ...
    m_d_ub + k_plot * sf_d_ub; m_d0_ub + k_plot * sf_d0_ub; ...
    m_both_ub + k_plot * sf_both_ub]) * 1.02];

%% Tabbed figure
fig = figure('Color', 'w', 'Position', [80, 80, 900, 560], ...
    'Name', 'Michaelis-Menten GP: VO, deriv, and upper bound');
tg = uitabgroup(fig);

panels = struct( ...
    'm', {m_unc, m_vo, m_deriv, m_voderiv, m_up, m_vo_ub, m_d_ub, m_d0_ub, m_both_ub}, ...
    'sf', {sf_unc, sf_vo, sf_deriv, sf_voderiv, sf_up, sf_vo_ub, sf_d_ub, sf_d0_ub, sf_both_ub}, ...
    'title', {'Baseline GP', 'Virtual Obs GP', 'Virtual Deriv GP', ...
              'VO + Deriv GP', 'Upper-bound GP', 'VO + Upper-bound', ...
              'Deriv + Upper-bound', 'Deriv + v(0) + Upper-bound', ...
              'VO + Deriv + Upper-bound'}, ...
    'x_v', {[], x_virt, [], x_virt, [], x_virt, [], x_virt_zero, x_virt}, ...
    'y_v', {[], y_virt, [], y_virt, [], y_virt, [], y_virt_zero, y_virt}, ...
    'show_deriv', {false, false, true, true, false, false, true, true, true});

ax_list = gobjects(numel(panels), 1);
tab_list = gobjects(numel(panels), 1);
for p = 1:numel(panels)
    tab_list(p) = uitab(tg, 'Title', panels(p).title);
    ax_list(p) = axes('Parent', tab_list(p));
    x_d = [];
    if panels(p).show_deriv
        x_d = x_deriv;
    end
    plot_mm_bounds_panel(ax_list(p), panels(p).m, panels(p).sf, ...
        x_grid, y_true, x_obs, y_obs, k_plot, band_label, ylim_shared, x_max, Vmax, ...
        panels(p).x_v, panels(p).y_v, x_d);
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
fprintf('Baseline:         ell=%.4f, sf=%.4f, sn=%.4f | NLML=%.4f\n', ...
    exp(hyp_unc.cov(1)), exp(hyp_unc.cov(2)), exp(hyp_unc.lik), nlml_unc);
fprintf('Virtual Obs:      ell=%.4f, sf=%.4f | NLML=%.4f (n_aug=%d)\n', ...
    exp(hyp_vo.cov(1)), exp(hyp_vo.cov(2)), nlml_vo, numel(y_aug));
fprintf('Virtual Deriv:    ell=%.4f, sf=%.4f | NLML=%.4f\n', ...
    exp(hyp_deriv.cov(1)), exp(hyp_deriv.cov(2)), nlml_deriv);
fprintf('VO+deriv:         ell=%.4f, sf=%.4f | NLML=%.4f (n_aug=%d)\n', ...
    exp(hyp_voderiv.cov(1)), exp(hyp_voderiv.cov(2)), nlml_voderiv, numel(y_aug));
fprintf('Upper+tube:       ell=%.4f, sf=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g (upper=%.4g, data=%.4g)\n', ...
    exp(hyp_up.cov(1)), exp(hyp_up.cov(2)), nlml_up, exitflag_up, ...
    max(c_up), max(c_up(1:nC)), max(c_up(nC+1:end)));
fprintf('VO+upper:         ell=%.4f, sf=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g (upper=%.4g, data=%.4g)\n', ...
    exp(hyp_vo_ub.cov(1)), exp(hyp_vo_ub.cov(2)), nlml_vo_ub, exitflag_vo_ub, ...
    max(c_vo_ub), max(c_vo_ub(1:nC)), max(c_vo_ub(nC+1:end)));
fprintf('Deriv+upper:      ell=%.4f, sf=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g (upper=%.4g, data=%.4g)\n', ...
    exp(hyp_d_ub.cov(1)), exp(hyp_d_ub.cov(2)), nlml_d_ub, exitflag_d_ub, ...
    max(c_d_ub), max(c_d_ub(1:nC)), max(c_d_ub(nC+1:end)));
fprintf('Deriv+v(0)+upper: ell=%.4f, sf=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g (upper=%.4g, data=%.4g)\n', ...
    exp(hyp_d0_ub.cov(1)), exp(hyp_d0_ub.cov(2)), nlml_d0_ub, exitflag_d0_ub, ...
    max(c_d0_ub), max(c_d0_ub(1:nC)), max(c_d0_ub(nC+1:end)));
fprintf('VO+deriv+upper:   ell=%.4f, sf=%.4f | NLML=%.4f | exitflag=%d | max(c)=%.4g (upper=%.4g, data=%.4g)\n', ...
    exp(hyp_both_ub.cov(1)), exp(hyp_both_ub.cov(2)), nlml_both_ub, exitflag_both_ub, ...
    max(c_both_ub), max(c_both_ub(1:nC)), max(c_both_ub(nC+1:end)));

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

function [c, ceq] = pens_constraints_upper_fid(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x, y, X_c, k, y_max, epsilon)
% Upper bound on latent f at X_c; data-fidelity tube at training points.
hyp = theta_to_hyp(theta, hyp_tpl);
nC = numel(X_c);
xstar = [X_c(:); x(:)];
[ymu, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xstar);
m_xc = fmu(1:nC);
s_xc = sqrt(max(fs2(1:nC), 0));
c_upper = m_xc + k .* s_xc - y_max;
y_star = ymu(nC+1:end);
c_data = abs(y(:) - y_star) - epsilon;
c = [c_upper(:); c_data(:)];
ceq = [];
end

function [c, ceq] = pens_constraints_upper_hetero(theta, hyp_tpl, x, y, noise_var, ...
    X_c, k, y_max, epsilon, x_data, y_data)
% Upper bound + data-fidelity on heteroscedastic VO posterior.
hyp = theta_to_hyp(theta, hyp_tpl);
nC = numel(X_c);
xstar = [X_c(:); x_data(:)];
[ymu, ~, fmu, fs2] = gp_seiso_hetero_noise('pred', hyp, x, y, noise_var, xstar);
m_xc = fmu(1:nC);
s_xc = sqrt(max(fs2(1:nC), 0));
c_upper = m_xc + k .* s_xc - y_max;
y_star = ymu(nC+1:end);
c_data = abs(y_data(:) - y_star) - epsilon;
c = [c_upper(:); c_data(:)];
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

function [c, ceq] = pens_constraints_upper_deriv(theta, hyp_tpl, x, y, x_d, y_d, sn_deriv, ...
    noise_var, X_c, k, y_max, epsilon, x_data, y_data)
% Upper bound on latent f at X_c; data-fidelity tube at real training points.
%   mu_f + k*sigma_f <= y_max
%   |y - y*(x)| <= epsilon
hyp = theta_to_hyp(theta, hyp_tpl);
nC = numel(X_c);
xstar = [X_c(:); x_data(:)];
[ymu, ~, fmu, fs2] = gp_seiso_deriv_obs('pred', hyp, x, y, x_d, y_d, xstar, ...
    sn_deriv, true, noise_var);
m_xc = fmu(1:nC);
s_xc = sqrt(max(fs2(1:nC), 0));
c_upper = m_xc + k .* s_xc - y_max;
y_star = ymu(nC+1:end);
c_data = abs(y_data(:) - y_star) - epsilon;
c = [c_upper(:); c_data(:)];
ceq = [];
end

function plot_mm_bounds_panel(ax, m, sf, ...
    x_grid, y_true, x_obs, y_obs, k_plot, band_label, ylim_shared, x_max, Vmax, ...
    x_virt_zero, y_virt_zero, x_deriv)
if nargin < 13
    x_virt_zero = [];
    y_virt_zero = [];
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
if ~isempty(x_virt_zero)
    scatter(ax, x_virt_zero(:), y_virt_zero(:), 90, 'd', ...
        'MarkerFaceColor', [0.85, 0.85, 0.85], 'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.5, 'DisplayName', 'Virtual observations');
end
yh0 = yline(ax, 0, 'k:', 'v_0 = 0', 'Alpha', 0.5);
yh0.Annotation.LegendInformation.IconDisplayStyle = 'off';
yhV = yline(ax, Vmax, 'k:', 'V_{max}', 'Alpha', 0.5);
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
xlabel(ax, '[S] (mM)', 'FontSize', 18);
ylabel(ax, 'v_0 (\muM/s)', 'FontSize', 18);
end
