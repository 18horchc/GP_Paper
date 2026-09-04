% Paper figure: Lotka-Volterra GP comparison (naive independent Per+SE vs. LMC).
% Naive GP fits each state independently with k = covPeriodic + covSEiso.
% The Linear Model of Coregionalization (LMC) couples prey and predator
% through Q=2 shared latent processes with the same temporal kernels:
%   k_1 = periodic kernel with a short length scale (oscillation)
%   k_2 = SE with a long length scale (smooth shared trend)
% Each latent process q gets a full-rank 2x2 coregionalization matrix B_q,
% built with GPML's covDiscrete (B_q = L_q'*L_q, PSD by construction). A short
% periodic length scale and a long SE length scale are encouraged with
% Gaussian hyperpriors via infPrior; the base-kernel signal variances are
% clamped to 1 so the cross-output magnitude lives entirely in B_q.
% Naive and LMC fits are shown as tabs in one figure (prey / predator).
% % Pensoneault lower bound + data-fidelity tube currently commented out.
% Standalone shared legend for LaTeX / Inkscape.
% Requires gp_nlml_cov_only.m on the path (problems/) and the GPML toolbox.
clear; clc; close all;

%% Lotka-Volterra parameters
alpha = 1.1;   % prey growth rate
beta  = 0.4;   % predation rate
delta = 0.1;   % predator reproduction
gamma = 0.4;   % predator death rate
prey0 = 10;
pred0 = 5;

odefun = @(t, z) [ ...
    alpha * z(1) - beta * z(1) * z(2); ...
    delta * z(1) * z(2) - gamma * z(2)];
z0 = [prey0; pred0];

%% Ground truth (dense ode45 solve, ~3 cycles)
t_min = 0;
t_max = 30;
x_grid = linspace(t_min, t_max, 600)';
[~, z_grid] = ode45(odefun, x_grid, z0);
y_true_grid = max(z_grid, 0);   % [n_grid x 2] -> columns [prey, predator]

%% Training data (homoscedastic: sigma_n = noise_frac * SD of true curve)
% % --- Shared-grid sampling (original) ---
% rng(100);
% n_train = 10;
% x_train = linspace(t_min, t_max, n_train)';   % shared sample times for both states
%
% y_true_train = interp1(x_grid, y_true_grid, x_train, 'pchip');
% y_true_train = max(y_true_train, 0);
%
% noise_frac_prey = 0.01;   % 1% of SD of true prey curve
% noise_frac_pred = 0.30;   % 30% of SD of true predator curve
% sd_true_prey = std(y_true_grid(:, 1));
% sd_true_pred = std(y_true_grid(:, 2));
% sn_prey = noise_frac_prey * sd_true_prey;   % y ~ N(y_true, sigma_n^2)
% sn_pred = noise_frac_pred * sd_true_pred;
%
% y_train_prey = y_true_train(:, 1) + sn_prey * randn(n_train, 1);
% y_train_pred = y_true_train(:, 2) + sn_pred * randn(n_train, 1);
%
% fprintf('Synthetic LV data: n=%d per state on [%.0f, %.0f]\n', n_train, t_min, t_max);
% fprintf(['Homoscedastic noise: sigma_prey=%.4f (%.0f%% of SD of true prey curve = %.4f), ', ...
%     'sigma_pred=%.4f (%.0f%% of SD of true pred curve = %.4f)\n'], ...
%     sn_prey, 100 * noise_frac_prey, sd_true_prey, sn_pred, 100 * noise_frac_pred, sd_true_pred);

% --- Dense predator, sparse prey with a full-period gap ---
rng(100);
noise_frac_prey = 0.01;   % 1% of SD of true prey curve
noise_frac_pred = 0.30;   % 30% of SD of true predator curve
sd_true_prey = std(y_true_grid(:, 1));
sd_true_pred = std(y_true_grid(:, 2));
sn_prey = noise_frac_prey * sd_true_prey;   % y ~ N(y_true, sigma_n^2)
sn_pred = noise_frac_pred * sd_true_pred;

T_period = 10;   % ~one LV cycle (~30/3; matches periodic init p0)
gap_lo = (t_min + t_max) / 2 - T_period / 2;   % gap starts at 10
gap_hi = gap_lo + T_period;                     % gap ends at 20

n_pred_times = 25;                              % dense predator over full window
nRep_pred = 2;                                  % replicates at each predator time
t_pred = linspace(t_min, t_max, n_pred_times)';

n_prey_side = 3;                                % sparse prey on each side of the gap (6 total)
x_train_prey = [linspace(t_min, gap_lo, n_prey_side)'; ...
                linspace(gap_hi, t_max, n_prey_side)'];
n_prey = numel(x_train_prey);

y_true_prey = max(interp1(x_grid, y_true_grid(:, 1), x_train_prey, 'pchip'), 0);
y_true_pred_t = max(interp1(x_grid, y_true_grid(:, 2), t_pred, 'pchip'), 0);

x_train_pred = repelem(t_pred, nRep_pred);
y_true_pred = repelem(y_true_pred_t, nRep_pred);
n_pred = numel(x_train_pred);

% Truncated-normal: redraw if y < 0. GP likelihood is unchanged.
y_train_prey = y_true_prey + sn_prey * randn(n_prey, 1);
y_train_pred = y_true_pred + sn_pred * randn(n_pred, 1);
n_reject = 0;
bad_prey = y_train_prey < 0;
bad_pred = y_train_pred < 0;
while any(bad_prey) || any(bad_pred)
    n_reject = n_reject + sum(bad_prey) + sum(bad_pred);
    if any(bad_prey)
        y_train_prey(bad_prey) = y_true_prey(bad_prey) + sn_prey * randn(sum(bad_prey), 1);
    end
    if any(bad_pred)
        y_train_pred(bad_pred) = y_true_pred(bad_pred) + sn_pred * randn(sum(bad_pred), 1);
    end
    bad_prey = y_train_prey < 0;
    bad_pred = y_train_pred < 0;
end

fprintf('Synthetic LV data: n_prey=%d (gap [%.1f, %.1f]), n_pred=%d times x %d rep (%d obs) on [%.0f, %.0f]\n', ...
    n_prey, gap_lo, gap_hi, n_pred_times, nRep_pred, n_pred, t_min, t_max);
fprintf(['Homoscedastic noise: sigma_prey=%.4f (%.0f%% of SD of true prey curve = %.4f), ', ...
    'sigma_pred=%.4f (%.0f%% of SD of true pred curve = %.4f)\n'], ...
    sn_prey, 100 * noise_frac_prey, sd_true_prey, sn_pred, 100 * noise_frac_pred, sd_true_pred);
if n_reject > 0
    fprintf('Redrawn %d Gaussian draw(s) with y < 0 (truncated at population 0).\n', n_reject);
end

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

meanfunc = @meanZero;
likfunc  = @likGauss;

% Shared temporal-kernel inits (naive independent GPs and LMC).
ellLong  = (t_max - t_min) / 2;    % ~15: smooth trend over the window
ellShort = 0.5;                    % sub-period roughness for the oscillation
s2_ell   = 0.5^2;                  % hyperprior variance on log(ell)
p0       = 10;                     % period init (~30/3 cycles)

%% ===== Naive baseline: independent Per+SE GP per state =====
% Each state is centered and scaled (zero mean, unit std), matching LMC.
% sigma_n is the known per-state noise in those standardized units.
% Same temporal kernels as LMC (Per + SE) and the same short/long ell priors;
% signal variances are free (no coregionalization).
covIndep = {@covSum, {@covPeriodic, @covSEiso}};
fprintf('\n=== Naive (independent Per + SE) ===\n');
[per_prey.m, per_prey.sf, per_prey.hyp, per_prey.nlml] = fit_per_se_state( ...
    x_train_prey, y_train_prey, sn_prey, p0, ellShort, ellLong, s2_ell, ...
    x_grid, meanfunc, covIndep, likfunc);
[per_pred.m, per_pred.sf, per_pred.hyp, per_pred.nlml] = fit_per_se_state( ...
    x_train_pred, y_train_pred, sn_pred, p0, ellShort, ellLong, s2_ell, ...
    x_grid, meanfunc, covIndep, likfunc);

%% ===== LMC: Q=2 latent processes with full-rank coregionalization =====
fprintf('\n=== LMC (Q=2, periodic + SE, full-rank B_q) ===\n');

% Standardize each output (zero mean, unit std) so a single Gaussian noise is
% well-scaled across the two states; predictions are un-standardized below.
mu_prey = mean(y_train_prey);  sd_prey = std(y_train_prey);
mu_pred = mean(y_train_pred);  sd_pred = std(y_train_pred);

LABEL_PREY = 1; LABEL_PRED = 2;
x_aug = [ [x_train_prey; x_train_pred], ...
          [LABEL_PREY * ones(n_prey, 1); LABEL_PRED * ones(n_pred, 1)] ];
y_aug = [ (y_train_prey - mu_prey) / sd_prey; (y_train_pred - mu_pred) / sd_pred ];

% Composed LMC kernel: sum over q of [ base kernel over time ] .* [ B_q over label ]
covLMC = {@covSum, { ...
    {@covProd, { {@covMask, {1, @covPeriodic}}, {@covMask, {2, {@covDiscrete, 2}}} }}, ...
    {@covProd, { {@covMask, {1, @covSEiso}},    {@covMask, {2, {@covDiscrete, 2}}} }} }};

% B_q Cholesky init: B_q = 0.5*I so each output gets unit variance summed over q.
% covDiscrete hyp = [log(L11); L21; log(L22)] with B = L'*L.
Lchol0 = [log(sqrt(0.5)); 0; log(sqrt(0.5))];

hyp_lmc.mean = [];
hyp_lmc.cov  = [ log(ellShort); log(p0); 0;  Lchol0; ...   % Periodic: ell1, p1, sf1(clamped), B1
                 log(ellLong); 0;            Lchol0 ];      % SE: ell2, sf2(clamped), B2
hyp_lmc.lik  = log(0.1);           % single Gaussian noise (standardized units), optimized

% Priors: encourage short/long length scales, clamp base-kernel signal variances.
prior.cov = { {@priorGauss, log(ellShort), s2_ell}, [], @priorClamped, [], [], [], ...
              {@priorGauss, log(ellLong), s2_ell}, @priorClamped, [], [], [] };
inffunc_lmc = {@infPrior, @infGaussLik, prior};

fprintf('Optimizing LMC hyperparameters (infPrior + infGaussLik)...\n');
hyp_lmc = minimize(hyp_lmc, @gp, -1000, inffunc_lmc, meanfunc, covLMC, likfunc, x_aug, y_aug);
nlml_lmc = gp(hyp_lmc, inffunc_lmc, meanfunc, covLMC, likfunc, x_aug, y_aug);

% Predict each state on the grid (carrying its label), then un-standardize.
x_te_prey = [x_grid, LABEL_PREY * ones(size(x_grid))];
x_te_pred = [x_grid, LABEL_PRED * ones(size(x_grid))];
[~, ~, fmu_p, fs2_p] = gp(hyp_lmc, inffunc_lmc, meanfunc, covLMC, likfunc, x_aug, y_aug, x_te_prey);
[~, ~, fmu_q, fs2_q] = gp(hyp_lmc, inffunc_lmc, meanfunc, covLMC, likfunc, x_aug, y_aug, x_te_pred);

lmc_prey.m  = mu_prey + sd_prey * fmu_p(:);
lmc_prey.sf = sd_prey * sqrt(max(fs2_p(:), 0));
lmc_pred.m  = mu_pred + sd_pred * fmu_q(:);
lmc_pred.sf = sd_pred * sqrt(max(fs2_q(:), 0));

% %% Pensoneault lower bound + data-fidelity tube on LMC (prey only)
% eta = 0.022;
% k_pens = -sqrt(2) * erfinv(2 * eta - 1);
% n_constraint = 41;
% X_c = linspace(t_min, t_max, n_constraint)';
% eps_prey = 75 * sn_prey;   % |y - y*(x)| <= epsilon at prey training points
% nTry = 2000;
% nMultistart = 10;
% opts_pens = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
%     'EnableFeasibilityMode', true, 'Display', 'off', ...
%     'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
%     'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);
%
% scale = struct('mu_prey', mu_prey, 'sd_prey', sd_prey, ...
%     'mu_pred', mu_pred, 'sd_pred', sd_pred, ...
%     'LABEL_PREY', LABEL_PREY, 'LABEL_PRED', LABEL_PRED, ...
%     'eps_prey', eps_prey);
% [lb_lmc, ub_lmc] = bound_box_lmc(t_min, t_max);
%
% fprintf('\n=== Pensoneault lower bound + data tube, LMC (prey only) ===\n');
% fprintf('eta = %.3g%% | k = %.4f | X_c: %d points on [%.0f, %.0f]\n', ...
%     100 * eta, k_pens, n_constraint, t_min, t_max);
% fprintf('epsilon_prey = %.4f (2*sn)\n', eps_prey);
%
% lmc_b = fit_lmc_lower_bound(x_aug, y_aug, hyp_lmc, X_c, k_pens, x_grid, scale, ...
%     inffunc_lmc, meanfunc, covLMC, likfunc, lb_lmc, ub_lmc, opts_pens, nTry, nMultistart, 48);
% lmc_prey_b = lmc_b.prey;
% lmc_pred_b = lmc_b.pred;

%% Tabbed figure (one tab per method × state)
k_plot = 2;
band_label = sprintf('\\pm %g\\sigma_f', k_plot);
col_prey = [0.00, 0.45, 0.74];   % blue
col_pred = [0.85, 0.16, 0.16];   % red

% Include the true curve; omit bound-GP bands so they cannot squash the axis.
% ylo_prey = min([0; y_train_prey(:); y_true_grid(:, 1); ...
%     per_prey.m - k_plot * per_prey.sf; lmc_prey.m - k_plot * lmc_prey.sf]);
% yhi_prey = max([y_train_prey(:); y_true_grid(:, 1); ...
%     per_prey.m + k_plot * per_prey.sf; lmc_prey.m + k_plot * lmc_prey.sf]);
% pad_prey = 0.05 * (yhi_prey - ylo_prey);
% ylim_prey = [ylo_prey - pad_prey, yhi_prey + pad_prey];
% 
% ylo_pred = min([0; y_train_pred(:); y_true_grid(:, 2); ...
%     per_pred.m - k_plot * per_pred.sf; lmc_pred.m - k_plot * lmc_pred.sf]);
% yhi_pred = max([y_train_pred(:); y_true_grid(:, 2); ...
%     per_pred.m + k_plot * per_pred.sf; lmc_pred.m + k_plot * lmc_pred.sf]);
% pad_pred = 0.05 * (yhi_pred - ylo_pred);
% ylim_pred = [ylo_pred - pad_pred, yhi_pred + pad_pred];

ylim_prey = [-5, 15];
ylim_pred = [-5, 15];

panels(1) = struct('fit', per_prey, 'y_true', y_true_grid(:, 1), ...
    'x_obs', x_train_prey, 'y_obs', y_train_prey, 'col', col_prey, ...
    'ylim', ylim_prey, 'state', 'Prey', 'show_zero', false, ...
    'tab', 'Naive Prey', 'title', 'Baseline GP (Independent Per + SE) — Prey', ...
    'fname', 'LV_LMC_Naive_Prey.eps');
panels(2) = struct('fit', per_pred, 'y_true', y_true_grid(:, 2), ...
    'x_obs', x_train_pred, 'y_obs', y_train_pred, 'col', col_pred, ...
    'ylim', ylim_pred, 'state', 'Predator', 'show_zero', false, ...
    'tab', 'Naive Pred', 'title', 'Baseline GP (Independent Per + SE) — Predator', ...
    'fname', 'LV_LMC_Naive_Predator.eps');
panels(3) = struct('fit', lmc_prey, 'y_true', y_true_grid(:, 1), ...
    'x_obs', x_train_prey, 'y_obs', y_train_prey, 'col', col_prey, ...
    'ylim', ylim_prey, 'state', 'Prey', 'show_zero', false, ...
    'tab', 'LMC Prey', 'title', 'LMC GP (Q=2: Periodic + SE) — Prey', ...
    'fname', 'LV_LMC_LMC_Prey.eps');
panels(4) = struct('fit', lmc_pred, 'y_true', y_true_grid(:, 2), ...
    'x_obs', x_train_pred, 'y_obs', y_train_pred, 'col', col_pred, ...
    'ylim', ylim_pred, 'state', 'Predator', 'show_zero', false, ...
    'tab', 'LMC Pred', 'title', 'LMC GP (Q=2: Periodic + SE) — Predator', ...
    'fname', 'LV_LMC_LMC_Predator.eps');
% panels(5) = struct('fit', lmc_prey_b, 'y_true', y_true_grid(:, 1), ...
%     'x_obs', x_train_prey, 'y_obs', y_train_prey, 'col', col_prey, ...
%     'ylim', ylim_prey, 'state', 'Prey', 'show_zero', true, ...
%     'tab', 'LMC Bound Prey', 'title', 'LMC + Lower Bound + Tube — Prey', ...
%     'fname', 'LV_LMC_LMC_Bound_Prey.eps');
% panels(6) = struct('fit', lmc_pred_b, 'y_true', y_true_grid(:, 2), ...
%     'x_obs', x_train_pred, 'y_obs', y_train_pred, 'col', col_pred, ...
%     'ylim', ylim_pred, 'state', 'Predator', 'show_zero', false, ...
%     'tab', 'LMC Bound Pred', 'title', 'LMC + Bound (prey lower bound + tube) — Predator', ...
%     'fname', 'LV_LMC_LMC_Bound_Predator.eps');

fig = figure('Color', 'w', 'Position', [80, 80, 900, 560], ...
    'Name', 'Lotka-Volterra GP: independent Per+SE vs LMC');
tg = uitabgroup(fig);
ax_list = gobjects(numel(panels), 1);
tab_list = gobjects(numel(panels), 1);
for pidx = 1:numel(panels)
    tab_list(pidx) = uitab(tg, 'Title', panels(pidx).tab);
    ax = axes('Parent', tab_list(pidx));
    ax.Layer = 'top';
    ax.FontSize = 16;
    hold(ax, 'on'); grid(ax, 'on');
    plot_state(ax, x_grid, panels(pidx).y_true, panels(pidx).fit, ...
        panels(pidx).x_obs, panels(pidx).y_obs, panels(pidx).col, ...
        k_plot, panels(pidx).state, band_label);
    if panels(pidx).show_zero
        yline(ax, 0, 'k:', 'HandleVisibility', 'off');
    end
    xlabel(ax, 't', 'FontSize', 16);
    ylabel(ax, 'Population', 'FontSize', 16);
    title(ax, panels(pidx).title, 'Interpreter', 'none', 'FontSize', 16);
    xlim(ax, [t_min, t_max]);
    ylim(ax, panels(pidx).ylim);
    ax_list(pidx) = ax;
end

%% Standalone shared legend (for LaTeX / Inkscape)
figL = figure('Color', 'w', 'Position', [100, 100, 900, 80], ...
    'Name', 'LV LMC shared legend');
axL = axes('Parent', figL, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
    'Position', [0 0 1 1]);
hold(axL, 'on');
hL = gobjects(8, 1);
hL(1) = fill(axL, nan, nan, col_prey, 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', 'Prey 95% CI');
hL(2) = plot(axL, nan, nan, '-', 'Color', col_prey, 'LineWidth', 1.5, ...
    'DisplayName', 'Prey True Model');
hL(3) = plot(axL, nan, nan, '--', 'Color', col_prey, 'LineWidth', 2, ...
    'DisplayName', 'Prey GP Mean');
hL(4) = plot(axL, nan, nan, 'o', 'Color', col_prey, 'MarkerFaceColor', col_prey, ...
    'MarkerEdgeColor', 'k', 'MarkerSize', 5, 'DisplayName', 'Prey Obs Data');
hL(5) = fill(axL, nan, nan, col_pred, 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', 'Predator 95% CI');
hL(6) = plot(axL, nan, nan, '-', 'Color', col_pred, 'LineWidth', 1.5, ...
    'DisplayName', 'Predator True Model');
hL(7) = plot(axL, nan, nan, '--', 'Color', col_pred, 'LineWidth', 2, ...
    'DisplayName', 'Predator GP Mean');
hL(8) = plot(axL, nan, nan, 'o', 'Color', col_pred, 'MarkerFaceColor', col_pred, ...
    'MarkerEdgeColor', 'k', 'MarkerSize', 5, 'DisplayName', 'Predator Obs Data');
lgd = legend(axL, hL, 'Orientation', 'horizontal', 'NumColumns', 4);
lgd.FontSize = 16;
lgd.ItemTokenSize = [20, 12];
drawnow;
figL.Units = 'pixels';
lgd.Units = 'pixels';
lp = lgd.Position;
margin = 6;
figL.Position(3:4) = [lp(3) + 2 * margin, lp(4) + 2 * margin];
lgd.Position = [margin, margin, lp(3), lp(4)];

%% Save Naive/LMC panels as EPS
plot_dir = 'C:\Users\chorc\OneDrive\Documents\Stroke Research\Gaussian Processes\Bio_Inf_GP_Code\results\plots\Paper Draft 2\Interacting Species';
if ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end
for i = 1:numel(ax_list)
    ax_list(i).Toolbar.Visible = 'off';
    disableDefaultInteractivity(ax_list(i));
    drawnow;
    out_path = fullfile(plot_dir, panels(i).fname);
    exportgraphics(ax_list(i), out_path, 'ContentType', 'vector');
    fprintf('Saved %s\n', out_path);
end
legend_path = fullfile(plot_dir, 'LV_LMC_legend.eps');
exportgraphics(figL, legend_path, 'ContentType', 'vector', 'BackgroundColor', 'white');
fprintf('Saved %s\n', legend_path);

%% Console report
fprintf('\n--- Fitted hyperparameters ---\n');
fprintf('Naive  Prey: Per ell=%.4f, p=%.4f, sf=%.4f | SE ell=%.4f, sf=%.4f | sn(std)=%.4f | NLML=%.4f\n', ...
    exp(per_prey.hyp.cov(1)), exp(per_prey.hyp.cov(2)), exp(per_prey.hyp.cov(3)), ...
    exp(per_prey.hyp.cov(4)), exp(per_prey.hyp.cov(5)), ...
    exp(per_prey.hyp.lik), per_prey.nlml);
fprintf('Naive  Pred: Per ell=%.4f, p=%.4f, sf=%.4f | SE ell=%.4f, sf=%.4f | sn(std)=%.4f | NLML=%.4f\n', ...
    exp(per_pred.hyp.cov(1)), exp(per_pred.hyp.cov(2)), exp(per_pred.hyp.cov(3)), ...
    exp(per_pred.hyp.cov(4)), exp(per_pred.hyp.cov(5)), ...
    exp(per_pred.hyp.lik), per_pred.nlml);

ell_per = exp(hyp_lmc.cov(1));
per1    = exp(hyp_lmc.cov(2));
ell_se  = exp(hyp_lmc.cov(7));
B1 = chol2cov(hyp_lmc.cov(4:6));
B2 = chol2cov(hyp_lmc.cov(9:11));
sn_lmc = exp(hyp_lmc.lik);
fprintf('LMC:   periodic ell=%.4f, p=%.4f | SE ell=%.4f | sigma_n(std)=%.4f | NLML=%.4f\n', ...
    ell_per, per1, ell_se, sn_lmc, nlml_lmc);
fprintf('LMC:   B_1 (periodic, oscillation)  = [%.4f %.4f; %.4f %.4f]\n', B1(1,1), B1(1,2), B1(2,1), B1(2,2));
fprintf('LMC:   B_2 (SE, shared trend)       = [%.4f %.4f; %.4f %.4f]\n', B2(1,1), B2(1,2), B2(2,1), B2(2,2));

% ell_per_b = exp(lmc_b.hyp.cov(1));
% per1_b    = exp(lmc_b.hyp.cov(2));
% ell_se_b  = exp(lmc_b.hyp.cov(7));
% B1b = chol2cov(lmc_b.hyp.cov(4:6));
% B2b = chol2cov(lmc_b.hyp.cov(9:11));
% fprintf('LMC+bound: periodic ell=%.4f, p=%.4f | SE ell=%.4f | sigma_n(std)=%.4f | NLML=%.4f\n', ...
%     ell_per_b, per1_b, ell_se_b, exp(lmc_b.hyp.lik), lmc_b.nlml);
% fprintf('LMC+bound: exitflag=%d | max(c)=%.4g | lower prey=%.4g | tube=%.4g\n', ...
%     lmc_b.exitflag, lmc_b.max_c, lmc_b.max_c_prey, lmc_b.max_c_data);
% fprintf('LMC+bound: B_1 = [%.4f %.4f; %.4f %.4f]\n', B1b(1,1), B1b(1,2), B1b(2,1), B1b(2,2));
% fprintf('LMC+bound: B_2 = [%.4f %.4f; %.4f %.4f]\n', B2b(1,1), B2b(1,2), B2b(2,1), B2b(2,2));

%% ----- local functions -----
function [m, sf, hyp, nlml] = fit_per_se_state(x, y, sn, p0, ellShort, ellLong, s2_ell, ...
    x_grid, meanfunc, covfunc, likfunc)
% Independent k = covPeriodic + covSEiso. Center and scale y (zero mean,
% unit std), matching LMC; sn is converted to standardized units.
% Short/long length-scale hyperpriors match LMC; period and both sf free.
% Predictions are returned in original units.
% hyp.cov = [log(ell_per); log(p); log(sf_per); log(ell_se); log(sf_se)].
x = x(:); y = y(:);
mu = mean(y); sd = std(y);
if sd < eps, sd = 1; end
y = (y - mu) / sd;
sn = sn / sd;
sf0 = sqrt(0.5);   % unit variance split across the two kernels (like LMC B_q = 0.5 I)
sn_fixed = log(sn);
hyp_cov0 = [log(ellShort); log(p0); log(sf0); log(ellLong); log(sf0)];
prior.cov = { {@priorGauss, log(ellShort), s2_ell}, [], [], ...
              {@priorGauss, log(ellLong), s2_ell}, [] };
inffunc = {@infPrior, @infGaussLik, prior};
hyp_cov = minimize(hyp_cov0, @gp_nlml_cov_only, -300, sn_fixed, ...
    inffunc, meanfunc, covfunc, likfunc, x, y);
hyp = struct('mean', [], 'cov', hyp_cov(:), 'lik', sn_fixed);
nlml = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y);
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, x_grid(:));
m = mu + sd * fmu(:);
sf = sd * sqrt(max(fs2(:), 0));
end

function B = chol2cov(hyp)
% Reconstruct a 2x2 covDiscrete coregionalization matrix B = L'*L from its
% Cholesky hyperparameters [log(L11); L21; log(L22)].
L = zeros(2);
L(triu(true(2))) = hyp(:);
L(1:3:end) = exp(diag(L));
B = L' * L;
end

%{
function [lb, ub] = bound_box_lmc(t_min, t_max)
% theta = [cov(11); lik]; covDiscrete sf entries 3 and 8 are clamped at log(1)=0.
span = max(t_max - t_min, 1);
lb = [log(0.05); log(2); 0; log(0.05); -5; log(0.05); ...
      log(1);    0;      log(0.05); -5; log(0.05); log(1e-4)];
ub = [log(10); log(max(t_max, 10)); 0; log(5); 5; log(5); ...
      log(2 * span); 0; log(5); 5; log(5); log(2)];
end

function hyp = theta_to_lmc_hyp(theta, hyp_tpl)
hyp = hyp_tpl;
n_cov = numel(hyp_tpl.cov);
hyp.cov = theta(1:n_cov);
hyp.lik = theta(n_cov+1);
hyp.mean = [];
end

function out = fit_lmc_lower_bound(x_aug, y_aug, hyp_unc, X_c, k, x_grid, scale, ...
    inffunc, meanfunc, covfunc, likfunc, hyp_lb, hyp_ub, opts, nTry, nMultistart, rng_seed)
% Joint LMC NLML min s.t. mu_f - k*sigma_f >= 0 for prey only (original units)
% and |y - y*| <= eps_prey at prey training points. Predator unconstrained.
theta_unc = [hyp_unc.cov(:); hyp_unc.lik(:)];
n_theta = numel(theta_unc);
if numel(hyp_lb) ~= n_theta || numel(hyp_ub) ~= n_theta
    error('LMC bound_box length (%d / %d) must match theta (%d).', ...
        numel(hyp_lb), numel(hyp_ub), n_theta);
end
hyp_tpl = hyp_unc;

objfun = @(theta) gp(theta_to_lmc_hyp(theta, hyp_tpl), inffunc, meanfunc, covfunc, likfunc, x_aug, y_aug);
nonlcon = @(theta) pens_constraints_lmc_lower(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x_aug, y_aug, X_c, k, scale);

theta_unc_box = min(max(theta_unc, hyp_lb), hyp_ub);
fprintf('  Multistart: %d random starts\n', nTry);
feasible_starts = zeros(n_theta, 0);
best_feas_nlml = inf;
best_feas_theta = nan(n_theta, 1);
rng(rng_seed);
for t = 1:nTry
    theta_try = hyp_lb + rand(n_theta, 1) .* (hyp_ub - hyp_lb);
    try
        [c_try, ~] = nonlcon(theta_try);
        is_feas = all(isfinite(c_try)) && max(c_try) <= 0;
    catch
        is_feas = false;
    end
    if is_feas
        feasible_starts = [feasible_starts, theta_try]; %#ok<AGROW>
        nlml_try = objfun(theta_try);
        if nlml_try < best_feas_nlml
            best_feas_nlml = nlml_try;
            best_feas_theta = theta_try;
        end
    end
end
nFeas = size(feasible_starts, 2);
fprintf('  Feasible random starts: %d / %d\n', nFeas, nTry);
if nFeas > 0
    nlml_feas = arrayfun(@(j) objfun(feasible_starts(:, j)), 1:nFeas);
    [~, ord] = sort(nlml_feas, 'ascend');
    starts_for_fmincon = feasible_starts(:, ord(1:min(nMultistart, nFeas)));
else
    fprintf('  No feasible random start; using projected unconstrained LMC theta.\n');
    starts_for_fmincon = theta_unc_box;
end
starts_for_fmincon = [theta_unc_box, starts_for_fmincon];
starts_for_fmincon = starts_for_fmincon(:, 1:min(nMultistart + 1, size(starts_for_fmincon, 2)));

best_nlml = inf;
theta_opt = nan(n_theta, 1);
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
    fprintf('  Warning: no successful fmincon run; using fallback theta.\n');
end

hyp = theta_to_lmc_hyp(theta_opt, hyp_tpl);
[c_final, ~] = nonlcon(theta_opt);
nC = numel(X_c);
out.hyp = hyp;
out.nlml = nlml;
out.exitflag = exitflag;
out.max_c = max(c_final);
out.max_c_prey = max(c_final(1:nC));
out.max_c_data = max(c_final(nC+1:end));
fprintf('  exitflag=%d | max(c)=%.6g (feasible if <= 0)\n', exitflag, out.max_c);
fprintf('    lower prey=%.6g | data-tube=%.6g\n', out.max_c_prey, out.max_c_data);

n_grid = numel(x_grid);
x_te = [x_grid, scale.LABEL_PREY * ones(n_grid, 1); ...
        x_grid, scale.LABEL_PRED * ones(n_grid, 1)];
[~, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x_aug, y_aug, x_te);
sf = sqrt(max(fs2(:), 0));
out.prey.m  = scale.mu_prey + scale.sd_prey * fmu(1:n_grid);
out.prey.sf = scale.sd_prey * sf(1:n_grid);
out.pred.m  = scale.mu_pred + scale.sd_pred * fmu(n_grid+1:end);
out.pred.sf = scale.sd_pred * sf(n_grid+1:end);
end

function [c, ceq] = pens_constraints_lmc_lower(theta, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x_aug, y_aug, X_c, k, scale)
% Prey only: mu_f - k*sigma_f >= 0 at X_c, and |y - y*(x)| <= eps_prey at
% prey training points. Predator unconstrained.
hyp = theta_to_lmc_hyp(theta, hyp_tpl);
nC = numel(X_c);
is_prey = x_aug(:, 2) == scale.LABEL_PREY;
x_prey = x_aug(is_prey, :);
y_prey = y_aug(is_prey);
n_prey = size(x_prey, 1);
x_c_prey = [X_c(:), scale.LABEL_PREY * ones(nC, 1)];
xstar = [x_c_prey; x_prey];
[ymu, ~, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x_aug, y_aug, xstar);
m_prey = scale.mu_prey + scale.sd_prey * fmu(1:nC);
s_prey = scale.sd_prey * sqrt(max(fs2(1:nC), 0));
c_lower = k .* s_prey - m_prey;
y_star = scale.mu_prey + scale.sd_prey * ymu(nC+1:nC+n_prey);
y_orig = scale.mu_prey + scale.sd_prey * y_prey;
c_data = abs(y_orig - y_star) - scale.eps_prey;
c = [c_lower(:); c_data(:)];
ceq = [];
end
%}

function plot_state(ax, x_grid, y_true, fit, x_train, y_train, col, k_plot, name, band_label)
xg = x_grid(:)';
m = fit.m(:); sf = fit.sf(:);
fill(ax, [xg, fliplr(xg)], [m + k_plot * sf; flipud(m - k_plot * sf)]', ...
    col, 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', sprintf('%s %s', name, band_label));
plot(ax, x_grid, m, '--', 'Color', col, 'LineWidth', 2, ...
    'DisplayName', sprintf('%s mean', name));
plot(ax, x_train, y_train, 'o', 'MarkerSize', 5, ...
    'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', ...
    'DisplayName', sprintf('%s data', name));
plot(ax, x_grid, y_true, '-', 'Color', col, 'LineWidth', 1.5, ...
    'DisplayName', sprintf('%s truth', name));
end
