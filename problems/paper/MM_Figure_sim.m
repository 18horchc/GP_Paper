function MM_Figure_sim
% Nested 4-model Monte Carlo ablation of encoded GPs on synthetic MM kinetics.
%
%   M0  baseline SE-GP (five assays only)
%   M1  M0 + boundary VO f(0)=0
%   M2  M1 + Solak virtual derivative observations (saturation)
%   M3  M2 + Pensoneault upper bound at Vmax (data-fidelity tube is part of
%       the constrained fitter, not a separate ablation arm)
%
% Nested paired comparisons (same y in replicate r):
%   M1-M0  effect of boundary information
%   M2-M1  additional effect of saturation (Solak) information
%   M3-M2  additional effect of the probabilistic upper bound
%   M3-M0  overall effect of the full encoding
%
% Primary metrics (lower is better), scored on x_grid (M=500) against the
% known noise-free MM curve f(x), using GPML latent fmu and fs2 (not ys2):
%   latent RMSE  — posterior-mean accuracy
%   latent NLPD  — Gaussian log loss of N(fmu, fs2); no sigma_n^2
% Percentile bands are 95% Monte Carlo ranges (2.5th-97.5th), not CIs for the mean.
%
% Checkpoint: local AppData first (not OneDrive), then best-effort copy to
% results/MM_ablation_MC.mat. Re-run to resume.

%% MM parameters (same as MM_Figure.m)
Vmax = 100;
Km   = 18;
mm_static = @(S) (Vmax .* S) ./ (Km + S);

%% Training locations and noise level (fixed across replicates)
x_max = 200;
noise_frac = 0.05;
x_train = [10; 30; 60; 90; 200];
n_train = numel(x_train);
v_true_at_train = mm_static(x_train);
y_domain_max = mm_static(x_max);
noise_sd_true = noise_frac * y_domain_max;

x_col = x_train(:);
nG = 500;
nMod = 4;
model_names = {'M0_baseline', 'M1_boundary', 'M2_boundary_deriv', 'M3_full'};

%% Fixed observation noises (not optimized; not tuned to RMSE / true curve)
sigma_data = noise_sd_true;
sigma_VO_zero = 0.8 * noise_sd_true;
sn_deriv = 0.1;

%% Boundary virtual observation only (no interpolant saturation VO)
x_virt = 0;
y_virt = 0;

%% Solak virtual derivative observations (as in MM_Figure.m)
x_deriv = [150; 175; 200];
y_deriv = zeros(size(x_deriv));

%% Ground truth curve
x_grid = linspace(0, x_max, nG);
y_true = mm_static(x_grid);
y_true = y_true(:);

%% Pensoneault constraint grid
eta = 0.022;
k   = -sqrt(2) * erfinv(2 * eta - 1);
n_constraint = 800;
X_c = linspace(0, x_max, n_constraint)';
y_max = Vmax;
epsilon = 2 * noise_sd_true;
con_tol = 1e-4;

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

ell_bounds_lo = 0.05;
ell_ub = x_max;
sf_bounds = [0.05, max(15, Vmax)];
hyp_lb = log([ell_bounds_lo; sf_bounds(1)]);
hyp_ub = log([ell_ub; sf_bounds(2)]);
meanfunc = @meanZero;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;
sn_fixed = log(noise_sd_true);

opts_pens = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
    'EnableFeasibilityMode', true, 'Display', 'off', ...
    'ConstraintTolerance', con_tol, 'OptimalityTolerance', 1e-4, ...
    'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);
nTry = 2000;
nMultistart = 10;

% Nine fixed log-space starts, reused in every replicate; plus one data-scale start.
rng(40);
fixed_starts = hyp_lb + rand(2, 9) .* (hyp_ub - hyp_lb);

%% Simulation controls
nRep = 1000;
use_parfor = license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel'));

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
out_dir = fullfile(repo_root, 'results');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
out_mat = fullfile(out_dir, 'MM_ablation_MC.mat');
out_csv_models = fullfile(out_dir, 'MM_ablation_MC_models.csv');
out_csv_contrasts = fullfile(out_dir, 'MM_ablation_MC_contrasts.csv');

% Primary checkpoint lives off OneDrive so HDF5 rewrite is not locked/truncated.
la = getenv('LOCALAPPDATA');
if isempty(la)
    la = tempdir;
end
local_dir = fullfile(la, 'Bio_Inf_GP_Code');
if ~exist(local_dir, 'dir')
    mkdir(local_dir);
end
out_mat_local = fullfile(local_dir, 'MM_ablation_MC.mat');
fprintf('Checkpoint (local): %s\n', out_mat_local);
fprintf('Checkpoint (copy):  %s\n', out_mat);

rmse_met = nan(nRep, nMod);
nlpd_met = nan(nRep, nMod);
fmu_all  = nan(nRep, nG, nMod);
fs2_all  = nan(nRep, nG, nMod);
ell_hat  = nan(nRep, nMod);
sf_hat   = nan(nRep, nMod);
fit_status = nan(nRep, 1);
fmincon_success = false(nRep, 1);
feasible_start_found = false(nRep, 1);
fallback_used = false(nRep, 1);
max_constraint_violation = nan(nRep, 1);

r_start = 1;
S = try_load_checkpoint({out_mat_local, out_mat, ...
    fullfile(local_dir, 'MM_ablation_MC_metrics.mat')}, nRep, nMod);
if ~isempty(S)
    rmse_met = S.rmse_met;
    nlpd_met = S.nlpd_met;
    fit_status = S.fit_status(:);
    if isfield(S, 'fmu_all') && isequal(size(S.fmu_all), [nRep, nG, nMod])
        fmu_all = S.fmu_all;
    end
    if isfield(S, 'fs2_all') && isequal(size(S.fs2_all), [nRep, nG, nMod])
        fs2_all = S.fs2_all;
    end
    if isfield(S, 'ell_hat') && isequal(size(S.ell_hat), [nRep, nMod])
        ell_hat = S.ell_hat;
    end
    if isfield(S, 'sf_hat') && isequal(size(S.sf_hat), [nRep, nMod])
        sf_hat = S.sf_hat;
    end
    if isfield(S, 'fmincon_success')
        fmincon_success = logical(S.fmincon_success(:));
    end
    if isfield(S, 'feasible_start_found')
        feasible_start_found = logical(S.feasible_start_found(:));
    end
    if isfield(S, 'fallback_used')
        fallback_used = logical(S.fallback_used(:));
    end
    if isfield(S, 'max_constraint_violation')
        max_constraint_violation = S.max_constraint_violation(:);
    end
    done = all(isfinite(rmse_met), 2) & all(isfinite(nlpd_met), 2) ...
        & isfinite(fit_status);
    if any(done)
        r_start = find(~done, 1, 'first');
        if isempty(r_start)
            r_start = nRep + 1;
        end
    end
    fprintf('Resuming from replicate %d / %d\n', min(r_start, nRep), nRep);
end

fprintf(['MM_Figure_sim ablation: nRep=%d | n=%d | sigma_n=%.4f | nTry=%d | X_c=%d | ', ...
    'use_parfor=%d\n'], nRep, n_train, noise_sd_true, nTry, numel(X_c), use_parfor);
fprintf('M1 boundary VO: v(0)=0 (sigma_VO=%.4g)\n', sigma_VO_zero);
fprintf('M2/M3 Solak deriv: %d sites at S=[%s], v''=0 | sn_deriv=%.4g\n', ...
    numel(x_deriv), strjoin(compose('%.0f', x_deriv), ', '), sn_deriv);
fprintf('M3: eta=%.3g%% | k=%.4f | epsilon=%.4g | no Pensoneault f'' inequality\n', ...
    100 * eta, k, epsilon);
fprintf('Metrics: latent RMSE and latent NLPD vs known f (M=%d); lower is better.\n', nG);

%% Replicate loop
if r_start <= nRep && use_parfor
    try
        if isempty(gcp('nocreate'))
            parpool;
        end
        gpml_c = char(gpml_folder_name);
        problems_c = fileparts(fileparts(mfilename('fullpath')));
        pctRunOnAll(['addpath(genpath(''' gpml_c '''));']);
        pctRunOnAll(['addpath(''' problems_c ''');']);
    catch
        use_parfor = false;
        fprintf('parpool failed; falling back to serial loop.\n');
    end
end

if r_start <= nRep && use_parfor
    idx = r_start:nRep;
    nLeft = numel(idx);
    tmp_rmse = nan(nLeft, nMod);
    tmp_nlpd = nan(nLeft, nMod);
    tmp_fmu  = nan(nLeft, nG, nMod);
    tmp_fs2  = nan(nLeft, nG, nMod);
    tmp_ell  = nan(nLeft, nMod);
    tmp_sf   = nan(nLeft, nMod);
    tmp_status = nan(nLeft, 1);
    tmp_fmin = false(nLeft, 1);
    tmp_feas = false(nLeft, 1);
    tmp_fall = false(nLeft, 1);
    tmp_viol = nan(nLeft, 1);
    parfor t = 1:nLeft
        tmp = run_one_replicate(idx(t), x_col, v_true_at_train, noise_sd_true, ...
            x_virt, y_virt, sigma_data, sigma_VO_zero, x_deriv, y_deriv, sn_deriv, ...
            x_grid, y_true, sn_fixed, meanfunc, covfunc, likfunc, inffunc, ...
            hyp_lb, hyp_ub, X_c, k, y_max, epsilon, opts_pens, nTry, nMultistart, ...
            fixed_starts, con_tol);
        tmp_rmse(t, :) = tmp.rmse;
        tmp_nlpd(t, :) = tmp.nlpd;
        tmp_fmu(t, :, :) = reshape(tmp.fmu, 1, nG, 4);
        tmp_fs2(t, :, :) = reshape(tmp.fs2, 1, nG, 4);
        tmp_ell(t, :) = tmp.ell;
        tmp_sf(t, :) = tmp.sf;
        tmp_status(t) = tmp.fit_status;
        tmp_fmin(t) = tmp.fmincon_success;
        tmp_feas(t) = tmp.feasible_start_found;
        tmp_fall(t) = tmp.fallback_used;
        tmp_viol(t) = tmp.max_constraint_violation;
    end
    rmse_met(idx, :) = tmp_rmse;
    nlpd_met(idx, :) = tmp_nlpd;
    fmu_all(idx, :, :) = tmp_fmu;
    fs2_all(idx, :, :) = tmp_fs2;
    ell_hat(idx, :) = tmp_ell;
    sf_hat(idx, :) = tmp_sf;
    fit_status(idx) = tmp_status;
    fmincon_success(idx) = tmp_fmin;
    feasible_start_found(idx) = tmp_feas;
    fallback_used(idx) = tmp_fall;
    max_constraint_violation(idx) = tmp_viol;
    save_checkpoint(out_mat_local, out_mat, pack_checkpoint(rmse_met, nlpd_met, ...
        fmu_all, fs2_all, ell_hat, sf_hat, fit_status, fmincon_success, ...
        feasible_start_found, fallback_used, max_constraint_violation, nRep, model_names));
else
    for r = r_start:nRep
        t0 = tic;
        tmp = run_one_replicate(r, x_col, v_true_at_train, noise_sd_true, ...
            x_virt, y_virt, sigma_data, sigma_VO_zero, x_deriv, y_deriv, sn_deriv, ...
            x_grid, y_true, sn_fixed, meanfunc, covfunc, likfunc, inffunc, ...
            hyp_lb, hyp_ub, X_c, k, y_max, epsilon, opts_pens, nTry, nMultistart, ...
            fixed_starts, con_tol);
        rmse_met(r, :) = tmp.rmse;
        nlpd_met(r, :) = tmp.nlpd;
        fmu_all(r, :, :) = reshape(tmp.fmu, 1, nG, 4);
        fs2_all(r, :, :) = reshape(tmp.fs2, 1, nG, 4);
        ell_hat(r, :) = tmp.ell;
        sf_hat(r, :) = tmp.sf;
        fit_status(r) = tmp.fit_status;
        fmincon_success(r) = tmp.fmincon_success;
        feasible_start_found(r) = tmp.feasible_start_found;
        fallback_used(r) = tmp.fallback_used;
        max_constraint_violation(r) = tmp.max_constraint_violation;
        save_checkpoint(out_mat_local, out_mat, pack_checkpoint(rmse_met, nlpd_met, ...
            fmu_all, fs2_all, ell_hat, sf_hat, fit_status, fmincon_success, ...
            feasible_start_found, fallback_used, max_constraint_violation, nRep, model_names));
        fprintf(['Rep %d / %d  (%.1f s)  RMSE [%s]  NLPD [%s]  M3 status=%d\n'], ...
            r, nRep, toc(t0), sprintf('%.3g ', tmp.rmse), sprintf('%.3g ', tmp.nlpd), ...
            tmp.fit_status);
    end
end

%% Paired Monte Carlo summary (do not average inside the replicate loop)
pair_more = [2; 3; 4; 4];
pair_less = [1; 2; 3; 1];
pair_lab = {'M1 vs M0'; 'M2 vs M1'; 'M3 vs M2'; 'M3 vs M0'};
nPair = numel(pair_more);

fprintf('\n=== Nested ablation summary (%d replicates) ===\n', nRep);
fprintf(['Latent RMSE / latent NLPD vs known MM curve on x_grid (M=%d).\n', ...
    'Percentiles are a 95%% Monte Carlo range (2.5th-97.5th), not a CI for the mean.\n', ...
    'Delta = more_encoded - less_encoded (negative => improvement).\n', ...
    'I_RMSE = 100*(RMSE_less - RMSE_more)/RMSE_less (positive => improvement).\n\n'], nG);

fprintf('--- Per-model latent metrics ---\n');
fprintf('%-22s  %10s  %10s  %10s  %10s  %10s\n', ...
    'Metric', 'mean', 'median', 'std', '2.5%', '97.5%');
s_rmse = cell(nMod, 1);
s_nlpd = cell(nMod, 1);
for j = 1:nMod
    s_rmse{j} = summarize_vec(rmse_met(:, j));
    s_nlpd{j} = summarize_vec(nlpd_met(:, j));
    print_summary_row(['RMSE ' model_names{j}], s_rmse{j});
    print_summary_row(['NLPD ' model_names{j}], s_nlpd{j});
end

fprintf('\nM3 constrained-fit status counts (0=feas fmincon, 1=feas fallback, 2=infeas fallback):\n');
for st = 0:2
    fprintf('  status %d: %d / %d\n', st, sum(fit_status == st), nRep);
end

fprintf('\n--- Paired nested contrasts ---\n');
d_rmse = nan(nRep, nPair);
d_nlpd = nan(nRep, nPair);
I_rmse = nan(nRep, nPair);
p_rmse = nan(nPair, 1);
p_nlpd = nan(nPair, 1);
s_d_rmse = cell(nPair, 1);
s_d_nlpd = cell(nPair, 1);
s_I_rmse = cell(nPair, 1);
for p = 1:nPair
    im = pair_more(p);
    il = pair_less(p);
    d_rmse(:, p) = rmse_met(:, im) - rmse_met(:, il);
    d_nlpd(:, p) = nlpd_met(:, im) - nlpd_met(:, il);
    I_rmse(:, p) = 100 * (rmse_met(:, il) - rmse_met(:, im)) ./ max(rmse_met(:, il), 1e-12);
    p_rmse(p) = mean(rmse_met(:, im) < rmse_met(:, il));
    p_nlpd(p) = mean(nlpd_met(:, im) < nlpd_met(:, il));
    s_d_rmse{p} = summarize_vec(d_rmse(:, p));
    s_d_nlpd{p} = summarize_vec(d_nlpd(:, p));
    s_I_rmse{p} = summarize_vec(I_rmse(:, p));
    fprintf('%s\n', pair_lab{p});
    print_summary_row('  dRMSE', s_d_rmse{p});
    print_summary_row('  I_RMSE(%)', s_I_rmse{p});
    print_summary_row('  dNLPD', s_d_nlpd{p});
    fprintf('  P(better RMSE)=%.4f  P(better NLPD)=%.4f\n', p_rmse(p), p_nlpd(p));
end

% Paired nested interpretation (same y per replicate):
%   M1 vs M0  — boundary VO; M2 vs M1 — Solak saturation; M3 vs M2 — upper bound;
%   M3 vs M0  — full encoding vs baseline. Percentiles are a 95% MC range.
T_models = table(model_names(:), ...
    cellfun(@(s) s.mean, s_rmse), cellfun(@(s) s.median, s_rmse), ...
    cellfun(@(s) s.std, s_rmse), cellfun(@(s) s.lo, s_rmse), cellfun(@(s) s.hi, s_rmse), ...
    cellfun(@(s) s.mean, s_nlpd), cellfun(@(s) s.median, s_nlpd), ...
    cellfun(@(s) s.std, s_nlpd), cellfun(@(s) s.lo, s_nlpd), cellfun(@(s) s.hi, s_nlpd), ...
    'VariableNames', {'model', 'rmse_mean', 'rmse_median', 'rmse_std', ...
    'rmse_p2_5', 'rmse_p97_5', 'nlpd_mean', 'nlpd_median', 'nlpd_std', ...
    'nlpd_p2_5', 'nlpd_p97_5'});
writetable(T_models, out_csv_models);

T_contrasts = table(pair_lab, ...
    cellfun(@(s) s.median, s_d_rmse), cellfun(@(s) s.median, s_I_rmse), ...
    cellfun(@(s) s.lo, s_d_rmse), cellfun(@(s) s.hi, s_d_rmse), p_rmse, ...
    cellfun(@(s) s.median, s_d_nlpd), ...
    cellfun(@(s) s.lo, s_d_nlpd), cellfun(@(s) s.hi, s_d_nlpd), p_nlpd, ...
    'VariableNames', {'contrast', 'median_dRMSE', 'median_I_RMSE_pct', ...
    'dRMSE_p2_5', 'dRMSE_p97_5', 'frac_better_RMSE', ...
    'median_dNLPD', 'dNLPD_p2_5', 'dNLPD_p97_5', 'frac_better_NLPD'});
writetable(T_contrasts, out_csv_contrasts);

chk_final = pack_checkpoint(rmse_met, nlpd_met, fmu_all, fs2_all, ell_hat, sf_hat, ...
    fit_status, fmincon_success, feasible_start_found, fallback_used, ...
    max_constraint_violation, nRep, model_names);
chk_final.d_rmse = d_rmse;
chk_final.d_nlpd = d_nlpd;
chk_final.I_rmse = I_rmse;
chk_final.p_rmse = p_rmse;
chk_final.p_nlpd = p_nlpd;
chk_final.pair_lab = pair_lab;
chk_final.T_models = T_models;
chk_final.T_contrasts = T_contrasts;
saved_path = save_checkpoint(out_mat_local, out_mat, chk_final);
fprintf('\nSaved checkpoint %s\n', saved_path);
fprintf('Saved %s\n', out_csv_models);
fprintf('Saved %s\n', out_csv_contrasts);

addpath(fileparts(mfilename('fullpath')));
MM_Figure_sim_primary(out_dir, rmse_met, nlpd_met);
end

%% ----- local functions -----
function out = run_one_replicate(r, x_col, v_true_at_train, noise_sd_true, ...
    x_virt, y_virt, sigma_data, sigma_VO_zero, x_deriv, y_deriv, sn_deriv, ...
    x_grid, y_true, sn_fixed, meanfunc, covfunc, likfunc, inffunc, ...
    hyp_lb, hyp_ub, X_c, k, y_max, epsilon, opts_pens, nTry, nMultistart, ...
    fixed_starts, con_tol)

rng(100 + r);
y_col = v_true_at_train + noise_sd_true * randn(size(v_true_at_train));
y_col = y_col(:);

x_aug = [x_col; x_virt(:)];
y_aug = [y_col; y_virt(:)];
noise_var_aug = [sigma_data^2 * ones(numel(y_col), 1); ...
    sigma_VO_zero^2 * ones(numel(y_virt), 1)];

data_start = min(max(log([std(x_col); max(std(y_col), 1e-3)]), hyp_lb), hyp_ub);
starts = [data_start, fixed_starts];
hyp_tpl = struct('mean', [], 'cov', data_start, 'lik', sn_fixed);
nG = numel(x_grid);

% M0: baseline GP, five assays only
obj_m0 = @(hyp_cov) gp_nlml_cov_only(hyp_cov, sn_fixed, inffunc, meanfunc, covfunc, likfunc, x_col, y_col);
hyp_m0 = fit_minimize_multistart(obj_m0, hyp_tpl, starts, hyp_lb, hyp_ub);

% M1: boundary VO f(0)=0 (hetero NLML; gradient w.r.t. hyp.cov only)
obj_m1 = @(hyp_cov) hetero_nlml_cov_only(hyp_cov, sn_fixed, x_aug, y_aug, noise_var_aug);
hyp_m1 = fit_minimize_multistart(obj_m1, hyp_tpl, starts, hyp_lb, hyp_ub);

% M2: boundary + Solak derivative observations
obj_m2 = @(hyp_cov) gp_seiso_deriv_obs_nlml_cov_only(hyp_cov, sn_fixed, ...
    x_aug, y_aug, x_deriv, y_deriv, sn_deriv, noise_var_aug);
hyp_m2 = fit_minimize_multistart(obj_m2, hyp_tpl, starts, hyp_lb, hyp_ub);
theta_m2_box = min(max(hyp_m2.cov(:), hyp_lb), hyp_ub);

% M3: M2 likelihood + Pensoneault upper bound + data-fidelity tube
objfun_m3 = @(theta) gp_seiso_deriv_obs('nlml', theta_to_hyp(theta, hyp_m2), ...
    x_aug, y_aug, x_deriv, y_deriv, [], sn_deriv, true, noise_var_aug);
nonlcon_m3 = @(theta) pens_constraints_upper_tube(theta, hyp_m2, ...
    x_aug, y_aug, noise_var_aug, X_c, k, y_max, epsilon, x_col, y_col, ...
    x_deriv, y_deriv, sn_deriv);
[hyp_m3, ~, diag_m3] = fit_pens_constrained( ...
    objfun_m3, nonlcon_m3, hyp_m2, hyp_lb, hyp_ub, theta_m2_box, ...
    opts_pens, nTry, nMultistart, 45000 + r, false, con_tol);

hyps = {hyp_m0, hyp_m1, hyp_m2, hyp_m3};
fmu = nan(nG, 4);
fs2 = nan(nG, 4);
ell = nan(1, 4);
sf  = nan(1, 4);
rmse = nan(1, 4);
nlpd = nan(1, 4);

[~, ~, fmu0, fs20] = gp(hyp_m0, inffunc, meanfunc, covfunc, likfunc, ...
    x_col, y_col, x_grid(:));
[~, ~, fmu1, fs21] = gp_seiso_hetero_noise('pred', hyp_m1, x_aug, y_aug, noise_var_aug, x_grid(:));
[~, ~, fmu2, fs22] = gp_seiso_deriv_obs('pred', hyp_m2, ...
    x_aug, y_aug, x_deriv, y_deriv, x_grid(:), sn_deriv, true, noise_var_aug);
[~, ~, fmu3, fs23] = gp_seiso_deriv_obs('pred', hyp_m3, ...
    x_aug, y_aug, x_deriv, y_deriv, x_grid(:), sn_deriv, true, noise_var_aug);
fmu_cell = {fmu0, fmu1, fmu2, fmu3};
fs2_cell = {fs20, fs21, fs22, fs23};

for j = 1:4
    fmu(:, j) = fmu_cell{j}(:);
    fs2(:, j) = fs2_cell{j}(:);
    ell(j) = exp(hyps{j}.cov(1));
    sf(j)  = exp(hyps{j}.cov(2));
    [rmse(j), nlpd(j)] = score_latent(fmu(:, j), fs2(:, j), y_true);
end

out = struct();
out.rmse = rmse;
out.nlpd = nlpd;
out.fmu = fmu;
out.fs2 = fs2;
out.ell = ell;
out.sf = sf;
out.fit_status = diag_m3.fit_status;
out.fmincon_success = diag_m3.fmincon_success;
out.feasible_start_found = diag_m3.feasible_start_found;
out.fallback_used = diag_m3.fallback_used;
out.max_constraint_violation = diag_m3.max_constraint_violation;
end

function [nlml, dnlml] = hetero_nlml_cov_only(hyp_cov, sn_fixed, x, y, noise_var)
% Heteroscedastic SE-iso NLML with fixed per-row noise; optimize ell, sf only.
hyp = struct('mean', [], 'cov', hyp_cov(:), 'lik', sn_fixed);
if nargout > 1
    [nlml, dnlml_s] = gp_seiso_hetero_noise('nlml', hyp, x, y, noise_var);
    dnlml = dnlml_s.cov;
else
    nlml = gp_seiso_hetero_noise('nlml', hyp, x, y, noise_var);
end
end

function hyp = fit_minimize_multistart(obj_cov, hyp_tpl, starts, hyp_lb, hyp_ub)
% GPML minimize from a shared start set; retain lowest NLML; clip to box.
best_nlml = inf;
theta_opt = min(max(starts(:, 1), hyp_lb), hyp_ub);
for j = 1:size(starts, 2)
    theta0 = min(max(starts(:, j), hyp_lb), hyp_ub);
    try
        hyp_cov = minimize(theta0, obj_cov, -100);
        hyp_cov = min(max(hyp_cov(:), hyp_lb), hyp_ub);
        nlml_j = obj_cov(hyp_cov);
        if isfinite(nlml_j) && nlml_j < best_nlml
            best_nlml = nlml_j;
            theta_opt = hyp_cov;
        end
    catch
    end
end
if ~isfinite(best_nlml)
    theta_opt = min(max(starts(:, 1), hyp_lb), hyp_ub);
end
hyp = theta_to_hyp(theta_opt, hyp_tpl);
end

function [rmse, nlpd] = score_latent(fmu, fs2, f_true)
% Latent RMSE and latent NLPD vs known noise-free ground-truth f.
% Uses GPML fmu / fs2 only (do not add observation-noise variance).
fmu = fmu(:);
fs2 = fs2(:);
f_true = f_true(:);
rmse = sqrt(mean((fmu - f_true).^2));
var_floor = 1e-12;
fs2_safe = max(fs2, var_floor);
nlpd = mean( ...
    0.5 * log(2 * pi * fs2_safe) + ...
    0.5 * ((f_true - fmu).^2 ./ fs2_safe) );
end

function s = summarize_vec(v)
v = v(:);
s.mean = mean(v, 'omitnan');
s.median = col_percentile(v, 50);
s.std = std(v, 0, 'omitnan');
s.lo = col_percentile(v, 2.5);
s.hi = col_percentile(v, 97.5);
end

function print_summary_row(name, s)
fprintf('%-22s  %10.4g  %10.4g  %10.4g  %10.4g  %10.4g\n', ...
    name, s.mean, s.median, s.std, s.lo, s.hi);
end

function v = col_percentile(M, p)
n = size(M, 1);
Ms = sort(M, 1);
pos = 1 + (p / 100) * (n - 1);
lo = max(1, min(n, floor(pos)));
hi = max(1, min(n, ceil(pos)));
w = pos - lo;
v = zeros(1, size(M, 2));
for j = 1:size(M, 2)
    if lo == hi
        v(j) = Ms(lo, j);
    else
        v(j) = (1 - w) * Ms(lo, j) + w * Ms(hi, j);
    end
end
end

function [hyp_con, nlml_con, fit_diag] = fit_pens_constrained( ...
    objfun, nonlcon, hyp_tpl, hyp_lb, hyp_ub, theta_unc_box, opts_pens, ...
    nTry, nMultistart, rng_seed, verbose, con_tol)
if nargin < 11
    verbose = false;
end
if nargin < 12
    con_tol = 1e-4;
end

feasible_starts = zeros(2, 0);
best_feas_nlml = inf;
best_feas_theta = nan(2, 1);
rng(rng_seed);
for t = 1:nTry
    theta_try = hyp_lb + rand(2, 1) .* (hyp_ub - hyp_lb);
    [c_try, ~] = nonlcon(theta_try);
    if max(c_try) <= con_tol
        feasible_starts = [feasible_starts, theta_try]; %#ok<AGROW>
        nlml_try = objfun(theta_try);
        if nlml_try < best_feas_nlml
            best_feas_nlml = nlml_try;
            best_feas_theta = theta_try;
        end
    end
end
nFeas = size(feasible_starts, 2);
fit_diag = struct();
fit_diag.feasible_start_found = nFeas > 0;
if verbose
    fprintf('Feasible random starts: %d / %d\n', nFeas, nTry);
end

if nFeas > 0
    nlml_feas = arrayfun(@(j) objfun(feasible_starts(:, j)), 1:nFeas);
    [~, ord] = sort(nlml_feas, 'ascend');
    starts_for_fmincon = feasible_starts(:, ord(1:min(nMultistart, nFeas)));
else
    starts_for_fmincon = theta_unc_box;
end
starts_for_fmincon = [theta_unc_box, starts_for_fmincon];
starts_for_fmincon = starts_for_fmincon(:, 1:min(nMultistart + 1, size(starts_for_fmincon, 2)));

best_nlml = inf;
theta_opt = nan(2, 1);
nlml_con = nan;
exitflag_con = -99;
used_fmincon = false;
nStarts = size(starts_for_fmincon, 2);
for j = 1:nStarts
    theta0_j = starts_for_fmincon(:, j);
    [theta_j, nlml_j, ef_j] = fmincon(objfun, theta0_j, [], [], [], [], ...
        hyp_lb, hyp_ub, nonlcon, opts_pens);
    if isfinite(nlml_j) && nlml_j < best_nlml
        [c_j, ~] = nonlcon(theta_j);
        if max(c_j) <= con_tol
            best_nlml = nlml_j;
            theta_opt = theta_j;
            nlml_con = nlml_j;
            exitflag_con = ef_j;
            used_fmincon = true;
        end
    end
end

if ~isfinite(best_nlml)
    if nFeas > 0
        theta_opt = best_feas_theta;
        nlml_con = objfun(theta_opt);
        used_fmincon = false;
    else
        theta_opt = theta_unc_box;
        nlml_con = objfun(theta_opt);
        used_fmincon = false;
    end
    exitflag_con = -99;
end

hyp_con = theta_to_hyp(theta_opt, hyp_tpl);
[c_final, ~] = nonlcon(theta_opt);
viol = max(c_final);
feas = viol <= con_tol;
fit_diag.fmincon_success = used_fmincon && feas;
fit_diag.max_constraint_violation = viol;
fit_diag.fallback_used = ~fit_diag.fmincon_success;
if fit_diag.fmincon_success
    fit_diag.fit_status = 0;
elseif feas
    fit_diag.fit_status = 1;
else
    fit_diag.fit_status = 2;
end
fit_diag.exitflag = exitflag_con;
end

function hyp = theta_to_hyp(theta, hyp_tpl)
hyp = hyp_tpl;
hyp.cov = theta(1:2);
hyp.mean = [];
end

function [c, ceq] = pens_constraints_upper_tube(theta, hyp_tpl, x, y, noise_var, ...
    X_c, k, y_max, epsilon, x_data, y_data, x_d, y_d, sn_d)
% M3 constraints: Pensoneault upper bound on latent f at X_c, plus data-
% fidelity tube at the five real assay points. Solak deriv obs are in the
% GP posterior, not as extra inequalities.
hyp = theta_to_hyp(theta, hyp_tpl);
nC = numel(X_c);
xstar = [X_c(:); x_data(:)];
[ymu, ~, fmu, fs2] = gp_seiso_deriv_obs('pred', hyp, x, y, x_d, y_d, xstar, ...
    sn_d, true, noise_var);
m_xc = fmu(1:nC);
s_xc = sqrt(max(fs2(1:nC), 0));
c_upper = m_xc + k .* s_xc - y_max;
y_star = ymu(nC+1:end);
c_data = abs(y_data(:) - y_star) - epsilon;
c = [c_upper(:); c_data(:)];
ceq = [];
end

function chk = pack_checkpoint(rmse_met, nlpd_met, fmu_all, fs2_all, ell_hat, sf_hat, ...
    fit_status, fmincon_success, feasible_start_found, fallback_used, ...
    max_constraint_violation, nRep, model_names)
chk = struct();
chk.rmse_met = rmse_met;
chk.nlpd_met = nlpd_met;
chk.fmu_all = fmu_all;
chk.fs2_all = fs2_all;
chk.ell_hat = ell_hat;
chk.sf_hat = sf_hat;
chk.fit_status = fit_status;
chk.fmincon_success = fmincon_success;
chk.feasible_start_found = feasible_start_found;
chk.fallback_used = fallback_used;
chk.max_constraint_violation = max_constraint_violation;
chk.nRep = nRep;
chk.model_names = model_names;
end

function S = try_load_checkpoint(paths, nRep, nMod)
S = [];
for i = 1:numel(paths)
    p = paths{i};
    if isempty(p) || ~exist(p, 'file')
        continue;
    end
    try
        T = load(p);
    catch ME
        warning('MM_Figure_sim:BadCheckpoint', 'Skipping unreadable %s: %s', p, ME.message);
        continue;
    end
    if isfield(T, 'rmse_met') && isfield(T, 'nlpd_met') && isfield(T, 'fit_status') ...
            && isequal(size(T.rmse_met), [nRep, nMod]) && isequal(size(T.nlpd_met), [nRep, nMod]) ...
            && numel(T.fit_status) == nRep
        S = T;
        fprintf('Loaded checkpoint %s\n', p);
        return;
    end
end
end

function dest = save_checkpoint(local_mat, cloud_mat, chk)
% Atomic local save (off OneDrive). Best-effort copy to results/. Never abort the run.
dest = '';
try
    dest = save_atomic(local_mat, chk);
catch ME
    warning('MM_Figure_sim:LocalSave', 'Full local save failed (%s). Writing metrics-only.', ME.message);
    slim = struct();
    slim.rmse_met = chk.rmse_met;
    slim.nlpd_met = chk.nlpd_met;
    slim.ell_hat = chk.ell_hat;
    slim.sf_hat = chk.sf_hat;
    slim.fit_status = chk.fit_status;
    slim.fmincon_success = chk.fmincon_success;
    slim.feasible_start_found = chk.feasible_start_found;
    slim.fallback_used = chk.fallback_used;
    slim.max_constraint_violation = chk.max_constraint_violation;
    slim.nRep = chk.nRep;
    slim.model_names = chk.model_names;
    [d, n, ~] = fileparts(local_mat);
    try
        dest = save_atomic(fullfile(d, [n '_metrics.mat']), slim);
    catch ME2
        warning('MM_Figure_sim:MetricsSave', 'Metrics-only save also failed: %s', ME2.message);
        return;
    end
end
if isempty(dest)
    return;
end
try
    copyfile(dest, cloud_mat, 'f');
catch ME
    warning('MM_Figure_sim:CloudCopy', ...
        'Could not copy checkpoint to OneDrive path %s (%s). Local file is %s.', ...
        cloud_mat, ME.message, dest);
end
end

function dest = save_atomic(dest, chk)
d = fileparts(dest);
if ~isempty(d) && ~exist(d, 'dir')
    mkdir(d);
end
tmp = fullfile(d, sprintf('.chk_tmp_%d.mat', round(1e9 * rand)));
save(tmp, '-struct', 'chk', '-v7.3');
if exist(dest, 'file')
    try
        delete(dest);
    catch
    end
end
movefile(tmp, dest, 'f');
end

