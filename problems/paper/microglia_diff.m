% microglia_diff.m - Joint sum-difference GP on raw counts with soft
% Gaussian ordering anchors on d(t).
%
% Latents: s(t) common level (cells/mm^2), d(t) difference (cells/mm^2)
%   M1 = s + 0.5*d,  M2 = s - 0.5*d
%
% Model A: real M1/M2 observations only (hypers fit on real data).
% Model B: same hypers + Gaussian VO anchors on d(t).
% Figure 1: 2x2 phenotypes (A|B) x (full|averaged).
% Figure 2: d(t) and P(M1>M2) for each dataset (A vs B overlay).

clear; close all; clc;

%% ===== Configuration =====
k_plot    = 1.96;
max_iters = -200;
tgrid     = (0:0.1:14)';
rng(0);

% Soft Gaussian ordering anchors (cells/mm^2 difference units)
t_vo_early = [0.5; 1.5; 2.5; 3.5; 4.5; 5.5];
t_vo_cross = 6;
t_vo_late  = [6.5; 8; 10; 12; 14];
margin       = 50;    % target |d| in cells/mm^2 (d < 0 early, d > 0 late)
sigma_order  = 100;   % soft: 1-sigma uncertainty on each anchor
sigma_cross  = 50;    % tighter anchor at d(6) = 0

% Sensitivity grid (console table only)
margins_sens      = [20,  50,  100];
sigmas_order_sens = [50,  100, 200];
sigmas_cross_sens = [25,  50,  100];

%% ===== Data (day 5 restored) =====
newtime = [0, 1, 2, 3, 5, 7, 14];
datapointsM1 = [5, 27.5, 122.5, 139.8, 325, 445, 816.67];
newtimeM2 = [0, 1, 2, 3, 5, 7, 14];
datapointsM2 = [5, 78.33, 179.5, 126.4, 800, 319, 136.67];

timeM1 = [0, 1, 3, 5, 7, 14, ...
          3, 7, ...
          2, ...
          14, ...
          3, ...
          2, ...
          0, 1, 3, 7, 14, ...
          3, 7];
dataM1 = [0, 5, 375, 325, 600, 750, ...
          62, 55, ...
          120, ...
          400, ...
          102, ...
          125, ...
          10, 50, 100, 900, 1300, ...
          60, 225];

timeM2 = [0, 1, 3, 5, 7, 14, ...
          1, 3, 7, ...
          2, ...
          14, ...
          3, ...
          2, ...
          0, 1, 3, 7, 14, ...
          3, 7];
dataM2 = [0, 170, 300, 800, 600, 200, ...
          15, 15, 6, ...
          90, ...
          110, ...
          57, ...
          269, ...
          10, 50, 100, 400, 100, ...
          160, 270];

datasets = struct( ...
    'name', {'full', 'averaged'}, ...
    'timeM1', {timeM1(:), newtime(:)}, ...
    'dataM1', {dataM1(:), datapointsM1(:)}, ...
    'timeM2', {timeM2(:), newtimeM2(:)}, ...
    'dataM2', {dataM2(:), datapointsM2(:)});

t_vo = [t_vo_early; t_vo_cross; t_vo_late];
z_vo = [-margin * ones(numel(t_vo_early), 1); 0; margin * ones(numel(t_vo_late), 1)];
var_vo = [sigma_order^2 * ones(numel(t_vo_early), 1); ...
          sigma_cross^2; ...
          sigma_order^2 * ones(numel(t_vo_late), 1)];

fprintf('=== Joint sum-difference GP (raw counts) with soft d(t) anchors ===\n');
fprintf('VO early: %s | cross: %.3g | late: %s\n', ...
    mat2str(t_vo_early(:)'), t_vo_cross, mat2str(t_vo_late(:)'));
fprintf('margin=%.3g, sigma_order=%.3g, sigma_cross=%.3g\n', ...
    margin, sigma_order, sigma_cross);

%% ===== GPML setup (minimize only) =====
gpml_folder_name = "C:\Users\chorc\OneDrive\Documents\Stroke Research\Gaussian Processes\Old\gpml-matlab-master\gpml-matlab-master";
if ~exist('minimize', 'file')
    if exist(gpml_folder_name, 'dir')
        addpath(genpath(gpml_folder_name));
    else
        error('GPML toolbox missing at %s (need minimize)', gpml_folder_name);
    end
end
try
    startup;
catch
end

%% ===== Fit =====
results = struct([]);
for didx = 1:numel(datasets)
    ds = datasets(didx);
    fprintf('\n--- Dataset: %s ---\n', ds.name);

    fit = fit_joint_sumdiff(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, t_vo, z_vo, var_vo, max_iters, k_plot);

    results(didx).name = ds.name;
    results(didx).timeM1 = ds.timeM1;
    results(didx).dataM1 = ds.dataM1;
    results(didx).timeM2 = ds.timeM2;
    results(didx).dataM2 = ds.dataM2;
    results(didx).A = fit.A;
    results(didx).B = fit.B;
    results(didx).theta = fit.theta;
    results(didx).report = fit.report;

    report_dataset(ds.name, fit);
end

%% ===== Figure 1: 2x2 phenotypes (A | B) x (full | averaged) =====
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];
col_vo = [0.20, 0.45, 0.80];

figure('Color', 'w', 'Position', [40, 40, 1240, 900], ...
    'Name', 'Microglia: joint sum-diff GP (data-only vs anchored)');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for didx = 1:numel(results)
    ds = results(didx);
    methods = {ds.A, ds.B};
    method_titles = {'Model A (data only)', 'Model B (+ d anchors)'};
    for midx = 1:2
        nexttile;
        ax = gca; ax.Layer = 'top';
        hold on; grid on;
        fitm = methods{midx};
        plot_phenotype(ax, tgrid, fitm.M1, ds.timeM1, ds.dataM1, col_M1, 'M1');
        plot_phenotype(ax, tgrid, fitm.M2, ds.timeM2, ds.dataM2, col_M2, 'M2');
        if midx == 2
            for iv = 1:numel(t_vo)
                xline(ax, t_vo(iv), ':', 'Color', col_vo, 'LineWidth', 1.0, ...
                    'HandleVisibility', 'off');
            end
            plot(ax, NaN, NaN, ':', 'Color', col_vo, 'LineWidth', 1.2, ...
                'DisplayName', 'VO times');
        end
        xlabel('Time (days)');
        ylabel('cells/mm^2');
        title(sprintf('%s — %s', ds.name, method_titles{midx}), 'Interpreter', 'none');
        xlim([0, 14]);
        ylim_auto_from_fit(ax, fitm.M1, fitm.M2, ds.dataM1, ds.dataM2);
        legend('Location', 'northwest', 'FontSize', 8);
    end
end

%% ===== Figure 2: d(t) and P(M1 > M2) =====
figure('Color', 'w', 'Position', [80, 60, 1240, 900], ...
    'Name', 'Microglia: d(t) and ordering probability');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for didx = 1:numel(results)
    ds = results(didx);

    nexttile;
    ax = gca; hold on; grid on;
    plot_d_band(ax, tgrid, ds.A.d, [0.55, 0.55, 0.55], 'A: data only');
    plot_d_band(ax, tgrid, ds.B.d, [0.15, 0.35, 0.70], 'B: + anchors');
    yline(ax, 0, 'k:', 'HandleVisibility', 'off');
    xline(ax, t_vo_cross, 'k--', 'HandleVisibility', 'off');
    for iv = 1:numel(t_vo)
        xline(ax, t_vo(iv), ':', 'Color', col_vo, 'LineWidth', 0.8, ...
            'HandleVisibility', 'off');
    end
    xlabel('Time (days)');
    ylabel('d(t) = M1(t) - M2(t)  (cells/mm^2)');
    title(sprintf('%s — posterior d(t)', ds.name), 'Interpreter', 'none');
    xlim([0, 14]);
    legend('Location', 'best', 'FontSize', 8);

    nexttile;
    ax = gca; hold on; grid on;
    plot(ax, tgrid, ds.A.P_M1_gt_M2, '--', 'Color', [0.55, 0.55, 0.55], ...
        'LineWidth', 2, 'DisplayName', 'A: P(M1>M2)');
    plot(ax, tgrid, ds.B.P_M1_gt_M2, '-', 'Color', [0.15, 0.35, 0.70], ...
        'LineWidth', 2, 'DisplayName', 'B: P(M1>M2)');
    yline(ax, 0.5, 'k:', 'HandleVisibility', 'off');
    xline(ax, t_vo_cross, 'k--', 'HandleVisibility', 'off');
    xlabel('Time (days)');
    ylabel('P(M1 > M2)');
    title(sprintf('%s — ordering probability', ds.name), 'Interpreter', 'none');
    xlim([0, 14]);
    ylim([0, 1]);
    legend('Location', 'best', 'FontSize', 8);
end

%% ===== Sensitivity (fixed theta_hat; anchors only) =====
fprintf('\n=== Sensitivity (fixed theta from data-only fit) ===\n');
for didx = 1:numel(results)
    ds = results(didx);
    fprintf('\n[%s]\n', ds.name);
    fprintf('%8s %8s %8s | %10s %10s %10s %10s %12s\n', ...
        'margin', 'sig_ord', 'sig_crs', 'RMSE_B', 'Pearly<', 'Plate>', ...
        't_cross', 'max|dMed|');

    theta = ds.theta;
    A = ds.A;
    for im = 1:numel(margins_sens)
        for io = 1:numel(sigmas_order_sens)
            for ic = 1:numel(sigmas_cross_sens)
                m = margins_sens(im);
                so = sigmas_order_sens(io);
                sc = sigmas_cross_sens(ic);
                z_s = [-m * ones(numel(t_vo_early), 1); 0; m * ones(numel(t_vo_late), 1)];
                v_s = [so^2 * ones(numel(t_vo_early), 1); sc^2; so^2 * ones(numel(t_vo_late), 1)];

                B = predict_joint_model(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
                    tgrid, theta, t_vo, z_s, v_s, k_plot, true);

                rmse_b = phenotype_rmse(B, tgrid, ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2);
                p_early = mean(B.P_M1_lt_M2(tgrid <= 6));
                p_late  = mean(B.P_M1_gt_M2(tgrid >= 6));
                t_cross = crossing_time(tgrid, B.d.mu);
                dmed = max([abs(B.M1.mu - A.M1.mu); abs(B.M2.mu - A.M2.mu)]);

                fprintf('%8.3g %8.3g %8.3g | %10.4g %10.3f %10.3f %10.2f %12.4g\n', ...
                    m, so, sc, rmse_b, p_early, p_late, t_cross, dmed);
            end
        end
    end
end

fprintf('\nDone.\n');

%% ===== Local functions =====

function fit = fit_joint_sumdiff(timeM1, dataM1, timeM2, dataM2, ...
    tgrid, t_vo, z_vo, var_vo, max_iters, k_plot)
% Fit hypers on real data only; predict Model A and Model B.

timeM1 = timeM1(:); dataM1 = dataM1(:);
timeM2 = timeM2(:); dataM2 = dataM2(:);
y_M1 = max(dataM1, 0);
y_M2 = max(dataM2, 0);

t_latent_data = unique([timeM1; timeM2]);
[H_data, y_data] = build_data_obs(timeM1, y_M1, timeM2, y_M2, t_latent_data);
n1 = numel(y_M1);
n2 = numel(y_M2);

theta0 = init_theta(t_latent_data, y_M1, y_M2);
theta = fit_theta(theta0, t_latent_data, H_data, y_data, n1, n2, max_iters);
nlml_A = nlml_joint(theta, t_latent_data, H_data, y_data, n1, n2);

A = predict_joint_model(timeM1, dataM1, timeM2, dataM2, ...
    tgrid, theta, [], [], [], k_plot, false);
B = predict_joint_model(timeM1, dataM1, timeM2, dataM2, ...
    tgrid, theta, t_vo, z_vo, var_vo, k_plot, true);

fit.theta = theta;
fit.A = A;
fit.B = B;
fit.report.nlml_A = nlml_A;
fit.report.ell_s = exp(theta(1));
fit.report.sf_s  = exp(theta(2));
fit.report.ell_d = exp(theta(3));
fit.report.sf_d  = exp(theta(4));
fit.report.sn_M1 = exp(theta(5));
fit.report.sn_M2 = exp(theta(6));
fit.report.rmse_A = phenotype_rmse(A, tgrid, timeM1, dataM1, timeM2, dataM2);
fit.report.rmse_B = phenotype_rmse(B, tgrid, timeM1, dataM1, timeM2, dataM2);
fit.report.P_early_lt_A = mean(A.P_M1_lt_M2(tgrid <= 6));
fit.report.P_early_lt_B = mean(B.P_M1_lt_M2(tgrid <= 6));
fit.report.P_late_gt_A  = mean(A.P_M1_gt_M2(tgrid >= 6));
fit.report.P_late_gt_B  = mean(B.P_M1_gt_M2(tgrid >= 6));
fit.report.t_cross_A = crossing_time(tgrid, A.d.mu);
fit.report.t_cross_B = crossing_time(tgrid, B.d.mu);
end

function out = predict_joint_model(timeM1, dataM1, timeM2, dataM2, ...
    tgrid, theta, t_vo, z_vo, var_vo, k_plot, use_vo)
% Predict s,d then M1/M2 posteriors on raw count scale.

timeM1 = timeM1(:); dataM1 = dataM1(:);
timeM2 = timeM2(:); dataM2 = dataM2(:);
y_M1 = max(dataM1, 0);
y_M2 = max(dataM2, 0);

if use_vo
    t_latent = unique([timeM1; timeM2; t_vo(:)]);
else
    t_latent = unique([timeM1; timeM2]);
end
n_t = numel(t_latent);

[H_M1, H_M2] = build_H_phenotypes(timeM1, timeM2, t_latent);
y = [y_M1; y_M2];
sn_M1 = exp(theta(5));
sn_M2 = exp(theta(6));
R_diag = [sn_M1^2 * ones(numel(y_M1), 1); sn_M2^2 * ones(numel(y_M2), 1)];

if use_vo
    H_vo = build_H_vo(t_vo(:), t_latent);
    H = [H_M1; H_M2; H_vo];
    y = [y; z_vo(:)];
    R_diag = [R_diag; var_vo(:)];
else
    H = [H_M1; H_M2];
end

K_f = build_Kf(t_latent, theta);
R = diag(R_diag);
Ky = H * K_f * H' + R;
jit = 1e-8 * max(mean(diag(Ky)), 1);
Ky = Ky + jit * eye(size(Ky));

% Predict latents on tgrid
n_pred = numel(tgrid);
K_s_star = se_kernel(tgrid, t_latent, exp(theta(1)), exp(theta(2)));
K_d_star = se_kernel(tgrid, t_latent, exp(theta(3)), exp(theta(4)));
K_star_f = [K_s_star, zeros(n_pred, n_t); zeros(n_pred, n_t), K_d_star];

K_ss = se_kernel(tgrid, tgrid, exp(theta(1)), exp(theta(2)));
K_dd = se_kernel(tgrid, tgrid, exp(theta(3)), exp(theta(4)));
K_pp = [K_ss, zeros(n_pred); zeros(n_pred), K_dd];
K_pp = K_pp + 1e-8 * max(mean(diag(K_pp)), 1) * eye(size(K_pp));

K_star_y = K_star_f * H';
alpha = Ky \ y;
mu_f = K_star_y * alpha;
V = Ky \ K_star_y';
Sigma_f = K_pp - K_star_y * V;
Sigma_f = 0.5 * (Sigma_f + Sigma_f');

mu_s = mu_f(1:n_pred);
mu_d = mu_f(n_pred+1:end);
vs = diag(Sigma_f(1:n_pred, 1:n_pred));
vd = diag(Sigma_f(n_pred+1:end, n_pred+1:end));
csd = diag(Sigma_f(1:n_pred, n_pred+1:end));

var_M1 = max(vs + 0.25 * vd + csd, 0);
var_M2 = max(vs + 0.25 * vd - csd, 0);
sd_M1 = sqrt(var_M1);
sd_M2 = sqrt(var_M2);
mu_M1 = mu_s + 0.5 * mu_d;
mu_M2 = mu_s - 0.5 * mu_d;

out.M1 = pack_raw_fit(mu_M1, sd_M1, k_plot);
out.M2 = pack_raw_fit(mu_M2, sd_M2, k_plot);

sd_d = sqrt(max(vd, 0));
out.d.mu = mu_d;
out.d.sf = sd_d;
out.d.lo = mu_d - k_plot * sd_d;
out.d.hi = mu_d + k_plot * sd_d;
out.s.mu = mu_s;
out.s.sf = sqrt(max(vs, 0));

% P(d < 0) = P(M1 < M2), P(d > 0) = P(M1 > M2)
out.P_M1_lt_M2 = normcdf_local((0 - mu_d) ./ max(sd_d, 1e-12));
out.P_M1_gt_M2 = normcdf_local(mu_d ./ max(sd_d, 1e-12));
out.t_latent = t_latent;
out.use_vo = use_vo;
end

function theta = fit_theta(theta0, t_latent, H, y, n1, n2, max_iters)
obj = @(th) nlml_joint(th(:), t_latent, H, y, n1, n2);
th = minimize_vec(theta0(:), obj, max_iters);
theta = th(:);
end

function th = minimize_vec(th0, obj, max_iters)
% Minimize scalar objective over a real vector using GPML minimize.
hyp0.x = th0(:);
obj_h = @(h) pack_nlml(obj, h);
hyp = minimize(hyp0, obj_h, max_iters);
th = hyp.x(:);
end

function [nlml, dnlml] = pack_nlml(obj, hyp)
th = hyp.x(:);
nlml = obj(th);
if nargout > 1
    step = 1e-4;
    g = zeros(size(th));
    for i = 1:numel(th)
        tp = th; tm = th;
        tp(i) = tp(i) + step;
        tm(i) = tm(i) - step;
        g(i) = (obj(tp) - obj(tm)) / (2 * step);
    end
    dnlml = hyp;
    dnlml.x = g;
end
end

function nlml = nlml_joint(theta, t_latent, H, y, n1, n2)
K_f = build_Kf(t_latent, theta);
sn_M1 = exp(theta(5));
sn_M2 = exp(theta(6));
R_diag = [sn_M1^2 * ones(n1, 1); sn_M2^2 * ones(n2, 1)];
if numel(R_diag) ~= numel(y)
    error('nlml_joint:SizeMismatch', 'Expected %d observations, got %d', ...
        numel(R_diag), numel(y));
end

Ky = H * K_f * H' + diag(R_diag);
jit = 1e-8 * max(mean(diag(Ky)), 1);
Ky = Ky + jit * eye(size(Ky));
[L, p] = chol(Ky, 'lower');
if p ~= 0
    nlml = 1e6;
    return;
end
alpha = L' \ (L \ y);
n = numel(y);
nlml = 0.5 * (y' * alpha) + sum(log(diag(L))) + 0.5 * n * log(2 * pi);
end

function K_f = build_Kf(t_latent, theta)
ell_s = exp(theta(1)); sf_s = exp(theta(2));
ell_d = exp(theta(3)); sf_d = exp(theta(4));
n_t = numel(t_latent);
K_s = se_kernel(t_latent, t_latent, ell_s, sf_s);
K_d = se_kernel(t_latent, t_latent, ell_d, sf_d);
K_f = [K_s, zeros(n_t); zeros(n_t), K_d];
jit = 1e-8 * max(mean(diag(K_f)), 1);
K_f = K_f + jit * eye(size(K_f));
end

function K = se_kernel(x, z, ell, sf)
x = x(:); z = z(:);
r2 = (x - z.').^2 / (ell^2);
K = sf^2 * exp(-0.5 * r2);
end

function [H_data, y_data] = build_data_obs(timeM1, y_M1, timeM2, y_M2, t_latent)
[H_M1, H_M2] = build_H_phenotypes(timeM1, timeM2, t_latent);
H_data = [H_M1; H_M2];
y_data = [y_M1(:); y_M2(:)];
end

function [H_M1, H_M2] = build_H_phenotypes(timeM1, timeM2, t_latent)
n_t = numel(t_latent);
n1 = numel(timeM1);
n2 = numel(timeM2);
H_M1 = zeros(n1, 2 * n_t);
H_M2 = zeros(n2, 2 * n_t);
for i = 1:n1
    j = find_time_index(t_latent, timeM1(i));
    H_M1(i, j) = 1.0;
    H_M1(i, n_t + j) = 0.5;
end
for i = 1:n2
    j = find_time_index(t_latent, timeM2(i));
    H_M2(i, j) = 1.0;
    H_M2(i, n_t + j) = -0.5;
end
end

function H_vo = build_H_vo(t_vo, t_latent)
n_t = numel(t_latent);
n_v = numel(t_vo);
H_vo = zeros(n_v, 2 * n_t);
for i = 1:n_v
    j = find_time_index(t_latent, t_vo(i));
    H_vo(i, n_t + j) = 1.0;
end
end

function j = find_time_index(t_latent, t)
j = find(abs(t_latent - t) < 1e-12, 1);
if isempty(j)
    error('find_time_index:MissingTime', 'Time %.6g not in t_latent', t);
end
end

function theta0 = init_theta(t_latent, y_M1, y_M2)
ell0 = max(std(t_latent), 0.5);
y_all = [y_M1(:); y_M2(:)];
sf0 = max(std(y_all), 1.0);
sn0 = max(0.1 * sf0, 1.0);
% [log ell_s, log sf_s, log ell_d, log sf_d, log sn_M1, log sn_M2]
theta0 = log([ell0; sf0; ell0; sf0; sn0; sn0]);
end

function pheno = pack_raw_fit(mu, sd, k_plot)
mu = mu(:); sd = sd(:);
pheno.mu = mu;
pheno.sf = sd;
pheno.lo = mu - k_plot .* sd;
pheno.hi = mu + k_plot .* sd;
end

function rmse = phenotype_rmse(fit, tgrid, timeM1, dataM1, timeM2, dataM2)
% RMSE of posterior count median vs observed counts.
pred1 = interp1(tgrid(:), fit.M1.mu(:), timeM1(:), 'linear', 'extrap');
pred2 = interp1(tgrid(:), fit.M2.mu(:), timeM2(:), 'linear', 'extrap');
e = [pred1 - dataM1(:); pred2 - dataM2(:)];
rmse = sqrt(mean(e.^2));
end

function t_c = crossing_time(tgrid, mu_d)
% First zero crossing of mu_d after t=0 (linear interp); NaN if none.
tgrid = tgrid(:); mu_d = mu_d(:);
s = sign(mu_d);
s(s == 0) = 1;
idx = find(s(1:end-1) .* s(2:end) < 0, 1, 'first');
if isempty(idx)
    t_c = NaN;
    return;
end
t0 = tgrid(idx); t1 = tgrid(idx+1);
y0 = mu_d(idx); y1 = mu_d(idx+1);
t_c = t0 - y0 * (t1 - t0) / (y1 - y0);
end

function p = normcdf_local(z)
p = 0.5 * erfc(-z ./ sqrt(2));
end

function report_dataset(name, fit)
r = fit.report;
fprintf(['[%s] NLML_A=%.4f | ell_s=%.3g sf_s=%.3g | ell_d=%.3g sf_d=%.3g | ', ...
    'sn_M1=%.3g sn_M2=%.3g\n'], name, r.nlml_A, r.ell_s, r.sf_s, r.ell_d, r.sf_d, ...
    r.sn_M1, r.sn_M2);
fprintf('[%s] RMSE_A=%.4g RMSE_B=%.4g | Pearly(d<0) A=%.3f B=%.3f | Plate(d>0) A=%.3f B=%.3f\n', ...
    name, r.rmse_A, r.rmse_B, r.P_early_lt_A, r.P_early_lt_B, r.P_late_gt_A, r.P_late_gt_B);
fprintf('[%s] t_cross A=%.2f B=%.2f\n', name, r.t_cross_A, r.t_cross_B);
end

function plot_phenotype(ax, tgrid, fit, t_data, y_data, col, name)
tg = tgrid(:)';
fill(ax, [tg, fliplr(tg)], [fit.hi(:)', fliplr(fit.lo(:)')], col, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', sprintf('%s band', name));
plot(ax, tgrid, fit.mu, '--', 'Color', col, 'LineWidth', 2, ...
    'DisplayName', sprintf('%s mean', name));
scatter(ax, t_data, y_data, 36, 'filled', ...
    'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', ...
    'DisplayName', sprintf('%s data', name));
end

function plot_d_band(ax, tgrid, d, col, name)
tg = tgrid(:)';
fill(ax, [tg, fliplr(tg)], [d.hi(:)', fliplr(d.lo(:)')], col, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.18, ...
    'DisplayName', sprintf('%s band', name));
plot(ax, tgrid, d.mu, '-', 'Color', col, 'LineWidth', 2, ...
    'DisplayName', sprintf('%s mean', name));
end

function ylim_auto_from_fit(ax, fitM1, fitM2, dataM1, dataM2)
vals = [fitM1.lo(:); fitM1.hi(:); fitM1.mu(:); ...
        fitM2.lo(:); fitM2.hi(:); fitM2.mu(:); ...
        dataM1(:); dataM2(:)];
pad = 0.05 * max(range(vals), 1);
ylim(ax, [min(vals) - pad, max(vals) + pad]);
end
