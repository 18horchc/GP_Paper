% microglia_hetero.m — Homoscedastic SE-GP vs heteroscedastic observation noise.
% Same full and averaged M1/M2 datasets as the other microglia paper scripts.
% Independent SE kernels for M1 and M2.
%
% Tab 1 — fixed Day-0 / Day-5 construction:
%   - Day 0 (M1 and M2): fixed low noise (increase confidence)
%   - Day 5 (M2 only): fixed high noise (decrease confidence)
%   - Day 5 (M1) and all other times: one shared noise SD per phenotype,
%     learned with ell/sf by NLML
%
% Tab 2 — empirical per-time variance:
%   - sigma^2(t) = sample variance of replicates at that time (n>=2)
%   - n=1 times (including all averaged points) fall back to the pooled
%     empirical variance from times with n>=2, else var(y)
%   - ell/sf learned by NLML with that fixed diagonal noise
%
% Tab 3 — VHGPR (Lazaro-Gredilla & Titsias), independent fits on M1 and M2.
%
% Each tab: 2x2 — left = homoscedastic SE-GP, right = heteroscedastic;
%   top = full dataset, bottom = averaged dataset.
%   M1 and M2 plotted together on each axes.

clear; close all; clc;

%% ===== Configuration =====
k_plot     = 1.96;
max_iters  = -200;
vhgpr_iter = 40;
tgrid      = (0:0.1:14)';

% Fixed absolute observation noise SDs (count units)
sn_day0 = 5;     % low  -> higher confidence at t=0 (M1 and M2)
sn_day5 = 250;   % high -> lower confidence at t=5 (M2 only; M1 uses shared sn)

%% ===== Data (same as microglia.m / microglia_VO.m) =====
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
    'title', {'Full', 'Averaged'}, ...
    'timeM1', {timeM1(:), newtime(:)}, ...
    'dataM1', {dataM1(:), datapointsM1(:)}, ...
    'timeM2', {timeM2(:), newtimeM2(:)}, ...
    'dataM2', {dataM2(:), datapointsM2(:)});

%% ===== GPML / helpers =====
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
if ~exist('gp_seiso_hetero_noise', 'file')
    error('microglia_hetero:MissingHelper', ...
        'gp_seiso_hetero_noise.m not found on path.');
end
if ~exist('vhgpr_fit_predict', 'file')
    error('microglia_hetero:MissingVHGPR', ...
        'vhgpr_fit_predict.m not found on path.');
end

meanfunc = @meanZero;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

fprintf('=== Microglia: homoscedastic SE vs heteroscedastic SE ===\n');
fprintf(['Tab 1: sn_day0=%.4g (M1+M2), sn_day5=%.4g (M2 only); ' ...
    'M1 day 5 and all other days share one NLML sn per phenotype\n'], ...
    sn_day0, sn_day5);
fprintf('Tab 2: empirical sigma^2(t) from replicate variance (n=1 uses pooled fallback)\n');
fprintf('Tab 3: VHGPR independent on M1 and M2 (iter=%d)\n', vhgpr_iter);

%% ===== Fit =====
results = struct([]);
for didx = 1:numel(datasets)
    ds = datasets(didx);
    fprintf('\n--- Dataset: %s ---\n', ds.name);

    homo = fit_homo_pair(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, inffunc, meanfunc, covfunc, likfunc, max_iters, k_plot);
    het = fit_het_pair(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, sn_day0, sn_day5, max_iters, k_plot);
    emp = fit_emp_het_pair(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, max_iters, k_plot);
    vhg = fit_vhgpr_pair(ds.timeM1, ds.dataM1, ds.timeM2, ds.dataM2, ...
        tgrid, vhgpr_iter, k_plot);

    results(didx).name = ds.name;
    results(didx).title = ds.title;
    results(didx).timeM1 = ds.timeM1;
    results(didx).dataM1 = ds.dataM1;
    results(didx).timeM2 = ds.timeM2;
    results(didx).dataM2 = ds.dataM2;
    results(didx).homo = homo;
    results(didx).het = het;
    results(didx).emp = emp;
    results(didx).vhgpr = vhg;

    report_homo(ds.name, homo);
    report_het(ds.name, het, sn_day0, sn_day5);
    report_emp(ds.name, emp);
    report_vhgpr(ds.name, vhg);
end

%% ===== Tabbed figure: each tab 2x2 (homo | het) x (full | averaged) =====
col_M1 = [0.10, 0.10, 0.10];
col_M2 = [0.85, 0.16, 0.16];

fig = figure('Color', 'w', 'Position', [60, 60, 1240, 900], ...
    'Name', 'Microglia: homoscedastic SE vs heteroscedastic SE');
tg = uitabgroup(fig);

tab1 = plot_comparison_tab(tg, 'Day 0/5 hetero', results, tgrid, 'het', ...
    'Homoscedastic SE', ...
    sprintf('Hetero (sn_0=%.3g; M2 sn_5=%.3g)', sn_day0, sn_day5), ...
    col_M1, col_M2);
plot_comparison_tab(tg, 'Empirical sigma^2(t)', results, tgrid, 'emp', ...
    'Homoscedastic SE', 'Hetero (empirical sigma^2(t))', ...
    col_M1, col_M2);
plot_comparison_tab(tg, 'VHGPR', results, tgrid, 'vhgpr', ...
    'Homoscedastic SE', 'VHGPR', ...
    col_M1, col_M2);

tg.SelectedTab = tab1;
drawnow;

fprintf('\nDone.\n');

%% ===== Local functions =====

function tab = plot_comparison_tab(tg, tab_title, results, tgrid, right_field, ...
    left_title, right_title, col_M1, col_M2)

tab = uitab(tg, 'Title', tab_title);
tg.SelectedTab = tab;
tl = tiledlayout(tab, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for didx = 1:numel(results)
    ds = results(didx);
    methods = {ds.homo, ds.(right_field)};
    method_titles = { ...
        sprintf('%s — %s', ds.title, left_title), ...
        sprintf('%s — %s', ds.title, right_title)};
    for midx = 1:2
        ax = nexttile(tl);
        ax.Layer = 'top';
        hold(ax, 'on'); grid(ax, 'on');
        fit = methods{midx};
        plot_phenotype(ax, tgrid, fit.M1, ds.timeM1, ds.dataM1, col_M1, 'M1');
        plot_phenotype(ax, tgrid, fit.M2, ds.timeM2, ds.dataM2, col_M2, 'M2');
        xlabel(ax, 'Time (days)');
        ylabel(ax, 'cells/mm^2');
        title(ax, method_titles{midx}, 'Interpreter', 'none');
        xlim(ax, [0, 14]);
        ylim_auto_from_fit(ax, fit.M1, fit.M2, ds.dataM1, ds.dataM2);
        legend(ax, 'Location', 'northwest', 'FontSize', 8);
    end
end
end

function out = fit_homo_pair(timeM1, dataM1, timeM2, dataM2, tgrid, ...
    inffunc, meanfunc, covfunc, likfunc, max_iters, k_plot)

y1 = max(dataM1(:), 0);
y2 = max(dataM2(:), 0);
[hyp1, mu1, s21, nlml1] = fit_homo_gp(timeM1(:), y1, tgrid, ...
    inffunc, meanfunc, covfunc, likfunc, max_iters);
[hyp2, mu2, s22, nlml2] = fit_homo_gp(timeM2(:), y2, tgrid, ...
    inffunc, meanfunc, covfunc, likfunc, max_iters);

out.M1 = pack_fit(mu1, sqrt(max(s21, 0)), k_plot);
out.M2 = pack_fit(mu2, sqrt(max(s22, 0)), k_plot);
out.hyp_M1 = hyp1;
out.hyp_M2 = hyp2;
out.nlml = nlml1 + nlml2;
out.nlml_M1 = nlml1;
out.nlml_M2 = nlml2;
end

function [hyp, mu, s2, nlml] = fit_homo_gp(x, y, xs, inffunc, meanfunc, covfunc, likfunc, max_iters)
x = x(:); y = y(:); xs = xs(:);
ell0 = max(std(x), 0.5);
sf0  = max(std(y), 0.1);
sn0  = max(0.1 * sf0, 1e-3);
hyp = struct('mean', [], 'cov', log([ell0; sf0]), 'lik', log(sn0));
hyp = minimize(hyp, @gp, max_iters, inffunc, meanfunc, covfunc, likfunc, x, y);
nlml = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y);
[~, ~, mu, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
mu = mu(:);
s2 = s2(:);
end

function out = fit_het_pair(timeM1, dataM1, timeM2, dataM2, tgrid, ...
    sn_day0, sn_day5, max_iters, k_plot)

y1 = max(dataM1(:), 0);
y2 = max(dataM2(:), 0);
% M1: day-5 uses shared learned sn (pass empty sn_day5).
% M2: day-5 uses fixed high sn_day5.
fit1 = fit_het_gp(timeM1(:), y1, tgrid, sn_day0, [], max_iters);
fit2 = fit_het_gp(timeM2(:), y2, tgrid, sn_day0, sn_day5, max_iters);

out.M1 = pack_fit(fit1.mu, fit1.sf, k_plot);
out.M2 = pack_fit(fit2.mu, fit2.sf, k_plot);
out.hyp_M1 = fit1.hyp;
out.hyp_M2 = fit2.hyp;
out.sn_shared_M1 = fit1.sn_shared;
out.sn_shared_M2 = fit2.sn_shared;
out.nlml = fit1.nlml + fit2.nlml;
out.nlml_M1 = fit1.nlml;
out.nlml_M2 = fit2.nlml;
end

function fit = fit_het_gp(x, y, xs, sn_day0, sn_day5, max_iters)
% Optimize p = [log(ell); log(sf); log(sn_shared)].
% Day 0 / Day 5 noise SDs stay fixed; other rows share sn_shared.
x = x(:); y = y(:); xs = xs(:);
ell0 = max(std(x), 0.5);
sf0  = max(std(y), 0.1);
sn0  = max(0.1 * sf0, 1e-3);
p0 = [log(ell0); log(sf0); log(sn0)];

obj = @(p) nlml_shared_except(p, x, y, sn_day0, sn_day5);
p = minimize(p0, obj, max_iters);

sn_shared = exp(p(3));
noise_var = build_noise_var(x, sn_shared, sn_day0, sn_day5);
hyp = struct('mean', [], 'cov', p(1:2), 'lik', []);
nlml = gp_seiso_hetero_noise('nlml', hyp, x, y, noise_var);
[~, ~, fmu, fs2] = gp_seiso_hetero_noise('pred', hyp, x, y, noise_var, xs);

fit.hyp = hyp;
fit.nlml = nlml;
fit.sn_shared = sn_shared;
fit.mu = fmu(:);
fit.sf = sqrt(max(fs2(:), 0));
end

function [nlml, dnlml] = nlml_shared_except(p, x, y, sn_day0, sn_day5)
p = p(:);
sn_shared = exp(p(3));
noise_var = build_noise_var(x, sn_shared, sn_day0, sn_day5);
hyp = struct('mean', [], 'cov', p(1:2), 'lik', []);
nlml = gp_seiso_hetero_noise('nlml', hyp, x, y, noise_var);
if nargout > 1
    dnlml = zeros(size(p));
    step = 1e-4;
    for i = 1:numel(p)
        pp = p;
        pp(i) = p(i) + step;
        fp = nlml_shared_except_value(pp, x, y, sn_day0, sn_day5);
        pp(i) = p(i) - step;
        fm = nlml_shared_except_value(pp, x, y, sn_day0, sn_day5);
        dnlml(i) = (fp - fm) / (2 * step);
    end
end
end

function nlml = nlml_shared_except_value(p, x, y, sn_day0, sn_day5)
sn_shared = exp(p(3));
noise_var = build_noise_var(x, sn_shared, sn_day0, sn_day5);
hyp = struct('mean', [], 'cov', p(1:2), 'lik', []);
nlml = gp_seiso_hetero_noise('nlml', hyp, x, y, noise_var);
end

function noise_var = build_noise_var(x, sn_shared, sn_day0, sn_day5)
x = x(:);
noise_var = (sn_shared^2) * ones(numel(x), 1);
if ~isempty(sn_day0) && isfinite(sn_day0)
    noise_var(abs(x - 0) < 1e-12) = sn_day0^2;
end
if ~isempty(sn_day5) && isfinite(sn_day5)
    noise_var(abs(x - 5) < 1e-12) = sn_day5^2;
end
end

function pheno = pack_fit(mu, sf, k_plot)
pheno.mu = mu(:);
pheno.sf = sf(:);
pheno.lo = mu(:) - k_plot .* sf(:);
pheno.hi = mu(:) + k_plot .* sf(:);
end

function report_homo(dataset_name, fit)
fprintf(['[%s | homo] NLML=%.4f | ' ...
    'ell_M1=%.4f, ell_M2=%.4f | sf_M1=%.4f, sf_M2=%.4f | ' ...
    'sn_M1=%.4f, sn_M2=%.4f\n'], ...
    dataset_name, fit.nlml, ...
    exp(fit.hyp_M1.cov(1)), exp(fit.hyp_M2.cov(1)), ...
    exp(fit.hyp_M1.cov(2)), exp(fit.hyp_M2.cov(2)), ...
    exp(fit.hyp_M1.lik), exp(fit.hyp_M2.lik));
end

function report_het(dataset_name, fit, sn_day0, sn_day5)
fprintf(['[%s | het ] NLML=%.4f | ' ...
    'ell_M1=%.4f, ell_M2=%.4f | sf_M1=%.4f, sf_M2=%.4f | ' ...
    'sn_shared_M1=%.4f, sn_shared_M2=%.4f | sn_day0=%.4g, sn_day5_M2=%.4g (M1 day5=shared)\n'], ...
    dataset_name, fit.nlml, ...
    exp(fit.hyp_M1.cov(1)), exp(fit.hyp_M2.cov(1)), ...
    exp(fit.hyp_M1.cov(2)), exp(fit.hyp_M2.cov(2)), ...
    fit.sn_shared_M1, fit.sn_shared_M2, sn_day0, sn_day5);
end

function out = fit_emp_het_pair(timeM1, dataM1, timeM2, dataM2, tgrid, max_iters, k_plot)
y1 = max(dataM1(:), 0);
y2 = max(dataM2(:), 0);
fit1 = fit_emp_het_gp(timeM1(:), y1, tgrid, max_iters);
fit2 = fit_emp_het_gp(timeM2(:), y2, tgrid, max_iters);

out.M1 = pack_fit(fit1.mu, fit1.sf, k_plot);
out.M2 = pack_fit(fit2.mu, fit2.sf, k_plot);
out.hyp_M1 = fit1.hyp;
out.hyp_M2 = fit2.hyp;
out.nlml = fit1.nlml + fit2.nlml;
out.nlml_M1 = fit1.nlml;
out.nlml_M2 = fit2.nlml;
out.emp_M1 = fit1;
out.emp_M2 = fit2;
end

function fit = fit_emp_het_gp(x, y, xs, max_iters)
% ell/sf via NLML; diagonal noise = empirical replicate variance at each t.
x = x(:); y = y(:); xs = xs(:);
[noise_var, t_unique, s2_unique, n_per_t, used_fallback] = empirical_time_noise(x, y);

ell0 = max(std(x), 0.5);
sf0  = max(std(y), 0.1);
hyp0 = struct('mean', [], 'cov', log([ell0; sf0]), 'lik', []);
obj = @(h) gp_seiso_hetero_noise('nlml', h, x, y, noise_var);
hyp = minimize(hyp0, obj, max_iters);
nlml = obj(hyp);
[~, ~, fmu, fs2] = gp_seiso_hetero_noise('pred', hyp, x, y, noise_var, xs);

fit.hyp = hyp;
fit.nlml = nlml;
fit.mu = fmu(:);
fit.sf = sqrt(max(fs2(:), 0));
fit.t_unique = t_unique;
fit.s2_unique = s2_unique;
fit.n_per_t = n_per_t;
fit.used_fallback = used_fallback;
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

function report_emp(dataset_name, fit)
fprintf(['[%s | emp ] NLML=%.4f | ' ...
    'ell_M1=%.4f, ell_M2=%.4f | sf_M1=%.4f, sf_M2=%.4f\n'], ...
    dataset_name, fit.nlml, ...
    exp(fit.hyp_M1.cov(1)), exp(fit.hyp_M2.cov(1)), ...
    exp(fit.hyp_M1.cov(2)), exp(fit.hyp_M2.cov(2)));
print_emp_sn(dataset_name, 'M1', fit.emp_M1);
print_emp_sn(dataset_name, 'M2', fit.emp_M2);
end

function print_emp_sn(dataset_name, pheno, emp)
sn = sqrt(emp.s2_unique);
mark = repmat(' ', size(sn));
mark(emp.used_fallback) = '*';
txt = arrayfun(@(t, s, n, m) sprintf('t=%.0f:sn=%.3g(n=%d)%s', t, s, n, m), ...
    emp.t_unique, sn, emp.n_per_t, mark, 'UniformOutput', false);
fprintf('  [%s | emp %s] %s  (* = n=1 pooled fallback)\n', ...
    dataset_name, pheno, strjoin(txt, ', '));
end

function out = fit_vhgpr_pair(timeM1, dataM1, timeM2, dataM2, tgrid, vhgpr_iter, k_plot)
y1 = max(dataM1(:), 0);
y2 = max(dataM2(:), 0);
fit1 = fit_vhgpr_one(timeM1(:), y1, tgrid, vhgpr_iter);
fit2 = fit_vhgpr_one(timeM2(:), y2, tgrid, vhgpr_iter);

out.M1 = pack_fit(fit1.fmu, sqrt(max(fit1.fs2, 0)), k_plot);
out.M2 = pack_fit(fit2.fmu, sqrt(max(fit2.fs2, 0)), k_plot);
out.raw_M1 = fit1;
out.raw_M2 = fit2;
out.mean_sigma_n_M1 = mean(fit1.sigma_n);
out.mean_sigma_n_M2 = mean(fit2.sigma_n);
end

function fit = fit_vhgpr_one(x, y, xs, vhgpr_iter)
fit = vhgpr_fit_predict(x, y, xs, struct('iter', vhgpr_iter));
end

function report_vhgpr(dataset_name, fit)
fprintf(['[%s | vhgpr] done | mean sigma_n_M1=%.4g, mean sigma_n_M2=%.4g ' ...
    '(latent fmu/fs2 bands)\n'], ...
    dataset_name, fit.mean_sigma_n_M1, fit.mean_sigma_n_M2);
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

function ylim_auto_from_fit(ax, fitM1, fitM2, dataM1, dataM2)
vals = [fitM1.lo(:); fitM1.hi(:); fitM1.mu(:); ...
        fitM2.lo(:); fitM2.hi(:); fitM2.mu(:); ...
        dataM1(:); dataM2(:)];
pad = 0.05 * max(range(vals), 1);
ylim(ax, [min(vals) - pad, max(vals) + pad]);
end
