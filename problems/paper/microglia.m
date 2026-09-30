% microglia.m — two independent GPs on the full and averaged microglia data.
% Archived predecessor: archive/microglia_v3.m (kernel x mean grid).
%
% Datasets (both include the Day 35 Suenaga et al. 2015 points):
%   Full    - all replicates, as in SINDy.m
%   Average - one mean per day, as in SINDy_avg.m
%
% Models, fit independently for M1 and M2 on the raw count scale (not z-scored):
%   Naive SE - zero mean, covSEiso, one learned homoscedastic sn (GPML)
%   NN emp   - zero-mean covNNone with fixed empirical heteroscedastic noise,
%              as in SINDy.m / SINDy_avg.m (gp_nn_hetero_noise; only ell, sf).
%              On the average series the noise is the replicate sigma^2(t)
%              from the full dataset (n = 1 days, including Day 35, use the
%              pooled variance).
%
% Figures 1-2: full data. Figures 3-4: averaged data.
% Shared legend drawn once at the end.

clear; close all; clc

%% ===== Configuration =====
k_plot    = 1.96;
max_iters = -200;
tgrid     = (0:0.1:35)';

%% ===== Data: full replicates (SINDy.m) =====
timeM1_full = [0, 1, 3, 5, 7, 14, ...
               3, 7, ...
               2, ...
               14, ...
               3, ...
               3, ...
               0, 1, 3, 7, 14, ...
               3, 7, ...
               35];
dataM1_full = [0, 5, 375, 325, 600, 750, ...
               62, 55, ...
               120, ...
               400, ...
               102, ...
               125, ...
               10, 50, 100, 900, 1300, ...
               60, 225, ...
               550];

timeM2_full = [0, 1, 3, 5, 7, 14, ...
               1, 3, 7, ...
               2, ...
               14, ...
               3, ...
               3, ...
               0, 1, 3, 7, 14, ...
               3, 7, ...
               35];
dataM2_full = [0, 170, 300, 800, 600, 200, ...
               15, 15, 6, ...
               90, ...
               110, ...
               57, ...
               269, ...
               10, 50, 100, 400, 100, ...
               160, 270, ...
               50];

timeM1_full = timeM1_full(:);
dataM1_full = max(dataM1_full(:), 0);
timeM2_full = timeM2_full(:);
dataM2_full = max(dataM2_full(:), 0);

%% ===== Data: averaged series (SINDy_avg.m) =====
timeM1_avg = [0, 1, 2, 3, 5, 7, 14, 35]';
dataM1_avg = [5, 27.5, 122.5, 139.8, 325, 445, 816.67, 550]';
timeM2_avg = [0, 1, 2, 3, 5, 7, 14, 35]';
dataM2_avg = [5, 78.33, 179.5, 126.4, 800, 319, 136.67, 50]';

timeM1_avg = timeM1_avg(:);
dataM1_avg = max(dataM1_avg(:), 0);
timeM2_avg = timeM2_avg(:);
dataM2_avg = max(dataM2_avg(:), 0);

full_ds = struct( ...
    'name', 'full', ...
    'M1', struct('t', timeM1_full, 'y', dataM1_full), ...
    'M2', struct('t', timeM2_full, 'y', dataM2_full));
avg_ds = struct( ...
    'name', 'average', ...
    'M1', struct('t', timeM1_avg, 'y', dataM1_avg), ...
    'M2', struct('t', timeM2_avg, 'y', dataM2_avg));

%% ===== GPML setup =====
gpml_folder_name = "C:\Users\chorc\OneDrive\Documents\Stroke Research\Gaussian Processes\Old\gpml-matlab-master\gpml-matlab-master";
if ~exist('minimize', 'file')
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
addpath(fileparts(fileparts(mfilename('fullpath'))));  % problems/ for gp_nn_hetero_noise
if ~exist('gp_nn_hetero_noise', 'file')
    error('microglia:MissingHelper', 'gp_nn_hetero_noise.m not found on path.');
end

meanfunc = @meanZero;
cov_se   = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

sty.col_M1 = [0.10, 0.10, 0.10];
sty.col_M2 = [0.85, 0.16, 0.16];
sty.band_label = '95% CI';

%% ===== Fit =====
fprintf('=== Microglia: naive SE and SINDy NN on full and averaged data ===\n');

fprintf('\n--- Full replicates: naive SE (zero mean, homoscedastic) ---\n');
full_se = fit_dataset_se(full_ds, tgrid, inffunc, meanfunc, cov_se, likfunc, max_iters, k_plot);

fprintf('\n--- Full replicates: zero-mean NN, empirical heteroscedastic noise ---\n');
full_nn = fit_dataset_nn(full_ds, tgrid, max_iters, k_plot, []);

fprintf('\n--- Averaged series: naive SE (zero mean, homoscedastic) ---\n');
avg_se = fit_dataset_se(avg_ds, tgrid, inffunc, meanfunc, cov_se, likfunc, max_iters, k_plot);

fprintf('\n--- Averaged series: zero-mean NN, empirical noise from full replicates ---\n');
avg_nn = fit_dataset_nn(avg_ds, tgrid, max_iters, k_plot, full_ds);

%% ===== Figures =====
fig1 = make_fit_figure(full_se, tgrid, sty, ...
    'Figure 1 - Full data: naive SE (zero mean, homoscedastic)', [60, 60, 1240, 900]); %#ok<NASGU>
fig2 = make_fit_figure(full_nn, tgrid, sty, ...
    'Figure 2 - Full data: NN, empirical heteroscedastic noise', [100, 60, 1240, 900]); %#ok<NASGU>
fig3 = make_fit_figure(avg_se, tgrid, sty, ...
    'Figure 3 - Averaged data: naive SE (zero mean, homoscedastic)', [140, 60, 1240, 900]); %#ok<NASGU>
fig4 = make_fit_figure(avg_nn, tgrid, sty, ...
    'Figure 4 - Averaged data: NN, empirical heteroscedastic noise', [180, 60, 1240, 900]); %#ok<NASGU>
make_shared_legend(sty.col_M1, sty.col_M2, sty.band_label, 'Microglia shared legend');

%% ===== Local functions =====

function fits = fit_dataset_se(ds, tgrid, inffunc, meanfunc, covfunc, likfunc, max_iters, k_plot)
fits = struct();
for name = {'M1', 'M2'}
    ph = ds.(name{1});
    [hyp, mu, s2] = fit_naive_se(ph.t, ph.y, tgrid, inffunc, meanfunc, covfunc, likfunc, max_iters);
    nlml = gp(hyp, inffunc, meanfunc, covfunc, likfunc, ph.t, ph.y);
    fit = pack_fit(mu, s2, k_plot);
    fit.hyp = hyp;
    fit.nlml = nlml;
    fit.ell = exp(hyp.cov(1));
    fit.sf  = exp(hyp.cov(2));
    fit.sn  = exp(hyp.lik);
    fit.t = ph.t;
    fit.y = ph.y;
    fits.(name{1}) = fit;
    fprintf('%s: NLML=%.4f | ell=%.4f | sf=%.4f | sn=%.4f\n', ...
        name{1}, fit.nlml, fit.ell, fit.sf, fit.sn);
end
end

function fits = fit_dataset_nn(ds, tgrid, max_iters, k_plot, noise_source)
% noise_source empty: empirical variance of this dataset's own y.
% otherwise: empirical variance of noise_source, mapped onto this dataset's times.
fits = struct();
for name = {'M1', 'M2'}
    ph = ds.(name{1});
    if isempty(noise_source)
        [noise_var, t_u, ~, ~, fb] = empirical_time_noise(ph.t, ph.y);
    else
        src = noise_source.(name{1});
        [~, t_u, s2_u, ~, fb] = empirical_time_noise(src.t, src.y);
        noise_var = map_emp_noise_to_times(ph.t, t_u, s2_u);
    end
    [hyp, mu, s2, nlml] = fit_nn_emp(ph.t, ph.y, tgrid, noise_var, max_iters);
    fit = pack_fit(mu, s2, k_plot);
    fit.hyp = hyp;
    fit.nlml = nlml;
    fit.ell = exp(hyp.cov(1));
    fit.sf  = exp(hyp.cov(2));
    fit.noise_var = noise_var;
    fit.t = ph.t;
    fit.y = ph.y;
    fits.(name{1}) = fit;
    fprintf('%s: NLML=%.4f | ell=%.4f | sf=%.4f\n', name{1}, fit.nlml, fit.ell, fit.sf);
    fprintf('Empirical sn(t)=sqrt(s2) (*=pooled):\n');
    print_emp_schedule(name{1}, ph.t, noise_var, t_u, fb);
end
end

function [hyp, mu, s2] = fit_naive_se(x, y, xs, inffunc, meanfunc, covfunc, likfunc, max_iters)
% Zero-mean SE-iso with one learned sn. Same start as the commented SINDy.m path.
x = x(:); y = y(:); xs = xs(:);
ell0 = max(std(x), 0.5);
sf0  = max(std(y), 0.1);
sn0  = 0.1 * sf0;
if sn0 <= 0
    sn0 = 0.1;
end
hyp.mean = [];
hyp.cov  = log([ell0; sf0]);
hyp.lik  = log(sn0);
hyp = minimize(hyp, @gp, max_iters, inffunc, meanfunc, covfunc, likfunc, x, y);
[~, ~, mu, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
mu = mu(:);
s2 = s2(:);
end

function [hyp, mu, s2, nlml] = fit_nn_emp(x, y, xs, noise_var, max_iters)
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

function fit = pack_fit(mu, s2, k_plot)
sf = sqrt(max(s2(:), 0));
fit.mu = mu(:);
fit.sf = sf;
fit.lo = mu(:) - k_plot .* sf;
fit.hi = mu(:) + k_plot .* sf;
end

function fig = make_fit_figure(fits, tgrid, sty, fig_name, fig_pos)
fig = figure('Color', 'w', 'Position', fig_pos, 'Name', fig_name);
ax = axes('Parent', fig);
ax.Layer = 'top';
ax.FontSize = 24;
hold(ax, 'on'); grid(ax, 'off'); box(ax, 'on');
plot_phenotype(ax, tgrid, fits.M1, sty.col_M1, 'M1', sty.band_label);
plot_phenotype(ax, tgrid, fits.M2, sty.col_M2, 'M2', sty.band_label);
xlabel(ax, 'Time (days)', 'FontSize', 24);
ylabel(ax, 'cells/mm^2', 'FontSize', 24);
xlim(ax, [tgrid(1), tgrid(end)]);
title(ax, fig_name, 'FontSize', 16, 'Interpreter', 'none');
ylim_auto_from_fit(ax, fits.M1, fits.M2);
end

function plot_phenotype(ax, tgrid, fit, col, name, band_label)
tg = tgrid(:)';
fill(ax, [tg, fliplr(tg)], [fit.hi(:)', fliplr(fit.lo(:)')], col, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', sprintf('%s %s', name, band_label));
plot(ax, tgrid, fit.mu, '--', 'Color', col, 'LineWidth', 2, ...
    'DisplayName', sprintf('%s GP Mean', name));
scatter(ax, fit.t, fit.y, 36, 'filled', ...
    'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', ...
    'DisplayName', sprintf('%s Obs Data', name));
end

function ylim_auto_from_fit(ax, fitM1, fitM2)
vals = [fitM1.lo(:); fitM1.hi(:); fitM1.mu(:); fitM1.y(:); ...
        fitM2.lo(:); fitM2.hi(:); fitM2.mu(:); fitM2.y(:)];
pad = 0.05 * max(range(vals), 1);
ylim(ax, [min(vals) - pad, max(vals) + pad]);
end

function figL = make_shared_legend(col_M1, col_M2, band_label, fig_name)
figL = figure('Color', 'w', 'Position', [100, 100, 900, 80], 'Name', fig_name);
axL = axes('Parent', figL, 'Visible', 'off', 'XLim', [0 1], 'YLim', [0 1], ...
    'Position', [0 0 1 1]);
hold(axL, 'on');
hL = gobjects(6, 1);
hL(1) = fill(axL, nan, nan, col_M1, 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', sprintf('M1 %s', band_label));
hL(2) = plot(axL, nan, nan, '--', 'Color', col_M1, 'LineWidth', 2, ...
    'DisplayName', 'M1 GP Mean');
hL(3) = plot(axL, nan, nan, 'o', 'Color', col_M1, 'MarkerFaceColor', col_M1, ...
    'MarkerEdgeColor', 'k', 'MarkerSize', 5, 'DisplayName', 'M1 Obs Data');
hL(4) = fill(axL, nan, nan, col_M2, 'EdgeColor', 'none', 'FaceAlpha', 0.15, ...
    'DisplayName', sprintf('M2 %s', band_label));
hL(5) = plot(axL, nan, nan, '--', 'Color', col_M2, 'LineWidth', 2, ...
    'DisplayName', 'M2 GP Mean');
hL(6) = plot(axL, nan, nan, 'o', 'Color', col_M2, 'MarkerFaceColor', col_M2, ...
    'MarkerEdgeColor', 'k', 'MarkerSize', 5, 'DisplayName', 'M2 Obs Data');
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
end

function noise_var = map_emp_noise_to_times(t, t_unique, s2_unique)
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
% One line per unique time.
[t_show, ~, ic] = unique(t);
for i = 1:numel(t_show)
    j = find(abs(t_unique - t_show(i)) < 1e-12, 1);
    mark = ' ';
    if isempty(j) || used_fallback(j)
        mark = '*';
    end
    nv = noise_var(find(ic == i, 1));
    fprintf('  %s t=%g: sn=%.4f%s\n', name, t_show(i), sqrt(nv), mark);
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
