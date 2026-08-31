function MM_Figure_sim_primary(out_dir, rmse_met, nlpd_met)
% Plot-only primary figure: baseline GP (M0) vs encoded GP (M3).
%
% Uses the nested ablation stored by MM_Figure_sim:
%   rmse_met, nlpd_met  — nRep x 4, columns M0, M1, M2, M3
% Primary comparison is M3 vs M0 (full encoding vs baseline), the same
% paired contrast computed in MM_Figure_sim (I_RMSE, dNLPD, p_better).
% Does not refit any GP.
%
% Percentile bands are a 95% Monte Carlo range (2.5th-97.5th), not a CI
% for the mean.

if nargin < 1 || isempty(out_dir)
    repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    out_dir = fullfile(repo_root, 'results');
end
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

if nargin < 3 || isempty(rmse_met) || isempty(nlpd_met)
    candidates = { ...
        fullfile(out_dir, 'MM_ablation_MC.mat'); ...
        fullfile(getenv('LOCALAPPDATA'), 'Bio_Inf_GP_Code', 'MM_ablation_MC.mat'); ...
        fullfile(getenv('LOCALAPPDATA'), 'Bio_Inf_GP_Code', 'MM_ablation_MC_metrics.mat')};
    S = [];
    src = '';
    for i = 1:numel(candidates)
        p = candidates{i};
        if ~exist(p, 'file')
            continue;
        end
        try
            T = load(p, 'rmse_met', 'nlpd_met');
        catch
            continue;
        end
        if isfield(T, 'rmse_met') && isfield(T, 'nlpd_met')
            S = T;
            src = p;
            break;
        end
    end
    if isempty(S)
        error('MM_Figure_sim_primary:NoResults', ...
            ['No readable MM_ablation_MC.mat with rmse_met/nlpd_met. ', ...
             'Run MM_Figure_sim first. This script does not refit GPs.']);
    end
    rmse_met = S.rmse_met;
    nlpd_met = S.nlpd_met;
    fprintf('Loaded paired MC metrics from %s\n', src);
end

if size(rmse_met, 2) < 4 || size(nlpd_met, 2) < 4 ...
        || size(rmse_met, 1) ~= size(nlpd_met, 1)
    error('MM_Figure_sim_primary:BadShape', ...
        'rmse_met and nlpd_met must be nRep x 4 (got %s and %s).', ...
        mat2str(size(rmse_met)), mat2str(size(nlpd_met)));
end

%% M0 baseline vs M3 encoded (same pairing as MM_Figure_sim pair M3 vs M0)
rmse_m0 = rmse_met(:, 1);
rmse_m3 = rmse_met(:, 4);
nlpd_m0 = nlpd_met(:, 1);
nlpd_m3 = nlpd_met(:, 4);
n_raw = size(rmse_met, 1);

ok = isfinite(rmse_m0) & isfinite(rmse_m3) & isfinite(nlpd_m0) & isfinite(nlpd_m3);
nonpos = ok & ((rmse_m0 <= 0) | (rmse_m3 <= 0));
ok = ok & ~nonpos;
if sum(~isfinite(rmse_m0) | ~isfinite(rmse_m3) | ~isfinite(nlpd_m0) | ~isfinite(nlpd_m3)) > 0
    warning('MM_Figure_sim_primary:IncompleteReplicates', ...
        'Dropped %d incomplete replicate(s) with NaN/Inf in RMSE or latent NLPD.', ...
        sum(~isfinite(rmse_m0) | ~isfinite(rmse_m3) | ~isfinite(nlpd_m0) | ~isfinite(nlpd_m3)));
end
if any(nonpos)
    warning('MM_Figure_sim_primary:NonpositiveRMSE', ...
        'Dropped %d replicate(s) with non-positive RMSE.', sum(nonpos));
end
if ~any(ok)
    error('MM_Figure_sim_primary:NoValidReplicates', ...
        'No paired replicates with finite RMSE and latent NLPD.');
end

rep_id = find(ok);
rmse_m0 = rmse_m0(ok);
rmse_m3 = rmse_m3(ok);
nlpd_m0 = nlpd_m0(ok);
nlpd_m3 = nlpd_m3(ok);
nRep = numel(rep_id);
fprintf('Paired replicates used in figure: %d / %d stored rows\n', nRep, n_raw);

% Same formulas as MM_Figure_sim (more = M3, less = M0)
delta_rmse = rmse_m3 - rmse_m0;
I_rmse = 100 * (rmse_m0 - rmse_m3) ./ max(rmse_m0, 1e-12);
delta_nlpd = nlpd_m3 - nlpd_m0;
p_better_rmse = mean(rmse_m3 < rmse_m0);
p_better_nlpd = mean(nlpd_m3 < nlpd_m0);

s_rmse_m0 = summarize_vec(rmse_m0);
s_rmse_m3 = summarize_vec(rmse_m3);
s_nlpd_m0 = summarize_vec(nlpd_m0);
s_nlpd_m3 = summarize_vec(nlpd_m3);
s_d_rmse  = summarize_vec(delta_rmse);
s_I_rmse  = summarize_vec(I_rmse);
s_d_nlpd  = summarize_vec(delta_nlpd);

%% Figure (white, tiledlayout 2x2; identity/zero lines as dashed black)
col_pt = [0.00, 0.00, 0.80];   % same blue as MM_Figure ground-truth line
col_ref = [0, 0, 0];
fs_ax = 12;
fs_ti = 13;
fs_ann = 10;

fig = figure('Color', 'w', 'Units', 'inches', 'Position', [0.8, 0.8, 10.2, 8.0], ...
    'PaperPositionMode', 'auto', 'Name', 'MM primary MC: M0 vs M3', ...
    'DefaultAxesFontSize', fs_ax, 'DefaultAxesLineWidth', 0.8, ...
    'DefaultAxesBox', 'on', 'DefaultAxesTickDir', 'out');
tl = tiledlayout(fig, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

axA = nexttile(tl, 1);
plot_identity_scatter(axA, rmse_m0, rmse_m3, col_pt, col_ref, false);
xlabel(axA, 'Baseline GP RMSE');
ylabel(axA, 'Encoded GP RMSE');
title(axA, 'A. RMSE by replicate', 'FontSize', fs_ti, 'FontWeight', 'normal');
annotate_encoded_better(axA, p_better_rmse, fs_ann);

axB = nexttile(tl, 2);
n_out_nlpd = plot_identity_scatter(axB, nlpd_m0, nlpd_m3, col_pt, col_ref, true);
xlabel(axB, 'Baseline GP latent NLPD');
ylabel(axB, 'Encoded GP latent NLPD');
title(axB, 'B. Latent NLPD by replicate', 'FontSize', fs_ti, 'FontWeight', 'normal');
annotate_encoded_better(axB, p_better_nlpd, fs_ann);
if n_out_nlpd > 0
    text(axB, 0.98, 0.04, sprintf('%d point(s) outside displayed range', n_out_nlpd), ...
        'Units', 'normalized', 'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'bottom', 'FontSize', fs_ann - 1, 'Color', [0.3 0.3 0.3]);
end

axC = nexttile(tl, 3);
plot_paired_distribution(axC, I_rmse, col_pt, col_ref);
xlabel(axC, 'RMSE improvement (%)');
title(axC, 'C. Paired RMSE improvement', 'FontSize', fs_ti, 'FontWeight', 'normal');
text(axC, 0.03, 0.97, sprintf(['Median improvement = %.1f%%\n', ...
    '95%% MC range = [%.1f, %.1f]%%\n', ...
    'Encoded better = %.1f%%'], ...
    s_I_rmse.median, s_I_rmse.lo, s_I_rmse.hi, 100 * p_better_rmse), ...
    'Units', 'normalized', 'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', 'FontSize', fs_ann, 'Interpreter', 'none');
text(axC, 0.97, 0.04, 'Positive favors encoded GP', 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
    'FontSize', fs_ann - 1, 'Color', [0.3 0.3 0.3], 'Interpreter', 'none');

axD = nexttile(tl, 4);
plot_paired_distribution(axD, delta_nlpd, col_pt, col_ref);
xlabel(axD, 'Encoded − baseline latent NLPD');
title(axD, 'D. Paired NLPD difference', 'FontSize', fs_ti, 'FontWeight', 'normal');
text(axD, 0.03, 0.97, sprintf(['Median \\DeltaNLPD = %.3f\n', ...
    '95%% MC range = [%.3f, %.3f]\n', ...
    'Encoded better = %.1f%%'], ...
    s_d_nlpd.median, s_d_nlpd.lo, s_d_nlpd.hi, 100 * p_better_nlpd), ...
    'Units', 'normalized', 'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', 'FontSize', fs_ann, 'Interpreter', 'tex');
text(axD, 0.97, 0.04, 'Negative favors encoded GP', 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
    'FontSize', fs_ann - 1, 'Color', [0.3 0.3 0.3], 'Interpreter', 'none');

%% Save
out_pdf = fullfile(out_dir, 'MM_Figure_sim_primary.pdf');
out_png = fullfile(out_dir, 'MM_Figure_sim_primary.png');
out_csv = fullfile(out_dir, 'MM_Figure_sim_paired_metrics.csv');
exportgraphics(fig, out_pdf, 'ContentType', 'vector', 'BackgroundColor', 'white');
exportgraphics(fig, out_png, 'Resolution', 300, 'BackgroundColor', 'white');

T = table(rep_id, rmse_m0, rmse_m3, delta_rmse, I_rmse, ...
    nlpd_m0, nlpd_m3, delta_nlpd, ...
    'VariableNames', {'replicate', 'rmse_baseline', 'rmse_encoded', ...
    'rmse_difference', 'rmse_percent_improvement', ...
    'nlpd_baseline', 'nlpd_encoded', 'nlpd_difference'});
writetable(T, out_csv);

fprintf('\nSaved %s\n', out_pdf);
fprintf('Saved %s\n', out_png);
fprintf('Saved %s\n', out_csv);

fprintf('\n=== Primary MC figure summary (%d paired replicates) ===\n', nRep);
fprintf('Number of paired replicates: %d\n', nRep);
fprintf('Baseline RMSE median [2.5%%, 97.5%%]: %.4g [%.4g, %.4g]\n', ...
    s_rmse_m0.median, s_rmse_m0.lo, s_rmse_m0.hi);
fprintf('Encoded RMSE median [2.5%%, 97.5%%]:  %.4g [%.4g, %.4g]\n', ...
    s_rmse_m3.median, s_rmse_m3.lo, s_rmse_m3.hi);
fprintf('Median RMSE percent improvement [2.5%%, 97.5%%]: %.2f [%.2f, %.2f]\n', ...
    s_I_rmse.median, s_I_rmse.lo, s_I_rmse.hi);
fprintf('Percent of replicates encoded GP has lower RMSE: %.1f%%\n', 100 * p_better_rmse);
fprintf('Baseline NLPD median [2.5%%, 97.5%%]: %.4g [%.4g, %.4g]\n', ...
    s_nlpd_m0.median, s_nlpd_m0.lo, s_nlpd_m0.hi);
fprintf('Encoded NLPD median [2.5%%, 97.5%%]:  %.4g [%.4g, %.4g]\n', ...
    s_nlpd_m3.median, s_nlpd_m3.lo, s_nlpd_m3.hi);
fprintf('Median ΔNLPD [2.5%%, 97.5%%]: %.4g [%.4g, %.4g]\n', ...
    s_d_nlpd.median, s_d_nlpd.lo, s_d_nlpd.hi);
fprintf('Percent of replicates encoded GP has lower NLPD: %.1f%%\n', 100 * p_better_nlpd);
fprintf('Mean RMSE Δ=%.4g (sd=%.4g); mean NLPD Δ=%.4g (sd=%.4g)\n', ...
    s_d_rmse.mean, s_d_rmse.std, s_d_nlpd.mean, s_d_nlpd.std);
end

%% ----- local functions (summaries match MM_Figure_sim) -----
function n_out = plot_identity_scatter(ax, x, y, col_pt, col_ref, allow_robust)
x = x(:);
y = y(:);
hold(ax, 'on');
if allow_robust
    [lo, hi, n_out] = scatter_display_limits(x, y);
else
    lo_hi = [min([x; y]), max([x; y])];
    pad = 0.04 * max(lo_hi(2) - lo_hi(1), eps);
    lo = lo_hi(1) - pad;
    hi = lo_hi(2) + pad;
    n_out = 0;
end
plot(ax, [lo, hi], [lo, hi], '--', 'Color', col_ref, 'LineWidth', 1.5, ...
    'HandleVisibility', 'off');
scatter(ax, x, y, 22, col_pt, 'filled', ...
    'MarkerFaceAlpha', 0.28, 'MarkerEdgeColor', col_pt, ...
    'MarkerEdgeAlpha', 0.40, 'LineWidth', 0.4);
xlim(ax, [lo, hi]);
ylim(ax, [lo, hi]);
axis(ax, 'equal');
xlim(ax, [lo, hi]);
ylim(ax, [lo, hi]);
grid(ax, 'off');
hold(ax, 'off');
end

function [lo, hi, n_out] = scatter_display_limits(x, y)
allv = [x(:); y(:)];
raw_lo = min(allv);
raw_hi = max(allv);
p_lo = col_percentile(allv, 0.5);
p_hi = col_percentile(allv, 99.5);
span = max(p_hi - p_lo, eps);
if (raw_hi - raw_lo) > 5 * span
    pad = 0.06 * span;
    lo = p_lo - pad;
    hi = p_hi + pad;
    n_out = sum(x < lo | x > hi | y < lo | y > hi);
else
    pad = 0.04 * max(raw_hi - raw_lo, eps);
    lo = raw_lo - pad;
    hi = raw_hi + pad;
    n_out = 0;
end
end

function annotate_encoded_better(ax, p_better, fs)
text(ax, 0.04, 0.96, sprintf('Encoded better: %.1f%%', 100 * p_better), ...
    'Units', 'normalized', 'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', 'FontSize', fs, 'Interpreter', 'none');
end

function plot_paired_distribution(ax, vals, col_pt, col_ref)
vals = vals(:);
n = numel(vals);
hold(ax, 'on');
boxchart(ax, ones(n, 1), vals, 'Orientation', 'horizontal', ...
    'BoxFaceColor', col_pt, 'BoxFaceAlpha', 0.28, ...
    'WhiskerLineColor', [0.25 0.25 0.25], 'LineWidth', 1.0, ...
    'MarkerStyle', 'none', 'BoxWidth', 0.45);
rng_state = rng;
rng(1);
jit = 0.14 * (rand(n, 1) - 0.5);
rng(rng_state);
scatter(ax, vals, 1 + jit, 14, col_pt, 'filled', ...
    'MarkerFaceAlpha', 0.22, 'MarkerEdgeColor', col_pt, ...
    'MarkerEdgeAlpha', 0.35, 'LineWidth', 0.3);
xline(ax, 0, '--', 'Color', col_ref, 'LineWidth', 1.5);
ylim(ax, [0.45, 1.55]);
yticks(ax, []);
ax.YAxis.Visible = 'off';
grid(ax, 'off');
hold(ax, 'off');
end

function s = summarize_vec(v)
v = v(:);
s.mean = mean(v, 'omitnan');
s.median = col_percentile(v, 50);
s.std = std(v, 0, 'omitnan');
s.lo = col_percentile(v, 2.5);
s.hi = col_percentile(v, 97.5);
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
