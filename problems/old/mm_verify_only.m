% Numeric constraint verification from mm_last_run.mat (inactive baseline).
% Requires the old ell_ub sweep and save(..., 'mm_last_run.mat') in michaelis_menten.m.
here = fileparts(mfilename('fullpath'));
mat_path = fullfile(here, 'mm_last_run.mat');
if ~isfile(mat_path)
    error('mm_verify_only:NoMat', 'Run michaelis_menten.m first to create %s', mat_path);
end
S = load(mat_path);
gpml_folder_name = "C:\Users\chorc\OneDrive\Documents\Stroke Research\Gaussian Processes\Old\gpml-matlab-master\gpml-matlab-master";
if ~exist('gp', 'file')
    addpath(genpath(gpml_folder_name));
end
try, startup; catch, end

fprintf('Loaded %s\n', mat_path);
args = {S.hyp_tpl, S.inffunc, S.meanfunc, S.covfunc, S.likfunc, S.x_col, S.y_col, ...
    S.X_c, S.k, S.k_mono, S.y_max, S.Vmax, S.Km};
mm_verify_saved_theta(S.sens(end).theta_opt, S.ell_ub_sweep(end), args{:});
mm_verify_saved_theta(S.sens(1).theta_opt, S.ell_ub_sweep(1), args{:});

function mm_verify_saved_theta(theta_opt, ell_ub_label, hyp_tpl, inffunc, meanfunc, covfunc, likfunc, ...
    x_col, y_col, X_c, k, k_mono, y_max, Vmax, Km)
hyp = theta_to_hyp_saved(theta_opt, hyp_tpl);
if isempty(k_mono), k_mono = k; end
Xg = X_c(:);
nC = numel(Xg);
[ymu, ys2, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x_col, y_col, Xg);
ymu = ymu(:); fmu = fmu(:);
sy = sqrt(max(ys2(:), 0)); sf = sqrt(max(fs2(:), 0));
sn = exp(hyp.lik(1));
fprintf('\n--- Optimum (ell_ub=%g): ell=%.4f sf=%.4f sn=%.4f ---\n', ell_ub_label, ...
    exp(theta_opt(1)), exp(theta_opt(2)), exp(theta_opt(3)));
fprintf('[Bounds] max|ymu-fmu|=%.3e | sy-sf: min=%.4g med=%.4g max=%.4g | sn=%.4g\n', ...
    max(abs(ymu - fmu)), min(sy-sf), median(sy-sf), max(sy-sf), sn);
fprintf('[Bounds] coded nonneg max(c)=%.6g | latent max(k*sf-fmu)=%.6g\n', ...
    max(k*sy-ymu), max(k*sf-fmu));
fprintf('[Bounds] coded upper max(c)=%.6g | latent max(fmu+k*sf-y_max)=%.6g\n', ...
    max(ymu+k*sy-y_max), max(fmu+k*sf-y_max));
[m_deriv, s2_deriv] = gp_seiso_deriv_pred_saved(hyp, x_col, y_col, Xg);
s_deriv = sqrt(max(s2_deriv(:), 0));
c_mono_inc = k_mono .* s_deriv - m_deriv(:);
c_mono_dec = m_deriv(:) + k_mono .* s_deriv;
mm_d = Vmax * Km ./ (Km + Xg).^2;
fprintf('[Mono] MM dV/d[S]: min=%.4g max=%.4g | m_deriv min=%.4g max=%.4g | #m_deriv<0: %d/%d\n', ...
    min(mm_d), max(mm_d), min(m_deriv), max(m_deriv), sum(m_deriv < 0), nC);
fprintf('[Mono] coded max(c_inc)=%.6g | WRONG decr max(m+k*s)=%.6g | corr(m_deriv,MM'')=%.4f\n', ...
    max(c_mono_inc), max(c_mono_dec), corr(m_deriv(:), mm_d));
idx_show = unique(round(linspace(1, nC, min(5, nC))));
fprintf('[Sample] S     ymu    fmu    sy      sf     m_deriv  MM_dv/dS\n');
for j = idx_show
    fprintf('  %5.2f %6.3f %6.3f %6.4f %6.4f %8.4f %8.4f\n', ...
        Xg(j), ymu(j), fmu(j), sy(j), sf(j), m_deriv(j), mm_d(j));
end
end

function hyp = theta_to_hyp_saved(theta, hyp_tpl)
hyp = hyp_tpl;
hyp.cov = theta(1:2);
hyp.lik = theta(3);
if isfield(hyp_tpl, 'mean') && ~isempty(hyp_tpl.mean)
    hyp.mean = theta(4:end);
else
    hyp.mean = [];
end
end

function [m_deriv, s2_deriv] = gp_seiso_deriv_pred_saved(hyp, x, y, X_c)
ell = exp(hyp.cov(1)); sf2 = exp(2 * hyp.cov(2)); sn2 = exp(2 * hyp.lik(1));
x = x(:); y = y(:); X_c = X_c(:); n = numel(x); m = numel(X_c);
if isempty(hyp.mean), mu = 0; else, mu = hyp.mean; end
ytil = y - mu;
dxx = (x - x.') ./ ell;
K = sf2 * exp(-0.5 * dxx.^2);
Ky = K + sn2 * eye(n);
R = X_c - x.';
Kxc = sf2 * exp(-0.5 * (R ./ ell).^2);
K_df = -Kxc .* (R ./ (ell^2));
k_dd_diag = (sf2 / ell^2) * ones(m, 1);
alpha = Ky \ ytil;
m_deriv = K_df * alpha;
L = chol(Ky, 'lower');
B = L \ K_df';
s2_deriv = max(k_dd_diag - sum(B.^2, 1).', 0);
end
