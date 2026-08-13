function model = mlhgp_seiso(x, y, hyp0_f, opts)
%MLHGP_SEISO Most-likely heteroscedastic GP (Kersting-style) with SE-iso kernels.
%   model = mlhgp_seiso(x, y, hyp0_f)
%   model = mlhgp_seiso(x, y, hyp0_f, opts)
%
%   Alternates fitting a mean GP with fixed diag(noise_var) and a noise GP on
%   log squared residuals, until max_iter or relative change below tol.
%   Mean GP uses meanZero; noise GP (g = log sigma^2) uses meanConst so the
%   overall log-noise level is not shrunk toward zero.
%
%   opts fields (all optional):
%     max_iter, tol, minimize_n, ell_g_min, ell_g_max, ell_g0, sn_g, eps_resid
%
%   model fields: hyp_f, hyp_g, noise_var, z_train, n_iter, nlml_hist, opts,
%                 meanfunc_g
%   Use mlhgp_predict_noise(model, x, xs) for sigma_n^2 at test inputs.

x = x(:);
y = y(:);
if nargin < 3 || isempty(hyp0_f)
    hyp0_f = struct('mean', [], 'cov', log([std(x); std(y)]), 'lik', []);
end
if nargin < 4 || isempty(opts)
    opts = struct();
end
opts = mlhgp_default_opts(opts, y);

meanfunc_f = @meanZero;
meanfunc_g = @meanConst;
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

% Init: homoscedastic GP (learn sn)
sn0 = log(max(std(y) * 0.1, 1e-3));
hyp_homo = struct('mean', [], 'cov', hyp0_f.cov(:), 'lik', sn0);
hyp_homo = minimize(hyp_homo, @gp, opts.minimize_n, inffunc, meanfunc_f, covfunc, likfunc, x, y);
[~, ~, fmu_tr, ~] = gp(hyp_homo, inffunc, meanfunc_f, covfunc, likfunc, x, y, x);

hyp_f = struct('mean', [], 'cov', hyp_homo.cov(:), 'lik', []);
% Noise GP: constant mean on log-variance scale (set from first z below)
hyp_g = struct('mean', 0, 'cov', log([opts.ell_g0; 1]), 'lik', log(opts.sn_g));
nlml_hist = nan(opts.max_iter, 1);
noise_var = max((y - fmu_tr(:)).^2, opts.eps_resid);

for it = 1:opts.max_iter
    noise_var_prev = noise_var;

    % Fit noise GP to log empirical residual variances
    z = log(noise_var);
    if it == 1
        hyp_g.mean = mean(z);
    end
    hyp_g = minimize(hyp_g, @gp, opts.minimize_n, inffunc, meanfunc_g, covfunc, likfunc, x, z);
    hyp_g.cov(1) = min(max(hyp_g.cov(1), log(opts.ell_g_min)), log(opts.ell_g_max));
    [~, ~, gmu, ~] = gp(hyp_g, inffunc, meanfunc_g, covfunc, likfunc, x, z, x);
    noise_var = max(exp(gmu(:)), opts.eps_resid);

    % Refit mean GP with heteroscedastic R
    obj_f = @(h) gp_seiso_hetero_noise('nlml', h, x, y, noise_var);
    hyp_f = minimize(hyp_f, obj_f, opts.minimize_n);
    nlml_hist(it) = obj_f(hyp_f);

    % Empirical residuals for next noise targets
    [~, ~, fmu_tr, ~] = gp_seiso_hetero_noise('pred', hyp_f, x, y, noise_var, x);
    resid2 = max((y - fmu_tr(:)).^2, opts.eps_resid);

    rel = norm(noise_var - noise_var_prev) / max(norm(noise_var_prev), eps);
    if rel < opts.tol
        break;
    end
    noise_var = resid2;
end

% Final smoothed noise from last residual targets
z_train = log(max((y - fmu_tr(:)).^2, opts.eps_resid));
hyp_g = minimize(hyp_g, @gp, opts.minimize_n, inffunc, meanfunc_g, covfunc, likfunc, x, z_train);
hyp_g.cov(1) = min(max(hyp_g.cov(1), log(opts.ell_g_min)), log(opts.ell_g_max));
[~, ~, gmu, ~] = gp(hyp_g, inffunc, meanfunc_g, covfunc, likfunc, x, z_train, x);
noise_var = max(exp(gmu(:)), opts.eps_resid);
hyp_f = minimize(hyp_f, @(h) gp_seiso_hetero_noise('nlml', h, x, y, noise_var), opts.minimize_n);

model = struct();
model.hyp_f = hyp_f;
model.hyp_g = hyp_g;
model.meanfunc_g = meanfunc_g;
model.noise_var = noise_var(:);
model.z_train = z_train(:);
model.n_iter = it;
model.nlml_hist = nlml_hist(1:it);
model.opts = opts;
end

function opts = mlhgp_default_opts(opts, y)
if ~isfield(opts, 'max_iter') || isempty(opts.max_iter), opts.max_iter = 10; end
if ~isfield(opts, 'tol') || isempty(opts.tol), opts.tol = 1e-3; end
if ~isfield(opts, 'minimize_n') || isempty(opts.minimize_n), opts.minimize_n = -100; end
if ~isfield(opts, 'ell_g_min') || isempty(opts.ell_g_min), opts.ell_g_min = 1.0; end
if ~isfield(opts, 'ell_g_max') || isempty(opts.ell_g_max), opts.ell_g_max = 10; end
if ~isfield(opts, 'ell_g0') || isempty(opts.ell_g0), opts.ell_g0 = 3.0; end
if ~isfield(opts, 'sn_g') || isempty(opts.sn_g), opts.sn_g = 0.5; end
if ~isfield(opts, 'eps_resid') || isempty(opts.eps_resid)
    opts.eps_resid = 1e-6 * max(var(y), 1);
end
end
