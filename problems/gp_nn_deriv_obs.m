function varargout = gp_nn_deriv_obs(mode, hyp, x, y, x_d, y_d, xs, sn_deriv, fix_sn_obs, noise_var)
%GP_NN_DERIV_OBS Solak-style GP with function + derivative observations (NN / covNNone).
%   Same API as gp_seiso_deriv_obs, but uses the neural-network (arcsin) kernel.
%
%   nlml = gp_nn_deriv_obs('nlml', hyp, x, y, x_d, y_d, [], sn_deriv)
%   nlml = gp_nn_deriv_obs('nlml', hyp, x, y, x_d, y_d, [], sn_deriv, true)
%   [ymu, ys2, fmu, fs2] = gp_nn_deriv_obs('pred', hyp, x, y, x_d, y_d, xs, sn_deriv)
%   [m_deriv, s2_deriv] = gp_nn_deriv_obs('deriv', hyp, x, y, x_d, y_d, xs, sn_deriv)
%   Optional noise_var (n_x x 1) overrides homoscedastic function-obs noise from hyp.lik.
%
%   Inputs are 1-D (columns). Kernel matches GPML covNNone (bias-augmented arcsin).

if nargin < 9
    fix_sn_obs = false;
end
if nargin < 10
    noise_var = [];
end

switch lower(mode)
    case 'nlml'
        [varargout{1:nargout}] = nlml_core(hyp, x, y, x_d, y_d, sn_deriv, fix_sn_obs, noise_var);
    case 'pred'
        [varargout{1}, varargout{2}, varargout{3}, varargout{4}] = ...
            pred_core(hyp, x, y, x_d, y_d, xs, sn_deriv, noise_var);
    case 'deriv'
        [varargout{1}, varargout{2}] = deriv_pred_core(hyp, x, y, x_d, y_d, xs, sn_deriv, noise_var);
    otherwise
        error('gp_nn_deriv_obs:UnknownMode', 'Unknown mode: %s', mode);
end
end

function [nlml, dnll] = nlml_core(hyp, x, y, x_d, y_d, sn_deriv, fix_sn_obs, noise_var)
if nargin < 7
    fix_sn_obs = false;
end
if nargin < 8
    noise_var = [];
end
[L, ~, z, nTot] = factor_Ky(hyp, x, y, x_d, y_d, sn_deriv, noise_var);
alpha = L' \ (L \ z);
nlml = 0.5 * (z' * alpha) + sum(log(diag(L))) + 0.5 * nTot * log(2 * pi);

if nargout > 1
    nCov = 2;
    if fix_sn_obs || ~isempty(noise_var)
        p = hyp.cov(:);
        fix_sn_obs = true;
    else
        p = [hyp.cov(:); hyp.lik(:)];
    end
    dnll_vec = zeros(numel(p), 1);
    step = 1e-4;
    for i = 1:numel(p)
        hp = hyp;
        if i <= nCov
            hp.cov(i) = hp.cov(i) + step;
        else
            hp.lik(1) = hp.lik(1) + step;
        end
        nlml_p = nlml_value(hp, x, y, x_d, y_d, sn_deriv, noise_var);
        if i <= nCov
            hp.cov(i) = hp.cov(i) - 2 * step;
        else
            hp.lik(1) = hp.lik(1) - 2 * step;
        end
        nlml_m = nlml_value(hp, x, y, x_d, y_d, sn_deriv, noise_var);
        dnll_vec(i) = (nlml_p - nlml_m) / (2 * step);
    end
    dnll = hyp;
    dnll.cov = dnll_vec(1:nCov);
    if fix_sn_obs
        dnll.lik = [];
    else
        dnll.lik = dnll_vec(nCov + 1);
    end
    dnll.mean = [];
end
end

function nlml = nlml_value(hyp, x, y, x_d, y_d, sn_deriv, noise_var)
[L, ~, z, nTot] = factor_Ky(hyp, x, y, x_d, y_d, sn_deriv, noise_var);
alpha = L' \ (L \ z);
nlml = 0.5 * (z' * alpha) + sum(log(diag(L))) + 0.5 * nTot * log(2 * pi);
end

function [ymu, ys2, fmu, fs2] = pred_core(hyp, x, y, x_d, y_d, xs, sn_deriv, noise_var)
[L, ~, z, ~, ell, sf2, sn2] = factor_Ky(hyp, x, y, x_d, y_d, sn_deriv, noise_var);
x = x(:); x_d = x_d(:); xs = xs(:);
nS = numel(xs);

alpha = L' \ (L \ z);

K_xs_x = nn_Kff(xs, x, ell, sf2);
K_xs_xd = nn_Kfd(xs, x_d, ell, sf2);
K_star = [K_xs_x, K_xs_xd];

fmu = K_star * alpha;
V = L \ K_star';
k_diag = nn_Kff_diag(xs, ell, sf2);
fs2 = max(k_diag - sum(V.^2, 1).', 0);
fmu = fmu(:);
fs2 = fs2(:);
ymu = fmu;
ys2 = fs2 + sn2;
end

function [m_deriv, s2_deriv] = deriv_pred_core(hyp, x, y, x_d, y_d, xs, sn_deriv, noise_var)
[L, ~, z, ~, ell, sf2] = factor_Ky(hyp, x, y, x_d, y_d, sn_deriv, noise_var);
x = x(:); x_d = x_d(:); xs = xs(:);

alpha = L' \ (L \ z);

K_d_x = nn_Kdf(xs, x, ell, sf2);
K_d_xd = nn_Kdd(xs, x_d, ell, sf2);
K_star = [K_d_x, K_d_xd];

m_deriv = K_star * alpha;
V = L \ K_star';
k_dd_diag = nn_Kdd_diag(xs, ell, sf2);
s2_deriv = max(k_dd_diag - sum(V.^2, 1).', 0);
m_deriv = m_deriv(:);
s2_deriv = s2_deriv(:);
end

function [L, Ky, z, nTot, ell, sf2, sn2] = factor_Ky(hyp, x, y, x_d, y_d, sn_deriv, noise_var)
jitter_scale = 1;
for attempt = 1:12
    [Ky, z, nTot, ell, sf2, sn2] = build_Ky(hyp, x, y, x_d, y_d, sn_deriv, noise_var, jitter_scale);
    [L, p] = chol(Ky, 'lower');
    if p == 0
        return;
    end
    jitter_scale = jitter_scale * 10;
end
error('gp_nn_deriv_obs:CholFailed', ...
    'Cholesky failed after adaptive jitter escalation (final scale=%g).', jitter_scale);
end

function [Ky, z, nTot, ell, sf2, sn2] = build_Ky(hyp, x, y, x_d, y_d, sn_deriv, noise_var, jitter_scale)
if nargin < 7
    noise_var = [];
end
if nargin < 8
    jitter_scale = 1;
end
x = x(:); y = y(:); x_d = x_d(:); y_d = y_d(:);
n = numel(x); m = numel(x_d);
nTot = n + m;

ell = exp(hyp.cov(1));
sf2 = exp(2 * hyp.cov(2));
sn2 = exp(2 * hyp.lik(1));
sn_deriv2 = sn_deriv^2;

K_ff = nn_Kff(x, x, ell, sf2);
K_fd = nn_Kfd(x, x_d, ell, sf2);
K_df = nn_Kdf(x_d, x, ell, sf2);
K_dd = nn_Kdd(x_d, x_d, ell, sf2);

K_aug = [K_ff, K_fd; K_df, K_dd];
diag_mean = mean(diag(K_aug));
if ~isfinite(diag_mean) || diag_mean <= 0
    diag_mean = 1;
end
jitter0 = max(1e-10, 1e-12 * diag_mean);
jitter = jitter_scale * jitter0;
max_jitter = max(1e-4, 1e-4 * diag_mean);
if jitter > max_jitter
    jitter = max_jitter;
end

if isempty(noise_var)
    sn_obs = sn2 * ones(n, 1);
else
    sn_obs = noise_var(:);
    if numel(sn_obs) ~= n
        error('gp_nn_deriv_obs:BadNoiseVar', ...
            'noise_var must have length numel(x)=%d, got %d.', n, numel(sn_obs));
    end
end
noise = [sn_obs; sn_deriv2 * ones(m, 1)];
Ky = K_aug + diag(noise + jitter);
z = [y; y_d];
end

% ---- Neural-network (covNNone) blocks for 1-D inputs ----
% k(u,v) = sf2 * asin(A), A = (1+uv) / sqrt((ell^2+1+u^2)(ell^2+1+v^2))

function [A, den, su, sv, ell2] = nn_A(xa, xb, ell)
ell2 = ell^2;
su = 1 + xa.^2;
sv = 1 + xb.^2;
S = 1 + xa * xb.';
den = sqrt(ell2 + su) * sqrt(ell2 + sv).';
A = S ./ den;
A = max(min(A, 1 - 1e-12), -1 + 1e-12);
end

function K = nn_Kff(xa, xb, ell, sf2)
xa = xa(:); xb = xb(:);
A = nn_A(xa, xb, ell);
K = sf2 * asin(A);
end

function k = nn_Kff_diag(x, ell, sf2)
x = x(:);
ell2 = ell^2;
su = 1 + x.^2;
A = su ./ (ell2 + su);
A = max(min(A, 1 - 1e-12), -1 + 1e-12);
k = sf2 * asin(A);
end

function K = nn_Kfd(xa, xb, ell, sf2)
% cov(f(xa), df/dx(xb)) = dk/dv
xa = xa(:); xb = xb(:);
[A, den, su, ~, ell2] = nn_A(xa, xb, ell);
g = 1 ./ sqrt(1 - A.^2);
% A_v(i,j) = (ell2+su(i)) * (xa(i)*(ell2+1) - xb(j)) / den(i,j)^3
Av = bsxfun(@times, (ell2 + su), bsxfun(@minus, xa * (ell2 + 1), xb.')) ./ (den.^3);
K = sf2 * g .* Av;
end

function K = nn_Kdf(xa, xb, ell, sf2)
% cov(df/dx(xa), f(xb)) = dk/du
xa = xa(:); xb = xb(:);
[A, den, ~, sv, ell2] = nn_A(xa, xb, ell);
g = 1 ./ sqrt(1 - A.^2);
% A_u(i,j) = (ell2+sv(j)) * (xb(j)*(ell2+1) - xa(i)) / den(i,j)^3
Au = bsxfun(@times, (ell2 + sv).', bsxfun(@minus, (ell2 + 1) * xb.', xa)) ./ (den.^3);
K = sf2 * g .* Au;
end

function K = nn_Kdd(xa, xb, ell, sf2)
% cov(df/dx(xa), df/dx(xb)) = d^2 k / (du dv)
xa = xa(:); xb = xb(:);
[A, den, su, sv, ell2] = nn_A(xa, xb, ell);
g = 1 ./ sqrt(1 - A.^2);
oneA2 = 1 - A.^2;

Au = bsxfun(@times, (ell2 + sv).', bsxfun(@minus, (ell2 + 1) * xb.', xa)) ./ (den.^3);
Av = bsxfun(@times, (ell2 + su), bsxfun(@minus, xa * (ell2 + 1), xb.')) ./ (den.^3);
Auv = ((ell2 + 1)^2 + xa * xb.') ./ (den.^3);

K = sf2 * g .* (A .* Au .* Av ./ oneA2 + Auv);
end

function k = nn_Kdd_diag(x, ell, sf2)
x = x(:);
ell2 = ell^2;
su = 1 + x.^2;
den = ell2 + su;
A = su ./ den;
A = max(min(A, 1 - 1e-12), -1 + 1e-12);
g = 1 ./ sqrt(1 - A.^2);
Au = (ell2 .* x) ./ (den.^2);
Auv = ((ell2 + 1).^2 + x.^2) ./ (den.^3);
k = sf2 * g .* (A .* Au.^2 ./ (1 - A.^2) + Auv);
end
