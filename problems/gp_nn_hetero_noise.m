function varargout = gp_nn_hetero_noise(mode, hyp, x, y, noise_var, xs, noise_var_star)
%GP_NN_HETERO_NOISE NN (covNNone) GP with fixed per-row observation noise.
%   Same API as gp_seiso_hetero_noise, using the neural-network arcsin kernel.
%   nlml = gp_nn_hetero_noise('nlml', hyp, x, y, noise_var)
%   [ymu, ys2, fmu, fs2] = gp_nn_hetero_noise('pred', hyp, x, y, noise_var, xs)
%   K_y = K_f + diag(noise_var). Optimizes hyp.cov only (ell, sf); hyp.lik ignored.

if nargin < 7
    noise_var_star = [];
end

switch lower(mode)
    case 'nlml'
        [varargout{1:nargout}] = nlml_core(hyp, x, y, noise_var);
    case 'pred'
        [varargout{1}, varargout{2}, varargout{3}, varargout{4}] = ...
            pred_core(hyp, x, y, noise_var, xs, noise_var_star);
    otherwise
        error('gp_nn_hetero_noise:UnknownMode', 'Unknown mode: %s', mode);
end
end

function [nlml, dnll] = nlml_core(hyp, x, y, noise_var)
[Ky, z, nTot] = build_Ky(hyp, x, y, noise_var);
L = chol(Ky, 'lower');
alpha = L' \ (L \ z);
nlml = 0.5 * (z' * alpha) + sum(log(diag(L))) + 0.5 * nTot * log(2 * pi);

if nargout > 1
    dnll_vec = zeros(2, 1);
    step = 1e-4;
    for i = 1:2
        hp = hyp;
        hp.cov(i) = hp.cov(i) + step;
        nlml_p = nlml_value(hp, x, y, noise_var);
        hp.cov(i) = hp.cov(i) - 2 * step;
        nlml_m = nlml_value(hp, x, y, noise_var);
        dnll_vec(i) = (nlml_p - nlml_m) / (2 * step);
    end
    dnll = hyp;
    dnll.cov = dnll_vec;
    dnll.lik = [];
    dnll.mean = [];
end
end

function nlml = nlml_value(hyp, x, y, noise_var)
[Ky, z, nTot] = build_Ky(hyp, x, y, noise_var);
L = chol(Ky, 'lower');
alpha = L' \ (L \ z);
nlml = 0.5 * (z' * alpha) + sum(log(diag(L))) + 0.5 * nTot * log(2 * pi);
end

function [ymu, ys2, fmu, fs2] = pred_core(hyp, x, y, noise_var, xs, noise_var_star)
if nargin < 6
    noise_var_star = [];
end
[Ky, z, ~, ell, sf2] = build_Ky(hyp, x, y, noise_var);
x = x(:);
xs = xs(:);
nS = numel(xs);

L = chol(Ky, 'lower');
alpha = L' \ (L \ z);

K_star = nn_Kff(xs, x, ell, sf2);
fmu = K_star * alpha;
V = L \ K_star';
ell2 = ell^2;
su = 1 + xs.^2;
A = su ./ (ell2 + su);
A = max(min(A, 1 - 1e-12), -1 + 1e-12);
k_diag = sf2 * asin(A);
fs2 = max(k_diag - sum(V.^2, 1).', 0);
fmu = fmu(:);
fs2 = fs2(:);
ymu = fmu;
if isempty(noise_var_star)
    ys2 = fs2;
else
    noise_var_star = noise_var_star(:);
    if numel(noise_var_star) ~= nS
        error('gp_nn_hetero_noise:BadNoiseVarStar', ...
            'noise_var_star must have length numel(xs)=%d, got %d.', nS, numel(noise_var_star));
    end
    ys2 = fs2 + max(noise_var_star, 0);
end
end

function [Ky, z, nTot, ell, sf2] = build_Ky(hyp, x, y, noise_var)
x = x(:);
y = y(:);
noise_var = noise_var(:);
nTot = numel(x);

ell = exp(hyp.cov(1));
sf2 = exp(2 * hyp.cov(2));

K_f = nn_Kff(x, x, ell, sf2);
jitter = 1e-8 * mean(diag(K_f));
Ky = K_f + diag(noise_var + jitter);
z = y;
end

function K = nn_Kff(xa, xb, ell, sf2)
xa = xa(:); xb = xb(:);
ell2 = ell^2;
su = 1 + xa.^2;
sv = 1 + xb.^2;
S = 1 + xa * xb.';
A = S ./ (sqrt(ell2 + su) * sqrt(ell2 + sv).');
A = max(min(A, 1 - 1e-12), -1 + 1e-12);
K = sf2 * asin(A);
end
