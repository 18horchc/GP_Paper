function varargout = gp_seiso_deriv_obs(mode, hyp, x, y, x_d, y_d, xs, sn_deriv, fix_sn_obs)
%GP_SEISO_DERIV_OBS Solak-style GP with function + derivative observations (SE-iso).
%   nlml = gp_seiso_deriv_obs('nlml', hyp, x, y, x_d, y_d, [], sn_deriv)
%   nlml = gp_seiso_deriv_obs('nlml', hyp, x, y, x_d, y_d, [], sn_deriv, true)
%       with fix_sn_obs true, hyp.lik is fixed and NLML gradients are w.r.t. hyp.cov only.
%   [ymu, ys2, fmu, fs2] = gp_seiso_deriv_obs('pred', hyp, x, y, x_d, y_d, xs, sn_deriv)
%   [m_deriv, s2_deriv] = gp_seiso_deriv_obs('deriv', hyp, x, y, x_d, y_d, xs, sn_deriv)

if nargin < 9
    fix_sn_obs = false;
end

switch lower(mode)
    case 'nlml'
        [varargout{1:nargout}] = nlml_core(hyp, x, y, x_d, y_d, sn_deriv, fix_sn_obs);
    case 'pred'
        [varargout{1}, varargout{2}, varargout{3}, varargout{4}] = ...
            pred_core(hyp, x, y, x_d, y_d, xs, sn_deriv);
    case 'deriv'
        [varargout{1}, varargout{2}] = deriv_pred_core(hyp, x, y, x_d, y_d, xs, sn_deriv);
    otherwise
        error('gp_seiso_deriv_obs:UnknownMode', 'Unknown mode: %s', mode);
end
end

function [nlml, dnll] = nlml_core(hyp, x, y, x_d, y_d, sn_deriv, fix_sn_obs)
if nargin < 7
    fix_sn_obs = false;
end
[Ky, z, nTot] = build_Ky(hyp, x, y, x_d, y_d, sn_deriv);
L = chol(Ky, 'lower');
alpha = L' \ (L \ z);
nlml = 0.5 * (z' * alpha) + sum(log(diag(L))) + 0.5 * nTot * log(2 * pi);

if nargout > 1
    nCov = 2;
    if fix_sn_obs
        p = hyp.cov(:);
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
        nlml_p = nlml_value(hp, x, y, x_d, y_d, sn_deriv);
        if i <= nCov
            hp.cov(i) = hp.cov(i) - 2 * step;
        else
            hp.lik(1) = hp.lik(1) - 2 * step;
        end
        nlml_m = nlml_value(hp, x, y, x_d, y_d, sn_deriv);
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

function nlml = nlml_value(hyp, x, y, x_d, y_d, sn_deriv)
[Ky, z, nTot] = build_Ky(hyp, x, y, x_d, y_d, sn_deriv);
L = chol(Ky, 'lower');
alpha = L' \ (L \ z);
nlml = 0.5 * (z' * alpha) + sum(log(diag(L))) + 0.5 * nTot * log(2 * pi);
end

function [ymu, ys2, fmu, fs2] = pred_core(hyp, x, y, x_d, y_d, xs, sn_deriv)
[Ky, z, ~, ell, sf2, sn2] = build_Ky(hyp, x, y, x_d, y_d, sn_deriv);
x = x(:); x_d = x_d(:); xs = xs(:);
n = numel(x); m = numel(x_d); nS = numel(xs);

L = chol(Ky, 'lower');
alpha = L' \ (L \ z);

K_xs_x = seiso_Kff(xs, x, ell, sf2);
K_xs_xd = seiso_Kfd(xs, x_d, ell, sf2);
K_star = [K_xs_x, K_xs_xd];

fmu = K_star * alpha;
V = L \ K_star';
k_diag = sf2 * ones(nS, 1);
fs2 = max(k_diag - sum(V.^2, 1).', 0);
fmu = fmu(:);
fs2 = fs2(:);
ymu = fmu;
ys2 = fs2 + sn2;
end

function [m_deriv, s2_deriv] = deriv_pred_core(hyp, x, y, x_d, y_d, xs, sn_deriv)
[Ky, z, ~, ell, sf2] = build_Ky(hyp, x, y, x_d, y_d, sn_deriv);
x = x(:); x_d = x_d(:); xs = xs(:);
nS = numel(xs);

L = chol(Ky, 'lower');
alpha = L' \ (L \ z);

K_d_x = seiso_Kdf(xs, x, ell, sf2);
K_d_xd = seiso_Kdd(xs, x_d, ell, sf2);
K_star = [K_d_x, K_d_xd];

m_deriv = K_star * alpha;
V = L \ K_star';
k_dd_diag = (sf2 / ell^2) * ones(nS, 1);
s2_deriv = max(k_dd_diag - sum(V.^2, 1).', 0);
m_deriv = m_deriv(:);
s2_deriv = s2_deriv(:);
end

function [Ky, z, nTot, ell, sf2, sn2] = build_Ky(hyp, x, y, x_d, y_d, sn_deriv)
x = x(:); y = y(:); x_d = x_d(:); y_d = y_d(:);
n = numel(x); m = numel(x_d);
nTot = n + m;

ell = exp(hyp.cov(1));
sf2 = exp(2 * hyp.cov(2));
sn2 = exp(2 * hyp.lik(1));
sn_deriv2 = sn_deriv^2;
jitter = 1e-10;

K_ff = seiso_Kff(x, x, ell, sf2);
K_fd = seiso_Kfd(x, x_d, ell, sf2);
K_df = seiso_Kdf(x_d, x, ell, sf2);
K_dd = seiso_Kdd(x_d, x_d, ell, sf2);

K_aug = [K_ff, K_fd; K_df, K_dd];
noise = [sn2 * ones(n, 1); sn_deriv2 * ones(m, 1)];
Ky = K_aug + diag(noise + jitter);
z = [y; y_d];
end

function K = seiso_Kff(xa, xb, ell, sf2)
R = xa - xb.';
K = sf2 * exp(-0.5 * (R ./ ell).^2);
end

function K = seiso_Kfd(xa, xb, ell, sf2)
% cov(f(xa), df/dx(xb)) = ∂k(xa, xb)/∂xb
R = xa - xb.';
Kxc = sf2 * exp(-0.5 * (R ./ ell).^2);
K = Kxc .* (R ./ ell^2);
end

function K = seiso_Kdf(xa, xb, ell, sf2)
% cov(df/dx(xa), f(xb)) = ∂k(xa, xb)/∂xa
R = xa - xb.';
Kxc = sf2 * exp(-0.5 * (R ./ ell).^2);
K = -Kxc .* (R ./ ell^2);
end

function K = seiso_Kdd(xa, xb, ell, sf2)
% cov(df/dx(xa), df/dx(xb)) = ∂²k/(∂xa ∂xb)
R = xa - xb.';
r2 = (R ./ ell).^2;
K = (sf2 / ell^2) * (1 - r2) .* exp(-0.5 * r2);
end
