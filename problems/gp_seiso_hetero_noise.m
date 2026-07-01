function varargout = gp_seiso_hetero_noise(mode, hyp, x, y, noise_var, xs)
%GP_SEISO_HETERO_NOISE SE-iso GP with fixed per-row observation noise.
%   nlml = gp_seiso_hetero_noise('nlml', hyp, x, y, noise_var)
%   [ymu, ys2, fmu, fs2] = gp_seiso_hetero_noise('pred', hyp, x, y, noise_var, xs)
%
%   K_y = K_f + diag(noise_var). Optimizes hyp.cov only (ell, sf); hyp.lik ignored.


%This switch dispatches behavior based on mode: it calls either nlml_core to 
% compute negative log marginal likelihood results or pred_core to compute 
% predictions, routing outputs into varargout; unknown modes raise an error.
switch lower(mode)
    case 'nlml'
        [varargout{1:nargout}] = nlml_core(hyp, x, y, noise_var);
    case 'pred'
        [varargout{1}, varargout{2}, varargout{3}, varargout{4}] = ...
            pred_core(hyp, x, y, noise_var, xs);
    otherwise
        error('gp_seiso_hetero_noise:UnknownMode', 'Unknown mode: %s', mode);
end
end

% fit by min NLML, and compute via Cholesky factor
% (same in principle as baseline GPML fit, just using the custom Ky)
% returns scalar and gradient
function [nlml, dnll] = nlml_core(hyp, x, y, noise_var)
[Ky, z, nTot] = build_Ky(hyp, x, y, noise_var);
L = chol(Ky, 'lower');
alpha = L' \ (L \ z);
nlml = 0.5 * (z' * alpha) + sum(log(diag(L))) + 0.5 * nTot * log(2 * pi);

%this section estimates derivatives of NLML with respect to log(ell) and
%log (sigma_f) via central finite differences
% Could consider replacing this with analytic gradient. Would need to work
% out the trace formula/kernel algebra
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

%same as nlml_core, but only returns scalar - used in nlml_core
function nlml = nlml_value(hyp, x, y, noise_var)
[Ky, z, nTot] = build_Ky(hyp, x, y, noise_var);
L = chol(Ky, 'lower');
alpha = L' \ (L \ z);
nlml = 0.5 * (z' * alpha) + sum(log(diag(L))) + 0.5 * nTot * log(2 * pi);
end

%prediction step uses standard GP posterior (same as baseline), but with
%the custom Ky
function [ymu, ys2, fmu, fs2] = pred_core(hyp, x, y, noise_var, xs)
[Ky, z, ~, ell, sf2] = build_Ky(hyp, x, y, noise_var);
x = x(:);
xs = xs(:);
nS = numel(xs);

L = chol(Ky, 'lower');
alpha = L' \ (L \ z);

K_star = seiso_Kff(xs, x, ell, sf2);
fmu = K_star * alpha;
V = L \ K_star';
k_diag = sf2 * ones(nS, 1);
fs2 = max(k_diag - sum(V.^2, 1).', 0);
fmu = fmu(:);
fs2 = fs2(:);
ymu = fmu;
ys2 = fs2;
end

% build custom cov with different noise along diag
function [Ky, z, nTot, ell, sf2] = build_Ky(hyp, x, y, noise_var)
x = x(:);
y = y(:);
noise_var = noise_var(:); %this is the vector (sigma_1^2,...sigma_n^2)
nTot = numel(x);

ell = exp(hyp.cov(1));
sf2 = exp(2 * hyp.cov(2));

K_f = seiso_Kff(x, x, ell, sf2);
jitter = 1e-8 * mean(diag(K_f)); %for numerical stabiity of the Cholesky factorization
Ky = K_f + diag(noise_var + jitter);
z = y;
end

%build SE kernel
function K = seiso_Kff(xa, xb, ell, sf2)
R = xa - xb.';
K = sf2 * exp(-0.5 * (R ./ ell).^2);
end
