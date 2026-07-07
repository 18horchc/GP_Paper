function f = sample_gp_se(t, ell, sf, jitter)
%SAMPLE_GP_SE Draw one zero-mean GP sample with an SE kernel on a 1D grid.
%
%   f = sample_gp_se(t, ell)                draws f ~ GP(0, k_SE) with sf = 1
%   f = sample_gp_se(t, ell, sf)            uses signal std sf
%   f = sample_gp_se(t, ell, sf, jitter)    adds jitter*eye for stability
%
%   The covariance is the squared-exponential
%       k(t,t') = sf^2 * exp( -0.5 * (t - t')^2 / ell^2 )
%   and the sample is generated via a Cholesky factor of K + jitter*I.
%
%   Used to build the shared latent function u(t) in the ICM toy demo.

if nargin < 3 || isempty(sf),     sf = 1;        end
if nargin < 4 || isempty(jitter), jitter = 1e-10; end

t = t(:);
n = numel(t);

D2 = (t - t').^2;
K  = sf^2 * exp(-0.5 * D2 / ell^2) + jitter * eye(n);

Lc = chol(K, 'lower');
f  = Lc * randn(n, 1);
end
