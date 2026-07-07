function [K, dK] = covICM_SE(hyp, x, z)
%COVICM_SE Intrinsic coregionalization model (ICM) covariance over (time, output).
%
%   Multi-output GP covariance for two outputs sharing a single squared-
%   exponential (SE) latent temporal process:
%
%       k( (t,d), (t',d') ) = B(d,d') * exp( -0.5 * (t - t')^2 / ell^2 )
%
%   where B is a 2x2 positive semidefinite coregionalization matrix and the
%   base SE signal variance is CLAMPED to 1. The overall output magnitudes
%   live entirely in B (a separate SE sigma_f would be non-identifiable with
%   the diagonal of B), so B carries all cross-output structure. This is the
%   intrinsic coregionalization model: one latent kernel (SE over time) times
%   one coregionalization matrix (LMC with Q = 1).
%
%   Inputs follow the standard GPML covariance interface:
%       x(:,1) = time
%       x(:,2) = output label (1 or 2)
%   and likewise for z when supplied.
%
%   Hyperparameters (hyp is a 4x1 vector):
%       hyp(1) = log(ell)
%       hyp(2) = b1        -> L(1,1) = exp(b1)
%       hyp(3) = b21       -> L(2,1) = b21
%       hyp(4) = b2        -> L(2,2) = exp(b2)
%   with the Cholesky factor
%       L = [ exp(b1)  0        ;
%             b21      exp(b2) ]
%   and B = L*L' (PSD by construction).
%
%   Usage (GPML directional-derivative convention, matching covDiscrete.m):
%       s        = covICM_SE()               % report number of hyperparameters ('4')
%       K        = covICM_SE(hyp, x)         % training covariance (symmetric)
%       Ks       = covICM_SE(hyp, x, z)      % cross covariance train-by-test
%       kss      = covICM_SE(hyp, x, 'diag') % self variances (diagonal)
%       [K, dK]  = covICM_SE(hyp, x)         % dK(Q) returns d tr(Q'*K)/d hyp
%
%   See also covDiscrete, LV_LMC (composed LMC kernel) and covSEiso.

% Report the number of hyperparameters (GPML calls with no arguments).
if nargin < 2
    K = '4';
    return;
end
if nargin < 3
    z = [];
end

xeqz = isempty(z);                 % training covariance if z is empty
dg   = strcmp(z, 'diag');          % self-variance (diagonal) requested

ell = exp(hyp(1));
b1  = hyp(2);
b21 = hyp(3);
b2  = hyp(4);

L = [exp(b1), 0; b21, exp(b2)];
B = L * L';                        % 2x2 PSD coregionalization matrix

t  = x(:, 1);
dx = fix(x(:, 2));

% --- SE factor over time and the coregionalization lookup B(d_i, d_j) ---
if dg                              % diagonal: t == t', d == d'
    Kse  = ones(size(t));          % exp(0) = 1
    Bfac = B(sub2ind(size(B), dx, dx));
    D2   = zeros(size(t));         % (t - t')^2 = 0 on the diagonal
elseif xeqz                        % symmetric training covariance
    D2   = (t - t').^2;            % pairwise squared distances
    Kse  = exp(-0.5 * D2 / ell^2);
    Bfac = B(dx, dx);
else                               % cross covariance train-by-test
    tz   = z(:, 1);
    dz   = fix(z(:, 2));
    D2   = (t - tz').^2;
    Kse  = exp(-0.5 * D2 / ell^2);
    Bfac = B(dx, dz);
end

K = Bfac .* Kse;

if nargout > 1
    dK = @(Q) dirder(Q, hyp, ell, D2, Kse, dx, z, dg, xeqz);
end
end

function [dhyp, dx_deriv] = dirder(Q, hyp, ell, D2, Kse, dx, z, dg, xeqz)
% Directional derivative: dhyp(i) = trace(Q' * dK/dhyp_i) = sum(Q .* dK_i).
b1  = hyp(2);
b21 = hyp(3);
b2  = hyp(4);
L   = [exp(b1), 0; b21, exp(b2)];

% (1) d/d log(ell): dK = B(d,d') .* Kse .* (D2/ell^2).
Bfac = coreg_lookup(L * L', dx, z, dg, xeqz);
dhyp = zeros(4, 1);
dhyp(1) = sum(sum(Q .* (Bfac .* Kse .* (D2 / ell^2))));

% (2-4) d/d b1, b21, b2: dK = dB(d,d') .* Kse, with dB = dL*L' + L*dL'.
dL_list = {[exp(b1), 0; 0, 0], [0, 0; 1, 0], [0, 0; 0, exp(b2)]};
for k = 1:3
    dB = dL_list{k} * L' + L * dL_list{k}';
    dBfac = coreg_lookup(dB, dx, z, dg, xeqz);
    dhyp(1 + k) = sum(sum(Q .* (dBfac .* Kse)));
end

if nargout > 1
    dx_deriv = zeros(numel(dx), size([dx, dx], 2));  % input derivatives unused
end
end

function Bfac = coreg_lookup(M, dx, z, dg, xeqz)
% Look up the 2x2 matrix M at the (row-label, col-label) pairs of the block.
if dg
    Bfac = M(sub2ind(size(M), dx, dx));
elseif xeqz
    Bfac = M(dx, dx);
else
    dz   = fix(z(:, 2));
    Bfac = M(dx, dz);
end
end
