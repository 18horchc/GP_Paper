function out = vhgpr_fit_predict(x, y, xs, opts)
%VHGPR_FIT_PREDICT Path-isolated wrapper around Lazaro-Gredilla & Titsias VHGPR.
%   out = vhgpr_fit_predict(x, y, xs)
%   out = vhgpr_fit_predict(x, y, xs, opts)
%
%   Temporarily adds problems/vhgpr to the path (authors' MATLAB code from
%   IPL-UV/simpleR), runs vhgpr_ui, then restores the path so GPML's
%   minimize/covSEiso/sq_dist are not permanently shadowed.
%
%   Cite: M. Lazaro-Gredilla and M. K. Titsias, "Variational Heteroscedastic
%   Gaussian Process Regression," ICML 2011.
%
%   opts.iter  (default 40)  VHGPR optimizer iterations
%
%   out fields: fmu, fs2, ymu, ys2, mut_g, s2_g, sigma_n, LambdaTheta

x = x(:);
y = y(:);
xs = xs(:);
if nargin < 4 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'iter') || isempty(opts.iter)
    opts.iter = 40;
end

this_dir = fileparts(mfilename('fullpath'));
vhgpr_dir = fullfile(this_dir, 'vhgpr');
if ~exist(fullfile(vhgpr_dir, 'vhgpr_ui.m'), 'file')
    error('vhgpr_fit_predict:MissingPackage', ...
        'VHGPR package not found at %s', vhgpr_dir);
end

% Dummy test targets (vhgpr_ui uses them only for NMSE/NLPD reporting)
ys_dummy = zeros(size(xs));

old_path = path;
cleanup = onCleanup(@() path(old_path));
addpath(vhgpr_dir);

% Suppress verbose display from vhgpr_ui if possible
[~, ~, Ey, Vy, mutst, diagSigmatst, atst, diagCtst, LambdaTheta] = ...
    vhgpr_ui(x, y, xs, ys_dummy, opts.iter);

% Predictive observation noise: E[exp(g)] = exp(mu_g + s2_g/2)
noise_var_star = exp(mutst(:) + diagSigmatst(:) / 2);

out = struct();
out.fmu = atst(:);
out.fs2 = max(diagCtst(:), 0);
out.ymu = Ey(:);
out.ys2 = max(Vy(:), 0);
out.mut_g = mutst(:);
out.s2_g = max(diagSigmatst(:), 0);
out.sigma_n = sqrt(max(noise_var_star, 0));
out.noise_var = noise_var_star;
out.LambdaTheta = LambdaTheta;
end
