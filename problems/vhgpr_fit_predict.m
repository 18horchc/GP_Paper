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
%   opts.iter          (default 40)  VHGPR optimizer iterations
%   opts.warm          prior out struct (LambdaTheta, loghyperGP) to skip vhgpr_ui
%   opts.X_c, opts.k   if both set, fmincon Pensoneault lower bound on latent f:
%                      mu_f(x_c) - k*sigma_f(x_c) >= 0
%   opts.opts_fmincon  fmincon options for the bound (optional)
%
%   out fields: fmu, fs2, ymu, ys2, mut_g, s2_g, sigma_n, LambdaTheta,
%               loghyperGP; bound runs also set exitflag, max_c, mv_bound

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

use_warm = isfield(opts, 'warm') && isstruct(opts.warm) ...
    && isfield(opts.warm, 'LambdaTheta') && isfield(opts.warm, 'loghyperGP') ...
    && ~isempty(opts.warm.LambdaTheta) && ~isempty(opts.warm.loghyperGP);
do_bound = isfield(opts, 'X_c') && ~isempty(opts.X_c) ...
    && isfield(opts, 'k') && ~isempty(opts.k);

old_path = path;
cleanup = onCleanup(@() path(old_path));
addpath(vhgpr_dir);

covfuncSignal = {'covSum', {'covSEisoj', 'covConst'}};
covfuncNoise  = {'covSum', {'covSEisoj', 'covNoise'}};

if use_warm
    LambdaTheta = opts.warm.LambdaTheta(:);
    loghyperGP = opts.warm.loghyperGP(:);
    Ey = [];
    Vy = [];
    mutst = [];
    diagSigmatst = [];
    atst = [];
    diagCtst = [];
else
    ys_dummy = zeros(size(xs));
    [~, ~, Ey, Vy, mutst, diagSigmatst, atst, diagCtst, LambdaTheta, loghyperGP] = ...
        vhgpr_ui(x, y, xs, ys_dummy, opts.iter);
    LambdaTheta = LambdaTheta(:);
    loghyperGP = loghyperGP(:);
end

D = size(x, 2);
lengthscales = loghyperGP(1:D);
scale_row = exp(lengthscales(:)');
x_s = x ./ (ones(size(x, 1), 1) * scale_row);
xs_s = xs ./ (ones(size(xs, 1), 1) * scale_row);

out_extra = struct('exitflag', [], 'max_c', [], 'mv_bound', []);
if do_bound
    X_c = opts.X_c;
    if size(X_c, 2) ~= D
        X_c = X_c(:);
        if D ~= 1
            error('vhgpr_fit_predict:BadXc', ...
                'X_c must have %d columns to match x.', D);
        end
    end
    Xc_s = X_c ./ (ones(size(X_c, 1), 1) * scale_row);
    k_pens = opts.k;
    if isfield(opts, 'opts_fmincon') && ~isempty(opts.opts_fmincon)
        opts_fmincon = opts.opts_fmincon;
    else
        opts_fmincon = optimoptions('fmincon', 'Algorithm', 'interior-point', ...
            'EnableFeasibilityMode', true, 'Display', 'off', ...
            'SpecifyObjectiveGradient', true, ...
            'ConstraintTolerance', 1e-4, 'OptimalityTolerance', 1e-4, ...
            'MaxFunctionEvaluations', 10000, 'MaxIterations', 2000);
    end
    objfun = @(lt) vhgpr_obj(lt, covfuncSignal, covfuncNoise, x_s, y);
    nonlcon = @(lt) vhgpr_pens_con(lt, covfuncSignal, covfuncNoise, x_s, y, Xc_s, k_pens);
    try
        [lt_opt, fval, ef] = fmincon(objfun, LambdaTheta, [], [], [], [], [], [], ...
            nonlcon, opts_fmincon);
    catch me
        fprintf('  Warning: VHGPR bound fmincon error (%s); using unconstrained LambdaTheta.\n', ...
            me.message);
        lt_opt = [];
        fval = inf;
        ef = -99;
    end
    if isempty(lt_opt) || any(~isfinite(lt_opt(:)))
        fprintf('  Warning: VHGPR bound fmincon failed; using unconstrained LambdaTheta.\n');
        lt_opt = LambdaTheta;
        fval = objfun(LambdaTheta);
        ef = -99;
    end
    LambdaTheta = lt_opt(:);
    [c_final, ~] = nonlcon(LambdaTheta);
    out_extra.exitflag = ef;
    out_extra.max_c = max(c_final);
    out_extra.mv_bound = fval;
end

need_pred = use_warm || do_bound || isempty(atst);
if need_pred
    [Ey, Vy, mutst, diagSigmatst, atst, diagCtst] = ...
        vhgpr(LambdaTheta, covfuncSignal, covfuncNoise, 0, x_s, y, xs_s);
end

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
out.loghyperGP = loghyperGP;
out.exitflag = out_extra.exitflag;
out.max_c = out_extra.max_c;
out.mv_bound = out_extra.mv_bound;
end

function [f, g] = vhgpr_obj(lt, cov1, cov2, X, y)
lt = lt(:);
try
    if nargout > 1
        [f, g] = vhgpr(lt, cov1, cov2, 0, X, y);
        g = g(:);
    else
        f = vhgpr(lt, cov1, cov2, 0, X, y);
    end
    if ~isfinite(f)
        f = 1e12;
        if nargout > 1
            g = zeros(size(lt));
        end
    end
catch
    f = 1e12;
    if nargout > 1
        g = zeros(size(lt));
    end
end
end

function [c, ceq] = vhgpr_pens_con(lt, cov1, cov2, X, y, Xc, k)
ceq = [];
nC = size(Xc, 1);
try
    [~, ~, ~, ~, atst, diagCtst] = vhgpr(lt(:), cov1, cov2, 0, X, y, Xc);
    c = k .* sqrt(max(diagCtst(:), 0)) - atst(:);
    if numel(c) ~= nC || any(~isfinite(c))
        c = 1e6 * ones(nC, 1);
    end
catch
    c = 1e6 * ones(nC, 1);
end
end
