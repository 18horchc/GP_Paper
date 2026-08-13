function noise_var_star = mlhgp_predict_noise(model, x, xs)
%MLHGP_PREDICT_NOISE Predict sigma_n^2(xs) from an mlhgp_seiso model.
%   noise_var_star = mlhgp_predict_noise(model, x_train, xs)

x = x(:);
xs = xs(:);
if isfield(model, 'meanfunc_g') && ~isempty(model.meanfunc_g)
    meanfunc = model.meanfunc_g;
else
    meanfunc = @meanConst;  % match current mlhgp_seiso default
end
covfunc  = @covSEiso;
likfunc  = @likGauss;
inffunc  = @infGaussLik;
[~, ~, gmu, ~] = gp(model.hyp_g, inffunc, meanfunc, covfunc, likfunc, ...
    x, model.z_train, xs);
noise_var_star = max(exp(gmu(:)), model.opts.eps_resid);
end
