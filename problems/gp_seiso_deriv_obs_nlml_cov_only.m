function [nlml, dnlml] = gp_seiso_deriv_obs_nlml_cov_only(hyp_cov, sn_fixed, x, y, x_d, y_d, sn_deriv, noise_var)
%GP_SEISO_DERIV_OBS_NLML_COV_ONLY Solak NLML with fixed function-obs noise; optimize ell, sf.
%   sn_fixed is log(sigma_n) for training y; sn_deriv is fixed derivative-obs noise (linear).
%   Optional noise_var (n_x x 1 variances) overrides homoscedastic function-obs noise.

if nargin < 8
    noise_var = [];
end

hyp = struct('mean', [], 'cov', hyp_cov(:), 'lik', sn_fixed);
[nlml, dnlml_s] = gp_seiso_deriv_obs('nlml', hyp, x, y, x_d, y_d, [], sn_deriv, true, noise_var);
if nargout > 1
    dnlml = dnlml_s.cov;
end
end
