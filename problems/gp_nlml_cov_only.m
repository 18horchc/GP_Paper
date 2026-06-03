function [nlml, dnlml] = gp_nlml_cov_only(hyp_cov, sn_fixed, varargin)
%GP_NLML_COV_ONLY GPML NLML with fixed likGauss noise; optimize ell, sf only.
%   [nlml, dnlml] = gp_nlml_cov_only(hyp_cov, log(sn), inf, mean, cov, lik, x, y)
%
%   sn_fixed is log(sigma_n). hyp_cov is [log(ell); log(sf)]. hyp.lik is not optimized.

hyp = struct('mean', [], 'cov', hyp_cov(:), 'lik', sn_fixed);
[nlml, dnlml_s] = gp(hyp, varargin{:});
if nargout > 1
    dnlml = dnlml_s.cov;
end
end
