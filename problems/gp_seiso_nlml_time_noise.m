function [nlml, dnlml] = gp_seiso_nlml_time_noise(p, x, y, t_unique)
%GP_SEISO_NLML_TIME_NOISE NLML for SE-iso GP with one noise SD per unique time.
%   [nlml, dnlml] = gp_seiso_nlml_time_noise(p, x, y, t_unique)
%
%   p = [log(ell); log(sf); log(sigma_1); ...; log(sigma_Nu)]
%   Each unique time t_unique(i) shares sigma_i across all replicates at that time.
%   K_y = K_f + diag(noise_var). Gradients via central finite differences.

x = x(:);
y = y(:);
t_unique = t_unique(:);
nU = numel(t_unique);
if numel(p) ~= 2 + nU
    error('gp_seiso_nlml_time_noise:BadP', ...
        'p must have length 2+numel(t_unique)=%d, got %d.', 2 + nU, numel(p));
end

nlml = nlml_at(p, x, y, t_unique);
if nargout > 1
    dnlml = zeros(size(p));
    step = 1e-4;
    for i = 1:numel(p)
        pp = p;
        pp(i) = p(i) + step;
        fp = nlml_at(pp, x, y, t_unique);
        pp(i) = p(i) - step;
        fm = nlml_at(pp, x, y, t_unique);
        dnlml(i) = (fp - fm) / (2 * step);
    end
end
end

function nlml = nlml_at(p, x, y, t_unique)
hyp = struct('mean', [], 'cov', p(1:2), 'lik', []);
noise_var = expand_time_noise(x, t_unique, exp(2 * p(3:end)));
nlml = gp_seiso_hetero_noise('nlml', hyp, x, y, noise_var);
end

function noise_var = expand_time_noise(x, t_unique, var_unique)
x = x(:);
t_unique = t_unique(:);
var_unique = var_unique(:);
noise_var = zeros(size(x));
for i = 1:numel(t_unique)
    noise_var(abs(x - t_unique(i)) < 1e-12) = var_unique(i);
end
end
