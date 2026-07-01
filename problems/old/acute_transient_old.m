function problem = acute_transient_old(cfg)
%ACUTE_TRANSIENT Ground truth + sampled noisy training data.
% Replace model equation with your preferred acute transient form if needed.

%sample points with options for specific, irregular time points random
%irregular, or uniform time points
if isfield(cfg,'x_train') && ~isempty(cfg.x_train)
    x_train = sort(cfg.x_train(:));
else
    mode = 'uniform';
    if isfield(cfg,'sample_mode') && ~isempty(cfg.sample_mode)
        mode = cfg.sample_mode;
    end

    switch lower(mode)
        case 'uniform'
            x_train = linspace(cfg.xmin, cfg.xmax, cfg.n_train)';

        case 'random'
            x_train = sort(cfg.xmin + (cfg.xmax-cfg.xmin)*rand(cfg.n_train,1));

        otherwise
            error('Unknown sample_mode: %s', mode);
    end
end

x_grid  = linspace(cfg.xmin, cfg.xmax, cfg.n_grid)';

% Default acute transient parameters
p.b = 5;
p.A = 20;
p.c = 0.6;

% Ground-truth function
f = @(x) p.b + p.A *x.* exp(-p.c .* x);

y_true_grid  = f(x_grid);
y_true_train = f(x_train);

if ~isfield(cfg, 'noise_model') || isempty(cfg.noise_model)
    cfg.noise_model = 'additive';
end

switch lower(string(cfg.noise_model))
    case "additive"
        y_train = y_true_train + cfg.noise_std .* randn(size(y_true_train));
    case "proportional"
        rel = cfg.noise_std;  % interpret as fraction (e.g., 0.05 = 5%)
        sigma_i = rel .* max(abs(y_true_train), 1e-6);
        y_train = y_true_train + sigma_i .* randn(size(y_true_train));
    otherwise
        error('Unknown noise_model: %s. Use ''additive'' or ''proportional''.', cfg.noise_model);
end

problem.name = 'acute_transient';
problem.params = p;
problem.f_true = f;

problem.x_train = x_train;
problem.y_train = y_train;

problem.x_grid = x_grid;
problem.y_true_grid = y_true_grid;
end