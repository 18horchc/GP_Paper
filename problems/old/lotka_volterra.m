function problem = lotka_volterra(cfg)
%LOTKA_VOLTERRA Generate prey/predator toy trajectories and noisy samples.
% Two-state truth generated from ODE:
%   dprey/dt = alpha*prey - beta*prey*predator
%   dpred/dt = delta*prey*predator - gamma*predator

% State-specific sample times
[x_train_prey, x_train_pred] = build_state_sample_times(cfg);

x_grid = linspace(cfg.xmin, cfg.xmax, cfg.n_grid)';

% Default Lotka-Volterra parameters (can be overridden from cfg.lv_params)
p.alpha = 1.1; %prey growth rate, higher means prey populates faster
p.beta = 0.4; %predation rate, how severely predator numbers reduce prey numbers
p.delta = 0.1; %predator reproduction, how prey pop size directly impacts pred pop growth
p.gamma = 0.4; %pred death, higher means pred die faster without food
p.prey0 = 10;
p.pred0 = 5;

if isfield(cfg, 'lv_params') && ~isempty(cfg.lv_params)
    p = merge_structs_local(p, cfg.lv_params);
end

odefun = @(t, z) [ ...
    p.alpha * z(1) - p.beta * z(1) * z(2); ...
    p.delta * z(1) * z(2) - p.gamma * z(2)];

z0 = [p.prey0; p.pred0];

% Dense truth on x_grid
[~, z_grid] = ode45(odefun, x_grid, z0);
y_true_grid = max(z_grid, 0); % keep nonnegative for toy biology setting

% Truth at sampled times (state-specific)
y_true_train_prey = interp1(x_grid, y_true_grid(:, 1), x_train_prey, 'pchip');
y_true_train_pred = interp1(x_grid, y_true_grid(:, 2), x_train_pred, 'pchip');
y_true_train_prey = max(y_true_train_prey, 0);
y_true_train_pred = max(y_true_train_pred, 0);

if ~isfield(cfg, 'noise_model') || isempty(cfg.noise_model)
    cfg.noise_model = 'additive';
end

switch lower(string(cfg.noise_model))
    case "additive"
        y_train_prey = y_true_train_prey + cfg.noise_std .* randn(size(y_true_train_prey));
        y_train_pred = y_true_train_pred + cfg.noise_std .* randn(size(y_true_train_pred));
    case "proportional"
        rel = cfg.noise_std;
        sigma_prey = rel .* max(abs(y_true_train_prey), 1e-6);
        sigma_pred = rel .* max(abs(y_true_train_pred), 1e-6);
        y_train_prey = y_true_train_prey + sigma_prey .* randn(size(y_true_train_prey));
        y_train_pred = y_true_train_pred + sigma_pred .* randn(size(y_true_train_pred));
    otherwise
        error('Unknown noise_model: %s. Use ''additive'' or ''proportional''.', cfg.noise_model);
end

problem.name = 'lotka_volterra';
problem.params = p;
problem.state_names = {'Prey', 'Predator'};

% Canonical asymmetric observation fields
problem.x_train_prey = x_train_prey;
problem.y_train_prey = y_train_prey;
problem.x_train_pred = x_train_pred;
problem.y_train_pred = y_train_pred;

problem.x_grid = x_grid;
problem.y_true_grid = y_true_grid;   % [n_grid x 2]

% Backward-compatible shared fields when times match exactly
if numel(x_train_prey) == numel(x_train_pred) && all(abs(x_train_prey - x_train_pred) < 1e-12)
    problem.x_train = x_train_prey;
    problem.y_train = [y_train_prey, y_train_pred];
else
    problem.x_train = [];
    problem.y_train = [];
end
end

function out = merge_structs_local(a, b)
out = a;
if isempty(b), return; end
f = fieldnames(b);
for i = 1:numel(f)
    k = f{i};
    out.(k) = b.(k);
end
end

function [x_prey, x_pred] = build_state_sample_times(cfg)
if isfield(cfg, 'x_train_prey') && ~isempty(cfg.x_train_prey)
    x_prey = sort(cfg.x_train_prey(:));
else
    mode_prey = get_state_mode(cfg, 'prey');
    n_prey = get_state_n(cfg, 'prey');
    x_prey = sample_times(mode_prey, n_prey, cfg.xmin, cfg.xmax);
end

if isfield(cfg, 'x_train_pred') && ~isempty(cfg.x_train_pred)
    x_pred = sort(cfg.x_train_pred(:));
else
    mode_pred = get_state_mode(cfg, 'pred');
    n_pred = get_state_n(cfg, 'pred');
    x_pred = sample_times(mode_pred, n_pred, cfg.xmin, cfg.xmax);
end
end

function mode = get_state_mode(cfg, which_state)
mode = 'uniform';
if isfield(cfg, 'sample_mode') && ~isempty(cfg.sample_mode)
    mode = cfg.sample_mode;
end

switch lower(which_state)
    case 'prey'
        if isfield(cfg, 'sample_mode_prey') && ~isempty(cfg.sample_mode_prey)
            mode = cfg.sample_mode_prey;
        end
    case 'pred'
        if isfield(cfg, 'sample_mode_pred') && ~isempty(cfg.sample_mode_pred)
            mode = cfg.sample_mode_pred;
        end
end
end

function n = get_state_n(cfg, which_state)
n = cfg.n_train;
switch lower(which_state)
    case 'prey'
        if isfield(cfg, 'n_train_prey') && ~isempty(cfg.n_train_prey)
            n = cfg.n_train_prey;
        end
    case 'pred'
        if isfield(cfg, 'n_train_pred') && ~isempty(cfg.n_train_pred)
            n = cfg.n_train_pred;
        end
end
n = max(1, round(n));
end

function x = sample_times(mode, n, xmin, xmax)
switch lower(string(mode))
    case "uniform"
        x = linspace(xmin, xmax, n)';
    case "random"
        x = sort(xmin + (xmax - xmin) * rand(n, 1));
    otherwise
        error('Unknown sample mode: %s', mode);
end
end
