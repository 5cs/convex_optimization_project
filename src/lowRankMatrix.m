m = 100;
d = 400;
d1 = 20;
d2 = 20;
valid = 20;

X = (randn(d) + 1j*randn(d))/sqrt(2);
[Q,R] = qr(X);
R = diag(diag(R)./abs(diag(R)));
U = Q*R*Q;

% x = randi([0, 1], valid, 1);
% x(x == 0) = -1;
% x = vertcat(x, zeros(d-valid, 1));
% x = x(randperm(length(x)));
Q1 = orth(randn(20, 1));
Q2 = orth(randn(20, 1));
x = Q1 * Q2';

rank(x)


mu0 = 0.1;
m0 = 160;
% ro = valid/d;
ro = 0.05;
delta0 = stat_dim(ro, mu0, x) * d
sigma = 0.1;
scale = 1;


[costs0, risks0] = constant_smooth(mu0/scale, delta0, sigma, d, U, x)


num = 13;
ms = 160:10:280;
mus = zeros(num, 1);

% constant risk, alpha = 0
mus = calc_mu(ms, mu0, delta0, ro, x, d, 0, num)
% mus = [0.1000, 0.3300, 0.5600, 0.8100, 1.0700, 1.3500, 1.6500, 1.9800, 2.3500, 2.7700];
mus = mus/scale;
[costs1, risks1] = constant_risk(mus, ro, sigma, d, U, x)

% balanced scheme, alpha = 0.9
mus = calc_mu(ms, mu0, delta0, ro, x, d, 0.9, num)
% mus = [0.1000, 0.1800, 0.2600, 0.3400, 0.4200, 0.4800, 0.5400, 0.6000, 0.65, 0.7]
mus = mus/scale;
[costs2, risks2] = constant_risk(mus, ro, sigma, d, U, x)


figure
ax1 = subplot(1, 2, 1);
plot(ax1, ms, costs0, '-ro', ms, costs1, '-md', ms, costs2, '-bx', 'LineWidth', 1.5);
title(ax1, 'Cost vs. sample size (1-norm)');
xlabel(ax1, 'Sample size (m)');
ylabel(ax1, 'Cost');

ax2 = subplot(1, 2, 2);
plot(ax2, ms, risks0, '-ro', ms, risks1, '-md', ms, risks2, '-bx', 'LineWidth', 1.5);
title(ax2,'Risk vs. sample size (1-norm)');
xlabel(ax2, 'Sample size (m)');
ylabel(ax2, 'Risk');

h = legend('Constant smooth', 'Constant risk', 'Alpha 0.9');
set(h, 'FontSize', 15);


% constant smooth
function [costs, risks] = constant_smooth(mu0, delta0, sigma, d, U, x)
costs = zeros(1, 13);
risks = zeros(1, 13);

N  = 200;
i = 1;
for m = 160:10:280
    for round = 1:N
        k = randperm(d);
        A = U(k(1:m),:);
        b = A * reshape(x, [], 1);

        noise = sigma*randn(length(b), 1);
        b = b + noise;
        eps = sigma * sqrt(m - delta0);
        
        [tmp, k] = auslender_teboulle(A, b, eps, mu0);
        tmp = reshape(tmp, [], 1);
        
        costs(i) = costs(i) + k*m*d;
        risks(i) = risks(i) + norm(A*(tmp-reshape(x, [], 1)))^2 / m;
    end
    i = i+1;
end
costs = costs/N;
risks = risks/N;
end

% constant risk
function [costs, risks] = constant_risk(mus, ro, sigma, d, U, x)
costs = zeros(1, 13);
risks = zeros(1, 13);

N = 200;
i = 1;
for m = 160:10:280
    for round = 1:N
%         A = randn(m, d);
%         A = orth(A.').';
        k = randperm(d);
        A = U(k(1:m),:);
        b = A * reshape(x, [], 1);

        noise = sigma*randn(length(b), 1);
        b = b + noise;

        delta = stat_dim(ro, mus(i), reshape(x, [], 1)) * d;
        eps = sigma * sqrt(m - delta);

        [tmp, k] = auslender_teboulle(A, b, eps, mus(i));
        tmp = reshape(tmp, [], 1);

        costs(i) = costs(i) + k*m*d;
        risks(i) = risks(i) + norm(A*(tmp-reshape(x, [], 1)))^2 / m;
    end
    i = i+1;
end
costs = costs/N;
risks = risks/N;
end


function [ret, k] = auslender_teboulle(A, b, eps, u)
L_u = norm(A)^2 / u;
z = zeros(length(b), 1);
z_d = z;
theta = 1;

x = 0;
c = 0;
while true
    y = (1 - theta) * z + theta * z_d;
    
    % [U, S, V] = svd(vec2mat(A'*y, 20));
    [U, S, V] = svd(A'*y);
    
    x = u * (U * diag(wthresh(diag(S), 's', 1)) * V');
    
    Z = z_d + ((b - A*reshape(x, [], 1))/(theta*L_u));
    T = eps / (theta*L_u);
    z_d = max(1-T/norm(Z), 0) * Z;
    
    z = (1 - theta) * z + theta * z_d;
    theta = 2/(1 + sqrt(1 + 4/(theta).^2));
    
    c = c+1;
    
    % error = norm(A * reshape(x, [], 1) - b)
    xx = abs(norm(A * reshape(x, [], 1) - b) - eps)/eps;
    if xx < 1e-3
        break
    end
end
ret = x;
k = c;
end


function ret = stat_dim(ro, u, x)
f = @(tau) (ro + (1-ro)*(ro * (1 + tau^2 * (1 + u * norm(x))^2) + (1-ro)/(12*pi) * (24 * (1 + tau^2) * acos(tau/2) - tau * (26 + tau^2) * sqrt(4-tau^2))));
[tau, fval] = fminbnd(f, 0, 2);
ret = fval;
end


function [mus] = calc_mu(ms, mu0, delta0, ro, x, d, alpha, num)
mus(1) = mu0;
mu = mu0;
step_size = 0.01;
c = 1;
for i = 2:num
    mu = mu + step_size;

    if alpha ~= 0
        m0 = ms(1) + (ms(i)-ms(1))^alpha;
    else
        m0 = ms(1);
    end

    c
    while true
        delta = stat_dim(ro, mu, x)*d;
        if delta/ms(i) > delta0/m0
            break
        end
        mu = mu + step_size;
    end
    c = c+1;
    mu = mu - step_size;
    mus(i) = mu;
end
end

