m = 100;
d = 400;
valid = 20;

X = (randn(d) + 1j*randn(d))/sqrt(2);
[Q,R] = qr(X);
R = diag(diag(R)./abs(diag(R)));
U = Q*R*Q;

% A = randn(d, d);
% oA = orth(A.').';
% U = oA;

x = randi([0, 1], valid, 1);
x(x == 0) = -1;
x = vertcat(x, zeros(d-valid, 1));
x = x(randperm(length(x)));


mu0 = 0.1;
m0 = 100;
ro = valid/d;
delta0 = stat_dim(ro, mu0, x) * d
sigma = 0.5;
scale = 1;


% [costs0, risks0] = constant_smooth(mu0/scale, delta0, sigma, d, U, x)


num = 10;
ms = 100:20:280;
mus = zeros(num, 1);

% constant risk, alpha = 0.8
mus = calc_mu(ms, mu0, delta0, ro, x, d, 0.8, num)
% mus = [0.1000, 0.3300, 0.5600, 0.8100, 1.0700, 1.3500, 1.6500, 1.9800, 2.3500, 2.7700];
mus = mus/scale;
[costs1, risks1, iters1] = constant_risk(mus, ro, sigma, d, U, x)
mus1 = mus;

% balanced scheme, alpha = 0.9
mus = calc_mu(ms, mu0, delta0, ro, x, d, 0.9, num)
% mus = [0.1000, 0.1800, 0.2600, 0.3400, 0.4200, 0.4800, 0.5400, 0.6000, 0.65, 0.7]
mus = mus/scale;
[costs2, risks2, iters2] = constant_risk(mus, ro, sigma, d, U, x)
mus2 = mus;


% balanced scheme, alpha = 1, constant smooth
mus = calc_mu(ms, mu0, delta0, ro, x, d, 1, num)
% mus = [0.1000, 0.1800, 0.2600, 0.3400, 0.4200, 0.4800, 0.5400, 0.6000, 0.65, 0.7]
mus = mus/scale;
[costs3, risks3, iters3] = constant_risk(mus, ro, sigma, d, U, x)
mus3 = mus;


% iters1
% iters2
% iters3
% 
% iters1 .* mus1
% iters2 .* mus2
% iters3 .* mus3

figure
ax1 = subplot(1, 2, 1);
% plot(ax1, ms, costs0, ms, costs1, ms, costs2);
plot(ax1, ms, costs1, ms, costs2, ms, costs3);
title(ax1, 'Cost vs. sample size (1-norm)');
ylabel(ax1, 'Cost');

ax2 = subplot(1, 2, 2);
% plot(ax2, ms, risks0, ms, risks1, ms, risks2);
plot(ax2, ms, risks1, ms, risks2, ms, risks3);
title(ax2,'Risk vs. sample size (1-norm)')
ylabel(ax2,'Risk')

% h = legend('Constant smooth', 'Constant risk', 'Alpha 0.9');
h = legend('Alpha 0.8', 'Alpha 0.9', 'Alpha 1.0');
% set(h, 'FontSize', 15);


% constant smooth
function [costs, risks] = constant_smooth(mu0, delta0, sigma, d, U, x)
costs = zeros(1, 10);
risks = zeros(1, 10);

N = 50;
i = 1;
for m = 100:20:280
    
    for round = 1:N
        k = randperm(d);
        A = U(k(1:m),:);
        b = A * x;
        noise = sigma*randn(length(b), 1);
        b = b + noise;
        eps = sigma * sqrt(m - delta0);
        
        [tmp, k] = auslender_teboulle(A, b, eps, mu0);
        
        costs(i) = costs(i) + k*m*d;
        risks(i) = risks(i) + norm(A*(tmp-x))^2 / m; 
    end
    i = i+1;
end
costs = costs/N;
risks = risks/N;
end

% constant risk
function [costs, risks, iters] = constant_risk(mus, ro, sigma, d, U, x)
costs = zeros(1, 10);
risks = zeros(1, 10);
iters = zeros(1, 10);

N = 50;
i = 1;
for m = 100:20:280
    for round = 1:N
        k = randperm(d);
        A = U(k(1:m),:);
        b = A * x;
        noise = sigma*randn(length(b), 1);
        b = b + noise;
        
        delta = stat_dim(ro, mus(i), x) * d;
        eps = sigma * sqrt(m - delta);
        
        [tmp, k] = auslender_teboulle(A, b, eps, mus(i));
        
        costs(i) = costs(i) + k*m*d;
        risks(i) = risks(i) + norm(A*(tmp-x))^2 / m;
        iters(i) = iters(i) + k;
    end
    i = i+1;
end
costs = costs/N;
risks = risks/N;
iters = iters/N;
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
    x = u * wthresh(A' * y, 's', 1);
    % x = soft(A'*y, 1);
    
    Z = z_d + ((b - A*x)/(theta*L_u));
    T = eps / (theta*L_u);
    z_d = max(1-T/norm(Z), 0) * Z;
    
    z = (1 - theta) * z + theta * z_d;
    theta = 2/(1 + sqrt(1 + 4/(theta).^2));
    
    c = c+1;
    
    % error = norm(A * x - b);
    xx = abs(norm(A * x - b) - eps)/eps;
    if xx < 1e-3
        break
    end
end
ret = x;
k = c;
end


function ret = stat_dim(ro, u, x)
f = @(tau) (ro * (1 + tau^2 * (1 + u*norm(x, inf))^2) + (1-ro) * (2/pi)^(1/2) * integral(@(z) (z-tau).^2.*exp(-z.^2/2), tau, inf));
[tau, fval] = fminbnd(f, 0, 100);
ret = fval;
end


function [mus] = calc_mu(ms, mu0, delta0, ro, x, d, alpha, num)
mus(1) = mu0;
mu = mu0;
step_size = 0.01;
for i = 2:num
    mu = mu + step_size;

    if alpha ~= 0
        m0 = ms(1) + (ms(i)-ms(1))^alpha;
    else
        m0 = ms(1);
    end

    while true
        delta = stat_dim(ro, mu, x)*d;
        if delta/ms(i) > delta0/m0
            break
        end
        mu = mu + step_size;
    end
    mu = mu - step_size;
    mus(i) = mu;
end
end

