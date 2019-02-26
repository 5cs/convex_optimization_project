m = 1000;
d = 4000;
valid = 200;
% A = randn(m, d);
% oA = orth(A.').';
% A = oA;
% A = qr(oA);
% nA = bsxfun(@rdivide, oA, sqrt(sum(oA.^2zeros(n-valid, 1), 2))); % redundant

X = (randn(d) + 1j*randn(d))/sqrt(2);
[Q,R] = qr(X);
R = diag(diag(R)./abs(diag(R)));
U = Q*R*Q;

x = randi([0, 1], valid, 1);
x(x == 0) = -1;
x = vertcat(x, zeros(d-valid, 1));
x = x(randperm(length(x)));


mu0 = 0.01;
m0 = 1000;
ro = valid/d;
delta0 = stat_dim(ro, mu0, x) * d
sigma = 0.01;

% constant smooth
mu0 = mu0/8;
for m = 1000:200:3800
    
    k = randperm(d);
    A = U(k(1:m),:);

    b = A * x;

    noise = sigma*randn(length(b), 1);
    b = b + noise;

    eps = sigma * sqrt(m - delta0);
    % calc_mu(3800, mu0, delta0, m0, ro, x, d)
    [tmp, k] = auslender_teboulle(A, b, eps, mu0);

%     tmp;
%     my_x = real(tmp);
%     my_x(my_x<-0.2) = -1;
%     my_x(my_x>0.2) = 1;
%     my_x(abs(my_x)<=0.2) = 0;
%     my_x;
% 
%     dis = norm(my_x - x)
%     em_err0 = norm(A*my_x - b)

    cost = k*m*4000
    square_prediction_error = norm(A*(tmp-x))^2 / m

end


% alpha = 0.9;
% mus = zeros(14, 1);
% mus(1) = mu0;
% i = 1;
% for m = 1200:200:3800
%     mus(i+1) = calc_mu(m, mus(i), delta0, m0, ro, x, d, alpha)
%     i = i+1;
% end

% mus = [0.1000
%     0.3300
%     0.5600
%     0.8100
%     1.0700
%     1.3500
%     1.6500
%     1.9800
%     2.3500
%     2.7700
%     3.2500
%     3.8200
%     4.5100
%     5.4000
%     6.6200];
% mus = mus/8;
% mus = [0.0100
%     0.2200
%     0.4300
%     0.6500
%     0.8800
%     1.1300
%     1.3900
%     1.6700
%     1.9700
%     2.3100
%     2.6900
%     3.1100
%     3.6100
%     4.2000
%     4.9300];
% mus = mus/8;

% mus = [0.0100
%     0.0800
%     0.1600
%     0.2300
%     0.3000
%     0.3600
%     0.4100
%     0.4600
%     0.5100
%     0.5500
%     0.6000
%     0.6300
%     0.6700
%     0.7100
%     0.7400];
% mus = mus/8;
% i = 1;
% % variable mu (constant risk or balanced scheme)
% for m = 1000:200:3800
%     
%     k = randperm(d);
%     A = U(k(1:m),:);
% 
%     b = A * x;
% 
%     noise = sigma*randn(length(b), 1);
%     b = b + noise;
% 
%     delta = stat_dim(ro, mus(i), x) * d;
%     eps = sigma * sqrt(m - delta);
% 
%     [tmp, k] = auslender_teboulle(A, b, eps, mus(i));
%     i = i+1;
% 
%     cost = k*m*4000
%     square_prediction_error = norm(A*(tmp-x))^2 / m
% 
% end


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
    
    z = z_d + ((b - A*x)/(theta*L_u));
    t = eps / (theta*L_u);
    if 1 - t/(norm(z)) > 0
        z_d = (1 - t/norm(z)) * z;
    else
        z_d = 0 * z;
    end
    
    z = (1 - theta) * z + theta * z_d;
    theta = 2/(1 + sqrt(1 + 4/(theta).^2));
    
    c = c+1;
    
    % error = norm(A * x - b);
    xx = abs(norm(A * x - b) - eps)/eps;
    if xx < 1e-2
        break
    end
end
ret = x;
k = c;
end


function ret = stat_dim(ro, u, x)
delta = 0.001;
c = 0;
tau = delta;
m = inf;
while c < 1000000
    r = val(ro, tau, u, x);
    tau = tau + delta;
    c = c+1;
    
    if m > r
        m = r;
    end
    
    if m < r % reach the minimu value, break the loop
        break
    end
end
ret = m;
end


function ret = val(ro, tau, u, x)
fun = @(z) (z-tau).^2.*exp(-z.^2/2);
ret = ro * (1 + tau^2 * (1 + u*norm(x, inf))^2);
ret = ret + (1-ro) * (2/pi)^(1/2) * integral(fun, tau, inf);
end


function ret = calc_mu(m, mu0, delta0, m0, ro, x, d, alpha)
mu = mu0;
step_size = 0.01;
mu = mu + step_size;

if alpha ~= 0
    m0 = m0 + (m-m0)^alpha;
end

while true
    delta = stat_dim(ro, mu, x)*d;
    if delta/m > delta0/m0
        break
    end
    mu = mu + step_size;
end
mu = mu - step_size;
ret = mu
end


function o = f_mu(x, u)
if ~isvector(x)
    error('Input must be vector');
end
o = norm(x, 1) + u/2 * norm(x)^2;
end

function ret = linear_regression(A, b)
[m, d] = size(A);
w = zeros(d, 1);
step_size = 0.001;
while norm(A'*(A*w-b)) > 0.0001
    w = w - step_size * (A'*(A*w-b));
    error = norm(A*w - b)
end
end

function ret = soft(x, tau)
ret = sign(x) .* max(abs(x)-tau, 0);
end


