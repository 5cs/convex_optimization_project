Q = [1 -1/2; -1/2 2];
f = [-1, 0]';
A = [1 2; 1 -4; 5 76];
b = [-2 -3 1]';

cvx_begin
    variable x(2)
    dual variable lambda
    minimize(quad_form(x, Q) + f'*x)
    subject to
        lambda: A*x <= b
cvx_end

p_star = cvx_optval