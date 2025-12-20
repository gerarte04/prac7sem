$$
c(y_i^n) \frac{y_i^{n+1} - y_i^n}{\tau} = \frac{1}{h^2} \left[ a_{i+1}(y^n) (y_{i+1}^n - y_i^n) - a_i(y^n) (y_i^n - y_{i-1}^n) \right] + f(y_i^n)
$$

$$
c(y_i^j) \frac{y_i^{j+1} - y_i^j}{\tau} = \frac{1}{h^2} \left[ a_{i+1}(y^j) (y_{i+1}^{j+1} - y_i^{j+1}) - a_i(y^j) (y_i^{j+1} - y_{i-1}^{j+1}) \right] + f(y_i^j)
$$

$$a_i(y^j) = k\left(\frac{y_{i-1}^j + y_i^j}{2}\right)$$

$$-sigma*a_i*y_{i-1}^{j+1} + (c_i + sigma*(a_{i+1} + a_i))*y_i^{j+1} - sigma*a_{i+1}*y_{i+1}^{j+1} = ...
$$