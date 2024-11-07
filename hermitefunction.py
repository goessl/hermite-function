import numpy as np
from numpy.polynomial.hermite import hermval, hermfit
from scipy.special import factorial, erf
from functools import cached_property
from operator import mul
from vector import Vector, vecsub



class HermiteFunction(Vector):
    """A Hermite function series class."""
    
    #creation stuff
    def __init__(self, coef, herm_coef=None):
        super().__init__(coef)
        #coefficients of the Hermite polynomial part of this series
        if herm_coef is not None:
            self.herm_coef = herm_coef
        else:
            if self:
                self.herm_coef = self.coef / HermiteFunction._factors(len(self))
            else:
                self.herm_coef = np.zeros(1, dtype=int)
    
    @staticmethod
    def rand(deg):
        """Return a random Hermite function series.
        
        See `Vector.rand`.
        """
        return HermiteFunction(Vector.rand(deg+1))
    
    @staticmethod
    def randn(deg, normed=True):
        """Return a random Hermite function series.
        
        See `Vector.randn`.
        """
        return HermiteFunction(Vector.randn(deg+1, normed))
    
    @staticmethod
    def _factors(n):
        i = np.arange(n)
        return np.sqrt(2**i * factorial(i) * np.sqrt(np.pi))
    
    @staticmethod
    def fit(X, y, deg):
        """Return a Hermite function series fitted to the given data."""
        #https://de.wikipedia.org/wiki/Multiple_lineare_Regression
        herm_coef = hermfit(X, y/np.exp(-X**2/2), deg)
        return HermiteFunction(herm_coef * HermiteFunction._factors(deg+1),
                herm_coef)
    
    
    #function stuff
    @property
    def deg(self):
        """Return the degree of this Hermite function series."""
        return len(self) - 1
    
    def __call__(self, x):
        return np.exp(-x**2/2) * hermval(x, self.herm_coef)
    
    
    #calculus stuff
    def der(self, n=1):
        """Return the n-th derivative of this Hermite function series."""
        k = np.arange(len(self)+n)
        for _ in range(n):
            self = HermiteFunction(vecsub(map(mul, self<<1, np.sqrt((k+1)/2)),
                                          map(mul, self>>1, np.sqrt(k/2))))
        return self
    
    def antider(self):
        """Return the antiderivative and residual factor.
        
        Returns F, r so that the antiderivative is of form
        `F(x) + r*HermiteFunction.zeroth_antiderivative(x)`
        where F is also a Hermite function series.
        """
        tmp = list(self)
        F = (len(self)-1) * [0]
        for i in reversed(range(1, len(self))):
            F[i-1] -= tmp[i] * np.sqrt(2/i)
            tmp[i-2] += tmp[i] * np.sqrt((i-1)/i)
        return HermiteFunction(F), tmp[0]
    
    @staticmethod
    def zeroth_antiderivative(x):
        """Evaluation of the antiderivative of the zeroth Hermite function."""
        return np.sqrt(np.sqrt(np.pi)/2) * (erf(x/np.sqrt(2)) + 1)
    
    
    #additional stuff
    def fourier(self):
        """Return the Fourier transform of this Hermite function series."""
        return HermiteFunction((-1j)**n * c for n, c in enumerate(self))
    
    @cached_property
    def kin(self):
        """Return the kinetic energy of this Hermite function series."""
        #return -(self @ self.der(2)) / 2
        return self.der().absq() / 2
    
    
    #python stuff
    def __str__(self):
        s = f'{self[0]:.1f} h_0'
        for i, c in enumerate(self[1:]):
            s += f' + {c:.1f} h_{i+1}'
        return s



if __name__ == '__main__':
    from scipy.integrate import cumulative_trapezoid
    
    x = np.linspace(-4, +4, 1000)
    
    
    #HermiteFunction
    #HermiteFunction.rand
    
    #fitting
    for _ in range(100):
        deg = np.random.randint(0, 20)
        f = HermiteFunction.rand(deg)
        fit = HermiteFunction.fit(x, f(x), deg)
        assert np.allclose(f.coef, fit.coef)
    
    
    
    f = HermiteFunction.randn(20)
    #length
    assert len(f) == 21
    #indexing
    f[5]
    assert f[999] == 0
    #iterating
    for c in f:
        pass
    #comparison
    assert f != HermiteFunction(21)
    #shifting
    assert (f<<1).deg == 19 and (f>>1).deg == 21
    
    
    
    #norm
    assert np.isclose(abs(f), 1)
    #dot
    assert np.isclose(f @ HermiteFunction(21), 0)
    
    
    
    #addition & subtraction
    for _ in range(100):
        f = HermiteFunction.rand(np.random.randint(0, 20))
        g = HermiteFunction.rand(np.random.randint(0, 20))
        assert np.allclose((f+g)(x), f(x)+g(x))
        assert np.allclose((f-g)(x), f(x)-g(x))
    
    #scalar multiplication and division
    for _ in range(100):
        f = HermiteFunction.rand(np.random.randint(0, 20))
        c = np.random.rand()
        assert np.allclose((c*f)(x), c*(f(x)))
        assert np.allclose((f/c)(x), (f(x))/c)
    
    
    
    f = HermiteFunction.rand(20)
    #degree
    assert f.deg == 20
    #calling
    f(x)
    
    #derivative
    def der_num(x, y, n=1):
        """Nummerical differentiation."""
        for _ in range(n):
            y = np.diff(y) / np.diff(x)
            x = (x[1:] + x[:-1]) / 2
        return x, y
    
    for _ in range(100):
        f = HermiteFunction.rand(np.random.randint(0, 20))
        
        assert np.allclose(f.der()(der_num(x, f(x))[0]),
                der_num(x, f(x))[1], atol=1e-3)
    
    #antiderivative
    for _ in range(100):
        f = HermiteFunction.rand(np.random.randint(0, 5))
        F, r = f.antider()
        assert np.allclose(F(x) + r * HermiteFunction.zeroth_antiderivative(x),
                cumulative_trapezoid(f(x), x, initial=0), atol=1e-1)
    
    #fourier
    def fourier(y, x):
        #https://stackoverflow.com/a/24077914
        dx = (max(x)-min(x)) / len(x)
        w = np.fft.fftshift(np.fft.fftfreq(len(x), dx)) * 2*np.pi
        g = np.fft.fftshift(np.fft.fft(y))
        g *= dx * np.exp(-complex(0,1)*w*min(x)) / np.sqrt(2*np.pi)
        return w, g
    
    for _ in range(100):
        f = HermiteFunction.rand(np.random.randint(0, 5))
        Fx, Ff = fourier(f(x), x)
        assert np.allclose(f.fourier()(Fx), Ff, atol=1e-1)
    
    #kinetic energy
    def kin_num(x, y):
        """Nummeric kinetic energy."""
        x, y_lapl = der_num(x, y, 2)
        y = (y[2:] + 2*y[1:-1] + y[:-2]) / 4 #mid y twice to broadcast to y_lapl
        return -np.trapezoid(y*y_lapl, x) / 2
    
    for _ in range(100):
        f = HermiteFunction.randn(5)
        assert np.isclose(f.kin, kin_num(x, f(x)), atol=1e-2)
    
    
    
    #python stuff
    str(f)
