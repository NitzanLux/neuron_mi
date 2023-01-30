from fractions import Fraction
from mpmath import mp
_DEFAULT_PRECISION=100
class FractionPrecition(Fraction):
    def __float__(self, precision=None):
        if precision is None:
            return super().__float__()
        else:
            mp.dps = precision
            return mp(self.numerator)/mp(self.denominator)
    @staticmethod
    def log(number,precision=_DEFAULT_PRECISION):
        mp.dps=precision
        return mp.log(number.numerator)-mp.log(number.denominator)
