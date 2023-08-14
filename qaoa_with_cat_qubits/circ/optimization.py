from scipy import optimize
import multiprocessing
import numpy

__all__ = ['multistart']

def minimize(args):
    # function to minimize
    fun, x0, bounds, arg = args
    #x0 = numpy.array([x, y])
    res = optimize.minimize(fun, x0, args=arg, bounds=bounds, method="L-BFGS-B", )
    return res

def multistart(fun, bounds=None, startpoints: int = 10, processes=4, args=None):
    # Generating uniformly distributed points on a sphere
    betas = numpy.pi * numpy.random.uniform(size=startpoints) / 2
    alphas = numpy.arccos(2 * numpy.random.uniform(size=startpoints) - 1)
    # starting points
    x0 = list(zip(alphas,betas))
    args = [fun, x0]
    print(__name__)
    #if __name__ == '__main__':
    with multiprocessing.Pool(processes) as pool:
        res = pool.map(minimize, args)
        return res
