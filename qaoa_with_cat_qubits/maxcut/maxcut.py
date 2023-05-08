import numpy
import networkx

__all__ = ['MaxCut']

class MaxCut(object):
    def __init__(self,graph) -> None:
        self.graph = graph
        self.nodes = graph.nodes
        self.edges = graph.edges
        self.num_nodes = len(graph.nodes)
        self.num_edges = len(graph.edges)
        self.cost = self.get_cost()

    def get_cost(self) -> numpy.ndarray:
        """
        Returns the MaxCut cost values of a graph

        Args:
            graph (networkx.Graph): graph

        Returns:
            numpy.ndarray: The cost values as an 1D-array
        """
        def product(*args, repeat=1):
            # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
            # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
            pools = [list(pool) for pool in args] * repeat
            result = [[]]
            for pool in pools:
                result = [x + [y] for x in result for y in pool]
            for prod in result:
                yield list(prod)

        # Number of edges
        M = self.num_edges
        # Number of nodes
        N = self.num_nodes
        # Adjacency matrix
        A = networkx.adjacency_matrix(self.graph).todense()

        # Generate a list of all possible n‐tuples of elements from {1,-1} and
        # organize them as a (2^n x n) matrix. In other words create all
        # possible solutions to the problem.
        s = numpy.array(list(product([1, -1], repeat=N)))

        # Construct the the cost function for Max Cut: C=1/2*Sum(Z_i*Z_j)-M/2
        # Note: This is the minimization version
        return 1 / 2 * (numpy.diag(s@numpy.triu(A)@s.T) - M)
