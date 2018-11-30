import networkx as nx
import numpy as np


class SparsityMaskInitializer(object):
  """Sparsity mask initializer base class: all sparsity mask initializers inherit from this class.
  """

  def __call__(self, shape):
    raise NotImplementedError

  def get_config(self):
    """Returns the configuration of the initializer as a JSON-serializable dict.
    Returns:
      A JSON-serializable Python dict.
    """
    return {}


class BarabasiAlbert(SparsityMaskInitializer):
    """Initializes a Barabasi Albert graph and slices it to the required shape
    
    For details see:
        networkx.generators.random_graphs.barabasi_albert_graph in the networkx documentation

    Examples:
    
    ```python
        # Generate a Barabasi Albert graph with m = 1 and shape (5,10)
        ba = BarabasiAlbert(1)
        mask = ba((5,10))
    ```
    
    Arguments:
        m: Number of edges to attach from a new node to existing nodes
    """
    
    def __init__(self, m=5):
        self.m = m

    def __call__(self, shape):
        """
        Returns:
            2d numpy array containing the adjacency_matrix of the generated
            Barabasi Albert graph sliced to the requested shape
        """
        n = max(shape[0], shape[1])
        g = nx.generators.barabasi_albert_graph(n=n, m=self.m)
        a = nx.adjacency_matrix(g).toarray().astype(np.int32) + np.eye(n, dtype=np.int32)
        a[0:self.m,0:self.m] = 1        
        return a[:shape[0], :shape[1]]