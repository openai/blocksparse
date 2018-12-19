from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
import networkx as nx
import numpy as np
import six


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
    
    def get_config(self):
        return {'m': self.m}

# Compatibility aliases
barabasi_albert = BarabasiAlbert

# Utility functions
def serialize(initializer):
    return serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='initializer')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret initializer identifier: ' +
                         str(identifier))
