
import os.path
from tensorflow.python.platform.resource_loader import get_data_files_path
from tensorflow.python.framework.load_library import load_op_library

bs_module = load_op_library(os.path.join(get_data_files_path(), 'blocksparse_ops.so'))
