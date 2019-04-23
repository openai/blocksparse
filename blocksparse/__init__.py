__version__ = '1.13.1_master'

from blocksparse.utils import (
    _op_module,
    entropy_size,
    get_entropy,
    set_entropy,
    reset_scalar_constants,
    scalar_constant,
    ceil_div,
    reduce_mul,
    bst_conv_layout,
    bst_deconv_layout,
)
dw_matmul_large_n = _op_module.dw_matmul_large_n

from blocksparse.conv import (
    ConvEdgeBias,
    conv_edge_bias_init,
    deconv_edge_bias_init,
    cwise_linear,
)

from blocksparse.embed import (
    embedding_lookup,
)

from blocksparse.ewops import (
    add,
    multiply,
    subtract,
    divide,
    maximum,
    minimum,
    negative,
    reciprocal,
    square,
    sqrt,
    exp,
    log,
    sigmoid,
    tanh,
    relu,
    elu,
    gelu,
    swish,
    fast_gelu,
    filter_tensor,
    filter_tensor_op,
    scale_tensor,
    float_cast,
    dropout,
    concrete_gate,
    concrete_gate_infer,
    add_n8,
    add_n,
    replace_add_n,
    restore_add_n,
    bias_relu,
    fancy_gather,
    reduce_max,
    assign_add,
)

from blocksparse.grads import (
    gradients,
    recomputable,
)

from blocksparse.lstm import (
    fused_lstm_gates,
    split4,
    concat4,
    sparse_relu,
    FusedBasicLSTMCell,
    grouped_lstm,
    group_lstm_grads,
)

from blocksparse.matmul import(
    BlocksparseMatMul,
    SparseProj,
    block_reduced_full_dw,
    group_param_grads,
    get_bsmm_dx_ops,
)

# from blocksparse.nccl import (
#     allreduce,
#     group_allreduce,
#     sync_variables_op,
#     sync_globals_zero_init_op,
#     serialize_nccl_ops,
#     reduce_scatter,
#     all_gather,
# )

from blocksparse.norms import (
    layer_norm,
    batch_norm,
)

from blocksparse.optimize import (
    Ema,
    AdamOptimizer,
    AdafactorOptimizer,
    blocksparse_l2_decay,
    blocksparse_norm,
    blocksparse_prune,
    clip_by_global_norm,
    global_norm,
    adafactor2d_op,
    adafactor1d_op,
    adam_op,
    blocksparse_adam_op,
)

from blocksparse.quantize import (
    QuantizeSpec,
    quantize,
    log_stats,
)

from blocksparse.transformer import (
    BlocksparseTransformer,
    softmax,
    masked_softmax,
    softmax_cross_entropy,
    transpose_2d,
    transpose_0213,
    top_k,
    rectified_top_k,
    clear_bst_constants,
)
