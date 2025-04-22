# Re-export useful public API
from .utils import set_seed
from .process import process_folds, data_prepare, data_prepare_cv
from .metrics import (
    get_metrics_multiclass_case,
    get_metrics_multiclass_case_cv,
    get_metrics_multiclass_case_test,
)
from .core import (
    make_bsp,
    build_qsvm_qc,
    sin_cos, 
    get_from_d1,
    get_from_d2,
    renew_operand,
    data_partition,
    data_to_operand,
    operand_to_amp,
    get_kernel_matrix
)
