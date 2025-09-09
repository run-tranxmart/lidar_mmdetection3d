_base_ = [
    '../models/pointpillars_hv_secfpn_heavy_5class.py',
    '../datasets/batch10_aicc_5class.py',
    '../schedules/cyclic-40e.py',
    '../../../_base_/default_runtime.py'
]


find_unused_parameters = True
