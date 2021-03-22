
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

__config_estnet = dotdict({"branch_a_layer_1_size": 240,
        "branch_a_filter_1_size": (1, 1),
        "branch_a_max_pool": (8, 8),
        "branch_b_layer_1_size": 128,
        "branch_b_filter_1_size": (8, 8),
        "branch_b_layer_2_size": 256,
        "branch_b_filter_2_size": (4, 4),
        "combined_dense_1": 256,
        "combined_dense_2": 3,
        "add_histogram": False,
        "histogram_depth": 32})