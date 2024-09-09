from itertools import product

# combinations_long = {
#     "S": [512, 2048, 4096, 8192],
#     "B": [1, 1, 1, 1],
#     "NH": [1, 1, 1, 1],
#     "DHQK": [128, 128, 128, 128],
#     "DHHV": [128, 128, 128, 128],
# }
# combinations_long = {
#     "S": [128, 1024, 4096, 8192],
#     "B":  [1, 1, 1, 1],   # [2, 2, 2, 2],
#     "NH": [1, 1, 1, 1],  # [3, 3, 3, 3],
#     "DHQK": [16,16,16,16], #[5, 5, 5, 5],
#     "DHHV": [16,16,16,16], #[5, 5, 5, 5],
# }
combinations_long = {
    "S": [4096],  # [8192],
    "B": [1],  # [2, 2, 2, 2],
    "NH": [1],  # [3, 3, 3, 3],
    "DHQK": [16],  # [5, 5, 5, 5],
    "DHHV": [16],  # [5, 5, 5, 5],
}
combinations_long_list = [values for values in zip(*combinations_long.values())]
target_dtypes = ["float16", "bfloat16", "float32", "float64"]

final_combinations = [
    (*combinations, dtype)
    for combinations, dtype in product(combinations_long_list, target_dtypes)
]
