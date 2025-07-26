from rl_games_dnne.networks.tcnn_mlp import TcnnNetBuilder
from rl_games_dnne.algos_torch import model_builder

model_builder.register_network('tcnnnet', TcnnNetBuilder)