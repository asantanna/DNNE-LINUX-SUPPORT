

from rl_games_dnne.envs.connect4_network import ConnectBuilder
from rl_games_dnne.envs.test_network import TestNetBuilder
from rl_games_dnne.algos_torch import model_builder

model_builder.register_network('connect4net', ConnectBuilder)
model_builder.register_network('testnet', TestNetBuilder)