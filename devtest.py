import jax
from dataclasses import dataclass

@dataclass
class testclass:
    boop: int
    bop: int


badpytree = [testclass(1, 2), 1]

def inspector(leaf):
    print("stopped")
    print(leaf)

jax.tree_util.tree_map(inspector, badpytree)