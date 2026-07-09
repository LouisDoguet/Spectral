import jax
import jax.numpy as jnp
import equinox as eqx
import optax

CHECKS = {
    'OPT':{
        'SET':False, 
        'INIT':False
    }
}

class AlphaModel(eqx.Module):
    layer1 : eqx.nn.Conv1d
    layer2 : eqx.nn.Conv1d
    layer3 : eqx.nn.Conv1d

    def __init__(self, in_channels:int, width:int = 5, kernel_size:int = 3,key:int = 0):
        k1, k2, k3 = jax.random.split(key, num=3)
        self._checks = CHECKS
        self.opt = None
        self.layer1 = eqx.nn.Conv1d(in_channels, width, kernel_size=kernel_size, padding=1, key=k1)
        self.layer2 = eqx.nn.Conv1d(width, in_channels, kernel_size=kernel_size, padding=1, key=k2)
        self.layer3 = eqx.nn.Conv1d(in_channels,     1, kernel_size=kernel_size, padding=1, key=k3)

    def setOptimizer(self, optimizer=optax.adam, lr:float = 1e-3):
        self.opt = optimizer(lr)
        self._checks['OPT']['SET'] = True

    def __call__(self, x):
        x = jax.nn.relu(self.layer1(x))
        x = jax.nn.relu(self.layer2(x))
        return jax.nn.softplus(self.layer3(x))
    
    def train(self, epoch:int = 1000):
        if not self._checks['OPT']['SET']: raise ValueError("[OPTIMIZER] Use setOptimizer method before training")
        opt_state = self.opt.init(eqx.filter(self, eqx.is_array))