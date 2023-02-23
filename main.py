import jax
import jax.numpy as jnp

def func(x):
    return jnp.sin(x)

# won't run on OSX Apple Silicon
x = jnp.arange(2**15)
y = func(x)