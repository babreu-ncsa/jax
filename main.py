import jax
import jax.numpy as jnp
import time
import numpy as np

SIZE = 2**27
ITERATIONS = 2**12

def selu(x, alpha=1.67, lambda_=1.05):
    return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(SIZE)

# send one operation at a time to the accelerator
t = []
for _ in range(ITERATIONS):
    start = time.perf_counter()
    selu(x).block_until_ready()
    end = time.perf_counter()
    t.append(end-start)
t = np.array(t)
print(f"Average time (s): {t.mean()}")


# JIT compile the function
selu_jit = jax.jit(selu, backend="gpu")
# warm it up
selu_jit(x).block_until_ready()
t = []
for _ in range(ITERATIONS):
    start = time.perf_counter()
    selu_jit(x).block_until_ready()
    end = time.perf_counter()
    t.append(end-start)
t = np.array(t)
print(f"Average time with compilation (s): {t.mean()}")

