import numpy as np


def seeded_entropy():
    """A seed for a fresh Generator, drawn from the legacy global RNG.

    scipy's qmc samplers and np.random.default_rng() default to OS entropy, so they
    ignore np.random.seed -- which is what a `seed:` in the config sets. Seeding them
    from the legacy global instead keeps successive calls independent while making the
    whole sequence a deterministic function of that one seed. With no seed set the
    global is itself entropy-seeded, so unseeded behaviour is statistically unchanged.

    Note this cannot reach the AIES sampler: its proposals come from tf.random.uniform
    inside a jit_compile'd tf.function, and XLA discards the seed (TensorFlow warns
    about this at runtime). Fixing that needs tf.random.stateless_uniform.
    """
    return int(np.random.randint(0, 2**32 - 1))
