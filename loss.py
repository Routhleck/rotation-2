import jax.numpy as jnp
import jax


def communicability_loss(weight_matrix, comms_factor=1):
    '''
    Version of SE1 regulariser which combines the spatial and communicability parts in loss function.
    Additional comms_factor scales the communicability matrix.
    The communicability term used here is unbiased weighted communicability:
    Crofts, J. J., & Higham, D. J. (2009). A weighted communicability measure applied to complex brain networks. Journal of the Royal Society Interface, 6(33), 411-414.
    '''
    abs_weight_matrix = jnp.abs(weight_matrix)

    # Calulcate weighted communicability (see reference in docstring)
    stepI = jnp.sum(abs_weight_matrix, axis=1)
    stepII = jnp.pow(stepI, -0.5)
    stepIII = jnp.diag(stepII)
    stepIV = jax.scipy.linalg.expm(stepIII @ abs_weight_matrix @ stepIII)
    comms_matrix = stepIV.at[jnp.diag_indices(stepIV.shape[0])].set(0)

    # Multiply absolute weights with communicability weights
    comms_matrix = comms_matrix ** comms_factor
    comms_weight_matrix = jnp.multiply(abs_weight_matrix, comms_matrix)

    # Multiply comms weights matrix with distance tensor, scale the mean, and return as loss
    return jnp.sum(comms_weight_matrix)
