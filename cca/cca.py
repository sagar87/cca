import jax.numpy as jnp
import numpy as np
import numpyro as npy
import numpyro.distributions as dist

from cca.handler import Model


class CCA(Model):
    _latent_variables = ["W", "z", "α", "τ"]

    def __init__(
        self,
        Y,
        num_latent_dim,
        horsehoe_prior=True,
        factor_cov=0,
        loading_cov=0,
        z_scale=0.1,
        z_prior=False,
        α_conc=1e-3,
        α_rate=1e-3,
        handler_kwargs={
            "handler": "SVI",
            "num_samples": 500,
            "num_epochs": 50000,
            "lr": 0.001,
            "rng_key": 84684,
        },
    ):
        Model.__init__(self, **handler_kwargs)
        self.Y = Y
        self.Y_c = np.concatenate(Y, 1)

        # prior for gamma prior
        self.α_conc = α_conc
        self.α_rate = α_rate

        self.num_latent_dim = num_latent_dim
        self.num_modalities = len(Y)
        self.num_nodes = self.Y_c.shape[0]
        self.num_features = self.Y_c.shape[1]
        self.num_cells = self.Y_c.shape[2]
        self.horseshoe = horsehoe_prior
        self.factor_cov = factor_cov
        self.loading_cov = loading_cov

        self.z_prior = z_prior
        self.z_scale = z_scale

        self.identity = np.zeros((self.Y_c.shape[1], self.num_modalities))

        j = 0
        for i in range(self.num_modalities):
            _, n, _ = self.Y[i].shape
            self.identity[j : j + n, i] = np.ones(n)
            j += n

    def model(self):
        # ARD prior
        if self.horseshoe:
            τ = npy.sample("τ", dist.HalfCauchy(scale=jnp.ones(self.num_modalities)))
            λ = npy.sample(
                "λ",
                dist.HalfCauchy(
                    scale=jnp.ones(self.num_modalities * self.num_latent_dim)
                ),
            )
            α = (
                npy.deterministic(
                    "α",
                    τ.reshape(
                        self.num_modalities,
                        1,
                        1,
                    )
                    * λ.reshape(self.num_modalities, 1, self.num_latent_dim),
                )
                * jnp.ones((1, self.num_nodes, 1))
            )
        else:
            α = npy.sample(
                "α",
                dist.InverseGamma(
                    jnp.repeat(self.α_conc, self.num_modalities * self.num_latent_dim),
                    jnp.repeat(self.α_rate, self.num_modalities * self.num_latent_dim),
                ),
            )

            α = jnp.expand_dims(
                jnp.sqrt(α.reshape(self.num_modalities, self.num_latent_dim)), 1
            ) * jnp.ones((1, self.num_nodes, 1))

        α_sd = jnp.einsum("ji,ikl->jkl", self.identity, α)

        if self.factor_cov == 2:
            W0 = npy.sample(
                "W",
                dist.Normal(jnp.zeros(self.num_features * self.num_latent_dim), 1.0),
            )
        elif self.factor_cov == 1:
            W0 = npy.sample(
                "W",
                dist.MultivariateNormal(
                    jnp.zeros((self.num_latent_dim, self.num_features)),
                    jnp.stack(self.num_latent_dim * [jnp.eye(self.num_features)]),
                ),
            )
        else:
            W0 = npy.sample(
                "W",
                dist.Normal(
                    jnp.zeros((self.num_features, self.num_latent_dim)),
                    jnp.ones((self.num_features, self.num_latent_dim)),
                ),
            )

        W = (
            jnp.expand_dims(W0.reshape(self.num_features, self.num_latent_dim), 1)
            * α_sd
        )

        if self.loading_cov == 1:
            z = npy.sample(
                "z",
                dist.MultivariateNormal(
                    jnp.zeros((self.num_nodes * self.num_cells, self.num_latent_dim)),
                    jnp.stack(
                        [jnp.eye(self.num_latent_dim)] * self.num_nodes * self.num_cells
                    ),
                ),
            )
            z = z.T.reshape(self.num_latent_dim, self.num_nodes, self.num_cells)
        else:
            z = npy.sample(
                "z",
                dist.Normal(
                    jnp.zeros((self.num_latent_dim, self.num_nodes, self.num_cells)),
                    jnp.ones((self.num_latent_dim, self.num_nodes, self.num_cells)),
                ),
            )

            if self.z_prior:
                z_sigma = npy.sample(
                    "z_sigma",
                    dist.InverseGamma(
                        jnp.repeat(self.z_scale, self.num_latent_dim * self.num_nodes),
                        jnp.repeat(self.z_scale, self.num_latent_dim * self.num_nodes),
                    ),
                )

                z = z * z_sigma.reshape(self.num_latent_dim, self.num_nodes, 1)

        Y_hat = jnp.einsum("ijk,kjm->jim", W, z)

        σ = npy.sample("σ", dist.HalfCauchy(1.0))
        σ = self.identity @ σ.reshape(-1, 1)

        _ = npy.sample("Y", dist.Normal(Y_hat, σ), obs=self.Y_c)

    def guide(self):
        if self.horseshoe:
            τ_loc = npy.param("τ_loc", jnp.zeros(self.num_modalities))
            τ_scale = npy.param(
                "τ_scale",
                0.1 * jnp.eye(self.num_modalities),
                constraint=dist.constraints.lower_cholesky,
            )
            npy.sample(
                "τ",
                dist.TransformedDistribution(
                    dist.MultivariateNormal(τ_loc, scale_tril=τ_scale),
                    transforms=dist.transforms.ExpTransform(),
                ),
            )

            λ_loc = npy.param(
                "λ_loc", jnp.zeros(self.num_modalities * self.num_latent_dim)
            )
            λ_scale = npy.param(
                "λ_scale",
                0.1 * jnp.eye(self.num_modalities * self.num_latent_dim),
                constraint=dist.constraints.lower_cholesky,
            )
            npy.sample(
                "λ",
                dist.TransformedDistribution(
                    dist.MultivariateNormal(λ_loc, scale_tril=λ_scale),
                    transforms=dist.transforms.ExpTransform(),
                ),
            )

        else:
            α_loc = npy.param(
                "α_loc", jnp.zeros(self.num_modalities * self.num_latent_dim)
            )
            α_scale = npy.param(
                "α_scale",
                0.1 * jnp.eye(self.num_modalities * self.num_latent_dim),
                constraint=dist.constraints.lower_cholesky,
            )
            npy.sample(
                "α",
                dist.TransformedDistribution(
                    dist.MultivariateNormal(α_loc, scale_tril=α_scale),
                    transforms=dist.transforms.ExpTransform(),
                ),
            )

        if self.factor_cov == 2:
            W_loc = npy.param(
                "W_loc", jnp.zeros(self.num_features * self.num_latent_dim)
            )
            W_scale = npy.param(
                "W_scale",
                jnp.eye(self.num_features * self.num_latent_dim),
                constraint=dist.constraints.lower_cholesky,
            )
            npy.sample("W", dist.MultivariateNormal(W_loc, scale_tril=W_scale))
        elif self.factor_cov == 1:
            W_loc = npy.param(
                "W_loc", jnp.zeros((self.num_latent_dim, self.num_features))
            )
            W_scale = npy.param(
                "W_scale",
                jnp.stack(self.num_latent_dim * [jnp.eye(self.num_features)]),
                constraint=dist.constraints.lower_cholesky,
            )
            npy.sample("W", dist.MultivariateNormal(W_loc, scale_tril=W_scale))
        else:
            W_loc = npy.param(
                "W_loc", jnp.zeros((self.num_features, self.num_latent_dim))
            )
            W_scale = npy.param(
                "W_scale",
                jnp.ones((self.num_features, self.num_latent_dim)),
                constraint=dist.constraints.positive,
            )
            npy.sample("W", dist.Normal(W_loc, W_scale))

        σ_loc = npy.param("σ_loc", jnp.zeros(self.num_modalities))
        σ_scale = npy.param(
            "σ_scale",
            jnp.ones(self.num_modalities),
            constraint=dist.constraints.positive,
        )
        npy.sample(
            "σ",
            dist.TransformedDistribution(
                dist.Normal(σ_loc, σ_scale), transforms=dist.transforms.ExpTransform()
            ),
        )

        if self.z_prior:
            z_sigma_loc = npy.param(
                "z_sigma_loc", jnp.zeros(self.num_nodes * self.num_latent_dim)
            )
            z_sigma_scale = npy.param(
                "z_sigma_scale",
                0.1 * jnp.eye(self.num_nodes * self.num_latent_dim),
                constraint=dist.constraints.lower_cholesky,
            )
            npy.sample(
                "z_sigma",
                dist.TransformedDistribution(
                    dist.MultivariateNormal(z_sigma_loc, scale_tril=z_sigma_scale),
                    transforms=dist.transforms.ExpTransform(),
                ),
            )

        if self.loading_cov == 1:
            z_loc = npy.param(
                "z_loc",
                jnp.zeros((self.num_nodes * self.num_cells, self.num_latent_dim)),
            )
            z_scale = npy.param(
                "z_scale",
                jnp.stack(
                    [jnp.eye(self.num_latent_dim)] * self.num_nodes * self.num_cells
                ),
                constraint=dist.constraints.lower_cholesky,
            )
            npy.sample("z", dist.MultivariateNormal(z_loc, scale_tril=z_scale))
        else:
            z_loc = npy.param(
                "z_loc",
                jnp.zeros((self.num_latent_dim, self.num_nodes, self.num_cells)),
            )
            z_scale = npy.param(
                "z_scale",
                jnp.ones((self.num_latent_dim, self.num_nodes, self.num_cells)),
            )
            npy.sample("z", dist.Normal(z_loc, z_scale))

    def get_alpha(
        self,
    ):
        α = self.posterior.dist("α")
        if self.horseshoe:
            return α

        return jnp.sqrt(α)

    def get_z(self):
        z = self.posterior.dist("z")
        if self.loading_cov == 1:
            return np.rollaxis(
                z.T.reshape(self.num_latent_dim, self.num_nodes, self.num_cells, -1), 3
            )
        return z

    def get_W(self, modality=None):
        α0 = self.posterior.dist("α")
        α = jnp.expand_dims(α0.reshape(-1, self.num_modalities, self.num_latent_dim), 2)
        W0 = self.posterior.dist("W")
        W = jnp.expand_dims(
            W0.reshape(-1, self.num_features, self.num_latent_dim), 2
        ) * jnp.einsum("ji,mikl->mjkl", self.identity, jnp.sqrt(α))
        if modality is None:
            return W
        else:
            return W[:, self.identity[:, modality] == 1.0]

    def get_Y(self):
        z = self.get_z()
        W = self.get_W()

        return jnp.einsum("nijk,nkjm->njim", W, z)
