from cca.handler import Model
import numpy as np

class CCA(Model):
    _latent_variables = [
        'W', 'z', 'α', 'τ'
    ]

    def __init__(self, 
                 Y, 
                 num_latent_dim,
                 horsehoe_prior = True,
                 factor_cov = 2,
                 loading_cov = 1,
                 handler_kwargs={'handler': 'SVI', 'num_samples':500, 'num_epochs': 50000, 'lr': 0.001, 'rng_key': 84684}
                ):
        Model.__init__(self, **handler_kwargs)
        self.Y = Y        
        self.Y_c = np.concatenate(Y, 1)
        
        
        self.α0 = 1e-9
        self.β0 = 1e-9
        
        self.num_latent_dim = num_latent_dim
        self.num_modalities = len(Y)
        self.num_nodes = self.Y_c.shape[0]
        self.num_features = self.Y_c.shape[1]
        self.num_cells = self.Y_c.shape[2]
        self.horseshoe = horsehoe_prior
        self.factor_cov = factor_cov
        self.loading_cov = loading_cov
        
        
        self.I = np.zeros((self.Y_c.shape[1], self.num_modalities))
        
        j = 0
        for i in range(self.num_modalities):
            _, n, _ = self.Y[i].shape
            self.I[j:j+n, i] = np.ones(n)
            j += n
            
    def model(self):
        # ARD prior
        if self.horseshoe:
            t = npy.sample('t', dist.HalfCauchy(scale=jnp.ones(self.num_modalities)))
            λ = npy.sample('λ', dist.HalfCauchy(scale=jnp.ones(self.num_modalities * self.num_latent_dim)))
            α = npy.deterministic('α', t.reshape(self.num_modalities, 1, 1,) * λ.reshape(self.num_modalities, 1, self.num_latent_dim)) * jnp.ones((1, self.num_nodes, 1))
        else:            
#             α = npy.sample('α', dist.InverseGamma(
#                 jnp.tile(self.α0, (self.num_modalities, self.num_latent_dim)), 
#                 jnp.tile(self.β0, (self.num_modalities, self.num_latent_dim))))
            α = npy.sample('α', dist.InverseGamma(
                jnp.repeat(self.α0, self.num_modalities * self.num_latent_dim), 
                jnp.repeat(self.β0, self.num_modalities * self.num_latent_dim)))
      
            α = jnp.expand_dims(jnp.sqrt(α.reshape(self.num_modalities, self.num_latent_dim)), 1) * jnp.ones((1, self.num_nodes, 1))
        
        α_sd = jnp.einsum('ji,ikl->jkl', self.I, α)
        #W0 = npy.sample('W', dist.Normal(jnp.zeros((self.num_features, self.num_latent_dim)), 1.))
        if self.factor_cov == 2:
            W0 = npy.sample('W', dist.Normal(jnp.zeros(self.num_features * self.num_latent_dim), 1.))
        elif self.factor_cov == 1:
            W0 = npy.sample('W', dist.MultivariateNormal(
                jnp.zeros((self.num_latent_dim, self.num_features)), 
                jnp.stack(self.num_latent_dim * [jnp.eye(self.num_features)])))
        else:
            W0 = npy.sample('W', dist.Normal(
                jnp.zeros((self.num_features, self.num_latent_dim)), 
                jnp.ones((self.num_features, self.num_latent_dim))))

        W = jnp.expand_dims(W0.reshape(self.num_features, self.num_latent_dim), 1) * α_sd
        
        
        if self.loading_cov == 1:
            z = npy.sample('z', dist.MultivariateNormal(
                jnp.zeros((self.num_nodes * self.num_cells, self.num_latent_dim)),
                jnp.stack([jnp.eye(self.num_latent_dim)] * self.num_nodes * self.num_cells)
                ))
#             print(z[:2])
            z = z.T.reshape(self.num_latent_dim, self.num_nodes, self.num_cells)
#             print(z[...,0, :2])
        else:
            z = npy.sample('z', dist.Normal(
                jnp.zeros((self.num_latent_dim, self.num_nodes, self.num_cells)), 
                jnp.ones((self.num_latent_dim, self.num_nodes, self.num_cells))))
        
#         npy.deterministic('z', z)
        Y_hat = jnp.einsum('ijk,kjm->jim', W, z)

#         τ = npy.sample('τ', dist.Gamma(
#             jnp.repeat(1., self.num_modalities), 
#             jnp.repeat(1., self.num_modalities)))
        
# #         _ = npy.deterministic('τ', τ)
#         τ_sd = self.I @ (1. / jnp.sqrt(τ).reshape(-1, 1))
        τ = npy.sample('τ', dist.HalfCauchy(1.))
        τ = self.I @ τ.reshape(-1, 1)
        
        
        _ = npy.sample('Y', dist.Normal(Y_hat, τ), obs=self.Y_c)

        
    def guide(self):
        if self.horseshoe:
            t_loc = npy.param("t_loc", jnp.zeros(self.num_modalities))
            t_scale = npy.param("t_scale", 
                                .1 * jnp.eye(self.num_modalities),
                                constraint = dist.constraints.lower_cholesky)
            t = npy.sample("t", dist.TransformedDistribution(
                dist.MultivariateNormal(t_loc, scale_tril=t_scale),
                transforms = dist.transforms.ExpTransform()
            ))
    
            λ_loc = npy.param('λ_loc', jnp.zeros(self.num_modalities * self.num_latent_dim))
            λ_scale = npy.param('λ_scale', 
                                .1 * jnp.eye(self.num_modalities * self.num_latent_dim),
                                constraint=dist.constraints.lower_cholesky)
            #npy.sample('t', dist.HalfCauchy(scale=t_scale))
            npy.sample("λ", dist.TransformedDistribution(
                dist.MultivariateNormal(λ_loc, scale_tril = λ_scale),
                transforms = dist.transforms.ExpTransform()
                ))    
            
        else:
#             α_conc = npy.param('α_conc', 
#                                jnp.tile(1., (self.num_modalities, self.num_latent_dim)), 
#                                constraint=dist.constraints.positive)
#             α_rate = npy.param('α_rate', 
#                                jnp.tile(1., (self.num_modalities, self.num_latent_dim)), 
#                                constraint=dist.constraints.positive)
            α_loc = npy.param("α_loc", jnp.zeros(self.num_modalities * self.num_latent_dim))
            α_scale = npy.param("α_scale", .1 * jnp.eye(self.num_modalities * self.num_latent_dim), 
                                constraint = dist.constraints.lower_cholesky)
            npy.sample("α", dist.TransformedDistribution(
                dist.MultivariateNormal(α_loc, scale_tril = α_scale),
                transforms = dist.transforms.ExpTransform()
                ))
            #npy.sample('α', dist.InverseGamma(α_conc, α_rate))
        

        if self.factor_cov == 2:
            W_loc = npy.param('W_loc', jnp.zeros(self.num_features * self.num_latent_dim))
            W_scale = npy.param('W_scale', jnp.eye(self.num_features * self.num_latent_dim), constraint=dist.constraints.lower_cholesky)
            npy.sample(f'W', dist.MultivariateNormal(W_loc, scale_tril = W_scale))
        elif self.factor_cov == 1:
            W_loc = npy.param('W_loc', jnp.zeros((self.num_latent_dim, self.num_features)))
            W_scale = npy.param('W_scale', jnp.stack(self.num_latent_dim * [jnp.eye(self.num_features)]), constraint=dist.constraints.lower_cholesky)
            npy.sample(f'W', dist.MultivariateNormal(W_loc, scale_tril = W_scale))
        else:
            W_loc = npy.param('W_loc', jnp.zeros((self.num_features, self.num_latent_dim)))
            W_scale = npy.param('W_scale', jnp.ones((self.num_features, self.num_latent_dim)), constraint=dist.constraints.positive)
            npy.sample(f'W', dist.Normal(W_loc, W_scale))
            

#         τ_conc = npy.param('τ_conc', jnp.repeat(1., self.num_modalities), constraint=dist.constraints.positive)
#         τ_rate = npy.param('τ_conc', jnp.repeat(1., self.num_modalities), constraint=dist.constraints.positive)    
#         τ = npy.sample('τ', dist.Gamma(τ_conc, τ_rate))

        τ_loc = npy.param("τ_loc", jnp.zeros(self.num_modalities))
        τ_scale = npy.param("τ_scale", jnp.ones(self.num_modalities), constraint = dist.constraints.positive)
        npy.sample("τ", dist.TransformedDistribution(dist.Normal(τ_loc, τ_scale), transforms=dist.transforms.ExpTransform()))
        
        
        if self.loading_cov==1:
            z_loc = npy.param('z_loc', jnp.zeros((self.num_nodes * self.num_cells, self.num_latent_dim)))
            z_scale = npy.param('z_scale', jnp.stack([jnp.eye(self.num_latent_dim)] * self.num_nodes * self.num_cells), constraint=dist.constraints.lower_cholesky)
            z = npy.sample('z', dist.MultivariateNormal(z_loc, scale_tril = z_scale))
        else:
            z_loc = npy.param('z_loc', jnp.zeros((self.num_latent_dim, self.num_nodes, self.num_cells)))
            z_scale = npy.param('z_scale', jnp.ones((self.num_latent_dim, self.num_nodes, self.num_cells)))
            z = npy.sample('z', dist.Normal(z_loc, z_scale))

    def get_alpha(self,):        
        α = self.posterior.dist('α')
        if self.horseshoe:
            return α

        return jnp.sqrt(α)
        
    def get_z(self):
        z = self.posterior.dist('z')
        if self.loading_cov == 1:
            return np.rollaxis(z.T.reshape(self.num_latent_dim, self.num_nodes, self.num_cells, -1), 3)
        return z
        
    def get_W(self, modality=None):
        α0 = self.posterior.dist('α')
        α = jnp.expand_dims(α0.reshape(-1, self.num_modalities, self.num_latent_dim), 2)
        W0 = self.posterior.dist('W')
        W = jnp.expand_dims(W0.reshape(-1, self.num_features, self.num_latent_dim), 2) * jnp.einsum('ji,mikl->mjkl', self.I, jnp.sqrt(α))
        if modality is None:
            return W
        else:
            return W[:, self.I[:,modality]==1.]
    
    def get_Y(self):
        z = self.get_z()
        W = self.get_W()
        
        return jnp.einsum('nijk,nkjm->njim', W, z)
   