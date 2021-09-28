import dataclasses

import numpy as np
import scipy.stats as stats

from .utils import match_vectors

F = [
    lambda N: np.sin(np.arange(1, N + 1) / (N / 20)),
    lambda N: np.cos(np.arange(1, N + 1) / (N / 20)),
    lambda N: stats.norm().rvs(size=N),
    lambda N: 2 * (np.arange(1, N + 1) / N - 0.5),
    lambda N: 2 * (np.arange(1, N + 1) / N - 0.5) ** 2,
    lambda N: np.ones(N),
]


@dataclasses.dataclass
class Simulation:
    seed: int = 3535
    samples: int = 1
    cells: int = 500
    dims: tuple = (15, 7)
    latent_dim: int = 4
    modalities: int = 2
    min_active_dims: int = 1
    max_active_dims: int = 10

    def __post_init__(self):
        np.random.seed(self.seed)
        self.sigma = self._sigma()

        self.num_active_modalities = {}
        self.active_modalities = {}  #
        self.num_active_latent_dims = {}
        self.active_latent_dims = {}  # sample -> latent dim
        self._set_active_dims()
        self._active_dims()

        self.W = self._W()
        self.z = self._z()
        self.Y = self._Y()

    def _set_active_dims(self):
        num_active_modalities = np.random.choice(
            np.arange(1, self.modalities + 1), size=self.latent_dim
        )

        for l in range(self.latent_dim):
            active_modalities = np.random.choice(
                np.arange(self.modalities), replace=False, size=num_active_modalities[l]
            )
            self.num_active_modalities[l] = num_active_modalities[l]
            self.active_modalities[l] = active_modalities

    def _active_dims(self):
        num_active_dims = np.random.choice(
            np.arange(
                self.min_active_dims,
                np.min([self.max_active_dims, self.latent_dim + 1]),
            ),
            size=self.samples,
        )
        for s in range(self.samples):
            active_dims = np.random.choice(
                np.arange(self.latent_dim), replace=False, size=num_active_dims[s]
            )
            self.num_active_latent_dims[s] = num_active_dims[s]
            self.active_latent_dims[s] = active_dims

    def _sigma(self):
        return 1 / np.sqrt(stats.halfcauchy(1.0).rvs(size=self.modalities))

    def _W(self):

        W = {i: [] for i in range(self.modalities)}
        for l in range(self.latent_dim):
            for m in range(self.modalities):
                if m in self.active_modalities[l]:
                    W[m].append(stats.norm(0, 1).rvs(size=(self.dims[m])))
                else:
                    W[m].append(stats.norm(0, 0.001).rvs(size=(self.dims[m])))

        W = [np.stack(v).T for k, v in W.items()]
        return W

    def _z(self):
        """
        Generates factor loadings.
        """
        z = np.zeros((self.samples, self.cells, self.latent_dim))

        for s in range(self.samples):
            # lp loading pattern
            for d in self.active_latent_dims[s]:
                z[s, :, d] = stats.norm().rvs(size=self.cells)  # F[p](self.cells)

        return z

    def _Y(self):
        Y = []

        for m in range(self.modalities):
            n = (
                stats.norm(0, self.sigma[m])
                .rvs(size=(self.dims[m] * self.cells * self.samples))
                .reshape(self.samples, self.dims[m], self.cells)
            )
            Y.append(np.einsum("ij,jlk->kil", self.W[m], self.z.T) + n)

        return Y

    def match_factors(self, model, fill_array=True):
        _, idx, dist = match_vectors(
            np.concatenate(self.W, 0).T,
            np.median(model.get_W(), 0).squeeze().T,
            fill_array=fill_array,
        )
        return idx, dist


class SimulationFunction(Simulation):
    def _z(self):
        """
        Generates factor loadings.
        """
        z = np.zeros((self.samples, self.cells, self.latent_dim))

        for s in range(self.samples):
            # lp loading pattern
            func_id = np.random.choice(
                np.arange(len(F)), replace=False, size=self.num_active_latent_dims[s]
            )
            for i, d in zip(self.active_latent_dims[s], func_id):
                z[s, :, d] = F[i](self.cells)

        return z


# @dataclasses.dataclass
# class Simulation:
#     seed: int = 3535
#     samples: int = 1
#     cells: int = 500
#     dims: tuple = (15, 7)
#     latent_dim: int = 4
#     modalities: int = 2

#     def __post_init__(self):
#         np.random.seed(self.seed)
#         self.tau = self._tau()
#         self.alpha = self._alpha()
#         self.z, self.a = self._z()
#         self.W = self._W()
#         self.Y = self._Y()

#     def _z(self):
#         """
#         Generates factor loadings.
#         """
#         z = np.zeros((self.samples, self.cells, self.latent_dim))
#         a = np.zeros((self.samples, self.latent_dim))
#         for s in range(self.samples):
#             # lp loading pattern
#             lp = np.random.choice(len(F), size=self.latent_dim, replace=False)

#             for k, p in enumerate(lp):
#                 z[s, :, k] = F[p](self.cells)

#             a[s, :] = lp

#         return z, a

#     def _tau(self):
#         return np.random.uniform(1, 10, size=self.modalities)

#     def _alpha(self):
#         alpha = np.zeros((self.samples, self.modalities, self.latent_dim))
#         alpha_comb = np.random.choice(
#             [1, 1e6], size=self.modalities * self.latent_dim
#         ).reshape(self.modalities, self.latent_dim)
#         return alpha_comb

#     def _W(self):
#         W = []
#         for m in range(self.modalities):
#             W.append(
#                 stats.norm(0, 1 / np.sqrt(self.alpha[m])).rvs(
#                     size=(self.dims[m], self.latent_dim)
#                 )
#             )
#         # stats.norm()
#         return W

#     def _Y(self):
#         Y = []

#         for m in range(self.modalities):
#             n = (
#                 stats.norm(0, 1 / np.sqrt(self.tau)[m])
#                 .rvs(size=(self.dims[m] * self.cells * self.samples))
#                 .reshape(self.samples, self.dims[m], self.cells)
#             )
#             Y.append(np.einsum("ij,jlk->kil", self.W[m], self.z.T) + n)

#         return Y

#     def match_factors(self, model, fill_array=True):
#         _, idx, dist = match_vectors(
#             np.concatenate(self.W, 0).T,
#             np.median(model.get_W(), 0).squeeze().T,
#             fill_array=fill_array,
#         )
#         return idx, dist

#     def plot_factor(self, figsize=(3, 12)):
#         fig, ax = plt.subplots(self.modalities, 1, figsize=figsize)
#         for m in range(self.modalities):
#             sns.heatmap(self.W[m], cmap="RdBu", vmin=-2, vmax=2, ax=ax[m])
#             ax[m].set_title(f"Modality: {m+1}")
#             ax[m].set_ylabel(f"Factor dimension")
#         ax[-1].set_xlabel("Factor")

#     def plot_matched_sample(self, idx, model, fill_array=False, figsize=(8, 5)):
#         fig, ax = plt.subplots(self.latent_dim, 1, figsize=figsize)
#         sorting, _ = self.match_factors(model, fill_array=fill_array)
#         z = model.get_z()
#         z_med = np.median(z, 0)[:, idx][sorting]

#         z_ci = np.abs(
#             z_med - np.quantile(z, [0.025, 0.975], axis=0)[:, :, idx][:, sorting]
#         )  # model.posterior.ci('z')
#         print(z_ci.shape)

#         for d in range(self.latent_dim):
#             # sns.heatmap(self.W[m], cmap='RdBu', vmin=-2, vmax=2, ax=ax[m])
#             ax[d].plot(self.z[idx].T[d], "-")
#             #            ax[d].plot(z_med[d], '.')
#             ax[d].errorbar(np.arange(self.cells), z_med[d], yerr=z_ci[:, d], fmt=".")
#             ax[d].margins(x=0.01)
#             ax[d].set_ylabel(f"{d}")
#             ax[d].spines["top"].set_visible(False)
#             ax[d].spines["bottom"].set_visible(False)
#             ax[d].spines["right"].set_visible(False)
#             if d <= self.latent_dim:
#                 ax[d].set_xticks([])

#     def plot_matched_factor(self, model, figsize=(6, 12), fill_array=False):
#         fig, ax = plt.subplots(self.modalities, 2, figsize=figsize)

#         idx, _ = self.match_factors(model, fill_array=fill_array)

#         for m in range(self.modalities):
#             sns.heatmap(
#                 self.W[m], cmap="RdBu", vmin=-2, vmax=2, ax=ax[m, 0], cbar=False
#             )
#             ax[m, 0].set_title(f"Modality: {m+1}")
#             ax[m, 0].set_ylabel("Factor dimension")

#             sns.heatmap(
#                 np.median(model.get_W(m), 0).squeeze()[..., idx],
#                 cmap="RdBu",
#                 vmin=-2,
#                 vmax=2,
#                 ax=ax[m, 1],
#             )
#         ax[m, -1].set_xlabel("Factor")

#     def plot_matched_factor2(self, model, figsize=(6, 12), fill_array=False):
#         fig, ax = plt.subplots(self.modalities, 2, figsize=figsize)

#         idx, _ = self.match_factors(model, fill_array=fill_array)

#         for m in range(self.modalities):
#             sns.heatmap(
#                 self.W[m], cmap="RdBu", vmin=-2, vmax=2, ax=ax[m, 0], cbar=False
#             )
#             ax[m, 0].set_title(f"Modality: {m+1}")
#             ax[m, 0].set_ylabel("Factor dimension")

#             sns.heatmap(
#                 model.posterior.median("W_{m}s").squeeze()[:, idx],
#                 cmap="RdBu",
#                 ax=ax[m, 1],
#             )
#         ax[m, -1].set_xlabel("Factor")

#     def plot_sample(self, idx, figsize=(8, 5)):
#         fig, ax = plt.subplots(self.latent_dim, 1, figsize=figsize)
#         # S1.z[1].T
#         ax[0].set_title(f"Sample: {idx}")
#         for d in range(self.latent_dim):
#             # sns.heatmap(self.W[m], cmap='RdBu', vmin=-2, vmax=2, ax=ax[m])
#             ax[d].plot(self.z[idx].T[d], "-")
#             ax[d].margins(x=0.01)
#             ax[d].set_ylabel(f"{d}")
#             ax[d].spines["top"].set_visible(False)
#             ax[d].spines["bottom"].set_visible(False)
#             ax[d].spines["right"].set_visible(False)
#             if d <= self.latent_dim:
#                 ax[d].set_xticks([])

#         ax[-1].set_xlabel("Sample idx")
