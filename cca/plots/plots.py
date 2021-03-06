import matplotlib.pyplot as plt
import numpy as np

from .heatmap import heatmap


def plot_factor(sim, figsize=(4, 12)):
    fig, ax = plt.subplots(sim.modalities, 1, figsize=figsize)
    for m in range(sim.modalities):
        heatmap(sim.W[m], cmap="RdBu", vmin=-2, vmax=2, ax=ax[m])
        ax[m].set_title(f"Modality: {m+1}")
        ax[m].set_ylabel("Factor dimension")

    ax[-1].set_xlabel("Factor")


def plot_matched_factor(sim, model, figsize=(6, 12), fill_array=False):
    fig, ax = plt.subplots(sim.modalities, 2, figsize=figsize)

    idx, _ = sim.match_factors(model, fill_array=fill_array)

    for m in range(sim.modalities):
        heatmap(sim.W[m], cmap="RdBu", vmin=-2, vmax=2, ax=ax[m, 0], cbar=False)
        ax[m, 0].set_title(f"Modality: {m+1}")
        ax[m, 0].set_ylabel("Factor dimension")

        heatmap(
            np.median(model.get_W(m), 0).squeeze()[..., idx],
            cmap="RdBu",
            vmin=-2,
            vmax=2,
            ax=ax[m, 1],
        )
        ax[m, 1].set_title(f"Inferred: {m+1}")
        ax[m, 1].set_ylabel("Factor dimension")

    ax[m, -1].set_xlabel("Factor")


def plot_factor_bar(sim):
    fig, axes = plt.subplots(
        sim.modalities,
        sim.latent_dim,
        figsize=(sim.latent_dim * 3, sim.modalities * 2),
        sharey=True,
    )

    for i in range(sim.modalities * sim.latent_dim):
        j, k = np.unravel_index(i, (sim.modalities, sim.latent_dim))
        axes[j, k].bar(np.arange(sim.dims[j]), sim.W[j][:, k])

    return fig, axes


def plot_matched_factor_bar(sim, model, flip=[]):
    fig, axes = plt.subplots(
        sim.modalities,
        sim.latent_dim,
        figsize=(sim.latent_dim * 4, sim.modalities * 2),
        sharey=True,
    )
    idx, _ = sim.match_factors(model, fill_array=False)

    for i in range(sim.modalities * sim.latent_dim):
        j, k = np.unravel_index(i, (sim.modalities, sim.latent_dim))
        axes[j, k].bar(np.arange(sim.dims[j]) - 0.15, sim.W[j][:, k], width=0.3)
        y = np.median(model.get_W(j).squeeze()[..., idx], 0)[..., k]
        if k in flip:
            y *= -1
        axes[j, k].bar(np.arange(sim.dims[j]) + 0.15, y, width=0.3)
        axes[j, k].margins(x=0.01)


def plot_sample(sim, idx=0, figsize=(8, 5)):
    fig, ax = plt.subplots(sim.latent_dim, 1, figsize=figsize)
    # sim.z[1].T
    ax[0].set_title(f"Sample: {idx}")
    for d in range(sim.latent_dim):
        # sns.heatmap(sim.W[m], cmap='RdBu', vmin=-2, vmax=2, ax=ax[m])
        ax[d].plot(sim.z[idx].T[d], ".-")
        ax[d].margins(x=0.01)
        ax[d].set_ylabel(f"{d}")
        ax[d].spines["top"].set_visible(False)
        ax[d].spines["bottom"].set_visible(False)
        ax[d].spines["right"].set_visible(False)
        if d <= sim.latent_dim:
            ax[d].set_xticks([])

    ax[-1].set_xlabel("Sample idx")


def plot_matched_sample(sim, model, idx=0, fill_array=False, figsize=(8, 5)):
    fig, ax = plt.subplots(sim.latent_dim, 1, figsize=figsize)
    sorting, _ = sim.match_factors(model, fill_array=fill_array)
    z = model.get_z()
    z_med = np.median(z, 0)[:, idx][sorting]

    z_ci = np.abs(
        z_med - np.quantile(z, [0.025, 0.975], axis=0)[:, :, idx][:, sorting]
    )  # model.posterior.ci('z')
    # print(z_ci.shape)

    for d in range(sim.latent_dim):
        # sns.heatmap(sim.W[m], cmap='RdBu', vmin=-2, vmax=2, ax=ax[m])
        ax[d].plot(sim.z[idx].T[d], ".-")
        #            ax[d].plot(z_med[d], '.')
        ax[d].errorbar(np.arange(sim.cells), z_med[d], yerr=z_ci[:, d], fmt=".")
        ax[d].margins(x=0.01)
        ax[d].set_ylabel(f"{d}")
        ax[d].spines["top"].set_visible(False)
        ax[d].spines["bottom"].set_visible(False)
        ax[d].spines["right"].set_visible(False)
        if d <= sim.latent_dim:
            ax[d].set_xticks([])


def plot_matched_alphas(sim, model, fill_array=False):
    d, _ = sim.match_factors(model, fill_array=fill_array)
    M = len(sim.dims)
    fig, ax = plt.subplots(M, 1, figsize=(4, M * 2))
    for m in range(M):
        yt = [1 if m in v else 0 for k, v in sim.active_modalities.items()]
        ax[m].bar(np.arange(sim.latent_dim) - 0.15, yt, width=0.3)
        xp = np.arange(len(d)) if fill_array else np.arange(sim.latent_dim)
        ax[m].bar(xp + 0.15, np.median(model.get_alpha().squeeze(), 0)[m, d], width=0.3)
        ax[m].set_xticks(xp)


def plot_lambda(model):
    fig, axes = plt.subplots(
        model.num_modalities,
        model.num_latent_dim,
        figsize=(model.num_latent_dim * 1.5, model.num_modalities * 1),
        sharey=True,
    )

    L = model.posterior("??").reshape(-1, model.num_modalities, model.num_latent_dim)

    for i in range(model.num_latent_dim):
        for j in range(model.num_modalities):
            axes[j, i].hist(L[:, j, i])

    return fig, axes


def plot_matched_data(sim, model):
    med = np.median(model.get_Y(), 0).reshape(-1, 1)
    q = np.quantile(model.get_Y(), [0.025, 0.975], 0).reshape(2, -1).T
    ci = np.abs(q - med)
    identity = np.linspace(np.min(med), np.max(med))
    plt.errorbar(
        np.concatenate(sim.Y, 1).swapaxes(0, 1).reshape(-1),
        med.reshape(-1),
        yerr=ci.T,
        fmt=".",
        alpha=0.2,
    )
    plt.plot(identity, identity, zorder=100)
