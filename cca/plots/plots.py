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


def plot_sample(sim, idx, figsize=(8, 5)):
    fig, ax = plt.subplots(sim.latent_dim, 1, figsize=figsize)
    # S1.z[1].T
    ax[0].set_title(f"Sample: {idx}")
    for d in range(sim.latent_dim):
        # sns.heatmap(sim.W[m], cmap='RdBu', vmin=-2, vmax=2, ax=ax[m])
        ax[d].plot(sim.z[idx].T[d], "-")
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
        ax[d].plot(sim.z[idx].T[d], "-")
        #            ax[d].plot(z_med[d], '.')
        ax[d].errorbar(np.arange(sim.cells), z_med[d], yerr=z_ci[:, d], fmt=".")
        ax[d].margins(x=0.01)
        ax[d].set_ylabel(f"{d}")
        ax[d].spines["top"].set_visible(False)
        ax[d].spines["bottom"].set_visible(False)
        ax[d].spines["right"].set_visible(False)
        if d <= sim.latent_dim:
            ax[d].set_xticks([])
