import matplotlib.pyplot as plt


def visualize_ufp_onc_curves(history: dict, k_opt: int) -> None:
    """Generates a visual comparison of UFP-ONC performance measures for
    selecting the optimal number of clusters (Section II-C of Gath & Geva, 1989).

    Plots the three validity criteria as a function of k:
        - FHV  (Fuzzy HyperVolume)     → optimal k at the minimum.
        - D_PA (Average Partition Density) → optimal k at the maximum.
        - PD   (Partition Density)      → optimal k at the maximum.

    Args:
        history: Dict returned by train_ufp_onc with keys 'k', 'fhv', 'D_PA', 'PD'.
        k_opt:   Optimal number of clusters detected by the algorithm.
    """
    k_list = history['k']
    fhv    = history['fhv']
    D_PA   = history['D_PA']
    PD     = history['PD']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # ── FHV: minimum indicates optimal k
    ax1.plot(k_list, fhv, marker='o', linestyle='-', color='b')
    ax1.axvline(k_opt, color='b', linestyle='--', alpha=0.5, label=f'k_opt = {k_opt}')
    ax1.set_title("Fuzzy HyperVolume (FHV)", fontsize=14)
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("FHV")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── D_PA: maximum indicates optimal k
    ax2.plot(k_list, D_PA, marker='s', linestyle='-', color='g')
    ax2.axvline(k_opt, color='g', linestyle='--', alpha=0.5, label=f'k_opt = {k_opt}')
    ax2.set_title("Average Partition Density (D_PA)", fontsize=14)
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("D_PA")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── PD: maximum indicates optimal k
    ax3.plot(k_list, PD, marker='^', linestyle='-', color='r')
    ax3.axvline(k_opt, color='r', linestyle='--', alpha=0.5, label=f'k_opt = {k_opt}')
    ax3.set_title("Partition Density (PD)", fontsize=14)
    ax3.set_xlabel("Number of clusters (k)")
    ax3.set_ylabel("PD")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f"UFP-ONC Performance Measures  –  optimal k = {k_opt}", fontsize=15)
    plt.tight_layout()
    plt.show()