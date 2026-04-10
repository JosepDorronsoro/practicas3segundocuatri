import matplotlib.pyplot as plt


def plot_metrics(dims, mse_list, kl_list, js_list, tv_list):
    """
        Plot multiple density estimation error metrics as a function of dimension.

        Args:
            dims (list or array-like): Dimensions used in the experiments.
            mse_list (list or array-like): Mean squared error values for each dimension.
            kl_list (list or array-like): Kullback-Leibler divergence values for each dimension.
            js_list (list or array-like): Jensen-Shannon divergence values for each dimension.
            tv_list (list or array-like): Total variation distance values for each dimension.

        Returns:
            None: Displays the generated plots.
    """
    fig, axs = plt.subplots(2,2, figsize=(10,8))

    axs[0,0].plot(dims, mse_list, marker="o")
    axs[0,0].set_title("MSE")
    axs[0,0].set_yscale("log")  # log scale for better visibility of MSE

    axs[0,1].plot(dims, kl_list, marker="o")
    axs[0,1].set_title("KL divergence")
    axs[0,1].set_yscale("log")  # log scale for better visibility of KL divergence

    axs[1,0].plot(dims, js_list, marker="o")
    axs[1,0].set_title("Jensen-Shannon")
    axs[1,0].set_yscale("log")  # log scale for better visibility of JS divergence

    axs[1,1].plot(dims, tv_list, marker="o")
    axs[1,1].set_title("Total variation")
    axs[1,1].set_yscale("log")  # log scale for better visibility of TV distance

    for ax in axs.flat:
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Metric value (log scale)")
        ax.grid(True)

    plt.tight_layout()
    plt.show()