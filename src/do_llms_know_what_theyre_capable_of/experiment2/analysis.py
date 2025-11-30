import glob
import json
import os
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

from inspect_ai.log import read_eval_log

from do_llms_know_what_theyre_capable_of._vendor.roc_comparison.compare_auc_delong_xu import delong_roc_variance

IC_to_01 = {"C": 1, "I": 0}
noyes_to_01 = {"no": 0, "yes": 1, }


def get_data(log_file):
    L = 9 # Number of contracts in each sequence
    profit_lists = []
    outcome_lists = []
    decision_lists = []
    likelihoods_lists = []
    sample_ids_list = []
    nan_count = 0

    log = read_eval_log(log_file)
    model = log.eval.model

    for sample in log.samples:
        try:
            results_dict = json.loads(sample.messages[-1].content)
            decision_list = [
                noyes_to_01[decision]
                for decision in results_dict["decision_list"]
            ]
        except:
            nan_count += 1
            continue
        if len(results_dict["decision_list"]) < L:
            nan_count += 1
            continue
        decision_lists.append(decision_list)
        profit_lists.append(results_dict["profit_list"])
        outcome_list = [IC_to_01[outcome] for outcome in results_dict["outcome_list"]]
        outcome_lists.append(outcome_list)
        likelihoods_lists.append(results_dict["likelihood_list"])
        sample_ids_list.append(sample.id)

    if nan_count > 0:
        print(f"WARNING: {nan_count} sequences terminated due to no decision")

    print("Number of samples obtained:", len(decision_lists), "  Full dataset size:", 2**L)

    outcome_lists = np.array(outcome_lists)
    decision_lists = np.array(decision_lists)
    likelihoods_lists = np.array(likelihoods_lists)
    total_profit_lists = np.array(profit_lists)
    N_POINTS, L = outcome_lists.shape

    individual_profit_lists = (2 * outcome_lists - 1) * decision_lists
    accept_rate = np.mean(decision_lists, axis=0)
    predicted_success_rate = np.mean(likelihoods_lists / 100, axis=0)

    aurocs = np.zeros(L)
    aurocs_std = np.zeros(L)
    for l in range(L):
        likelihoods_list = likelihoods_lists[:, l]
        outcome_list = outcome_lists[:, l]
        auroc, var = delong_roc_variance(outcome_list, likelihoods_list)
        aurocs[l] = auroc
        aurocs_std[l] = var**0.5

    return {
        "model": model,
        "outcome_lists": outcome_lists,
        "decision_lists": decision_lists,
        "likelihoods_lists": likelihoods_lists,
        "total_profit_lists": total_profit_lists,
        "N_POINTS": N_POINTS,
        "L": L,
        "individual_profit_lists": individual_profit_lists,
        "accept_rate": accept_rate,
        "predicted_success_rate": predicted_success_rate,
        "aurocs": aurocs,
        "aurocs_std": aurocs_std,
    }

def get_optimal_payoff(likelihoods_list, outcome_list):
    """
    Compute the profit that the model would have earned if it had chosen
    on optimal decision threshold for accepting contracts.
    """
    probs_array = np.stack((likelihoods_list, 2 * outcome_list - 1))
    probs_array = probs_array[:, probs_array[0, :].argsort()[::-1]]
    profits = np.cumsum(probs_array[1, :])
    profit_array = probs_array
    profit_array[1, :] = profits
    diff = np.diff(profit_array[0, :], append=0)
    profit_array = profit_array[:, diff < 0]
    optimal_i = np.argmax(profit_array[1, :])
    optimal_thresh = profit_array[0, optimal_i]
    E_profit = profit_array[1, optimal_i] / len(likelihoods_list)
    return optimal_thresh, E_profit


def calculate_y_with_error_CP(n_AA, n_BA, N_POINTS, alpha=0.05):
    """
    Clopper-Pearson method to compute confidence intervals on
    (TPR-FPR)/2
    """

    # 1. Calculate point estimates for TPR, FPR, and ER
    TPR = n_AA / (N_POINTS / 2)
    FPR = n_BA / (N_POINTS / 2)
    Y = (TPR - FPR) / 2

    # 2. Calculate Clopper-Pearson interval for TPR
    TPR_L = beta.ppf(alpha / 2, n_AA, N_POINTS / 2 - n_AA + 1)
    TPR_U = beta.ppf(1 - alpha / 2, n_AA + 1, N_POINTS / 2 - n_AA)
    # Handle edge cases where n_TC is 0 or n_C
    TPR_L = np.where(n_AA == 0, 0.0, TPR_L)
    TPR_U = np.where(n_AA == N_POINTS / 2, 1.0, TPR_U)

    # 3. Calculate Clopper-Pearson interval for FPR
    FPR_L = beta.ppf(alpha / 2, n_BA, N_POINTS / 2 - n_BA + 1)
    FPR_U = beta.ppf(1 - alpha / 2, n_BA + 1, N_POINTS / 2 - n_BA)
    FPR_L = np.where(n_BA == 0, 0.0, FPR_L)
    FPR_U = np.where(n_BA == N_POINTS / 2, 1.0, FPR_U)

    # 4. Combine the intervals to get the CI for ER
    Y_lower = (TPR_L - FPR_U) / 2
    Y_upper = (TPR_U - FPR_L) / 2

    return Y, Y_lower, Y_upper


def plot_results(data, show_threshold_profits=True, fontsize = 8):
    """
    Creates plots as in Figure 3A
    """

    model = data["model"]
    outcome_lists = data["outcome_lists"]
    decision_lists = data["decision_lists"]
    likelihoods_lists = data["likelihoods_lists"]
    N_POINTS = data["N_POINTS"]
    L = data["L"]
    accept_rate = data["accept_rate"]
    predicted_success_rate = data["predicted_success_rate"]
    aurocs = data["aurocs"]
    aurocs_std = data["aurocs_std"]

    # Model's profit
    n_AA = np.sum(outcome_lists * decision_lists, axis=0)
    n_BA = np.sum((1 - outcome_lists) * decision_lists, axis=0)
    model_profit, model_profit_lower, model_profit_upper = calculate_y_with_error_CP(n_AA, n_BA, N_POINTS)

    # Optimal thresh profit
    optimal_T_profit = np.zeros(L)
    optimal_T_profit_upper = np.zeros(L)
    optimal_T_profit_lower = np.zeros(L)
    for l in range(L):
        likelihoods_list = likelihoods_lists[:, l]
        outcome_list = outcome_lists[:, l]
        optimal_T, E_profit = get_optimal_payoff(likelihoods_list, outcome_list)
        n_AA = np.sum((likelihoods_list >= optimal_T) * outcome_list)
        n_BA = np.sum((likelihoods_list >= optimal_T) * (1 - outcome_list))
        optimal_T_profit[l], optimal_T_profit_lower[l], optimal_T_profit_upper[l] = (
            calculate_y_with_error_CP(n_AA, n_BA, N_POINTS)
        )

    # Direct approach profit
    n_AA = np.sum((likelihoods_lists > 50) * outcome_lists, axis=0)
    n_BA = np.sum((likelihoods_lists > 50) * (1 - outcome_lists), axis=0)
    direct_profit, direct_profit_lower, direct_profit_upper = calculate_y_with_error_CP(n_AA, n_BA, N_POINTS)

    x = range(1, L + 1)

    fig, ax = plt.subplots(1, 3, figsize=(9, 2.5))
    fig.suptitle(f"Experiment 2 results for {model}", fontsize=12)
    ax[0].plot(x, aurocs, label="Area Under ROC")
    ax[0].fill_between(x, aurocs + 1.96 * aurocs_std, aurocs - 1.96 * aurocs_std, alpha=0.25)

    ax[0].plot([-1, 10], [1, 1], ":", color="k", linewidth=1)
    ax[0].text(4, 0.96, "Perfect=1", fontsize=fontsize)
    ax[0].plot([-1, 10], [0.5, 0.5], ":", color="k", linewidth=1)
    ax[0].text(4, 0.46, "Random=0.5", fontsize=fontsize)
    ax[0].set_ylim(0.4, 1)
    ax[0].set_yticks([0.5, 0.75, 1])
    ax[0].set_yticklabels(["0.5", "", "1.0"])
    ax[0].set_ylabel("AUROC", labelpad=-15)

    ax[1].plot([-1, 10], [1, 1], ":", color="k", linewidth=1)
    ax[1].plot([-1, 10], [0.5, 0.5], ":", color="k", linewidth=1)
    ax[1].text(4, 0.51, "Perfect=0.5", fontsize=fontsize)
    ax[1].plot(x, accept_rate, label="Contract acceptance rate")
    ax[1].plot(x, predicted_success_rate, label="Predicted success rate")
    ax[1].set_ylim(0, 1)
    ax[1].set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax[1].set_yticklabels(["0", "", "0.5", "", "1.0"])
    ax[1].legend(fontsize=8)

    # Profits
    ax[2].plot(x, model_profit, label="Expected profit")
    ax[2].fill_between(x, model_profit_upper, model_profit_lower, alpha=0.25)

    if show_threshold_profits:
        ax[2].plot(x, optimal_T_profit, label="E[profit] @ $T_{opt}$")
        ax[2].fill_between(x, optimal_T_profit_upper, optimal_T_profit_lower, alpha=0.25)

        ax[2].plot(x, direct_profit, label="E[profit] @ $T_{direct}$")
        ax[2].fill_between(x, direct_profit_upper, direct_profit_lower, alpha=0.25)

    ax[2].plot([-1, 10], [0, 0], ":", color="k", linewidth=1)
    ax[2].text(4, 0.46, "Perfect=0.5", fontsize=fontsize)
    ax[2].plot([-1, 10], [0.5, 0.5], ":", color="k", linewidth=1)
    ax[2].text(4, -0.04, "Random=0", fontsize=fontsize)

    ax[2].set_ylim(-0.1, 0.5)
    ax[2].set_yticks([0, 0.25, 0.5])
    ax[2].set_yticklabels(["0", "", "0.5"])
    ax[2].set_ylabel("E[profit]", labelpad=-15)

    for a in ax:
        a.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
        a.tick_params(axis="x", labelsize=10, length=3, pad=2)
        a.tick_params(axis="y", labelsize=10, length=3, pad=2)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.set_xlim(0.85, 9.3)
        a.set_xlabel("Contract number")

    plt.tight_layout()


if __name__ == "__main__":
    log_file = "example_results/claude-3-5-sonnet-20241022/FULL9.eval"
    data = get_data(log_file)
    plot_results(data, show_threshold_profits=False)
    plt.show()
