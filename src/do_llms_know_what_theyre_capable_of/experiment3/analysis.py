import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from inspect_ai.log import read_eval_log
from do_llms_know_what_theyre_capable_of.experiment3.utils.analysis_utils import get_data, numpy_likelihoods, auroc_with_delong


def plot_AUROC_step_by_step(log_file_dir):
    # Load eval logs
    log_file_dir = Path(log_file_dir)
    log_files = list(log_file_dir.glob("*.eval"))
    assert len(log_files)>=1, "No eval logs found."
    print("Loading eval files... (this might take a minute or two)")
    list_of_logs = [read_eval_log(log_file) for log_file in log_files]
    print(f"Loaded {len(list_of_logs)} .eval files")
    model = list_of_logs[0].eval.model
    assert all(log.eval.model == model for log in list_of_logs[1:])

    # Extract data from logs
    I_likelihood_lists, C_likelihood_lists, I_tool_call_count, C_tool_call_count, I_final_likelihoods, C_final_likelihoods, I_ids, C_ids, tool_call_limit = get_data(list_of_logs[0])
    for log in list_of_logs[1:]:
        I_likelihood_lists_i, C_likelihood_lists_i, I_tool_call_count_i, C_tool_call_count_i, I_final_likelihoods_i, C_final_likelihoods_i, I_ids_i, C_ids_i, tool_call_limit_i = get_data(log)
        assert tool_call_limit_i==tool_call_limit
        I_likelihood_lists.extend(I_likelihood_lists_i)
        C_likelihood_lists.extend(C_likelihood_lists_i)
        I_tool_call_count.extend(I_tool_call_count_i)
        C_tool_call_count.extend(C_tool_call_count_i)
        I_final_likelihoods.extend(I_final_likelihoods_i)
        C_final_likelihoods.extend(C_final_likelihoods_i)
        I_ids.extend(I_ids_i)
        C_ids.extend(C_ids_i)

    num_I = len(I_likelihood_lists)
    num_C = len(C_likelihood_lists)
    num_samples = num_I + num_C
    I_probs = numpy_likelihoods(I_likelihood_lists, tool_call_limit)/100
    C_probs = numpy_likelihoods(C_likelihood_lists, tool_call_limit)/100
    aucs, bounds, auc0, std0, final_absolute_auc, final_absolute_std, start_to_final_dif, start_to_final_bound = auroc_with_delong(C_probs, I_probs, C_final_likelihoods, I_final_likelihoods, tool_call_limit)

    # Print data in Figure 4B
    print("Model: ", model)
    print(f"Initial AUROC: {auc0} ± {1.96*std0} (95% CI, DeLong's method)")
    print(f"Final AUROC: {final_absolute_auc} ± {1.96*final_absolute_std} (95% CI, DeLong's method)")

    # Plot data as in Figure 4A
    fraction_predict_yes = (np.sum(I_probs>.5, axis=0) + np.sum(C_probs>.5, axis=0))/num_samples
    predicted_accuracy = (np.sum(I_probs, axis=0) + np.sum(C_probs, axis=0))/num_samples
    steps = range(1,71)
    plt.plot(steps, predicted_accuracy)
    plt.ylim(0,1)
    plt.xlabel("Step")
    plt.ylabel("Predicted success rate")
    plt.title(f"Experiment 3: Predicted success rate for {model}")

    # Plot data as in Figure 4C
    plt.figure()
    ax = plt.gca()
    ax.plot(steps,aucs-auc0, color="k")
    ax.errorbar(73,start_to_final_dif,start_to_final_bound[1]-start_to_final_dif,capsize=5, marker='s', color="k")
    ax.fill_between(steps,bounds[0,:],bounds[1,:],alpha=0.25)

    ax.set_ylim(-.17,.16)
    ax.set_xlim(-2,76)
    ax.plot([-20,70],[0,0],color="k",linestyle="--", linewidth=.8)
    ax.set_xticks(range(0,75,5))
    ax.set_yticks([-.1,0,.1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels([0,'','','','',25,'','','','',50,'','','',70])
    ax.set_title(f"Experiment 3: Step-by-step change in AUROC\nfor {model}")
    ax.set_xlabel("step")
    ax.set_ylabel("Change in AUROC\nrelative to step 1")
    plt.show()


if __name__=="__main__":
    log_file_dir = Path(__file__).parent / "example_results" / "gpt-41"
    plot_AUROC_step_by_step(log_file_dir)
