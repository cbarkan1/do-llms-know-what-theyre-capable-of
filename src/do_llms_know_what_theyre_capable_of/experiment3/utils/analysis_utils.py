import numpy as np
import scipy.stats
from do_llms_know_what_theyre_capable_of._vendor.roc_comparison.compare_auc_delong_xu import delong_roc_variance, compute_ground_truth_statistics, fastDeLong


def get_self_assess_data(sample):
    """
    Extract model's predictions from messages history
    """
    def float_or_none(x):
        if x=="None":
            print("Caught None")
            return None
        try:
            return float(x)
        except:
            raise ValueError(f"value {x} of type {type(x)} cannot be converted to float")

    data_message = sample.messages[-1]
    content = data_message.content
    assert content.startswith("SELF_ASSESS_DATA:")
    # Remove prefix and split into lines
    lines = content[len("SELF_ASSESS_DATA:"):].strip().split("\n")
    # Parse each line into key-value pairs
    self_assess_data = {}
    for line in lines:
        key, value = line.split(": ", 1)
        # Convert values to appropriate types
        if key == "likelihood_list":
            # Parse list of floats
            value = [float_or_none(x) for x in value.strip("[]").split(", ") if x]
        elif key in ["tool_call_count", "tool_call_limit"]:
            value = int(value)
        elif key == "final_likelihood":
            value = float_or_none(value)
        self_assess_data[key] = value
    return self_assess_data


def get_data(log):
    def is_valid(sample):
        is_scored = "swe_bench_scorer" in sample.scores
        has_self_assess_data = sample.messages[-1].content.startswith("SELF_ASSESS_DATA:")
        return is_scored, has_self_assess_data
    
    C_likelihood_lists = []
    I_likelihood_lists = []
    C_final_likelihoods = []
    I_final_likelihoods = []
    I_tool_call_count = []
    C_tool_call_count = []
    I_ids = []
    C_ids = []
    tool_call_limit = None
    for s in log.samples:
        is_scored, has_self_assess_data = is_valid(s)
        if is_scored and has_self_assess_data:
            self_assess_data = get_self_assess_data(s)
            score = s.scores["swe_bench_scorer"].value
            if score==1.0:
                C_likelihood_lists.append(self_assess_data["likelihood_list"])
                C_final_likelihoods.append(self_assess_data["final_likelihood"])
                C_tool_call_count.append(self_assess_data["tool_call_count"])
                C_ids.append(s.id)
            else:
                I_likelihood_lists.append(self_assess_data["likelihood_list"])
                I_final_likelihoods.append(self_assess_data["final_likelihood"])
                I_tool_call_count.append(self_assess_data["tool_call_count"])
                I_ids.append(s.id)
        else: # sample error or message limit
            pass
    
    tool_call_limit = self_assess_data["tool_call_limit"]
    return I_likelihood_lists, C_likelihood_lists, I_tool_call_count, C_tool_call_count, I_final_likelihoods, C_final_likelihoods, I_ids, C_ids, tool_call_limit


def numpy_likelihoods(likelihood_lists, tool_call_limit):
    """
    Converts list of likelihood lists to numpy array, with handling of likelihood
    lists of length < tool_call_limit
    """
    num_lists = len(likelihood_lists)
    # Every element of likelihoods should get overwritten in the for loop.
    # I've set the initial values to -1 so it will be obvious if something is not overwritten.
    likelihoods = -1*np.ones((num_lists, tool_call_limit)) 
    for i in range(num_lists):
        likelihoods[i,0:len(likelihood_lists[i])] = likelihood_lists[i][:]
        likelihoods[i,len(likelihood_lists[i]):] = likelihood_lists[i][-1]
    return likelihoods


def auroc_diff_CI(ground_truth, predictions_one, predictions_two, confidence_level=0.95):
    """
    Computes the confidence interval for the difference between two correlated ROC AUCs.
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model (e.g., at t-1),
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model (e.g., at t),
          np.array of floats of the probability of being class 1
       confidence_level: The confidence level for the interval (e.g., 0.95 for 95% CI)
    Returns:
       tuple: (auc_difference, (lower_bound, upper_bound))
              auc_difference = AUC(predictions_two) - AUC(predictions_one)
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    
    auc1 = aucs[0]  # AUC for predictions_one
    auc2 = aucs[1]  # AUC for predictions_two
    
    delta_auc = auc2 - auc1
    
    # Variance of the difference: Var(AUC2 - AUC1) = Var(AUC1) + Var(AUC2) - 2 * Cov(AUC1, AUC2)
    var_auc1 = delongcov[0,0]
    var_auc2 = delongcov[1,1]
    # Covariance is delongcov[0,1] or delongcov[1,0]
    cov_auc1_auc2 = delongcov[0,1] 
    
    var_delta_auc = var_auc1 + var_auc2 - 2 * cov_auc1_auc2

    se_delta_auc = np.sqrt(var_delta_auc)
    
    # Z-score for the desired confidence level (e.g., 1.96 for 95%)
    alpha = 1 - confidence_level
    z_critical = scipy.stats.norm.ppf(1 - alpha / 2)
    
    lower_bound = delta_auc - z_critical * se_delta_auc
    upper_bound = delta_auc + z_critical * se_delta_auc
    
    return delta_auc, (lower_bound, upper_bound)


def auroc_with_delong(C_likelihoods, I_likelihoods, C_final_likelihoods, I_final_likelihoods, tool_call_limit, start=0):
    """
    returns:
        aucs: E[auc] for each step
        bounds: 95% CI on the *change* in auc for each step relative to the start step
        auc0: auc for the start step
        std0: standard deviation for the start step
    """
    predictions0 = np.concatenate((C_likelihoods[:,start],I_likelihoods[:,start]))
    num_C, num_I = C_likelihoods.shape[0], I_likelihoods.shape[0]
    outcomes = np.concatenate((np.ones(num_C),np.zeros(num_I)))
    bounds = np.zeros((2, tool_call_limit))*np.nan
    aucs = np.zeros(tool_call_limit)*np.nan
    auc0,var0 = delong_roc_variance(outcomes, predictions0)
    aucs[0] = auc0
    bounds[:,start] = [0,0]
    std0 = np.sqrt(var0)
    for step in range(start+1,tool_call_limit):
        predictions_at_step = np.concatenate((C_likelihoods[:,step],I_likelihoods[:,step]))
        dif, bound = auroc_diff_CI(outcomes, predictions0, predictions_at_step, confidence_level=0.95)
        aucs[step] = auc0 + dif
        bounds[:,step] = bound[:]

    final_predictions = np.concatenate((C_final_likelihoods, I_final_likelihoods))
    final_absolute_auc, final_absolute_var = delong_roc_variance(outcomes, final_predictions)
    final_absolute_std = np.sqrt(final_absolute_var)
    start_to_final_dif, start_to_final_bound = auroc_diff_CI(outcomes, predictions0, final_predictions, confidence_level=0.95)

    return aucs, bounds, auc0, std0, final_absolute_auc, final_absolute_std, start_to_final_dif, start_to_final_bound
