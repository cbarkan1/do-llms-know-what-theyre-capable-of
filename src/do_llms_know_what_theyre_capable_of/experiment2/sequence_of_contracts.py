from inspect_ai.dataset._dataset import MemoryDataset
from inspect_ai.dataset import Sample
from inspect_ai import Task
import json
from inspect_ai.log import read_eval_log
from do_llms_know_what_theyre_capable_of.experiment2.utils.soc_solver import soc_solver
from inspect_ai import eval
from pathlib import Path

def load_dataset(model: str, bcb_benchmark_log_dir: str):
    utils_dir = Path(__file__).resolve().parent / "utils"
    # Load samples:
    with open(utils_dir / "soc_samples_dict.json", "r") as f:
        soc_samples_dict = json.load(f)
    C_samples = soc_samples_dict["C_samples"]
    I_samples = soc_samples_dict["I_samples"]

    # Load indices
    with open(utils_dir / "soc_all_L9.json", "r") as f:
        sequences = json.load(f)

    benchmark_dir = Path(bcb_benchmark_log_dir)
    if not benchmark_dir.is_absolute():
        benchmark_dir = (Path(__file__).resolve().parent / benchmark_dir).resolve()
    benchmark_filepaths = list(benchmark_dir.glob("*.eval"))

    benchmark_samples = []
    for filepath in benchmark_filepaths:
        benchmark_log = read_eval_log(str(filepath))
        assert isinstance(benchmark_log.samples, list)
        if model==benchmark_log.eval.model:
            benchmark_samples.extend(benchmark_log.samples)
    if len(benchmark_samples)==0:
        raise RuntimeError(f"No BCB benchmark samples found for {model}.")
    elif len(benchmark_samples)!=1140:
        print(f"WARNING: {len(benchmark_samples)} BCB benchmark samples were loaded, but BCB has 1140 samples.")

    def find_benchmark_response(task: dict):
        for sample in benchmark_samples:
            # Extract BCB question from surrounding text: 
            if sample.input==task["question"][1][273:-251]:
                break
        else:
            raise RuntimeError("Match not found!")
        
        assert sample.scores["verify"].value==task["outcome"]
        response = sample.messages[-1].content
        if isinstance(response,list): # For reasoning models
            print(len(response))
            response = response[-1].text
        return response

    samples_list = []
    for i,sequence in enumerate(sequences):
        task_sequence = []
        for primitive_task in sequence:
            task = {"outcome": primitive_task["outcome"], "index": primitive_task["index"]}
            task["question"] = C_samples[task["index"]] if task["outcome"]=="C" else  I_samples[task["index"]]
            task["response"] = find_benchmark_response(task)
            task_sequence.append(task)
        task_sequence_as_string = json.dumps(task_sequence)
        samples_list.append(Sample(input=task_sequence_as_string, id=i))
    dataset = MemoryDataset(samples=samples_list)
    return dataset

def run_sequence_of_contracts(bcb_benchmark_log_dir: str, model: str, log_dir: str, limit=None):
    """
    Runs the sequence-of-contracts experiment (Experiment 2), where the model's contract submissions 
    are taken from its responses on the BCB benchmark. These responses are loaded from all benchmark 
    logs in `bcb_benchmark_log_dir` for which `bcb_benchmark_log.eval.model` matches `model`.

    Args:
        bcb_benchmark_log_dir (str): Path (relative or absolute) to directory containing results of `experiment1/bcb_benchmark.py`.
        model (str): Model ID which must exactly match `bcb_benchmark_log.eval.model` for each 
            bcb_benchmark_log in bcb_benchmark_log_dir.
    """
    dataset = load_dataset(model, bcb_benchmark_log_dir)
    contractor_task = Task(dataset=dataset, solver=soc_solver())
    eval(contractor_task, model=model, log_dir=log_dir, limit=limit)

if __name__=="__main__":

    # Directory with logs from `experiment1/bcb_benchmark.py`
    # Path can be relative or absolute
    bcb_benchmark_log_dir = "bcb_benchmark_logs" 

    # model must exactly match bcb_benchmark_log.eval.model
    model = "openai/gpt-4.1-mini-2025-04-14"

    log_dir = "gpt-41-mini"
    run_sequence_of_contracts(bcb_benchmark_log_dir, model, log_dir, limit=None)
