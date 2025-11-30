from inspect_ai.dataset._dataset import MemoryDataset
from inspect_ai.dataset import Sample
from inspect_ai import Task, task
from inspect_ai import eval
from do_llms_know_what_theyre_capable_of.experiment1.utils.enforce_percent_solver import enforce_percent_solver, percentage
from do_llms_know_what_theyre_capable_of.experiment1.utils.my_big_code_bench import get_dataset

def run_prediction(model, log_dir, limit=None):
    bcb_dataset = get_dataset()
    samples_list = []
    for sample in bcb_dataset:
        prompt = (
            "The code below shows the beginning of a python function, with imports, a function signature, and a docstring that describes "
            "what the function should do. Please generate the full code, including the needed import statements and the complete function.\n\n"
            f"```python\n{sample.input}\n```"
        )
        samples_list.append(Sample(input = prompt, id=sample.id))
    dataset = MemoryDataset(samples=samples_list)

    @task
    def enforce_percent_task(dataset, scorer) -> Task:
        return Task(dataset=dataset,solver=enforce_percent_solver(),scorer=scorer)

    scorer = percentage()
    config = {"max_tokens": 25} # Cut off long responses (we really only need 1 or 2 tokens).
    predict_task = enforce_percent_task(dataset, scorer)

    eval(predict_task, model=model, log_dir=log_dir, limit=limit, **config)

if __name__=="__main__":
    model = "openai/gpt-4o"
    log_dir = "gpt-4o/predict/"
    run_prediction(model, log_dir, limit=None)