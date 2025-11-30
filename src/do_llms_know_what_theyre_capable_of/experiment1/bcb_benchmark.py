from do_llms_know_what_theyre_capable_of.experiment1.utils.my_big_code_bench import bigcodebench, get_dataset
from inspect_ai import eval

def run_benchmark(model, log_dir, limit=None):
    config = {"temperature":0.5}
    dataset = get_dataset()
    eval(bigcodebench(dataset), model=model, limit=limit, log_dir=log_dir, **config)

if __name__=="__main__":
    model = "openai/gpt-4o"
    log_dir = "gpt-4o/benchmark/"
    run_benchmark(model, log_dir, limit=None)
