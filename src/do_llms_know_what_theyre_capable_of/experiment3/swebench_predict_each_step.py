from inspect_ai import eval
from do_llms_know_what_theyre_capable_of.experiment3.utils.swebench_predict_task import swebench_predict_task


def run_swebench_predict(model, log_dir, limit=None):

    # if using an o-series model, name python tool `python3`
    if model.startswith("openai/o"):
        python_tool_name = "python3"
    else:
        python_tool_name = "python"

    kwargs = {"python_tool_name":python_tool_name, "tool_call_limit": 70, "message_limit": 1000}
    config = {"fail_on_error": 10} # Number of failed samples after which the eval fails

    if model=="anthropic/claude-3-7-sonnet-20250219":
        reasoning_tokens = 4096 # Our paper uses both 0 and 4096
        if reasoning_tokens > 0:
            config["reasoning_tokens"] = reasoning_tokens

    # Uncomment the line below to specify specific tasks to run:
    # kwargs["instance_ids"] = ["sphinx-doc__sphinx-7889"]
    kwargs["exclude_ids"] = ["django__django-15278"] # The docker image for this sample doesn't work
    task = swebench_predict_task(**kwargs)
    eval(task, model=model, log_dir=log_dir, limit=limit, **config)

if __name__=="__main__":
    model = "openai/gpt-4.1-nano"
    log_dir = "gpt-41-nano/"
    run_swebench_predict(model, log_dir, limit=1)
