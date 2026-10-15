import importlib


def get_environment(environment_name):
    if environment_name.startswith("go2"):
        module = "go2"
    elif environment_name.startswith("crazyflow_"):
        module = "crazyflow"
    else:
        module = "jaxgcrl"

    return importlib.import_module(f"chronosrl.environments.{module}")