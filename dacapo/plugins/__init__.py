import importlib

# import local plugin
globals()["local_plugin"] = importlib.import_module("local")
