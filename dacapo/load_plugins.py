import pkgutil

import dacapo.plugins


def import_plugins(name_space):
    discovered_plugins = [
        name
        for finder, name, ispkg in pkgutil.iter_modules(
            dacapo.plugins.__path__, dacapo.plugins.__name__ + "."
        )
    ]

    for name in discovered_plugins:
        print(f"Importing {name}")
        name_space[name.split(".")[0]] = __import__(name, globals={"__name__": __name__})
