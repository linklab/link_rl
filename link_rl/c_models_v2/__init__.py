from .a_model import model_registry as model_creators

import os

working_path = os.path.dirname(os.path.realpath(__file__))
file_list = os.listdir(working_path)
module_list = [
    file.replace(".py", "")
    for file in file_list
    if file.endswith(".py")
    and file.replace(".py", "") not in ["__init__",
                                        "base",
                                        "head",
                                        "utils"]
]

# naming_rule = lambda x: re.sub("([a-z])([A-Z])", r"\1_\2", x).lower()
for module_name in module_list:
    module_path = f"{__name__}.{module_name}"
    __import__(module_path)
    # for class_name, _class in inspect.getmembers(module, inspect.isclass):
    #     if module_path in str(_class):
    #         network_dict[naming_rule(class_name)] = _class
