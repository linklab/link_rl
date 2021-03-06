import inspect
import re


def show_tensor_info(*tensors):
    this_function_name = inspect.currentframe().f_code.co_name
    lcls = inspect.stack()

    outer = re.compile("\((.+)\)")

    arg_names = None
    for lcl in lcls:
        if lcl.code_context[0].startswith(this_function_name):
            w = outer.search(lcl.code_context[0].split(this_function_name)[1])
            arg_names = w.group(1).split(", ")

    for idx, arg_name in enumerate(arg_names):
        print("# {0}.shape: {1}".format(arg_name, tensors[idx].shape))