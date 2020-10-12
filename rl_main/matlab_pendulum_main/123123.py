from line_profiler import LineProfiler

def a():

    print("@")
    print("#")


lp = LineProfiler()
lp_wrapper = lp(a)
lp_wrapper()
lp.print_stats()