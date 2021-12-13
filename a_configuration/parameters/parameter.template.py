from a_configuration.parameters.parameter_preamble import *

###############
## CART_POLE ##
###############
# class Parameter(ParameterCartPoleDqn):
#     USE_WANDB = False
#     WANDB_ENTITY = "link-koreatech"

# class Parameter(ParameterCartPoleReinforce):
#     USE_WANDB = False
#     WANDB_ENTITY = "link-koreatech"

# class Parameter(ParameterCartPoleA2c):
#     USE_WANDB = False
#     WANDB_ENTITY = "link-koreatech"


##########
## PONG ##
##########
class Parameter(ParameterPongDqn):
    USE_WANDB = False
    WANDB_ENTITY = "link-koreatech"


if __name__ == "__main__":
    parameter = Parameter()
    print_basic_info(device=None, parameter=parameter)
