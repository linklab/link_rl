from a_configuration.parameter_preamble import *

###############
## CART_POLE ##
###############
# class Parameter(ParameterCartPoleDqn):
#     def __init__(self):
#         super(Parameter, self).__init__()
#         self.USE_WANDB = False
#         self.WANDB_ENTITY = "link-koreatech"

# class Parameter(ParameterCartPoleReinforce):
#     def __init__(self):
#         super(Parameter, self).__init__()
#         self.USE_WANDB = False
#         self.WANDB_ENTITY = "link-koreatech"
#

# class Parameter(ParameterCartPoleA2c):
#     def __init__(self):
#         super(Parameter, self).__init__()
#         self.USE_WANDB = False
#         self.WANDB_ENTITY = "link-koreatech"


######################
## CART_POLE_BULLET ##
######################
# class Parameter(ParameterCartPoleBulletDqn):
#     def __init__(self):
#         super(Parameter, self).__init__()
#         self.USE_WANDB = True
#         self.WANDB_ENTITY = "link-koreatech"


######################
## ANT_BULLET ##
######################
class Parameter(ParameterAntBulletA2c):
    def __init__(self):
        super(Parameter, self).__init__()
        self.USE_WANDB = False
        self.WANDB_ENTITY = "link-koreatech"


##########
## PONG ##
##########
# class Parameter(ParameterPongDqn):
#     def __init__(self):
#         super(Parameter, self).__init__()
#         self.USE_WANDB = False
#         self.WANDB_ENTITY = "link-koreatech"


# class Parameter(ParameterPongA2c):
#     def __init__(self):
#         super(Parameter, self).__init__()
#         self.USE_WANDB = False
#         self.WANDB_ENTITY = "link-koreatech"


if __name__ == "__main__":
    parameter = Parameter()
    print_basic_info(device=None, parameter=parameter)