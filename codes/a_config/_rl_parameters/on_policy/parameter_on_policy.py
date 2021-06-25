import enum


class OnPolicyActionType(enum.Enum):
    ACTION_STD = 0
    ACTION_STD_DECAY = 1


class PARAMETERS_ON_POLICY:
    TYPE_OF_ON_POLICY_ACTION = OnPolicyActionType.ACTION_STD_DECAY

