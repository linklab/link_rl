from codes.c_models.base_model import BaseModel


class DiscreteActionModel(BaseModel):
    def __init__(self, worker_id, params, device):
        super(DiscreteActionModel, self).__init__(worker_id, params, device)
