from codes.c_models.base_model import BaseModel


class ContinuousActionModel(BaseModel):
    def __init__(self, worker_id, params, device):
        super(ContinuousActionModel, self).__init__(worker_id, params, device)
