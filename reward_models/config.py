from pydantic import BaseModel


class Config(BaseModel):
    base_model: str
    dataset_name: str
    limit: int
    max_length: int
    batch_size: int
    lr: float
    epochs: int
