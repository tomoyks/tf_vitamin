from pydantic import BaseModel, Field
from typing import List


class ImageConfig(BaseModel):
    orig_shape: List[int] = Field([], max_items=3)
    reshape_size: List[int] = Field([], max_items=2)
