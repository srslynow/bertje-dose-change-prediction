from os import name
from typing import Optional, List
from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str
    title: Optional[str]


class ArticleClassPrediction(BaseModel):
    id: int
    name: str
    selected: bool
    probability: float


class PredictResponse(BaseModel):
    main_prediction: ArticleClassPrediction
    all_predictions: List[ArticleClassPrediction]
