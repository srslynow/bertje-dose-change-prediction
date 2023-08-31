import os

from fastapi import FastAPI

from transformers import LongformerConfig

from ner.config import device
from type_prediction.serializers import PredictRequest, PredictResponse
from type_prediction.model_pt_lightning import CaracalArticleTypePrediction

app = FastAPI()

model_path = os.environ.get('TYPE_PRED_MODEL')
# model_path = '/data/caracal-article-type/models/bert/20210702_1642'

# load config from the given path
config = LongformerConfig.from_pretrained(model_path)
# load model parameters from the given model path and pass the config for the metadata
model: CaracalArticleTypePrediction = CaracalArticleTypePrediction.from_pretrained(
    model_path, config=config
).to(device)
model.eval()


@app.post("/predict", response_model=PredictResponse)
def predict(predict_request: PredictRequest):
    main_prediction, subtype_prediction, all_predictions = model.predict(
        predict_request.text, predict_request.title
    )
    predict_response = PredictResponse(
        main_prediction=main_prediction,
        subtype_prediction=subtype_prediction,
        prediction=all_predictions,
    )
    return predict_response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8088)
