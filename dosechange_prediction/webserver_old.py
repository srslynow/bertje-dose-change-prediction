import os
import glob
import json
import logging
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F

from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer

# oneliner to select cuda if available, with a fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# load the model
app.logger.info('Loading BERT model')
model = BertForSequenceClassification.from_pretrained(os.environ.get('TYPE_PRED_MODEL')).to(device)
model.eval()

# initialize the tokenizer
app.logger.info('Loading tokenizer')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

id_to_label_map = {int(k): int(v) for k, v in model.config.id2label.items()}
label_to_id_map = {int(v): int(k) for k, v in model.config.label2id.items()}


app.logger.info('Ready')


def predict_text(text: str):
    subword_tokens = ['[CLS]'] + tokenizer.tokenize(text)
    if len(subword_tokens) > 500:
        subword_tokens = subword_tokens[:500]
    subword_tokens = subword_tokens + ['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(subword_tokens)
    # convert input to torch tensor
    x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    # create a segments tensor, we treat the entire input as sentence A -> int: 0
    s = torch.zeros_like(x, dtype=torch.long).to(device)
    # create an attention mask for separating input from padding
    # use the context slice to calculate the mask: -1's are padding
    m = torch.ones_like(x, dtype=torch.long).to(device).to(device)
    with torch.no_grad():
        # forward the data
        logits = model(x, s, m)[0]
    # # only select the first tokens of each wordpiece tokenized words
    # do a softmax to determine the confidence in each prediction
    scores = F.softmax(logits, dim=1)
    return scores


@app.route('/predict', methods=['POST'])
def predict():
    # For more information about CORS and CORS preflight requests, see
    # https://developer.mozilla.org/en-US/docs/Glossary/Preflight_request
    # for more information.

    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET,POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600',
        }
        return ('', 204, headers)
    else:
        request_json = request.get_json(silent=True)

        if 'text' in request_json:
            scores = predict_text(request_json['text'])
            if 'text_2' in request_json:
                scores_2 = predict_text(request_json['text_2'])
                scores = scores + scores_2
                scores = scores / scores.sum()
            # get the maximum prediction
            preds = torch.argmax(scores, dim=1)
            # send from gpu tensor to a (cpu) numpy array
            scores = scores.detach().to('cpu').numpy()[0].tolist()

            pred = int(preds.detach().to('cpu').numpy()[0])
            ret = {
                'status': 'success',
                'label': id_to_label_map[pred],
                'probability': float(scores[pred]),
                'full_prediction': [
                    {'label': id_to_label_map[i], 'probability': float(prob)}
                    for i, prob in enumerate(scores)
                ],
            }
        else:
            ret = {'status': 'failure', 'message': 'No text key provided'}
        # Set CORS headers for the main request
        headers = {'Access-Control-Allow-Origin': '*', 'Content-Type': 'application/json'}

        return (json.dumps(ret), 200, headers)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8088, debug=False)
