import joblib
import os
import json
import numpy as np

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "REPLACE_WITH_MODEL_BASE_NAME.joblib"))
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return np.array(input_data["data"])
    else:
        raise ValueError("This model only supports application/json input")

def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return prediction.tolist()

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps({"predictions": prediction})
    else:
        raise ValueError("This model only supports application/json output")
