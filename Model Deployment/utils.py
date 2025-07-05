import xgboost as xgb
import numpy as np
from skops.io import load, get_untrusted_types
import os

def load_model():
    base_path = os.path.dirname(__file__) 

    model_path = os.path.join(base_path, "xgb_coupon_model.json")
    preprocessor_path = os.path.join(base_path, "xgb_coupon_preprocessor.skops")
    threshold_path = os.path.join(base_path, "threshold.txt")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    untrusted = get_untrusted_types(preprocessor_path)
    preprocessor = load(preprocessor_path, trusted=untrusted)

    with open(threshold_path, "r") as f:
        threshold = float(f.read().strip())

    return model, preprocessor, threshold


def make_prediction(model, preprocessor, threshold, user_input):
    import pandas as pd

    input_df = pd.DataFrame([user_input])
    processed_input = preprocessor.transform(input_df)
    prob = model.predict_proba(processed_input)[0][1]

    prediction = prob > threshold
    return prediction, prob


def explain_prediction(model, preprocessor, user_input):
    import shap
    import pandas as pd

    input_df = pd.DataFrame([user_input])
    processed_input = preprocessor.transform(input_df)

    explainer = shap.Explainer(model.predict, processed_input)
    shap_values = explainer(processed_input)

    return shap_values, processed_input