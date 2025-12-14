import json
import pickle

# Load model - it's a tuple of (DictVectorizer, Model)
with open('model_C=1.0.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


def predict_single(customer):
    # Transform customer data using DictVectorizer, then predict
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    return float(y_pred)


def lambda_handler(event, context):
    print("Parameters:", event)
    
    # Handle both API Gateway and direct invocation
    if 'body' in event:
        body = json.loads(event['body'])
        customer = body['customer']
    else:
        customer = event['customer']
    
    prob = predict_single(customer)
    
    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }