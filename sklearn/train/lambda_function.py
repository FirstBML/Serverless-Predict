import pickle

with open('model_C=1.0.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    prob = model.predict_proba([customer])[0, 1]
    return float(prob)

def lambda_handler(event, context):    
    print("Parameters:", event)
    customer = event['customer']
    prob = predict_single(customer)
    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }