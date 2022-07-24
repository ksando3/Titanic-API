from predict import predict, load_model
from fastapi import FastAPI

app = FastAPI()

@app.get("/get-prediction")
def get_prediction(Pclass, SibSp, Parch, Sex_female, Sex_male):
    print("Hello")
    model = load_model()
    data = {"Pclass": [Pclass], "SibSp": [SibSp], "Parch": [Parch], "Sex_female": [Sex_female], "Sex_male": [Sex_male]}
    prediction = int(predict(model, data))
    if prediction == 1:
        return {"Prediction": "Survived"}
    else:
        return {"Prediction": "rip bozo"}   
    
