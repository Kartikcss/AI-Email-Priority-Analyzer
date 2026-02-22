from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

class EmailRequest(BaseModel):
    text: str

def generate_explanation(text):
    text = text.lower()

    if any(word in text for word in ["urgent", "asap", "deadline"]):
        return "Contains urgent keywords"
    elif any(word in text for word in ["meeting", "review", "discuss"]):
        return "General work-related content"
    elif any(word in text for word in ["offer", "discount", "newsletter"]):
        return "Promotional or low priority content"
    else:
        return "No strong indicators found"
    

    
@app.get("/")
def read_root():
    return FileResponse("static/index.html")



@app.post("/predict")
def predict(data: EmailRequest):
    vec = vectorizer.transform([data.text])

    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = round(max(prob) * 100, 2)

    return {
        "priority": pred.upper(),
        "confidence": confidence,
        "explanation": generate_explanation(data.text)
    }



