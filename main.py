from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load MongoDB details securely (use environment variables or .env in production)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://test:<ZqwvTB2YIrA6V5EO>@cluster0.be0jwqg.mongodb.net/")
DB_NAME = os.getenv("DB_NAME", "chatbot_db")

app = FastAPI(title="EEG Chatbot Backend")

# Connect to MongoDB Atlas with motor (async driver)
client = AsyncIOMotorClient(MONGODB_URI)
db = client[DB_NAME]

# Load EEG CNN-LSTM model once during startup
model = load_model("eeg_cnn_lstm_model.h5")

# Expected EEG input length (adjust based on your training)
EEG_INPUT_LENGTH = 512

# Pydantic schemas for request/response

class EEGReport(BaseModel):
    raw_values: List[float]

class ChatRequest(BaseModel):
    user_id: str
    eeg_report: EEGReport
    message: Optional[str] = None  # Optional user message

class ChatResponse(BaseModel):
    response_text: str
    prediction_class: str
    user_id: str

# Helper function to preprocess EEG input for model
def preprocess_eeg(raw_values: List[float]):
    # Normalize input (min-max 0-1 scaling) same as training scaler
    arr = np.array(raw_values)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    if len(arr) < EEG_INPUT_LENGTH:
        # Pad with zeros if shorter
        arr = np.pad(arr, (0, EEG_INPUT_LENGTH - len(arr)), mode='constant')
    else:
        # Truncate if longer
        arr = arr[:EEG_INPUT_LENGTH]
    return arr.reshape(1, EEG_INPUT_LENGTH, 1)

# Map model output index to label names, sync with your model training
LABEL_MAP = {0: "blink", 1: "math"}  # adjust this based on your LabelEncoder classes

@app.post("/predict/", response_model=ChatResponse)
async def predict(chat_request: ChatRequest):
    # Preprocess EEG data
    try:
        input_data = preprocess_eeg(chat_request.eeg_report.raw_values)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing EEG data: {e}")

    # Run prediction
    preds = model.predict(input_data)
    pred_index = preds.argmax(axis=1)[0]
    pred_class = LABEL_MAP.get(pred_index, "unknown")

    # Generate a simple chatbot reply based on prediction (customize as needed)
    if pred_class == "blink":
        response_text = "Your EEG indicates blinking activity. How can I assist you further?"
    elif pred_class == "math":
        response_text = "Your EEG shows math-related brain activity. Would you like to explore math problems?"
    else:
        response_text = "I'm unable to interpret your EEG report clearly. Please try again."

    # Save chat and prediction to MongoDB
    chat_doc = {
        "user_id": chat_request.user_id,
        "eeg_report": chat_request.eeg_report.raw_values,
        "prediction_class": pred_class,
        "response_text": response_text,
        "message": chat_request.message,
        "timestamp":  __import__("datetime").datetime.utcnow()
    }
    await db.chats.insert_one(chat_doc)

    return ChatResponse(
        response_text=response_text,
        prediction_class=pred_class,
        user_id=chat_request.user_id
    )

@app.get("/history/{user_id}", response_model=List[ChatResponse])
async def get_chat_history(user_id: str):
    # Retrieve chat history for user
    cursor = db.chats.find({"user_id": user_id}).sort("timestamp", -1).limit(20)
    chats = []
    async for doc in cursor:
        chats.append(ChatResponse(
            response_text=doc["response_text"],
            prediction_class=doc["prediction_class"],
            user_id=doc["user_id"]
        ))
    return chats

# Root endpoint for sanity check
@app.get("/")
async def root():
    return {"message": "EEG Chatbot Backend is running."}
