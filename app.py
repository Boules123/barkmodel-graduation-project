from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
import uuid
# from bark_model import bark_model  # import your class
import os

app = FastAPI()
bark = bark_model()  # load once

@app.post("/generate/")
async def generate(text: str = Form(...)):
    audio_array, sr = bark.generate_audio(text)
    file_name = f"output_{uuid.uuid4().hex}.wav"
    bark.save_audio(audio_array, sr, file_name)
    return FileResponse(file_name, media_type="audio/wav", filename=file_name)
