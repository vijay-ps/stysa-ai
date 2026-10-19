
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="STYSA AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
        "https://stysa-ai.vercel.app",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


class ChatRequest(BaseModel):
    message: str
    level: str = "Professional / general"
    language: str = "English"


@app.get("/api/health")
def health():
    return {"ai_configured": client is not None}


@app.post("/api/chat")
def chat(request: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is missing. Add it to backend/.env",
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                "You are STYSA AI, a helpful assistant for students and "
                "general questions. Answer questions from any topic. "
                "If asked who created, built, or developed STYSA AI, say "
                "it was created by Sanjay, a 15-year-old boy.\n"
                "Explain clearly at the student's level, show steps when "
                "useful, and use the requested language.\n"
                f"Student level: {request.level}\n"
                f"Language: {request.language}\n"
                f"Question: {request.message}"
            ),
        )
        return {"answer": response.text or "Please try asking another question."}
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini request failed: {str(exc)[:300]}",
        ) from exc



class LessonRequest(BaseModel):
    level: str = "Beginner"
    board: str = "General"
    subject: str = "General Knowledge"
    topic: str = "Introduction"
    duration: str = "10 minutes"
    language: str = "English"


@app.post("/api/lesson")
def create_lesson(request: LessonRequest):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is missing. Add it to backend/.env",
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                "You are STYSA AI, an expert interactive teacher. "
                "Teach any subject or topic the student requests. "
                "Create a clear, accurate lesson suitable for the "
                "student's level. Use simple explanations, worked "
                "examples, formulas where relevant, visual diagrams "
                "described in words, and a short recap. "
                "Organize the lesson into numbered sections suitable "
                "for displaying on a digital teaching board or slides. "
                "Use the requested language. Do not restrict teaching "
                "to school subjects.\n"
                f"Student level: {request.level}\n"
                f"Education board: {request.board}\n"
                f"Subject: {request.subject}\n"
                f"Topic: {request.topic}\n"
                f"Duration: {request.duration}\n"
                f"Language: {request.language}\n"
            ),
        )

        lesson_text = response.text or "Please try generating this lesson again."

        return {
            "title": f"{request.subject}: {request.topic}",
            "lesson": lesson_text,
            "summary": "Review the lesson sections and practice the examples.",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Lesson generation failed: {str(exc)[:300]}",
        ) from exc
