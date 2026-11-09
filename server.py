"""
Pinch: the voice agent server.

The browser sends what you said. Gemini decides which canvas "tool" to call
(add a note, draw a concept map, draw a chart, make flashcards, clear the canvas).
The browser then draws whatever Gemini chose.

This is "tool calling" (also called function calling). It is the same pattern the original
team used with the OpenAI Agent SDK, and it is one of the most useful patterns to know for
Forward Deployed Engineering interviews: the model does not draw anything itself, it picks an
action and fills in the details, and YOUR code does the action.

Why a server at all? So your API key stays on your computer and never sits in the web page.

Run:
    pip install -r requirements.txt
    export GEMINI_API_KEY="your-key"     (skip this to use mock mode)
    python server.py
    open http://localhost:8000 in Chrome
"""

import json
import os
import re
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
app = FastAPI()

# ---------------------------------------------------------------------------
# The tools Gemini is allowed to call. Each one is a JSON description of a function.
# Gemini reads these descriptions to decide what to do, so clear descriptions matter a lot.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "add_note",
        "description": "Put a short sticky note on the canvas. Use for definitions, answers, summaries, or examples.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "A 2 to 5 word heading"},
                "text": {"type": "string", "description": "The note body, under 60 words"},
            },
            "required": ["title", "text"],
        },
    },
    {
        "name": "draw_concept_map",
        "description": "Draw a concept map: one central idea with branches, each branch can have a few child ideas.",
        "parameters": {
            "type": "object",
            "properties": {
                "center": {"type": "string"},
                "branches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "children": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["label"],
                    },
                },
            },
            "required": ["center", "branches"],
        },
    },
    {
        "name": "draw_chart",
        "description": "Draw a simple bar or line chart from numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {"type": "string", "enum": ["bar", "line"]},
                "title": {"type": "string"},
                "labels": {"type": "array", "items": {"type": "string"}},
                "values": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["chart_type", "title", "labels", "values"],
        },
    },
    {
        "name": "make_flashcards",
        "description": "Make study flashcards. Front is a question, back is the answer. Pinch a card to flip it.",
        "parameters": {
            "type": "object",
            "properties": {
                "cards": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"front": {"type": "string"}, "back": {"type": "string"}},
                        "required": ["front", "back"],
                    },
                }
            },
            "required": ["cards"],
        },
    },
    {
        "name": "clear_canvas",
        "description": "Remove everything from the canvas.",
        "parameters": {"type": "object", "properties": {}},
    },
]

SYSTEM_INSTRUCTION = (
    "You are Pinch, a study assistant that works on a visual canvas. "
    "Always answer by calling one or more tools so something appears on the canvas. "
    "Prefer visuals: concept maps for topics with parts, charts for numbers, flashcards for "
    "'quiz me' or 'help me memorize'. Keep text short and simple. "
    "If study material is provided, base your answer on it."
)


class AgentRequest(BaseModel):
    text: str
    material: str = ""   # optional study material the student pasted


@app.post("/agent")
def agent(req: AgentRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        return {"mode": "mock", "actions": mock_actions(req.text)}
    try:
        return {"mode": "gemini", "actions": gemini_actions(req.text, req.material)}
    except Exception as error:
        return JSONResponse({"error": str(error)}, status_code=500)


def gemini_actions(text: str, material: str) -> list[dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = text if not material else f"STUDY MATERIAL:\n{material[:20000]}\n\nSTUDENT SAID: {text}"

    response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[types.Tool(function_declarations=TOOLS)],
            # We run the tools ourselves in the browser, so turn off automatic calling
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    actions = [{"name": call.name, "args": dict(call.args or {})} for call in (response.function_calls or [])]
    if not actions and response.text:   # model replied in plain text, show it as a note
        actions = [{"name": "add_note", "args": {"title": "Answer", "text": response.text[:400]}}]
    return json.loads(json.dumps(actions, default=list))  # make nested values plain JSON


def mock_actions(text: str) -> list[dict]:
    """Fake agent for testing with no key. Picks a tool from keywords."""
    t = text.lower()
    topic = re.sub(r"^(make|draw|show|create|give)( me)?( a| an)?\s*", "", t).strip() or "photosynthesis"
    if "clear" in t:
        return [{"name": "clear_canvas", "args": {}}]
    if "map" in t:
        return [{"name": "draw_concept_map", "args": {
            "center": "Photosynthesis",
            "branches": [
                {"label": "Inputs", "children": ["Sunlight", "Water", "CO2"]},
                {"label": "Outputs", "children": ["Glucose", "Oxygen"]},
                {"label": "Where", "children": ["Chloroplasts"]},
            ]}}]
    if "chart" in t or "graph" in t:
        return [{"name": "draw_chart", "args": {
            "chart_type": "bar", "title": "Sample: hours studied per day",
            "labels": ["Mon", "Tue", "Wed", "Thu", "Fri"], "values": [2, 3, 1.5, 4, 2.5]}}]
    if "quiz" in t or "flashcard" in t:
        return [{"name": "make_flashcards", "args": {"cards": [
            {"front": "What gas do plants take in?", "back": "Carbon dioxide"},
            {"front": "Where does photosynthesis happen?", "back": "In the chloroplasts"},
            {"front": "What does it make?", "back": "Glucose and oxygen"},
        ]}}]
    return [{"name": "add_note", "args": {"title": "Mock answer", "text": f"(mock mode) You asked about: {topic}"}}]


@app.get("/model")
def model():
    """Serves the trained gesture model if you made one in Stage 2 (see train_gestures.py)."""
    path = ROOT / "gesture_model.json"
    return FileResponse(path) if path.exists() else JSONResponse({}, status_code=404)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

if __name__ == "__main__":
    mode = "Gemini" if os.environ.get("GEMINI_API_KEY") else "MOCK (no GEMINI_API_KEY set)"
    print(f"Pinch running in {mode} mode. Open http://localhost:8000 in Chrome.")
    uvicorn.run(app, host="127.0.0.1", port=8000)
