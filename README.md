# Pinch (Stage 1: core loop)

A study canvas you control with your hands and voice. Built as a learning rebuild inspired by MindPad, the Best Overall Hack at HackPrinceton Fall 2025.
Original project: https://devpost.com/software/mindpad
Original code: https://github.com/krishgolcha/MindPad

## The idea

A study canvas you control with your hand and your voice. Point your finger to draw, pinch to
grab things, and say "make a concept map of the cell" or "quiz me on photosynthesis" to have
the AI build it on the canvas.

## Run it (needs your laptop's webcam, use Chrome)

```
pip install -r requirements.txt
python server.py                  # mock mode, no key needed
```

Open http://localhost:8000 and allow the camera.

To use real Gemini:

```
export GEMINI_API_KEY="your-key"          # Windows: set GEMINI_API_KEY=your-key
export GEMINI_MODEL="gemini-2.5-flash"    # change if Google has a newer model
python server.py
```

## Try these

- Hold up just your index finger and draw.
- Hold up index and middle fingers to move without drawing.
- Say or type "quiz me on photosynthesis", then pinch a flashcard quickly to flip it.
- Pinch and hold on any card to drag it.
- Click Notes, paste a chapter, then ask "make a concept map of this."
- Hold your open palm up for 1.5 seconds to clear your drawing.

## How it works (read in this order)

| Part | Where | What it does |
|------|-------|--------------|
| Hand tracking | `static/app.js` section 1 | MediaPipe finds 21 points on your hand every frame |
| Features | section 2 | Turns the points into 52 numbers that ignore where your hand is or how big it looks |
| Classifier | section 3 | Rules decide the gesture now. A trained network replaces them in Stage 2 |
| Smoothing | section 3 and 4 | A gesture must hold 3 frames, and the cursor is averaged so it does not shake |
| Actions | section 4 | Draw, move, grab, flip, clear |
| Voice | section 6 | Chrome turns speech into text and sends it to the server |
| Agent | `server.py` | Gemini picks a tool (note, concept map, chart, flashcards, clear) and fills in the details |
| Tools | section 7 | The browser draws whatever tool Gemini picked |

The key idea is in `server.py`: the AI never draws anything. It chooses an action from a list
you defined and fills in the details, and your code carries it out. That is tool calling, and
it is how most real AI agents work.

One change from the original: they used PyAutoGUI to move your actual computer mouse. This
version moves a cursor inside the web page instead. It is simpler, safer, and easier to demo.

## Roadmap to the original team's scale

**Stage 2: train your own gesture model (about an evening)**
- Open "Train gestures" at the bottom right. For each gesture, hold the R key for a few
  seconds while moving your hand around. Aim for 300+ samples per gesture.
- Download `gestures.json` into this folder and run `python train_gestures.py`.
- Restart the server. The app switches to your neural network automatically.
- Add a new gesture (thumbs up to undo, two-finger swipe to scroll) and see how much easier a
  trained model makes it compared to writing rules. That comparison is a great interview story.

**Stage 3: real-time voice agent (about a weekend)**
- Swap the one-shot request for a streaming voice conversation (Gemini Live API), so the agent
  can talk back while it draws.
- Let the agent see the canvas: send a summary of what is already on it so it can say
  "add a branch to the map" instead of making a new one.

**Stage 4: study material pipeline**
- Upload a PDF, split it into chunks, and only send the relevant chunks with each question
  (this is called retrieval-augmented generation).
- Generate a whole study session from a lecture: concept map, then flashcards, then a chart.

**Stage 5: polish like the original**
- Move to Next.js and TypeScript like their final version.
- Measure your gesture accuracy and response time. They reported under 200 milliseconds and 99%
  accuracy, so give yourself numbers to beat.
- Accessibility settings: adjustable gesture sensitivity and high contrast mode.

## Interview angle

This is a good system to explain out loud: a real-time input pipeline (camera to features to
classifier to action), a separate request pipeline (voice to agent to tool to render), and how
you kept them from getting in each other's way.
