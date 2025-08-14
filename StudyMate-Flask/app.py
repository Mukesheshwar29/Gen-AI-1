from __future__ import annotations
import os, uuid, io, datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from rag_engine import RAGEngine
from llm_ollama import OllamaLLM

# ---------------- Flask setup ----------------
load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'data')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

ALLOWED_EXTENSIONS = {'.pdf'}

# Single, simple in-memory store per session
# In production, persist these per user/session.
_engines = {}  # session_id -> RAGEngine
_histories = {}  # session_id -> list[dict]

def get_sid() -> str:
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]

def allowed(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- Routes ----------------

@app.route("/", methods=["GET"])
def home():
    sid = get_sid()
    history = _histories.get(sid, [])
    stats = None
    if sid in _engines and _engines[sid] is not None:
        stats = _engines[sid].stats()
    return render_template("index.html", history=history, stats=stats)

@app.route("/upload", methods=["POST"])
def upload():
    sid = get_sid()
    files = request.files.getlist("pdfs")
    if not files or files == [None]:
        flash("Please select one or more PDF files.", "warning")
        return redirect(url_for("home"))

    save_paths = []
    for f in files:
        if f and allowed(f.filename):
            fname = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{fname}")
            f.save(path)
            save_paths.append(path)
        else:
            flash(f"Unsupported file type: {getattr(f,'filename','(unknown)')}", "danger")

    if not save_paths:
        return redirect(url_for("home"))

    # Build / extend the engine
    engine = _engines.get(sid) or RAGEngine()
    engine.ingest_pdfs(save_paths)
    _engines[sid] = engine
    flash(f"Uploaded & indexed {len(save_paths)} PDF(s). You can ask questions now.", "success")
    return redirect(url_for("home"))

@app.route("/ask", methods=["POST"])
def ask():
    sid = get_sid()
    question = (request.form.get("question") or "").strip()
    if not question:
        flash("Type a question.", "warning")
        return redirect(url_for("home"))

    engine = _engines.get(sid)
    if not engine or engine.corpus_size() == 0:
        flash("Upload at least one PDF first.", "warning")
        return redirect(url_for("home"))

    # Retrieve context
    top_chunks = engine.retrieve(question, k=3)

    # Compose prompt
    prompt = OllamaLLM.build_prompt(question, top_chunks)
    llm = OllamaLLM(model_name="llama3")  # or any local Ollama model you have
    answer = llm.generate(prompt)

    # Save to history
    entry = {
        "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "contexts": top_chunks,
    }
    _histories.setdefault(sid, []).append(entry)
    return redirect(url_for("home"))

@app.route("/download", methods=["GET"])
def download():
    sid = get_sid()
    history = _histories.get(sid, [])
    if not history:
        flash("No Q&A history to download.", "warning")
        return redirect(url_for("home"))

    # Build a plain-text transcript
    buff = io.StringIO()
    buff.write("StudyMate Q&A Session Transcript\n")
    buff.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
    for i, h in enumerate(history, 1):
        buff.write(f"Q{i}: {h['question']}\n")
        buff.write(f"A{i}: {h['answer']}\n")
        buff.write("Referenced Paragraphs:\n")
        for j, c in enumerate(h["contexts"], 1):
            buff.write(f"  [{j}] ({c['source']})\n{c['text']}\n\n")
        buff.write("-" * 60 + "\n")
    data = buff.getvalue().encode("utf-8")
    return send_file(io.BytesIO(data), as_attachment=True,
                    download_name="studymate_transcript.txt",
                    mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
