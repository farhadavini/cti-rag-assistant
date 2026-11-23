#!/usr/bin/env python3
"""Minimal web UI for the CTI RAG assistant.

A single page: ask a question, get the answer with its sources. The heavy
lifting stays in query.answer(); this only renders it.

    pip install flask
    python web.py                 # http://127.0.0.1:5001
"""
import os

from flask import Flask, render_template_string, request

from query import answer

app = Flask(__name__)

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CTI RAG Assistant</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto;
         padding: 0 1rem; line-height: 1.5; }
  h1 { font-size: 1.3rem; }
  form { display: flex; gap: .5rem; margin-bottom: 1.5rem; }
  input[type=text] { flex: 1; padding: .6rem; font-size: 1rem; }
  button { padding: .6rem 1rem; font-size: 1rem; cursor: pointer; }
  .answer { white-space: pre-wrap; padding: 1rem; border: 1px solid #8884;
            border-radius: 8px; }
  .sources { margin-top: 1rem; font-size: .9rem; }
  .sources li { margin-bottom: .4rem; }
</style></head><body>
<h1>CTI RAG Assistant</h1>
<form method="post" action="/">
  <input type="text" name="q" placeholder="Ask about the CTI reports..."
         value="{{ q|e }}" autofocus>
  <button type="submit">Ask</button>
</form>
{% if answered %}
  <div class="answer">{{ answer_text }}</div>
  {% if sources %}
  <div class="sources"><strong>Sources ({{ sources|length }}):</strong>
    <ul>{% for s in sources %}<li>{{ s.file }}</li>{% endfor %}</ul>
  </div>
  {% endif %}
{% endif %}
</body></html>"""


@app.route("/", methods=["GET", "POST"])
def index():
    q = request.form.get("q", "").strip() if request.method == "POST" else ""
    ctx = {"q": q, "answered": False, "answer_text": "", "sources": []}
    if q:
        res = answer(q, include_sources=True)
        if isinstance(res, dict):
            ctx.update(answered=True, answer_text=res.get("answer", ""),
                       sources=res.get("sources", []))
        else:
            ctx.update(answered=True, answer_text=res)
    return render_template_string(_PAGE, **ctx)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "5001")))
