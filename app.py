# app.py
import logging
import os

from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS

from routes import recommend_bp

# ─── Load environment variables from .env ─────────────────────────────────────
load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

# ─── App factory ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

app.register_blueprint(recommend_bp, url_prefix="/api")


# ─── Health endpoints ─────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Recommender API is running", "status": "ok"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", 5000))

    app.run(host="0.0.0.0", port=port, debug=debug)
