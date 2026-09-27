"""
GRU + XAI — Web version of input_jawaban() / list_history() / tampilkan_penjelasan()
--------------------------------------------------------------------------------
This is NOT a mock. It loads your real trained artifacts (the "otak"/brain from
the notebook pipeline) and runs real inference + real SHAP explanations.

Run:
    pip install -r requirements.txt
    python app.py
Then open:  http://127.0.0.1:5000
"""

import os
import json
import pickle
import threading

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Response, render_template

# ------------------------------------------------------------------
# 1) Konfigurasi path artefak
#    Letakkan file-file ini SEJAJAR dengan app.py (folder yang sama),
#    atau ubah path di bawah ini sesuai lokasi Anda.
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gru_optik_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
ID2LABEL_PATH = os.path.join(BASE_DIR, "id2label.json")

# Opsional — hanya dipakai untuk membangun "background" SHAP yang lebih bermakna.
# Jika tidak ada, aplikasi tetap jalan (prediksi tetap 100% asli), hanya
# penjelasan SHAP memakai background netral.
DATASET_PATH = os.path.join(BASE_DIR, "dataset_gru.csv")
DATASET_DELIMITER = ";"

# Warna label untuk tampilan (silakan sesuaikan)
LABEL_COLORS = {
    "Paham Konsep": "#5ad1e6",
    "Miskonsepsi": "#f0708a",
    "Tidak Paham": "#f0b45a",
}
DEFAULT_COLOR = "#9297a8"

app = Flask(__name__)

# ------------------------------------------------------------------
# 2) Muat model & artefak SEKALI saat server start
# ------------------------------------------------------------------
missing = [p for p in [MODEL_PATH, TOKENIZER_PATH, ID2LABEL_PATH] if not os.path.exists(p)]
if missing:
    raise SystemExit(
        "File artefak berikut tidak ditemukan di folder ini:\n  - "
        + "\n  - ".join(missing)
        + "\n\nSalin file gru_optik_model.h5, tokenizer.pkl, dan id2label.json "
          "ke folder yang sama dengan app.py, lalu jalankan ulang."
    )

print("Memuat model...")
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model_def import build_gru_model
import json

with open("id2label.json", "r", encoding="utf-8") as f:
    id2label = json.load(f)

VOCAB_SIZE = 5000        # sesuai konfigurasi di notebook
EMBED_DIM = 128          # sesuai konfigurasi di notebook
MAXLEN = 22             # lihat catatan di bawah
NUM_CLASSES = len(id2label)

final_model = build_gru_model(VOCAB_SIZE, EMBED_DIM, MAXLEN, NUM_CLASSES)
final_model.load_weights(MODEL_PATH)   # MODEL_PATH tetap menunjuk ke gru_optik_model.h5

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(ID2LABEL_PATH, "r", encoding="utf-8") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}
num_classes = len(id2label)

# ------------------------------------------------------------------
# 3) Riwayat jawaban (in-memory) — setara history_jawaban di notebook
# ------------------------------------------------------------------
history_jawaban = []
history_lock = threading.Lock()


def predict_text(answer: str):
    """Sama persis dengan alur input_jawaban(): tokenize -> pad -> predict."""
    seq = tokenizer.texts_to_sequences([answer])
    pad = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post").astype(np.int32)
    probs = final_model.predict(pad, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "pred_idx": idx,
        "pred_label": id2label[idx],
        "confidence": float(probs[idx]),
        "probs": {id2label[i]: float(probs[i]) for i in range(num_classes)},
        "pad": pad,
    }


# ------------------------------------------------------------------
# 4) SHAP — setara _get_explainer_for_class / proses_batch / tampilkan_penjelasan
#    Dihitung sesuai permintaan (lewat tombol "Jelaskan"), bukan otomatis,
#    karena KernelExplainer perlu beberapa detik per jawaban.
# ------------------------------------------------------------------
_shap_ready = False
_shap_import_error = None
_class_explainers = {}
background_data = None


def _init_shap_background():
    global background_data, _shap_ready, _shap_import_error
    try:
        import shap  # noqa: F401
    except ImportError as e:
        _shap_import_error = str(e)
        print("shap tidak terpasang — fitur 'Jelaskan' dinonaktifkan. (" + _shap_import_error + ")")
        return

    try:
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH, sep=DATASET_DELIMITER)
            df = df.dropna(subset=["jawaban_siswa"]).reset_index(drop=True)
            n = min(20, len(df))
            sample = df["jawaban_siswa"].astype(str).sample(n=n, random_state=42).tolist()
            seqs = tokenizer.texts_to_sequences(sample)
            bg = pad_sequences(seqs, maxlen=MAXLEN, padding="post", truncating="post").astype(np.int32)
            print(f"Background SHAP disiapkan dari {n} baris '{os.path.basename(DATASET_PATH)}'.")
        else:
            print(
                f"'{os.path.basename(DATASET_PATH)}' tidak ditemukan di folder ini — "
                "memakai background netral (penjelasan tetap real, hanya kurang presisi)."
            )
            bg = np.zeros((5, MAXLEN), dtype=np.int32)
        background_data = bg
        _shap_ready = True
    except Exception as e:
        _shap_import_error = str(e)
        print("Gagal menyiapkan SHAP background:", e)


_init_shap_background()


def _get_explainer_for_class(c):
    import shap

    if c not in _class_explainers:
        def _f(x):
            return final_model.predict(x, verbose=0)[:, c : c + 1]

        _class_explainers[c] = shap.KernelExplainer(_f, background_data)
    return _class_explainers[c]


# ------------------------------------------------------------------
# 5) Routes
# ------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify(
        {
            "ok": True,
            "maxlen": MAXLEN,
            "classes": list(id2label.values()),
            "colors": {lbl: LABEL_COLORS.get(lbl, DEFAULT_COLOR) for lbl in id2label.values()},
            "shap_ready": _shap_ready,
            "shap_error": _shap_import_error,
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True) or {}
    answer = (data.get("jawaban") or "").strip()
    if not answer:
        return jsonify({"error": "Jawaban tidak boleh kosong."}), 400

    result = predict_text(answer)
    with history_lock:
        jawaban_id = len(history_jawaban) + 1
        history_jawaban.append(
            {
                "id": jawaban_id,
                "jawaban": answer,
                "prediksi": result["pred_label"],
                "keyakinan": result["confidence"],
                "pred_idx": result["pred_idx"],
                "processed_input": result["pad"].tolist(),
                "shap_values": None,
            }
        )

    return jsonify(
        {
            "id": jawaban_id,
            "jawaban": answer,
            "prediksi": result["pred_label"],
            "keyakinan": result["confidence"],
            "probs": result["probs"],
            "color": LABEL_COLORS.get(result["pred_label"], DEFAULT_COLOR),
        }
    )


@app.route("/api/history")
def api_history():
    search = request.args.get("search", "").strip().lower()
    limit = request.args.get("limit", "20")

    with history_lock:
        rows = sorted(history_jawaban, key=lambda r: r["id"], reverse=True)

    if search:
        rows = [r for r in rows if search in r["jawaban"].lower()]

    if limit and str(limit).lower() != "none":
        try:
            rows = rows[: int(limit)]
        except ValueError:
            pass

    out = [
        {
            "id": r["id"],
            "jawaban": r["jawaban"],
            "prediksi": r["prediksi"],
            "keyakinan": r["keyakinan"],
            "sudah_shap": r["shap_values"] is not None,
            "color": LABEL_COLORS.get(r["prediksi"], DEFAULT_COLOR),
        }
        for r in rows
    ]
    return jsonify(out)


@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    with history_lock:
        history_jawaban.clear()
    return jsonify({"ok": True})


@app.route("/api/history/export")
def api_history_export():
    with history_lock:
        rows = sorted(history_jawaban, key=lambda r: r["id"])
        df = pd.DataFrame(
            [
                {
                    "id": r["id"],
                    "jawaban": r["jawaban"],
                    "prediksi": r["prediksi"],
                    "keyakinan": round(r["keyakinan"], 3),
                    "sudah_SHAP": r["shap_values"] is not None,
                }
                for r in rows
            ]
        )
    csv_data = df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=history_jawaban.csv"},
    )


@app.route("/api/explain", methods=["POST"])
def api_explain():
    if not _shap_ready:
        return (
            jsonify(
                {
                    "error": "SHAP belum siap di server ini (library tidak terpasang atau gagal dimuat). "
                    "Prediksi tetap berjalan normal — fitur ini hanya untuk visualisasi kontribusi kata."
                }
            ),
            503,
        )

    data = request.get_json(force=True) or {}
    jawaban_id = data.get("id")
    nsamples = int(data.get("nsamples", 60))

    with history_lock:
        rec = next((r for r in history_jawaban if r["id"] == jawaban_id), None)
    if rec is None:
        return jsonify({"error": f"Jawaban #{jawaban_id} tidak ditemukan."}), 404

    pad = np.array(rec["processed_input"], dtype=np.int32)
    c = int(rec["pred_idx"])

    explainer_c = _get_explainer_for_class(c)
    shap_vals = explainer_c.shap_values(pad, nsamples=nsamples)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_val = np.array(shap_vals[0]).astype(np.float64).squeeze()

    index_word = tokenizer.index_word
    tokens = [index_word.get(int(t), "<PAD>") for t in pad[0] if int(t) != 0]
    m = min(len(tokens), shap_val.shape[0])
    tokens = tokens[:m]
    shap_val = shap_val[:m]

    with history_lock:
        rec["shap_values"] = shap_val.tolist()

    contributions = sorted(
        [{"token": tok, "value": float(v)} for tok, v in zip(tokens, shap_val)],
        key=lambda x: abs(x["value"]),
        reverse=True,
    )

    return jsonify(
        {
            "id": jawaban_id,
            "jawaban": rec["jawaban"],
            "prediksi": rec["prediksi"],
            "keyakinan": rec["keyakinan"],
            "contributions": contributions,
        }
    )


if __name__ == "__main__":
    # debug=False supaya model tidak dimuat dua kali oleh auto-reloader Flask.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
