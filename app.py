from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "final_model.joblib"
METADATA_PATH = OUTPUT_DIR / "model_metadata.json"

# Reuse the exact preprocessing implementation from the training pipeline.
sys.path.insert(0, str(SRC_DIR))
from DrivebuddyAI_PavBhaji_Challenge import (  # noqa: E402
    NUMERIC_FEATURES,
    EMOJI_RE,
    HASHTAG_TOKEN_RE,
    MENTION_RE,
    URL_RE,
    clean_text,
)

APP_TITLE = "DrivebuddyAI — Pav Bhaji Text Classifier"

st.set_page_config(
    page_title="DrivebuddyAI Pav Bhaji Classifier",
    page_icon="🤖",
    layout="wide",
)


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def parse_tags(raw: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,\n]", raw or "") if x.strip()]


def build_inference_row(
    caption: str,
    tags_text: str,
    likes: float,
    comments_count: float,
    is_video: bool,
    has_location: bool,
) -> pd.DataFrame:
    tags = parse_tags(tags_text)
    raw_text = f"{caption or ''} {' '.join(tags)}".strip()
    strict_text = clean_text(
        raw_text,
        mask_target=True,
        strict_target_removal=True,
    )

    row = pd.DataFrame(
        [
            {
                "caption": caption or "",
                "tags": tags,
                "n_tags": len(tags),
                "likes": float(likes),
                "comments_count": float(comments_count),
                "is_video": bool(is_video),
                "has_location": bool(has_location),
                "n_hashtags_in_caption": len(HASHTAG_TOKEN_RE.findall(caption or "")),
                "n_mentions": len(MENTION_RE.findall(caption or "")),
                "n_urls": len(URL_RE.findall(caption or "")),
                "n_emojis": len(EMOJI_RE.findall(caption or "")),
                "caption_char_len": len(caption or ""),
                "caption_word_len": len((caption or "").split()),
                "text_clean_strict": strict_text,
            }
        ]
    )
    return row


def extract_local_explanation(model, row: pd.DataFrame, top_n: int = 6):
    """Explain a prediction for linear sklearn models using exact feature contributions."""
    try:
        clf = model.named_steps["clf"]
        preprocessor = model.named_steps["preprocess"]
        if not hasattr(clf, "coef_"):
            return []

        transformed = preprocessor.transform(row)
        coef = clf.coef_.ravel()

        if hasattr(preprocessor, "get_feature_names_out"):
            names = preprocessor.get_feature_names_out()
        else:
            text_names = preprocessor.named_transformers_["text"].get_feature_names_out()
            names = list(text_names) + list(NUMERIC_FEATURES)

        values = transformed.toarray()[0] if hasattr(transformed, "toarray") else transformed[0]
        contributions = values * coef

        rows = []
        for idx in contributions.argsort()[::-1]:
            if abs(contributions[idx]) < 1e-8:
                continue
            rows.append(
                {
                    "feature": str(names[idx]).replace("text__", "").replace("num__", ""),
                    "contribution": float(contributions[idx]),
                    "direction": "Pav Bhaji" if contributions[idx] > 0 else "Not Pav Bhaji",
                }
            )
            if len(rows) >= top_n:
                break
        return rows
    except Exception:
        return []


def model_summary(model, metadata: dict) -> dict:
    summary = dict(metadata)
    summary.setdefault("model_name", model.named_steps.get("clf", model).__class__.__name__)
    try:
        summary["feature_count"] = len(model.named_steps["preprocess"].get_feature_names_out())
    except Exception:
        summary["feature_count"] = None
    return summary


def render_sidebar(metadata: dict):
    with st.sidebar:
        st.markdown("### Project")
        st.caption("DrivebuddyAI / Roadzen Technologies")
        st.write("**Task:** Text-based Pav Bhaji classification")
        st.write(f"**Model:** {metadata.get('model_name', 'Final trained model')}")
        st.write("**Input:** Instagram-style text + metadata")

        st.divider()
        st.markdown("### Important")
        st.caption(
            "This demo intentionally does not use image pixels, CNNs, OCR, image embeddings, "
            "or computer vision."
        )


def main():
    metadata = load_metadata()
    model = load_model()
    render_sidebar(metadata)

    st.title(APP_TITLE)
    st.caption("Text-based Machine Learning Classification Demo")

    if model is None:
        st.error("The trained model artifact is missing.")
        st.info(
            "Run the training pipeline first so it creates `models/final_model.joblib`. "
            "The Streamlit app never retrains the model at startup."
        )
        st.code(
            "python src/DrivebuddyAI_PavBhaji_Challenge.py "
            "--data-dir <path_to_dataset> --output-dir outputs",
            language="bash",
        )
        st.stop()

    summary = model_summary(model, metadata)

    if "demo_caption" not in st.session_state:
        st.session_state.demo_caption = ""
        st.session_state.demo_tags = ""
        st.session_state.demo_likes = 120
        st.session_state.demo_comments = 18
        st.session_state.demo_video = False
        st.session_state.demo_location = True

    left, right = st.columns([1.25, 0.75], gap="large")

    with left:
        st.subheader("Post metadata")
        st.text_area(
            "Caption / description",
            key="demo_caption",
            height=170,
            placeholder="Example: buttery mashed vegetables served hot with toasted bread rolls...",
        )
        st.text_input(
            "Hashtags / tags",
            key="demo_tags",
            placeholder="streetfood, foodphotography, indianstreetfood",
            help="Separate multiple tags with commas.",
        )

        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Likes", min_value=0, step=1, key="demo_likes")
        with c2:
            st.number_input("Comments", min_value=0, step=1, key="demo_comments")

        c3, c4 = st.columns(2)
        with c3:
            st.checkbox("Post is a video", key="demo_video")
        with c4:
            st.checkbox("Post has a location", key="demo_location")

        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            if st.button("Load food example", use_container_width=True):
                st.session_state.demo_caption = (
                    "Buttery mashed vegetables with toasted bread rolls, onions, lemon and fresh coriander. "
                    "Classic Indian street food comfort."
                )
                st.session_state.demo_tags = "streetfood, foodphotography, indianstreetfood"
                st.session_state.demo_likes = 185
                st.session_state.demo_comments = 23
                st.session_state.demo_video = False
                st.session_state.demo_location = True
                st.rerun()
        with ex2:
            if st.button("Load non-target example", use_container_width=True):
                st.session_state.demo_caption = (
                    "Crispy masala dosa with coconut chutney and sambar, served fresh for breakfast."
                )
                st.session_state.demo_tags = "dosa, southindianfood, breakfast"
                st.session_state.demo_likes = 95
                st.session_state.demo_comments = 9
                st.session_state.demo_video = False
                st.session_state.demo_location = True
                st.rerun()
        with ex3:
            if st.button("Clear", use_container_width=True):
                st.session_state.demo_caption = ""
                st.session_state.demo_tags = ""
                st.session_state.demo_likes = 0
                st.session_state.demo_comments = 0
                st.session_state.demo_video = False
                st.session_state.demo_location = False
                st.rerun()

        if st.button("Predict", type="primary", use_container_width=True):
            row = build_inference_row(
                caption=st.session_state.demo_caption,
                tags_text=st.session_state.demo_tags,
                likes=st.session_state.demo_likes,
                comments_count=st.session_state.demo_comments,
                is_video=st.session_state.demo_video,
                has_location=st.session_state.demo_location,
            )
            try:
                pred = int(model.predict(row)[0])
                label = "Pav Bhaji" if pred == 1 else "Not Pav Bhaji"
                st.session_state.last_result = {
                    "label": label,
                    "row": row,
                    "score": None,
                }
                clf = model.named_steps.get("clf")
                if hasattr(clf, "predict_proba"):
                    st.session_state.last_result["score"] = float(model.predict_proba(row)[0, 1])
                    st.session_state.last_result["score_type"] = "probability"
                elif hasattr(model, "decision_function"):
                    st.session_state.last_result["score"] = float(model.decision_function(row)[0])
                    st.session_state.last_result["score_type"] = "decision score"
                else:
                    st.session_state.last_result["score_type"] = None
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    with right:
        st.subheader("Prediction")
        result = st.session_state.get("last_result")
        if result is None:
            st.info("Enter post information and click Predict.")
        else:
            if result["label"] == "Pav Bhaji":
                st.success("### PAV BHAJI")
            else:
                st.warning("### NOT PAV BHAJI")

            if result.get("score") is not None:
                if result.get("score_type") == "probability":
                    st.metric("Prediction confidence", f"{result['score']:.1%}")
                else:
                    st.metric("Model decision score", f"{result['score']:.4f}")

            explanations = extract_local_explanation(model, result["row"])
            if explanations:
                st.markdown("#### Important signals detected")
                st.dataframe(
                    pd.DataFrame(explanations),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption("Feature contributions explain the model score; they are not causal effects.")

        st.divider()
        st.subheader("Model information")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Final model", summary.get("model_name", "—"))
            st.metric("Training samples", summary.get("training_samples", "—"))
            st.metric("Features", summary.get("feature_count", "—"))
        with m2:
            cv_f1 = summary.get("cv_f1_mean")
            st.metric("Grouped CV F1", f"{cv_f1:.3f}" if isinstance(cv_f1, (int, float)) else "—")
            test_f1 = summary.get("test_f1")
            st.metric("Holdout F1", f"{test_f1:.3f}" if isinstance(test_f1, (int, float)) else "—")
            auc = summary.get("test_roc_auc")
            st.metric("Holdout ROC-AUC", f"{auc:.3f}" if isinstance(auc, (int, float)) else "—")

    with st.expander("Important: Dataset Leakage Control"):
        st.write(
            "The supplied challenge data was collected using Pav-Bhaji-related descriptions/hashtags. "
            "Direct target terms were therefore investigated as potential leakage. The final pipeline "
            "removes direct target terms such as 'pav', 'bhaji' and their common combinations before TF-IDF modeling."
        )

    with st.expander("How the model works"):
        st.markdown(
            "**Raw metadata** → **data cleaning** → **strict target-term removal** → "
            "**TF-IDF text features + structured metadata** → **final ML model** → **prediction**"
        )
        st.write(
            "The app loads the fitted sklearn pipeline saved by the training script. "
            "It does not retrain the model when Streamlit starts."
        )


if __name__ == "__main__":
    main()
