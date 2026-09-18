"""
DrivebuddyAI / Roadzen Technologies - Data Scientist Campus Placement Challenge
================================================================================
Task: Classify whether an Instagram post is about Pav Bhaji or not, using
TEXT / METADATA ONLY (no computer vision, no image pixels, no CNNs).

Author: Hardik

Run with:
    python DrivebuddyAI_PavBhaji_Challenge.py --data-dir ../dataset --output-dir ../outputs

All reported numbers in the accompanying report/notebook are produced by
running this exact script end-to-end. Nothing is hand-typed or fabricated.
"""

import argparse
import json
import os
import re
import string
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
    cross_validate,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

# Regex used to detect the direct target-identifying tokens ("pav bhaji" and
# common spelling/spacing variants + the hashtag form). Used ONLY for the
# leakage investigation and for building the leakage-controlled feature set.
TARGET_LEAK_PATTERN = re.compile(r"(?:pav[\s\-_]*bhaji|(?<![a-z])pav(?![a-z])|(?<![a-z])bhaji(?![a-z]))", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
def load_data(data_dir):
    """Load the raw JSON metadata and list image files per class folder.

    Returns
    -------
    records : list[dict]   raw JSON records (untouched)
    image_map : dict       {0: [filenames...], 1: [filenames...]}
    """
    json_path = os.path.join(data_dir, "pavbhaji.json")
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    image_map = {}
    for label in (0, 1):
        folder = os.path.join(data_dir, "images", str(label))
        files = [
            fn for fn in os.listdir(folder)
            if not fn.startswith(".") and not fn.startswith("._")
        ]
        image_map[label] = files

    return records, image_map


# ---------------------------------------------------------------------------
# 2. DATASET INSPECTION
# ---------------------------------------------------------------------------
def inspect_data(records, image_map, verbose=True):
    """Print/collect a schema & quality inventory of the raw dataset."""
    report = {}
    report["n_json_records"] = len(records)
    report["n_images_class0"] = len(image_map[0])
    report["n_images_class1"] = len(image_map[1])

    key_freq = Counter()
    for r in records:
        key_freq.update(r.keys())
    report["field_coverage"] = {k: v for k, v in key_freq.most_common()}

    ids = [r.get("id") for r in records]
    report["n_records"] = len(ids)
    report["n_unique_ids"] = len(set(ids))
    report["n_duplicate_id_records"] = len(ids) - len(set(ids))

    shortcodes = [r.get("shortcode") for r in records]
    report["n_unique_shortcodes"] = len(set(shortcodes))

    n_missing_caption = sum(
        1 for r in records
        if not r.get("edge_media_to_caption", {}).get("edges")
    )
    report["n_missing_caption"] = n_missing_caption

    n_null_location = sum(1 for r in records if not r.get("location"))
    report["n_null_location"] = n_null_location

    report["n_video_posts"] = sum(1 for r in records if r.get("is_video"))

    if verbose:
        print("=" * 70)
        print("DATASET INSPECTION SUMMARY")
        print("=" * 70)
        print(f"JSON records                 : {report['n_json_records']}")
        print(f"Images in images/0 (Not PB)   : {report['n_images_class0']}")
        print(f"Images in images/1 (Pav Bhaji): {report['n_images_class1']}")
        print(f"Unique record ids             : {report['n_unique_ids']} "
              f"(duplicates: {report['n_duplicate_id_records']})")
        print(f"Missing captions              : {report['n_missing_caption']}")
        print(f"Null location field           : {report['n_null_location']}")
        print(f"Video posts                   : {report['n_video_posts']}")
        print("\nField coverage (out of "
              f"{report['n_json_records']} records):")
        for k, v in report["field_coverage"].items():
            print(f"  {k:28s}: {v}")
    return report


# ---------------------------------------------------------------------------
# 3. LABEL / DATASET CONSTRUCTION (JSON <-> image join)
# ---------------------------------------------------------------------------
def _basename(url):
    if not url:
        return None
    return url.split("/")[-1].split("?")[0]


def build_dataset(records, image_map, verbose=True):
    """Deterministically join image filenames (folder = label) to JSON
    records via the CDN filename embedded in `display_url` / `thumbnail_src`
    / `thumbnail_resources[*].src`. This filename is a verbatim substring
    match to the files provided in images/0 and images/1, giving an exact,
    non-fuzzy join key.
    """
    filename_to_indices = {}
    for i, rec in enumerate(records):
        names = set()
        for key in ("display_url", "thumbnail_src"):
            b = _basename(rec.get(key))
            if b:
                names.add(b)
        for tr in rec.get("thumbnail_resources") or []:
            b = _basename(tr.get("src"))
            if b:
                names.add(b)
        for n in names:
            filename_to_indices.setdefault(n, []).append(i)

    rows = []
    unmatched_images = []
    for label in (0, 1):
        for fn in image_map[label]:
            idxs = filename_to_indices.get(fn)
            if not idxs:
                unmatched_images.append((label, fn))
                continue
            # every observed filename matched exactly one unique record id
            rec = records[idxs[0]]
            rows.append({"filename": fn, "label": label, "record": rec})

    matched_record_indices = set()
    for label in (0, 1):
        for fn in image_map[label]:
            idxs = filename_to_indices.get(fn)
            if idxs:
                matched_record_indices.add(idxs[0])
    n_unmatched_json = len(records) - len(matched_record_indices)

    if verbose:
        print("=" * 70)
        print("LABEL MAPPING VALIDATION")
        print("=" * 70)
        print("Join key: CDN filename embedded in display_url / "
              "thumbnail_src / thumbnail_resources[*].src")
        print(f"Total images across images/0 + images/1: "
              f"{len(image_map[0]) + len(image_map[1])}")
        print(f"Successfully matched to a JSON record    : {len(rows)}")
        print(f"Unmatched images (no JSON record found)  : "
              f"{len(unmatched_images)}")
        print(f"JSON records with no corresponding image : {n_unmatched_json} "
              f"(these are extra scraped posts not used for the labeled task)")
        print(f"Final usable labeled samples              : {len(rows)}")

    df = _rows_to_dataframe(rows)
    return df, {
        "n_matched": len(rows),
        "n_unmatched_images": len(unmatched_images),
        "unmatched_images": unmatched_images,
        "n_unmatched_json_records": n_unmatched_json,
        "n_final_samples": len(df),
    }


def _rows_to_dataframe(rows):
    data = []
    for row in rows:
        rec = row["record"]
        cap_edges = rec.get("edge_media_to_caption", {}).get("edges", [])
        caption = cap_edges[0]["node"]["text"] if cap_edges else ""
        loc = rec.get("location") or {}
        data.append({
            "filename": row["filename"],
            "label": row["label"],
            "record_id": rec.get("id"),
            "shortcode": rec.get("shortcode"),
            "owner_id": rec.get("owner", {}).get("id"),
            "caption": caption if caption else "",
            "tags": rec.get("tags") or [],
            "n_tags": len(rec.get("tags") or []),
            "likes": rec.get("edge_liked_by", {}).get("count",
                     rec.get("edge_media_preview_like", {}).get("count", np.nan)),
            "comments_count": rec.get("edge_media_to_comment", {}).get("count", np.nan),
            "is_video": bool(rec.get("is_video", False)),
            "video_view_count": rec.get("video_view_count", np.nan),
            "comments_disabled": rec.get("comments_disabled"),
            "location_name": loc.get("name"),
            "has_location": loc.get("name") is not None,
            "taken_at_timestamp": rec.get("taken_at_timestamp"),
            "width": rec.get("dimensions", {}).get("width", np.nan),
            "height": rec.get("dimensions", {}).get("height", np.nan),
        })
    df = pd.DataFrame(data)
    df["taken_at_dt"] = pd.to_datetime(df["taken_at_timestamp"], unit="s", errors="coerce")
    df["post_hour"] = df["taken_at_dt"].dt.hour
    df["post_dayofweek"] = df["taken_at_dt"].dt.dayofweek
    return df


# ---------------------------------------------------------------------------
# 4/5. DATA QUALITY + LEAKAGE DETECTION
# ---------------------------------------------------------------------------
def detect_leakage(df, verbose=True):
    """Investigate direct target-token leakage and other structural leakage
    (duplicate captions, duplicate owners, filename/id patterns)."""
    results = {}

    combined_text = (df["caption"].fillna("") + " " +
                      df["tags"].apply(lambda t: " ".join(t)))
    has_target_token = combined_text.apply(lambda t: bool(TARGET_LEAK_PATTERN.search(t)))
    by_class = df.groupby("label").apply(
        lambda g: has_target_token.loc[g.index].mean()
    )
    results["target_token_rate_by_class"] = by_class.to_dict()

    dup_caption_counts = df["caption"].value_counts()
    dup_captions = dup_caption_counts[dup_caption_counts > 1]
    results["n_duplicate_caption_groups"] = int((dup_captions).shape[0])
    results["n_rows_in_duplicate_captions"] = int(dup_captions.sum())

    dup_record_ids = df["record_id"].value_counts()
    results["n_duplicate_record_ids"] = int((dup_record_ids > 1).sum())

    owner_post_counts = df["owner_id"].value_counts()
    results["max_posts_per_owner"] = int(owner_post_counts.max())
    results["n_owners_with_multiple_posts"] = int((owner_post_counts > 1).sum())

    if verbose:
        print("=" * 70)
        print("DATA LEAKAGE INVESTIGATION")
        print("=" * 70)
        print("Direct target-token ('pav bhaji' variant) presence by class:")
        for cls, rate in results["target_token_rate_by_class"].items():
            label_name = "Pav Bhaji (1)" if cls == 1 else "Not Pav Bhaji (0)"
            print(f"  {label_name:18s}: {rate:.1%}")
        print("-> The token appears in almost ALL posts of BOTH classes "
              "(dataset was collected via pav-bhaji-related search/hashtags),")
        print("   so its raw presence/absence is NOT a usable leakage shortcut "
              "by itself. Still masked below to prevent the model from")
        print("   keying off the literal target word instead of real context.")
        print(f"\nDuplicate caption groups     : {results['n_duplicate_caption_groups']} "
              f"({results['n_rows_in_duplicate_captions']} rows involved)")
        print(f"Duplicate JSON record ids    : {results['n_duplicate_record_ids']}")
        print(f"Owners with >1 post in sample : {results['n_owners_with_multiple_posts']} "
              f"(max posts by one owner: {results['max_posts_per_owner']})")
        print("-> owner_id is therefore used as the GROUP key for a grouped "
              "train/test split, to avoid the same author's writing style")
        print("   appearing in both train and test.")

    return results


# ---------------------------------------------------------------------------
# 6. TEXT CLEANING
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_TOKEN_RE = re.compile(r"#(\w+)")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+", flags=re.UNICODE
)
NON_ALNUM_RE = re.compile(r"[^a-z0-9#\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text, mask_target=False, strict_target_removal=False):
    """Reusable, deterministic text cleaning function.

    - lowercases
    - strips URLs, @mentions
    - converts emojis to a neutral placeholder (keeps signal that an emoji
      was used without keeping literal pixel-derived meaning)
    - keeps hashtags as words (they carry real semantic content) but drops
      the '#' character so 'pavbhaji' and '#pavbhaji' are treated the same
    - collapses punctuation/whitespace
    - optionally masks/removes direct target-label tokens (leakage control)
    """
    if not isinstance(text, str) or text == "":
        return ""
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = EMOJI_RE.sub(" <emoji> ", t)
    t = t.replace("#", " ")
    t = NON_ALNUM_RE.sub(" ", t)
    t = MULTI_SPACE_RE.sub(" ", t).strip()
    if mask_target:
        replacement = " " if strict_target_removal else " <target_masked> "
        t = TARGET_LEAK_PATTERN.sub(replacement, t)
        t = MULTI_SPACE_RE.sub(" ", t).strip()
    return t


def clean_dataframe(df):
    """Apply cleaning + basic quality fixes. Returns a NEW dataframe;
    the original raw df/records are never mutated."""
    df = df.copy()
    df["caption"] = df["caption"].fillna("")
    df["n_hashtags_in_caption"] = df["caption"].apply(
        lambda t: len(HASHTAG_TOKEN_RE.findall(t)) if isinstance(t, str) else 0
    )
    df["n_mentions"] = df["caption"].apply(
        lambda t: len(MENTION_RE.findall(t)) if isinstance(t, str) else 0
    )
    df["n_urls"] = df["caption"].apply(
        lambda t: len(URL_RE.findall(t)) if isinstance(t, str) else 0
    )
    df["n_emojis"] = df["caption"].apply(
        lambda t: len(EMOJI_RE.findall(t)) if isinstance(t, str) else 0
    )
    df["caption_char_len"] = df["caption"].str.len()
    df["caption_word_len"] = df["caption"].apply(lambda t: len(t.split()))

    # combined raw text field (caption + hashtag list, deduplicated)
    df["text_raw"] = df.apply(
        lambda r: (r["caption"] + " " + " ".join(r["tags"])).strip(), axis=1
    )
    df["text_clean_naive"] = df["text_raw"].apply(lambda t: clean_text(t, mask_target=False))
    df["text_clean_masked"] = df["text_raw"].apply(lambda t: clean_text(t, mask_target=True))
    df["text_clean_strict"] = df["text_raw"].apply(
        lambda t: clean_text(t, mask_target=True, strict_target_removal=True)
    )

    # Build a composite leakage-control group that prevents BOTH the same
    # owner and the same normalized caption from crossing train/test or CV folds.
    # Caption duplicates across owners are connected into the same group.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for idx, row in df.iterrows():
        owner_key = f"owner::{row['owner_id']}" if pd.notna(row['owner_id']) else f"row::{idx}"
        union(owner_key, f"row::{idx}")
        cap = clean_text(row["caption"], mask_target=False) if isinstance(row["caption"], str) else ""
        if cap:
            union(f"caption::{cap}", f"row::{idx}")

    df["cv_group"] = [find(f"row::{i}") for i in range(len(df))]
    df["normalized_caption"] = df["caption"].apply(
        lambda x: clean_text(x, mask_target=False) if isinstance(x, str) else ""
    )

    # numeric cleanup
    for col in ["likes", "comments_count", "video_view_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["video_view_count"] = df["video_view_count"].fillna(0)
    df["likes"] = df["likes"].fillna(df["likes"].median())
    df["comments_count"] = df["comments_count"].fillna(df["comments_count"].median())

    # duplicate handling: drop exact duplicate (record_id) rows, keep first
    before = len(df)
    df = df.drop_duplicates(subset=["record_id"], keep="first").reset_index(drop=True)
    dropped = before - len(df)
    return df, dropped


# ---------------------------------------------------------------------------
# 7. EDA
# ---------------------------------------------------------------------------
def run_eda(df, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    figs_created = []

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["label"].value_counts().sort_index()
    labels = ["Not Pav Bhaji (0)", "Pav Bhaji (1)"]
    ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#4C72B0", "#DD8452"])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of posts")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v + 2, str(v), ha="center")
    plt.tight_layout()
    p = os.path.join(fig_dir, "01_class_distribution.png")
    plt.savefig(p); plt.close(); figs_created.append(p)

    # 2. Missing values
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(miss))))
    if len(miss) > 0:
        ax.barh(miss.index, miss.values, color="#C44E52")
        ax.set_xlabel("Fraction missing")
        ax.set_title("Missing Values by Column")
    else:
        ax.text(0.5, 0.5, "No missing values in engineered dataset",
                ha="center", va="center")
        ax.axis("off")
    plt.tight_layout()
    p = os.path.join(fig_dir, "02_missing_values.png")
    plt.savefig(p); plt.close(); figs_created.append(p)

    # 3. Text length distribution by class
    fig, ax = plt.subplots(figsize=(6, 4))
    for lbl, name, color in [(0, "Not Pav Bhaji", "#4C72B0"), (1, "Pav Bhaji", "#DD8452")]:
        sns.kdeplot(df.loc[df.label == lbl, "caption_word_len"], label=name, ax=ax, color=color, fill=True, alpha=0.3)
    ax.set_title("Caption Word-Length Distribution by Class")
    ax.set_xlabel("Words in caption")
    ax.legend()
    plt.tight_layout()
    p = os.path.join(fig_dir, "03_text_length_distribution.png")
    plt.savefig(p); plt.close(); figs_created.append(p)

    # 4. Important numeric feature distributions (likes, comments)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in [(axes[0], "likes", "Likes (log scale)"),
                            (axes[1], "comments_count", "Comments (log scale)")]:
        for lbl, name, color in [(0, "Not Pav Bhaji", "#4C72B0"), (1, "Pav Bhaji", "#DD8452")]:
            vals = df.loc[df.label == lbl, col]
            vals = np.log1p(vals)
            sns.kdeplot(vals, label=name, ax=ax, color=color, fill=True, alpha=0.3)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    p = os.path.join(fig_dir, "04_engagement_distributions.png")
    plt.savefig(p); plt.close(); figs_created.append(p)

    # 5. Top words (naive, unmasked) overall
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(stop_words="english", max_features=25)
    X = cv.fit_transform(df["text_clean_naive"])
    freqs = np.asarray(X.sum(axis=0)).ravel()
    top = sorted(zip(cv.get_feature_names_out(), freqs), key=lambda x: -x[1])[:20]
    fig, ax = plt.subplots(figsize=(6, 6))
    words, vals = zip(*top)
    ax.barh(words[::-1], vals[::-1], color="#55A868")
    ax.set_title("Top 20 Most Frequent Words (raw text, before leakage masking)")
    plt.tight_layout()
    p = os.path.join(fig_dir, "05_top_words.png")
    plt.savefig(p); plt.close(); figs_created.append(p)

    # 6. Top hashtags
    all_tags = Counter(t.lower() for tags in df["tags"] for t in tags)
    top_tags = all_tags.most_common(20)
    fig, ax = plt.subplots(figsize=(6, 6))
    tags_, vals = zip(*top_tags)
    ax.barh(list(tags_)[::-1], list(vals)[::-1], color="#8172B2")
    ax.set_title("Top 20 Hashtags")
    plt.tight_layout()
    p = os.path.join(fig_dir, "06_top_hashtags.png")
    plt.savefig(p); plt.close(); figs_created.append(p)

    # 7. Class-wise feature comparison (n_tags, n_hashtags, has_location)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    sns.boxplot(x="label", y="n_tags", data=df, ax=axes[0], palette=["#4C72B0", "#DD8452"])
    axes[0].set_title("Tag Count by Class")
    axes[0].set_xticklabels(["Not PB", "Pav Bhaji"])
    sns.boxplot(x="label", y="caption_word_len", data=df, ax=axes[1], palette=["#4C72B0", "#DD8452"])
    axes[1].set_title("Caption Length by Class")
    axes[1].set_xticklabels(["Not PB", "Pav Bhaji"])
    loc_rate = df.groupby("label")["has_location"].mean()
    axes[2].bar(["Not PB", "Pav Bhaji"], [loc_rate.get(0, 0), loc_rate.get(1, 0)], color=["#4C72B0", "#DD8452"])
    axes[2].set_title("Has Location Tag Rate")
    plt.tight_layout()
    p = os.path.join(fig_dir, "07_classwise_feature_comparison.png")
    plt.savefig(p); plt.close(); figs_created.append(p)

    return figs_created


# ---------------------------------------------------------------------------
# 9-11. FEATURES, SPLIT, MODELS, EVALUATION
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "likes", "comments_count", "n_tags", "n_hashtags_in_caption",
    "n_mentions", "n_urls", "n_emojis", "caption_char_len",
    "caption_word_len", "is_video", "has_location",
]


def split_data(df, text_col, group_col="cv_group", test_size=0.2, seed=RANDOM_SEED):
    """Stratified, GROUPED train/test split so that posts by the same
    Instagram owner never appear in both train and test (prevents
    author-style leakage), while keeping class balance as close as possible.
    """
    sgkf = StratifiedGroupKFold(n_splits=int(1 / test_size), shuffle=True, random_state=seed)
    y = df["label"].values
    groups = df[group_col].values
    train_idx, test_idx = next(sgkf.split(df, y, groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def build_pipeline(model, text_col):
    text_transformer = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    numeric_transformer = Pipeline([
        ("scale", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("text", text_transformer, text_col),
        ("num", numeric_transformer, NUMERIC_FEATURES),
    ])
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", model),
    ])
    return pipe


def get_candidate_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
        "LinearSVM": LinearSVC(class_weight="balanced", random_state=RANDOM_SEED),
        "MultinomialNB": MultinomialNB(),  # requires non-negative features; handled via separate text-only pipe
    }


def _has_predict_proba(estimator):
    return hasattr(estimator, "predict_proba")


def evaluate_model(pipe, X_test, y_test, model_name=""):
    y_pred = pipe.predict(X_test)
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    try:
        if _has_predict_proba(pipe.named_steps["clf"]):
            y_score = pipe.predict_proba(X_test)[:, 1]
        else:
            y_score = pipe.decision_function(X_test)
        metrics["roc_auc"] = roc_auc_score(y_test, y_score)
        metrics["pr_auc"] = average_precision_score(y_test, y_score)
    except Exception:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_pred


def cross_validate_models(train_df, text_col, seed=RANDOM_SEED, n_splits=5):
    """5-fold GROUP-AWARE CV on the training data only. Groups prevent
    owner/caption duplicates from crossing validation folds."""
    X = train_df
    y = train_df["label"].values
    groups = train_df["cv_group"].values
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results = []
    models = get_candidate_models()
    for name, model in models.items():
        if name == "MultinomialNB":
            # NB needs non-negative inputs -> text-only TF-IDF pipeline (no scaled numerics)
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)),
                ("clf", model),
            ])
            X_in = train_df[text_col]
        else:
            pipe = build_pipeline(model, text_col)
            X_in = train_df

        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        cv_res = cross_validate(pipe, X_in, y, cv=skf, groups=groups, scoring=scoring, n_jobs=1)
        row = {"model": name}
        for m in scoring:
            row[f"cv_{m}_mean"] = np.mean(cv_res[f"test_{m}"])
            row[f"cv_{m}_std"] = np.std(cv_res[f"test_{m}"])
        results.append(row)
    return pd.DataFrame(results)


def train_final_model(train_df, text_col, model_name="LogisticRegression"):
    models = get_candidate_models()
    model = models[model_name]
    pipe = build_pipeline(model, text_col)
    pipe.fit(train_df, train_df["label"].values)
    return pipe


def explain_model(pipe, top_n=20):
    """Extract top positive/negative coefficients for a linear final model."""
    clf = pipe.named_steps["clf"]
    preprocessor = pipe.named_steps["preprocess"]
    text_features = preprocessor.named_transformers_["text"].get_feature_names_out()
    num_features = np.array(NUMERIC_FEATURES)
    all_features = np.concatenate([text_features, num_features])

    if hasattr(clf, "coef_"):
        coefs = clf.coef_.ravel()
    else:
        return None

    order = np.argsort(coefs)
    top_neg = [(all_features[i], coefs[i]) for i in order[:top_n]]
    top_pos = [(all_features[i], coefs[i]) for i in order[::-1][:top_n]]
    return {"top_positive": top_pos, "top_negative": top_neg}


def predict_sample(pipe, sample_dict, text_col="text_clean_strict"):
    """Run inference on a single new post-metadata record.

    `sample_dict` should contain at least the same fields produced by
    build_dataset/clean_dataframe: caption, tags, likes, comments_count,
    n_tags, n_hashtags_in_caption, n_mentions, n_urls, n_emojis,
    caption_char_len, caption_word_len, is_video, has_location.
    """
    row = pd.DataFrame([sample_dict])
    if text_col not in row.columns:
        raw = (row.get("caption", [""])[0] or "") + " " + " ".join(row.get("tags", [[]])[0] or [])
        row[text_col] = clean_text(raw, mask_target=True, strict_target_removal=True)
    pred = pipe.predict(row)[0]
    label = "Pav Bhaji" if pred == 1 else "Not Pav Bhaji"
    clf = pipe.named_steps["clf"]
    if _has_predict_proba(clf):
        proba = pipe.predict_proba(row)[0, 1]
        return {"prediction": label, "probability_pav_bhaji": float(proba)}
    score = pipe.decision_function(row)[0]
    return {"prediction": label, "decision_score": float(score)}


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../dataset")
    parser.add_argument("--output-dir", default="../outputs")
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print("# STEP 1: LOAD DATA")
    print("#" * 70)
    records, image_map = load_data(args.data_dir)

    print("\n" + "#" * 70)
    print("# STEP 2: INSPECT DATA")
    print("#" * 70)
    inspect_report = inspect_data(records, image_map)

    print("\n" + "#" * 70)
    print("# STEP 3: BUILD LABELED DATASET (JSON <-> image join)")
    print("#" * 70)
    df, join_report = build_dataset(records, image_map)

    print("\n" + "#" * 70)
    print("# STEP 4: DATA CLEANING")
    print("#" * 70)
    df, n_dropped_dupes = clean_dataframe(df)
    print(f"Dropped {n_dropped_dupes} exact-duplicate records (same record_id).")
    print(f"Final dataset shape: {df.shape}")

    print("\n" + "#" * 70)
    print("# STEP 5: LEAKAGE DETECTION")
    print("#" * 70)
    leakage_report = detect_leakage(df)

    print("\n" + "#" * 70)
    print("# STEP 6: EDA")
    print("#" * 70)
    figs = run_eda(df, fig_dir)
    print(f"Saved {len(figs)} figures to {fig_dir}")

    print("\n" + "#" * 70)
    print("# STEP 7: TRAIN / TEST SPLIT (grouped by owner_id)")
    print("#" * 70)
    train_df, test_df = split_data(df, text_col="text_clean_strict", group_col="cv_group")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Train class balance: {train_df['label'].value_counts(normalize=True).to_dict()}")
    print(f"Test class balance : {test_df['label'].value_counts(normalize=True).to_dict()}")
    overlap = set(train_df["owner_id"].dropna()) & set(test_df["owner_id"].dropna())
    cap_overlap = (set(train_df["normalized_caption"]) & set(test_df["normalized_caption"])) - {""}
    group_overlap = set(train_df["cv_group"]) & set(test_df["cv_group"])
    print(f"Owner overlap between train/test: {len(overlap)} (should be 0)")
    print(f"Exact normalized caption overlap: {len(cap_overlap)} (should be 0)")
    print(f"Composite group overlap: {len(group_overlap)} (should be 0)")

    # ---------------- EXPERIMENT A: raw/naive text baseline (leakage check) ----------------
    print("\n" + "#" * 70)
    print("# STEP 8: EXPERIMENT A - RAW/NAIVE TEXT BASELINE (leakage probe)")
    print("#" * 70)
    naive_pipe = build_pipeline(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
        text_col="text_clean_naive",
    )
    naive_pipe.fit(train_df, train_df["label"])
    naive_metrics, naive_cm, _ = evaluate_model(naive_pipe, test_df, test_df["label"], "Naive (unmasked) LogReg")
    print("Naive baseline test metrics:", {k: round(v, 4) for k, v in naive_metrics.items() if k != "model"})
    naive_explain = explain_model(naive_pipe, top_n=15)
    print("Top positive tokens (naive/unmasked model):")
    for feat, coef in naive_explain["top_positive"][:15]:
        print(f"   {feat:25s} {coef:+.3f}")

    # ---------------- EXPERIMENT B: leakage-controlled model ----------------
    print("\n" + "#" * 70)
    print("# STEP 9: EXPERIMENT B - LEAKAGE-CONTROLLED MODEL")
    print("#" * 70)
    print("Using text_clean_strict (direct target terms removed) for all further modeling.")

    print("\n" + "#" * 70)
    print("# STEP 10: 5-FOLD CV MODEL COMPARISON (train set only)")
    print("#" * 70)
    cv_results = cross_validate_models(train_df, text_col="text_clean_strict")
    print(cv_results.round(4).to_string(index=False))
    cv_results.to_csv(os.path.join(args.output_dir, "cv_model_comparison.csv"), index=False)

    best_model_name = cv_results.sort_values("cv_f1_mean", ascending=False).iloc[0]["model"]
    print(f"\nSelected model (highest mean CV F1): {best_model_name}")

    print("\n" + "#" * 70)
    print("# STEP 11: TRAIN FINAL MODEL ON FULL TRAINING SET")
    print("#" * 70)
    final_pipe = train_final_model(train_df, text_col="text_clean_strict", model_name=best_model_name)

    print("\n" + "#" * 70)
    print("# STEP 12: FINAL TEST-SET EVALUATION")
    print("#" * 70)
    final_metrics, final_cm, final_preds = evaluate_model(final_pipe, test_df, test_df["label"], best_model_name)
    print("Final model test metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in final_metrics.items()})
    print("Confusion matrix:\n", final_cm)
    print("\nClassification report:\n", classification_report(test_df["label"], final_preds, target_names=["Not Pav Bhaji", "Pav Bhaji"]))

    # confusion matrix plot
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(final_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not PB", "Pav Bhaji"], yticklabels=["Not PB", "Pav Bhaji"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {best_model_name} (Final, Leakage-Controlled)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "08_confusion_matrix_final.png"))
    plt.close()

    # model comparison bar chart (CV F1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(cv_results["model"], cv_results["cv_f1_mean"], yerr=cv_results["cv_f1_std"], color="#4C72B0", capsize=4)
    ax.set_ylabel("Mean CV F1-score")
    ax.set_title("Model Comparison (5-fold CV, training set)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "09_model_comparison_cv_f1.png"))
    plt.close()

    print("\n" + "#" * 70)
    print("# STEP 13: MODEL INTERPRETABILITY")
    print("#" * 70)
    explanation = explain_model(final_pipe, top_n=20)
    if explanation:
        print("Top POSITIVE (-> Pav Bhaji) features:")
        for feat, coef in explanation["top_positive"]:
            print(f"   {feat:25s} {coef:+.3f}")
        print("\nTop NEGATIVE (-> Not Pav Bhaji) features:")
        for feat, coef in explanation["top_negative"]:
            print(f"   {feat:25s} {coef:+.3f}")
        fi_rows = (
            [{"feature": f, "coefficient": c, "direction": "Pav Bhaji"} for f, c in explanation["top_positive"]] +
            [{"feature": f, "coefficient": c, "direction": "Not Pav Bhaji"} for f, c in explanation["top_negative"]]
        )
        pd.DataFrame(fi_rows).to_csv(os.path.join(args.output_dir, "feature_importance.csv"), index=False)

        fig, ax = plt.subplots(figsize=(7, 7))
        feats = [f for f, _ in explanation["top_positive"][:15][::-1]] + [f for f, _ in explanation["top_negative"][:15][::-1]]
        vals = [c for _, c in explanation["top_positive"][:15][::-1]] + [c for _, c in explanation["top_negative"][:15][::-1]]
        colors = ["#DD8452" if v > 0 else "#4C72B0" for v in vals]
        ax.barh(feats, vals, color=colors)
        ax.set_title("Top Feature Coefficients (Final Model)")
        ax.set_xlabel("Coefficient (+ towards Pav Bhaji, - towards Not Pav Bhaji)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "10_feature_importance.png"))
        plt.close()

    print("\n" + "#" * 70)
    print("# STEP 14: ERROR ANALYSIS")
    print("#" * 70)
    test_df_eval = test_df.copy()
    test_df_eval["pred"] = final_preds
    fp = test_df_eval[(test_df_eval["label"] == 0) & (test_df_eval["pred"] == 1)]
    fn = test_df_eval[(test_df_eval["label"] == 1) & (test_df_eval["pred"] == 0)]
    print(f"False positives (predicted Pav Bhaji, actually not): {len(fp)}")
    print(f"False negatives (predicted Not, actually Pav Bhaji): {len(fn)}")
    error_cols = ["filename", "label", "pred", "caption", "n_tags", "likes"]
    errors_df = pd.concat([fp[error_cols], fn[error_cols]])
    errors_df.to_csv(os.path.join(args.output_dir, "error_analysis.csv"), index=False)
    if len(fp) > 0:
        print("\nSample false positive caption:")
        print("  " + fp.iloc[0]["caption"][:200].replace("\n", " "))
    if len(fn) > 0:
        print("\nSample false negative caption:")
        print("  " + fn.iloc[0]["caption"][:200].replace("\n", " "))

    print("\n" + "#" * 70)
    print("# STEP 15: EXAMPLE INFERENCE ON A HELD-OUT SAMPLE")
    print("#" * 70)
    example_row = test_df.iloc[0]
    example_dict = example_row.to_dict()
    result = predict_sample(final_pipe, example_dict, text_col="text_clean_strict")
    print(f"Held-out sample filename : {example_row['filename']}")
    print(f"True label               : {'Pav Bhaji' if example_row['label']==1 else 'Not Pav Bhaji'}")
    print(f"Prediction               : {result['prediction']}")
    if "decision_score" in result:
        print(f"Decision score            : {result['decision_score']:.4f}")
    else:
        print(f"P(Pav Bhaji)             : {result['probability_pav_bhaji']:.4f}")

    # save predictions.csv for full test set
    out_preds = test_df[["filename", "record_id", "label"]].copy()
    out_preds["predicted_label"] = final_preds
    out_preds.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    # save model results summary
    model_results_df = pd.DataFrame([naive_metrics, final_metrics])
    model_results_df.to_csv(os.path.join(args.output_dir, "model_results.csv"), index=False)

    # persist the exact fitted final pipeline for local inference / Streamlit
    model_dir = os.path.join(os.path.dirname(args.output_dir), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "final_model.joblib")
    joblib.dump(final_pipe, model_path)

    # Persist metrics/model metadata for the Streamlit app. Values are generated
    # from the same final run; the app does not retrain or hardcode metrics.
    metadata = {
        "model_name": best_model_name,
        "training_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "cv_f1_mean": float(cv_results.loc[cv_results["model"] == best_model_name, "cv_f1_mean"].iloc[0]),
        "cv_f1_std": float(cv_results.loc[cv_results["model"] == best_model_name, "cv_f1_std"].iloc[0]),
        "test_accuracy": float(final_metrics["accuracy"]),
        "test_precision": float(final_metrics["precision"]),
        "test_recall": float(final_metrics["recall"]),
        "test_f1": float(final_metrics["f1"]),
        "test_roc_auc": float(final_metrics["roc_auc"]),
        "test_pr_auc": float(final_metrics["pr_auc"]),
        "strict_target_removal": True,
        "text_column": "text_clean_strict",
        "numeric_features": NUMERIC_FEATURES,
        "model_artifact": "models/final_model.joblib",
    }
    with open(os.path.join(args.output_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved exact final model pipeline: {model_path}")
    print(f"Saved Streamlit metadata: {os.path.join(args.output_dir, 'model_metadata.json')}")

    print("\n" + "#" * 70)
    print("# FINAL DELIVERABLE SUMMARY")
    print("#" * 70)
    print(f"A. Dataset size                : {len(df)} labeled samples "
          f"({join_report['n_matched']} matched; {n_dropped_dupes} exact dupes dropped)")
    print(f"B. Class distribution           : {df['label'].value_counts().to_dict()}")
    print(f"C. Key features used            : TF-IDF(1-2gram) of strict-cleaned caption+hashtags; "
          f"likes, comments_count, n_tags, n_hashtags, n_mentions, n_urls, n_emojis, "
          f"caption length (char/word), is_video, has_location")
    print(f"D. Leakage discovered           : 'pav bhaji' token present in "
          f"~{leakage_report['target_token_rate_by_class']}; direct target terms removed before final modeling. "
          f"composite owner/caption groups used for split/CV.")
    print(f"E. Models evaluated             : {list(get_candidate_models().keys())}")
    print(f"F. Final model                  : {best_model_name} (leakage-controlled)")
    print(f"G. Final test metrics           : "
          f"Acc={final_metrics['accuracy']:.3f} P={final_metrics['precision']:.3f} "
          f"R={final_metrics['recall']:.3f} F1={final_metrics['f1']:.3f} "
          f"ROC-AUC={final_metrics['roc_auc']:.3f}")
    print(f"H. Naive (unmasked) vs controlled F1: {naive_metrics['f1']:.3f} vs {final_metrics['f1']:.3f}")
    print(f"I. Output files                 : {os.listdir(args.output_dir)}")
    print(f"J. Run command                  : python DrivebuddyAI_PavBhaji_Challenge.py "
          f"--data-dir <path_to_dataset> --output-dir <path_to_outputs>")

    return {
        "df": df, "train_df": train_df, "test_df": test_df,
        "final_pipe": final_pipe, "final_metrics": final_metrics,
        "naive_metrics": naive_metrics, "cv_results": cv_results,
        "leakage_report": leakage_report, "join_report": join_report,
        "best_model_name": best_model_name,
    }


if __name__ == "__main__":
    main()
