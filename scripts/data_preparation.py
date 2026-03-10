import pandas as pd
import json
import re
from pathlib import Path

# Paths
data_dir = Path("data/raw")
output_dir = Path("data/processed")
reports_dir = Path("reports")

output_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

# Load datasets
emotion_df = pd.read_csv(data_dir / "emotion-emotion_69k.csv")
esconv_df = pd.read_csv(data_dir / "ESConv-1k3_extracted.csv")

print(f"Emotion dataset shape: {emotion_df.shape}")
print(f"ESConv dataset shape: {esconv_df.shape}")

print("Emotion dataset columns:", emotion_df.columns)
print("ESConv dataset columns:", esconv_df.columns)

# -----------------------------
# Emotion dataset extraction
# -----------------------------
def extract_emotion_pairs(df):

    pairs = []

    # detect conversation column automatically
    possible_cols = ["dialogue", "conversation", "empathetic_dialogues", "utterance"]
    conv_col = None

    for c in possible_cols:
        if c in df.columns:
            conv_col = c
            break

    if conv_col is None:
        print("⚠ No conversation column found in emotion dataset")
        return pairs

    emotion_col = "emotion" if "emotion" in df.columns else None

    for _, row in df.iterrows():

        dialogue = str(row[conv_col])

        if pd.isna(dialogue):
            continue

        parts = re.split(r'(Customer :|Agent :|User :|Assistant :)', dialogue)

        parts = [
            p.strip() for p in parts
            if p.strip() not in ["Customer :", "Agent :", "User :", "Assistant :"]
        ]

        for i in range(0, len(parts)-1, 2):

            instruction = parts[i]
            response = parts[i+1]

            pairs.append({
                "instruction": instruction,
                "response": response,
                "emotion": row[emotion_col] if emotion_col else "unknown",
                "source": "emotion-emotion_69k"
            })

    return pairs


# -----------------------------
# ESConv dataset extraction
# -----------------------------
def extract_esconv_pairs(df):

    pairs = []

    dialog_col = "dialog" if "dialog" in df.columns else None
    emotion_col = "emotion_type" if "emotion_type" in df.columns else None

    if dialog_col is None:
        print("⚠ No dialog column found in ESConv dataset")
        return pairs

    for _, row in df.iterrows():

        dialog_str = row[dialog_col]

        if pd.isna(dialog_str):
            continue

        try:
            dialog = json.loads(dialog_str)
        except:
            continue

        seeker = None

        for turn in dialog:

            speaker = turn.get("speaker", "")
            text = turn.get("content", "")

            if speaker == "seeker":
                seeker = text

            elif speaker == "supporter" and seeker:

                pairs.append({
                    "instruction": seeker,
                    "response": text,
                    "emotion": row[emotion_col] if emotion_col else "unknown",
                    "source": "ESConv"
                })

                seeker = None

    return pairs


# Extract pairs
emotion_pairs = extract_emotion_pairs(emotion_df)
esconv_pairs = extract_esconv_pairs(esconv_df)

print(f"Emotion pairs: {len(emotion_pairs)}")
print(f"ESConv pairs: {len(esconv_pairs)}")

# Merge datasets
all_pairs = emotion_pairs + esconv_pairs
print(f"Total pairs before cleaning: {len(all_pairs)}")


# -----------------------------
# Safety Cleaning
# -----------------------------
unsafe_keywords = [
    "suicide","kill yourself","end your life","self-harm",
    "cutting","overdose","hang yourself","jump off",
    "shoot yourself","poison yourself"
]

def is_safe(text):

    text = str(text).lower()

    for word in unsafe_keywords:
        if word in text:
            return False

    return True


cleaned_pairs = [
    p for p in all_pairs
    if is_safe(p["instruction"]) and is_safe(p["response"])
]

print(f"Total pairs after cleaning: {len(cleaned_pairs)}")


# -----------------------------
# Save dataset
# -----------------------------
output_file = output_dir / "cleaned_dataset.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_pairs, f, indent=2, ensure_ascii=False)

print("Cleaned dataset saved to:", output_file)


# -----------------------------
# Report
# -----------------------------
report = f"""
Data Cleaning Report

Emotion pairs extracted: {len(emotion_pairs)}
ESConv pairs extracted: {len(esconv_pairs)}

Total pairs before cleaning: {len(all_pairs)}
Total pairs after cleaning: {len(cleaned_pairs)}

Unsafe keywords removed:
{', '.join(unsafe_keywords)}
"""

report_file = reports_dir / "data_cleaning_report.txt"

with open(report_file, "w", encoding="utf-8") as f:
    f.write(report)

print("Report saved to:", report_file)