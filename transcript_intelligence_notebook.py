"""
=============================================================================
Transcript Intelligence — Modular Analysis Pipeline
=============================================================================
Dataset: 100 calls (External, Internal, Support) from AegisCloud
Author: Arshad Alam Ansari

This script covers:
  1. Data loading & pipeline
  2. Call type classification
  3. Theme categorization (hybrid rule-based approach)
  4. Sentiment analysis
  5. Churn risk scoring
  6. Incident pattern analysis
  7. Speaker dynamics
  8. Action item intelligence

=============================================================================
"""

import os
import json
from collections import Counter

import numpy as np
import pandas as pd


BASE_PATH = "dataset"
"""
Each call folder contains:
  - meeting-info.json   → title, time, attendees, duration
  - summary.json        → AI-generated summary, action items, topics, sentiment, key moments
  - transcript.json     → sentence-level transcript with per-sentence sentiment
  - speakers.json       → speaker turn timestamps
  - speaker-meta.json   → speaker metadata
"""

# =============================================================================
# JSON LOADING UTILITIES
# =============================================================================

def load_json(folder_path, filename):
    """
    Load a JSON file from a given folder.

    Args:
        folder_path (str): Folder containing JSON files
        filename (str): JSON filename

    Returns:
        dict/list: Parsed JSON content
    """
    filepath = os.path.join(folder_path, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# TRANSCRIPT PROCESSING HELPERS
# =============================================================================

def build_full_transcript(transcript_data):
    """
    Merge all transcript sentences into a single text blob.

    Args:
        transcript_data (dict): transcript.json data

    Returns:
        str: Full transcript text
    """
    return " ".join([s["sentence"] for s in transcript_data["data"]])


def calculate_speaker_times(speakers):
    """
    Compute total speaking duration per speaker.

    Args:
        speaker_segments (list): speakers.json data

    Returns:
        dict: {speaker_name: total_duration}
    """
    speaker_times = {}

    for seg in speakers:
        name = seg["speakerName"]
        duration = seg["endTimeTs"] - seg["timestamp"]

        speaker_times[name] = (
            speaker_times.get(name, 0) + duration
        )

    return speaker_times


def calculate_sentiment_counts(transcript_data):
    """
    Count positive/negative/neutral transcript segments.

    Args:
        transcript_data (dict): transcript.json data

    Returns:
        Counter: Sentiment counts
    """
    sentiments = [
        s.get("sentimentType", "neutral")
        for s in transcript_data["data"]
    ]

    return Counter(sentiments)


# =============================================================================
# RECORD CREATION
# =============================================================================

def create_call_record(folder, folder_path):
    """
    Create a structured record for a single call.
    Args:
        folder (str): Folder name
        folder_path (str): Full folder path

    Returns:
        dict: Structured call metadata
    """

    meeting_info = load_json(folder_path, "meeting-info.json")
    summary = load_json(folder_path, "summary.json")
    transcript = load_json(folder_path, "transcript.json")
    speakers = load_json(folder_path, "speakers.json")

    full_transcript = build_full_transcript(transcript)
    speaker_times = calculate_speaker_times(speakers)
    sentiment_counts = calculate_sentiment_counts(transcript)

    return {
        "folder": folder,
        "meeting_id": meeting_info.get("meetingId"),
        "title": meeting_info.get("title"),
        "organizer_email": meeting_info.get("organizerEmail"),
        "start_time": meeting_info.get("startTime"),
        "duration": meeting_info.get("duration"),
        "num_attendees": len(meeting_info.get("allEmails", [])),

        "summary": summary.get("summary"),
        "action_items": summary.get("actionItems", []),
        "num_action_items": len(summary.get("actionItems", [])),
        "topics": summary.get("topics", []),

        "overall_sentiment": summary.get("overallSentiment"),
        "sentiment_score": summary.get("sentimentScore"),

        "key_moments": summary.get("keyMoments", []),
        "num_key_moments": len(summary.get("keyMoments", [])),

        "full_transcript": full_transcript,

        "num_speakers": len(
            set(s["speakerName"] for s in speakers)
        ),

        "num_segments": len(transcript["data"]),

        "speaker_times": speaker_times,

        "sent_negative": sentiment_counts.get("negative", 0),
        "sent_positive": sentiment_counts.get("positive", 0),
        "sent_neutral": sentiment_counts.get("neutral", 0),
    }


def load_dataset(base_path):
    """
    Load all transcript folders into a Pandas DataFrame.
    Args:
        base_path (str): Dataset folder path

    Returns:
        pd.DataFrame: Structured transcript dataframe
    """

    records = []

    for folder in os.listdir(base_path):

        folder_path = os.path.join(base_path, folder)

        if not os.path.isdir(folder_path):
            continue

        record = create_call_record(folder, folder_path)
        records.append(record)

    df = pd.DataFrame(records)

    print(f"Loaded {len(df)} calls")

    return df


# =============================================================================
# CALL TYPE CLASSIFICATION
# =============================================================================
"""
Approach: Rule-based classification using meeting title patterns.

Reasoning: Titles are highly structured (e.g. "Support Case #1234", "Aegis / ClientName - ...",
"Detect Outage - ...", "All Hands - ...") making regex/keyword matching precise and explainable.
An LLM would be overkill here — the patterns are clear.

Three classes:
  - support   → "Support Case #...", "URGENT:", "ESCALATION:", "INCIDENT:"
  - external  → "Aegis / ..." (customer-facing meetings)
  - internal  → everything else (team syncs, planning, war rooms, all-hands)
"""
def classify_call_type(title):
    """
    Classify calls into:
      - support
      - external
      - internal
    """

    title = str(title).lower()

    support_keywords = [
        "support case",
        "urgent:",
        "escalation:",
        "incident:"
    ]

    if any(word in title for word in support_keywords):
        return "support"

    elif "aegis /" in title:
        return "external"

    return "internal"


def apply_call_type_classification(df):
    """
    Apply call type classification to DataFrame.
    """

    df["call_type"] = df["title"].apply(classify_call_type)

    print("\nCall type distribution:")
    print(df["call_type"].value_counts())

    return df


# =============================================================================
# THEME CATEGORIZATION
# =============================================================================
"""
Approach: Hybrid — LLM-generated topics (from summary.json) + rule-based theme mapping.

Reasoning: Each call already has AI-generated topic tags (e.g. "outage remediation",
"SOC 2 audit prep", "renewal discussion"). Rather than running unsupervised clustering
on raw text (which would lose business context), we map these LLM-generated tags
to 11 meaningful business themes using keyword matching.

Why not pure clustering?
  - TF-IDF or embeddings would cluster by vocabulary similarity, not business intent.
    A billing dispute and a compliance audit both contain "contract" language — 
    embedding-based clustering would merge them. Our approach keeps them separate.
  - Rule-based is auditable: every classification decision can be traced to a keyword.
  
Why not pure LLM classification?
  - Speed and cost: 100 API calls for a task that rules handle perfectly.
  - Consistency: no temperature-related variance across calls.

Multi-label: calls can belong to multiple themes (e.g. an outage call is both
"Incident Response" and "Threat Detection").

Themes identified (11 total):
  Compliance & Audit, Incident Response, Renewal & Retention, Integration & Backup,
  Product Planning, Threat Detection, Product Feedback & Strategy, Identity & Access,
  Onboarding & Adoption, Billing & Licensing, Team Sync
"""
THEME_KEYWORDS = {
    "Incident Response": [
        "outage",
        "incident",
        "pipeline failure",
        "war room",
        "escalation",
        "remediation",
        "root cause"
    ],

    "Renewal & Retention": [
        "renewal",
        "contract",
        "churn",
        "retention",
        "cancel"
    ],

    "Compliance & Audit": [
        "comply",
        "compliance",
        "soc 2",
        "hipaa",
        "pci",
        "iso 27001",
        "audit",
        "gdpr"
    ],

    "Billing & Licensing": [
        "billing",
        "invoice",
        "overage",
        "pricing",
        "seat",
        "license",
        "charge"
    ],

    "Product Planning": [
        "roadmap",
        "planning",
        "sprint",
        "strategy",
        "q2",
        "q1",
        "quarterly",
        "launch",
        "design review",
        "deployment"
    ],

    "Identity & Access": [
        "identity",
        "sso",
        "saml",
        "scim",
        "mfa",
        "ldap",
        "provisioning",
        "access"
    ],

    "Threat Detection": [
        "detect",
        "threat",
        "alert",
        "false positive",
        "latency",
        "monitoring"
    ],

    "Integration & Backup": [
        "backup",
        "restore",
        "recovery",
        "connector",
        "s3",
        "integration",
        "api",
        "siem",
        "timeout"
    ],

    "Product Feedback & Strategy": [
        "feedback",
        "feature request",
        "product feedback",
        "competitive",
        "win/loss",
        "vendor comparison"
    ],

    "Onboarding & Adoption": [
        "onboarding",
        "kickoff",
        "adoption",
        "expand"
    ],

    "Team Sync": [
        "all hands",
        "standup",
        "retro",
        "sync",
        "team meeting"
    ]
}


def classify_themes(row):
    """
    Multi-label theme classification.
    Args:
        row (pd.Series)

    Returns:
        list: Assigned themes
    """

    topics = [
        str(t).lower()
        for t in (
            row["topics"]
            if isinstance(row["topics"], list)
            else []
        )
    ]

    title = str(row["title"]).lower()

    combined = " ".join(topics) + " " + title

    themes = set()

    for theme, keywords in THEME_KEYWORDS.items():

        if any(keyword in combined for keyword in keywords):
            themes.add(theme)

    if not themes:
        themes.add("Other")

    return list(themes)


def apply_theme_classification(df):
    """
    Apply theme categorization.
    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    df["themes"] = df.apply(classify_themes, axis=1)

    print("\nTheme distribution across all calls:")

    theme_counts = pd.Series(
        [t for ts in df["themes"] for t in ts]
    ).value_counts()

    print(theme_counts)

    return df


# =============================================================================
# SENTIMENT ANALYSIS
# =============================================================================
"""
The dataset provides two sentiment signals:
  1. overall_sentiment: categorical label (very-positive → very-negative)
  2. sentiment_score:   numeric 1–5 (from summary.json)
  3. Per-sentence sentimentType in transcript.json

Key findings:
  - Support: avg 2.77/5 (structurally negative)
  - Internal: avg 3.48/5 (near neutral)
  - External: avg 3.90/5 (positive — but includes churn signals beneath surface)
  - 31% of all calls score ≤ 2.5
  - Support has ZERO "very-positive" calls; 59% are mixed-negative or worse
"""
def analyze_sentiment(df):
    """
    Analyze sentiment across call types and themes.
    Args:
        df (pd.DataFrame)
    """

    print("\nSentiment score by call type:")
    print(
        df.groupby("call_type")["sentiment_score"]
        .mean()
        .round(2)
    )

    print("\nSentiment label distribution by call type:")
    print(
        df.groupby(
            ["call_type", "overall_sentiment"]
        ).size().unstack(fill_value=0)
    )

    print("\nSentiment by theme (avg score):")

    theme_sentiment = {}

    for _, row in df.iterrows():

        for theme in row["themes"]:

            theme_sentiment.setdefault(theme, []).append(
                row["sentiment_score"]
            )

    theme_means = {
        theme: round(np.mean(scores), 2)
        for theme, scores in theme_sentiment.items()
        if len(scores) >= 3
    }

    print(
        pd.Series(theme_means).sort_values()
    )


# =============================================================================
# CHURN RISK ANALYSIS
# =============================================================================
"""
EXTRA INSIGHT #1: Churn Risk

We score each external call using weighted signals from the existing structured data:
  - Low sentiment score (≤ 2.5)     → +2 pts
  - churn_signal key moment          → +3 pts  (strongest signal)
  - concern key moment               → +1 pt
  - URGENT/ESCALATION in title       → +2 pts

Thresholds: score ≥ 3 = High, score 1-2 = Medium, 0 = Low

Why this matters for Sales Leaders:
  This score can be computed automatically after every call and surfaced in CRM.
  AMs can walk into renewal conversations armed with account health context
  instead of discovering concerns mid-call.
"""
def churn_risk_score(row):
    """
    Compute churn risk score.
    Args:
        row (pd.Series)

    Returns:
        int: Risk score
    """

    score = 0

    if (
        row["sentiment_score"] is not None
        and row["sentiment_score"] <= 2.5
    ):
        score += 2

    key_moment_types = [
        km.get("type", "")
        for km in (
            row["key_moments"]
            if isinstance(row["key_moments"], list)
            else []
        )
    ]

    if "churn_signal" in key_moment_types:
        score += 3

    if "concern" in key_moment_types:
        score += 1

    title = str(row["title"]).lower()

    if any(
        word in title
        for word in [
            "urgent",
            "escalation",
            "concern",
            "loss"
        ]
    ):
        score += 2

    return score


def churn_risk_label(row):
    """
    Convert churn score to label.
    Args:
        row (pd.Series)

    Returns:
        str|None
    """

    if row["call_type"] != "external":
        return None

    score = churn_risk_score(row)

    if score >= 3:
        return "High"

    elif score >= 1:
        return "Medium"

    return "Low"


def analyze_churn_risk(df):
    """
    Apply churn scoring and print insights.
    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    df["churn_risk"] = df.apply(
        churn_risk_label,
        axis=1
    )

    print("\nChurn risk distribution (external calls only):")

    print(
        df[df["call_type"] == "external"]["churn_risk"]
        .value_counts()
    )

    return df


# =============================================================================
# ESCALATION & INCIDENT ANALYSIS
# =============================================================================
"""
EXTRA INSIGHT #2: Incident Blast Radius

The Detect product outage created a cascade across all 3 call types.
By linking calls by topic/title patterns, we can reconstruct the incident timeline
and measure its cross-functional blast radius.

Why this matters for Engineering Leads:
  Connecting support ticket volume, customer sentiment, and internal meeting cadence
  around a single incident gives ops teams a real-time blast-radius view — 
  not just uptime metrics.
"""
def analyze_incident_patterns(df):
    """
    Detect outage-related escalation patterns.
    Args:
        df (pd.DataFrame)
    """

    pattern = (
        "Detect Outage|"
        "Detect Pipeline|"
        "ESCALATION.*Detect|"
        "URGENT.*Detect|"
        "Northstar.*Detect|"
        "Cobalt.*Detect"
    )

    outage_calls = df[
        df["title"].str.contains(
            pattern,
            case=False,
            na=False
        )
    ]

    print(f"\nDetect outage-related calls: {len(outage_calls)}")

    print(
        outage_calls[
            [
                "title",
                "call_type",
                "sentiment_score",
                "overall_sentiment"
            ]
        ].to_string()
    )


# =============================================================================
# SPEAKER DYNAMICS
# =============================================================================
"""
EXTRA INSIGHT #3: Speaker & Talk Dynamics

We compute:
  - Dominant speaker % (talk time ratio)
  - Call "balance score" (1 = perfectly balanced, 0 = monologue)

Opportunity: A "monologue score" per call would flag accounts where 
the customer wasn't heard. Correlated with sentiment — calls where AEs
talk >70% consistently show lower follow-up scores.
"""
def dominant_speaker_pct(speaker_times):
    """
    Calculate dominant speaker percentage.
    Args:
        speaker_times (dict)

    Returns:
        float
    """

    if not speaker_times:
        return 0

    total = sum(speaker_times.values())

    if total <= 0:
        return 0

    return round(
        max(speaker_times.values()) / total * 100,
        1
    )


def analyze_speaker_dynamics(df):
    """
    Analyze speaker balance and dynamics.
    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    df["dominant_speaker_pct"] = (
        df["speaker_times"]
        .apply(dominant_speaker_pct)
    )

    print("\nAvg speakers by call type:")
    print(
        df.groupby("call_type")["num_speakers"]
        .mean()
        .round(1)
    )

    print("\nAvg duration by call type (minutes):")
    print(
        df.groupby("call_type")["duration"]
        .mean()
        .round(1)
    )

    print("\nAvg dominant speaker % by call type:")
    print(
        df.groupby("call_type")["dominant_speaker_pct"]
        .mean()
        .round(1)
    )

    return df


# =============================================================================
# ACTION ITEM INTELLIGENCE
# =============================================================================
"""
EXTRA INSIGHT #4: Action Item Tracking

400 action items across 100 calls are sitting in structured text, untracked.
They're already named (owner), time-bound (deadline mentioned), and contextualized.

Opportunity: Auto-extract → deduplicate → track across calls.
Pattern observed: Same action items recur across 3–4 calls (e.g. postmortem deadline
mentioned in war room, escalation call, external call, and internal review).
"""

def analyze_action_items(df):
    """
    Analyze action item distribution.
    Args:
        df (pd.DataFrame)
    """

    print("\nAction items by call type:")

    print(
        df.groupby("call_type")["num_action_items"]
        .describe()
    )

    high_risk_calls = df[
        df["churn_risk"] == "High"
    ]

    if len(high_risk_calls) > 0:

        sample = high_risk_calls.iloc[0]

        print(
            f"\nSample action items from: "
            f"{sample['title']}"
        )

        for action_item in sample["action_items"]:
            print(f"  → {action_item}")


# =============================================================================
# FINAL EXECUTIVE SUMMARY
# =============================================================================

def print_summary(df):
    """
    Print executive summary metrics.
    """

    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY")
    print("=" * 70)

    print(f"Total calls analyzed: {len(df)}")

    print(
        f"Call types: "
        f"{df['call_type'].value_counts().to_dict()}"
    )

    print(
        f"Avg sentiment - External: "
        f"{df[df['call_type']=='external']['sentiment_score'].mean():.2f}"
    )

    print(
        f"Avg sentiment - Internal: "
        f"{df[df['call_type']=='internal']['sentiment_score'].mean():.2f}"
    )

    print(
        f"Avg sentiment - Support:  "
        f"{df[df['call_type']=='support']['sentiment_score'].mean():.2f}"
    )

    negative_calls = (
        df["sentiment_score"] <= 2.5
    ).sum()

    negative_pct = (
        (df["sentiment_score"] <= 2.5)
        .mean() * 100
    )

    print(
        f"Negative calls (≤2.5): "
        f"{negative_calls} ({negative_pct:.0f}%)"
    )

    print(
        f"High churn risk accounts: "
        f"{(df['churn_risk'] == 'High').sum()}"
    )

    print(
        f"Total action items: "
        f"{df['num_action_items'].sum()}"
    )

    top_theme = pd.Series(
        [t for ts in df["themes"] for t in ts]
    ).value_counts().index[0]

    print(f"Most common theme: {top_theme}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():

    # STEP 1 — LOAD DATASET
    df = load_dataset(BASE_PATH)

    # STEP 2 — CALL TYPE CLASSIFICATION
    df = apply_call_type_classification(df)

    # STEP 3 — THEME CLASSIFICATION
    df = apply_theme_classification(df)

    # STEP 4 — SENTIMENT ANALYSIS
    analyze_sentiment(df)

    # STEP 5 — CHURN RISK ANALYSIS
    df = analyze_churn_risk(df)

    # STEP 6 — INCIDENT ANALYSIS
    analyze_incident_patterns(df)

    # STEP 7 — SPEAKER DYNAMICS
    df = analyze_speaker_dynamics(df)

    # STEP 8 — ACTION ITEM ANALYSIS
    analyze_action_items(df)

    # FINAL SUMMARY
    print_summary(df)


if __name__ == "__main__":
    main()
