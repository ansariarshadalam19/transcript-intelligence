# transcript-intelligence
Transcript Intelligence — a tool that helps different stakeholders across the company make better decisions using these transcripts. Think support leaders, sales managers, product managers, engineering leads — each would want something different from this tool.

# Transcript Intelligence — Take-Home Assignment

## Overview

This project implements a **Transcript Intelligence pipeline** for analyzing enterprise SaaS meeting transcripts across:

- Customer Support Calls
- External Customer Calls
- Internal Team Discussions

The objective is to transform raw transcript data into structured business intelligence that can help:

- Support Leaders identify recurring operational issues
- Sales Teams predict churn risk
- Product Managers understand customer feedback trends
- Engineering Leaders measure incident impact and escalation patterns

The project focuses on:
1. Theme / Topic Categorization
2. Sentiment Analysis
3. Additional Stakeholder Insights
4. Explainable & Auditable Processing Logic

---

# Dataset

The dataset contains approximately **100 meeting transcripts** from a fictional enterprise SaaS company, **AegisCloud**.

Each meeting folder contains:

```bash
dataset/
 └── <meeting-folder>/
      ├── meeting-info.json
      ├── summary.json
      ├── transcript.json
      ├── speakers.json
      └── speaker-meta.json
```

## File Descriptions

| File | Purpose |
|------|---------|
| meeting-info.json | Metadata (title, organizer, attendees, duration) |
| summary.json | AI-generated summary, topics, action items, sentiment |
| transcript.json | Sentence-level transcript + sentiment |
| speakers.json | Speaker turn timestamps |
| speaker-meta.json | Speaker metadata |

---

# Project Goals

The implementation addresses all required tasks from the assignment.

## 1. Transcript Processing & Theme Categorization

- Built a structured processing pipeline
- Classified calls into business-relevant themes
- Used a hybrid approach combining:
  - Existing AI-generated topic tags
  - Rule-based business mapping

---

## 2. Sentiment Analysis

Analyzed sentiment trends across:
- Support calls
- External customer calls
- Internal team calls

Compared:
- Numeric sentiment scores
- Overall sentiment labels
- Sentence-level sentiment signals

---

## 3. Additional Business Insights

Implemented additional intelligence layers including:
- Churn Risk Detection
- Incident Blast Radius Analysis
- Speaker Dynamics
- Action Item Intelligence

---

# Technical Approach

## Why a Hybrid Approach?

The dataset already contained:
- AI-generated summaries
- Topic tags
- Key moments
- Action items

Instead of re-running expensive LLM classification, the implementation combines:

### Existing LLM Metadata
Used existing AI-generated topics from `summary.json`

### Rule-Based Business Mapping
Mapped topics into meaningful business themes.

---

## Why Not Pure Clustering?

Traditional embedding clustering groups vocabulary similarity, not business intent.

Example:
- "billing contract issue"
- "compliance contract review"

Both contain similar vocabulary but represent completely different business workflows.

Rule-based mapping preserves explainability and business meaning.

---

## Why Not Pure LLM Classification?

- Higher cost
- Less deterministic
- Unnecessary for highly structured patterns

The chosen approach is:
- Fast
- Auditable
- Deterministic
- Easy to extend

---

# Pipeline Architecture

## Step 1 — Data Loading

The pipeline:
- Iterates through all transcript folders
- Loads JSON files
- Aggregates:
  - transcript text
  - sentiment signals
  - speaker timing
  - action items
  - topics
  - metadata

Outputs a unified Pandas DataFrame.

---

## Step 2 — Call Type Classification

Calls are classified into:

| Type | Logic |
|------|------|
| Support | Titles containing support/escalation keywords |
| External | Customer-facing "Aegis / Client" meetings |
| Internal | Everything else |

### Reasoning

Meeting titles were highly structured and predictable, making rule-based classification:
- Accurate
- Explainable
- Lightweight

---

## Step 3 — Theme Categorization

Implemented multi-label classification into 11 business themes:

| Theme |
|------|
| Compliance & Audit |
| Incident Response |
| Renewal & Retention |
| Integration & Backup |
| Product Planning |
| Threat Detection |
| Product Feedback & Strategy |
| Identity & Access |
| Onboarding & Adoption |
| Billing & Licensing |
| Team Sync |

### Multi-label Design

A single call may belong to multiple themes.

Example:
- A Detect outage escalation may map to:
  - Incident Response
  - Threat Detection

---

# Sentiment Analysis

The dataset provides:
- Overall sentiment labels
- Numeric sentiment scores (1–5)
- Sentence-level sentiment signals

## Key Findings

### Support Calls
- Average sentiment: ~2.77/5
- Structurally negative
- Strong correlation with outages/escalations

### External Calls
- Average sentiment: ~3.90/5
- Generally positive
- Hidden churn risk signals underneath positive conversations

### Internal Calls
- Average sentiment: ~3.48/5
- Mostly neutral operational coordination

### Important Observation
31% of all calls scored ≤ 2.5, indicating widespread negative operational pressure.

---

# Additional Insights

## 1. Churn Risk Scoring

Built a weighted heuristic churn scoring model.

### Signals Used

| Signal | Weight |
|--------|--------|
| Low sentiment | +2 |
| churn_signal key moment | +3 |
| concern key moment | +1 |
| URGENT/ESCALATION title | +2 |

### Output Labels
- High
- Medium
- Low

### Business Value

Can be integrated directly into CRM/account management workflows.

---

## 2. Incident Blast Radius

Detected systemic outage patterns by linking:
- Escalation calls
- Support tickets
- Internal war rooms
- Customer-facing meetings

### Key Finding

The "Detect" outage propagated across all three call types.

### Why It Matters

Engineering teams can measure:
- Operational impact
- Customer impact
- Escalation spread
- Cross-functional disruption

---

## 3. Speaker Dynamics

Computed:
- Number of speakers
- Dominant speaker %
- Talk-time imbalance

### Observation

Calls dominated by a single speaker (>70%) correlated with poorer sentiment outcomes.

### Potential Use Case

"Conversation balance" scoring for customer-facing teams.

---

## 4. Action Item Intelligence

Extracted:
- Action items
- Owners
- Contextual tasks

### Key Finding

~400 action items across 100 calls were already present but untracked.

### Opportunity

Build:
- Automatic task extraction
- Cross-call deduplication
- Jira/CRM integration

---

# Key Business Takeaways

## Compliance & Audit is Cross-Cutting

Compliance requirements impacted every department, making it the highest-value platform capability.

---

## Support Sentiment is Structurally Negative

Negative support calls were driven by systemic operational failures rather than isolated issues.

---

## Churn Risk is Immediately Actionable

43% of external accounts showed medium or high churn risk using only existing structured data.

---

## Action Items Are Underutilized

Significant operational intelligence already exists in transcript metadata but is not operationalized.

---

## The Dataset Is Richer Than It Appears

The transcript system already contains:
- speaker turns
- sentence-level sentiment
- key moments
- ownership signals
- escalation patterns

The challenge is not data collection — it is intelligence extraction.

---

# Tech Stack

- Python
- Pandas
- NumPy
- JSON Processing
- Rule-Based NLP
- Heuristic Scoring

---

# How to Run

## Install Dependencies

```bash
pip install pandas numpy
```

## Run Analysis

```bash
python transcript_intelligence_notebook.py
```

---

# Future Improvements

## Potential Enhancements

- Embedding-based semantic clustering
- Real-time transcript ingestion
- Streamlit dashboard
- CRM integration
- LLM-assisted summarization
- Predictive churn models
- Action item tracking system

---

# Deliverables

This submission includes:
- Slide Deck (leadership presentation)
- Analysis Code / Notebook
- README Documentation
- Video Walkthrough

---

# Final Thought

This project demonstrates that enterprise transcript systems already contain significant operational intelligence.

The real opportunity is not transcription itself —  
it is converting conversation data into:
- product insight
- customer risk signals
- operational awareness
- decision support systems.
