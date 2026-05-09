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

## Output
```
(venv) arshad@Arshads-MacBook-Pro transcript-intelligence % python transcript_intelligence_notebook.py
Loaded 100 calls

Call type distribution:
call_type
external    39
support     32
internal    29
Name: count, dtype: int64

Theme distribution across all calls:
Compliance & Audit             68
Incident Response              55
Renewal & Retention            41
Product Planning               37
Integration & Backup           36
Threat Detection               34
Product Feedback & Strategy    24
Identity & Access              23
Billing & Licensing            13
Onboarding & Adoption          10
Team Sync                      10
Name: count, dtype: int64

Sentiment score by call type:
call_type
external    3.90
internal    3.48
support     2.77
Name: sentiment_score, dtype: float64

Sentiment label distribution by call type:
overall_sentiment  mixed-negative  mixed-positive  negative  positive  very-negative  very-positive
call_type                                                                                          
external                        9              13         0         3              0             14
internal                        7              11         2         4              0              5
support                        17               9         2         0              2              2

Sentiment by theme (avg score):
Threat Detection               2.83
Incident Response              2.90
Renewal & Retention            3.14
Product Feedback & Strategy    3.42
Team Sync                      3.46
Identity & Access              3.60
Compliance & Audit             3.63
Integration & Backup           3.64
Billing & Licensing            3.69
Product Planning               3.81
Onboarding & Adoption          4.70
dtype: float64

Churn risk distribution (external calls only):
churn_risk
High      17
Medium    16
Low        6
Name: count, dtype: int64

Detect outage-related calls: 9
                                                                title call_type  sentiment_score overall_sentiment
1                                 Detect Outage - Root Cause Analysis  internal              2.4    mixed-negative
5                                   Detect Outage - Escalation Bridge  internal              1.8          negative
7   ESCALATION: Northstar Pharma - Detect Outage Impact on Compliance   support              2.1    mixed-negative
12                         Detect Outage - Customer Impact Assessment  internal              1.8          negative
22                               Detect Outage - Post-Incident Review  internal              2.8    mixed-negative
32              URGENT: Cobalt Software - Aegis Detect Dashboard Down   support              1.8          negative
66                       INCIDENT: Detect Pipeline Failure - War Room   support              1.8          negative
70                            Detect Outage - Remediation Plan Review  internal              2.4    mixed-negative
73            Aegis / Northstar Pharma - Urgent: Detect Outage Impact   support              2.1    mixed-negative

Avg speakers by call type:
call_type
external    2.9
internal    3.8
support     2.7
Name: num_speakers, dtype: float64

Avg duration by call type (minutes):
call_type
external    38.5
internal    31.4
support     19.3
Name: duration, dtype: float64

Avg dominant speaker % by call type:
call_type
external    47.6
internal    39.3
support     50.9
Name: dominant_speaker_pct, dtype: float64

Action items by call type:
           count      mean       std  min  25%  50%  75%  max
call_type                                                    
external    39.0  3.948718  0.223456  3.0  4.0  4.0  4.0  4.0
internal    29.0  4.000000  0.000000  4.0  4.0  4.0  4.0  4.0
support     32.0  3.968750  0.176777  3.0  4.0  4.0  4.0  4.0

Sample action items from: Aegis / Summit Trust - Platform Concerns Discussion
  → Maria Santos: Escalate the open MFA ticket today and provide Alicia with a real status update by end of day
  → Maria Santos: Schedule a technical Identity specialist call for next week and deliver a written summary with accountable action items, owners, and timelines afterward
  → Maria Santos: Send Alicia a formal post-incident report on the March Detect outage and arrange an engineering briefing for her CISO on the architecture changes
  → Maria Santos: Develop a formal remediation plan for all Aegis Identity issues including MFA enforcement, session timeout behavior, and support response quality

======================================================================
KEY FINDINGS SUMMARY
======================================================================
Total calls analyzed: 100
Call types: {'external': 39, 'support': 32, 'internal': 29}
Avg sentiment - External: 3.90
Avg sentiment - Internal: 3.48
Avg sentiment - Support:  2.77
Negative calls (≤2.5): 31 (31%)
High churn risk accounts: 17
Total action items: 397
Most common theme: Compliance & Audit
(venv) arshad@Arshads-MacBook-Pro transcript-intelligence % 
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
