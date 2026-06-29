# AI Candidate Ranking System - Redrob Challenge

This project is an AI-based candidate ranking system built for the Redrob recruitment challenge.

The goal is to rank candidates the way a strong recruiter would: not by simply matching keywords, but by understanding the job description, evaluating candidate profiles deeply, and generating a trusted shortlist of the most relevant candidates.

---

## Problem Statement

Recruiters often go through hundreds of candidate profiles and still miss the right person. Traditional keyword-based filters may fail because they cannot understand real candidate fit.

A candidate may mention the right keywords but still be a poor match. Another candidate may be highly suitable but may not use the exact same words as the job description.

This project solves that problem by ranking candidates based on multiple signals such as skills, experience, seniority, career history, behavioral signals, platform activity, logistics fit, and profile quality.

---

## Solution Overview

The system reads candidate data from a JSONL file and evaluates each candidate against the target job role.

Instead of relying only on keyword matching, the ranking pipeline uses a hybrid recruiter-style scoring approach. It checks whether the candidate genuinely fits the role based on their full profile.

The output is a ranked `submission.csv` file containing the top recommended candidates.

---

## Key Features

* Understands job requirements beyond simple keyword matching
* Evaluates candidate skills and experience
* Checks AI/ML, LLM, ranking, retrieval, and engineering relevance
* Considers behavioral signals such as platform activity and recruiter interest
* Considers logistics such as location, notice period, salary, and relocation
* Penalizes suspicious profiles, keyword stuffing, and inconsistent data
* Produces a valid ranked output file for submission

---

## Project Structure

```text
redrob-ai-candidate-ranking/
│
├── rank.py
├── submission.csv
├── README.md
├── .gitignore
├── redrob_ai_candidate_ranking_template_filled.pdf
└── redrob_ai_candidate_ranking_template_filled.pptx
```

---

## Files

| File                                               | Description                                                        |
| -------------------------------------------------- | ------------------------------------------------------------------ |
| `rank.py`                                          | Main candidate ranking pipeline                                    |
| `submission.csv`                                   | Final ranked output file containing recommended candidates         |
| `README.md`                                        | Project documentation                                              |
| `.gitignore`                                       | Prevents dataset, cache, and unnecessary files from being uploaded |
| `redrob_ai_candidate_ranking_template_filled.pdf`  | Final presentation deck in PDF format                              |
| `redrob_ai_candidate_ranking_template_filled.pptx` | Editable presentation deck                                         |

---

## How the System Works

The ranking system follows this workflow:

```text
Job Description + Candidate Dataset
        ↓
Candidate Profile Parsing
        ↓
Skill, Experience, and Seniority Scoring
        ↓
Behavioral and Platform Signal Evaluation
        ↓
Logistics Fit Evaluation
        ↓
Suspicious Profile and Keyword Stuffing Penalty
        ↓
Final Weighted Ranking
        ↓
Top 100 Candidate Shortlist
        ↓
submission.csv
```

---

## Ranking Methodology

The system combines multiple recruiter-style signals into a final ranking score.

### 1. Job Description Understanding

The system identifies important role requirements such as:

* AI/ML experience
* LLM and GenAI exposure
* Search, ranking, retrieval, or recommendation experience
* Python and backend engineering ability
* Product-building experience
* Seniority level
* Location and availability fit

### 2. Candidate Evaluation

Each candidate is evaluated using:

* Skills
* Current and past job titles
* Career history
* AI/ML relevance
* Product and engineering experience
* Behavioral activity
* Recruiter engagement signals
* GitHub or technical activity
* Salary expectation
* Notice period
* Location and relocation flexibility

### 3. Red Flag Detection

The system also applies penalties for weak or suspicious profiles, such as:

* Keyword stuffing
* Non-technical profiles using AI buzzwords
* Unrealistic experience
* Inconsistent salary information
* Weak career alignment
* Suspicious or low-quality profile data

### 4. Final Ranking

All signals are combined into a final weighted score. The highest-scoring candidates are selected as the final recommended shortlist.

---

## How to Run

### Step 1: Place the dataset file

Make sure the candidate dataset file is available locally.

Expected input file:

```text
candidates.jsonl
```

Example folder structure before running:

```text
redrob-ai-candidate-ranking/
│
├── rank.py
├── candidates.jsonl
└── README.md
```

---

### Step 2: Run the ranking script

Use this command:

```bash
python3 rank.py --candidates candidates.jsonl --out submission.csv
```

---

### Optional: Choose number of candidates

By default, the system outputs the top 100 candidates.

You can also specify the number manually:

```bash
python3 rank.py --candidates candidates.jsonl --out submission.csv --top-n 100
```

---

## Output

After running the script, the system generates:

```text
submission.csv
```

This file contains the ranked candidate shortlist in the required submission format.

---

## Validation

The generated output was validated using the official validator script.

Result:

```text
Submission is valid.
```

---

## Technologies Used

* Python
* JSONL file processing
* CSV generation
* Rule-based semantic scoring
* Hybrid candidate ranking logic
* Recruiter-style signal weighting

---

## Why This Approach

This solution was designed to be practical, explainable, and lightweight.

It does not depend only on exact keyword matches. Instead, it evaluates the full candidate profile and tries to identify who is genuinely suitable for the role.

The system is also explainable because every ranking decision is based on clear scoring signals and penalties.

---

## Submission Assets

This repository includes:

* Working ranking code
* Final ranked candidate output
* Presentation deck explaining the approach
* Documentation and run instructions

---

## Author

Bibek Roy
