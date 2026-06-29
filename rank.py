#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Discovery Ranker

CPU-only, no-network, standard-library implementation.
It streams candidates.jsonl, scores every candidate using a hybrid recruiter-style
feature model, and writes the top-N ranked CSV in the required format.

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import heapq
import json
import math
import re
from datetime import date
from pathlib import Path

REFERENCE_DATE = date(2026, 6, 29)

SERVICE_COMPANIES = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant",
    "Capgemini", "HCL", "Tech Mahindra", "Mindtree", "Mphasis",
}

PRODUCT_INDUSTRIES = {
    "Software", "AI/ML", "SaaS", "Fintech", "E-commerce", "Food Delivery",
    "EdTech", "HealthTech", "HealthTech AI", "Conversational AI", "AdTech",
    "Transportation", "Gaming", "Insurance Tech", "Internet", "Media",
    "Consumer Electronics", "AI Services", "Voice AI",
}

PREFERRED_CITIES = ("Pune", "Noida", "Gurgaon", "Delhi")
WELCOME_CITIES = (
    "Hyderabad", "Mumbai", "Bangalore", "Bengaluru", "Chennai", "Jaipur",
    "Kolkata", "Ahmedabad", "Chandigarh", "Coimbatore", "Indore", "Kochi",
    "Trivandrum", "Vizag", "Bhubaneswar",
)

TITLE_WEIGHT = {
    "Senior AI Engineer": 1.00,
    "Lead AI Engineer": 0.98,
    "Staff Machine Learning Engineer": 0.96,
    "Senior Machine Learning Engineer": 0.94,
    "Senior NLP Engineer": 0.95,
    "Search Engineer": 0.92,
    "Recommendation Systems Engineer": 0.92,
    "Applied ML Engineer": 0.88,
    "Senior Applied Scientist": 0.84,
    "Machine Learning Engineer": 0.82,
    "AI Engineer": 0.82,
    "NLP Engineer": 0.80,
    "ML Engineer": 0.72,
    "Senior Software Engineer (ML)": 0.70,
    "Senior Data Scientist": 0.65,
    "Data Scientist": 0.55,
    "Senior Software Engineer": 0.38,
    "Senior Data Engineer": 0.34,
    "Backend Engineer": 0.32,
    "Data Engineer": 0.32,
    "Analytics Engineer": 0.22,
    "Software Engineer": 0.18,
    "DevOps Engineer": 0.10,
    "Cloud Engineer": 0.10,
    "Full Stack Developer": 0.09,
    "Java Developer": 0.05,
    ".NET Developer": 0.05,
}

NONTECH_TITLES = {
    "HR Manager", "Marketing Manager", "Sales Executive", "Graphic Designer",
    "Content Writer", "Accountant", "Operations Manager", "Customer Support",
    "Business Analyst", "Project Manager", "Mechanical Engineer", "Civil Engineer",
}

PROFICIENCY_WEIGHT = {"beginner": 0.55, "intermediate": 0.75, "advanced": 1.0, "expert": 1.18}

SKILL_WEIGHT = {
    # Core IR / ranking skills
    "Information Retrieval": 3.5,
    "Information Retrieval Systems": 3.8,
    "Ranking Systems": 3.8,
    "Learning to Rank": 3.5,
    "Search & Discovery": 3.3,
    "Search Infrastructure": 3.0,
    "Search Backend": 2.6,
    "Recommendation Systems": 3.0,
    "Semantic Search": 3.2,
    "Vector Search": 3.2,
    "Embeddings": 3.1,
    "Vector Representations": 2.4,
    "Text Encoders": 2.2,
    "BM25": 2.5,
    "Content Matching": 2.0,
    "Indexing Algorithms": 2.0,
    # Infrastructure
    "Pinecone": 2.0,
    "FAISS": 2.1,
    "Qdrant": 2.0,
    "Milvus": 2.0,
    "Weaviate": 2.0,
    "pgvector": 1.7,
    "OpenSearch": 1.6,
    "Elasticsearch": 1.5,
    "Sentence Transformers": 2.4,
    # ML / LLM support skills
    "NLP": 2.0,
    "Natural Language Processing": 2.1,
    "LLMs": 1.7,
    "RAG": 1.8,
    "Fine-tuning LLMs": 1.4,
    "Python": 2.0,
    "Machine Learning": 1.4,
    "MLOps": 1.2,
    "MLflow": 0.8,
    "Feature Engineering": 1.2,
    "PyTorch": 0.8,
    "Hugging Face Transformers": 1.0,
    # Useful engineering support
    "Data Pipelines": 0.7,
    "Kafka": 0.5,
    "Spark": 0.5,
    "Airflow": 0.5,
    "Kubernetes": 0.5,
    "Docker": 0.3,
}

CORE_SKILLS = {
    "Information Retrieval", "Information Retrieval Systems", "Ranking Systems",
    "Learning to Rank", "Search & Discovery", "Recommendation Systems",
    "Semantic Search", "Vector Search", "Embeddings", "BM25", "Search Infrastructure",
}
VECTOR_SKILLS = {
    "Pinecone", "FAISS", "Qdrant", "Milvus", "Weaviate", "pgvector",
    "OpenSearch", "Elasticsearch", "Vector Search", "Embeddings", "Sentence Transformers",
}
EVALUATION_SKILLS = {
    "Ranking Systems", "Learning to Rank", "BM25", "Information Retrieval", "Information Retrieval Systems",
}
NEGATIVE_SPECIALTY_SKILLS = {
    "Computer Vision", "Image Classification", "Object Detection", "OpenCV", "YOLO", "CNN",
    "Speech Recognition", "ASR", "TTS", "GANs", "Diffusion Models",
}

# Each phrase group is a recruiter interpretation of the JD. It intentionally includes
# plain-language evidence, not only exact tool names.
PHRASE_GROUPS = {
    "prod_retrieval": [
        "production retrieval", "production search", "production ranking",
        "production recommendation", "deployed retrieval", "deployed search",
        "deployed ranking", "serving 10m", "serving 50m", "serving millions",
        "serving 35m", "serving 30m", "low latency", "index refresh", "embedding drift",
    ],
    "hybrid_retrieval": [
        "hybrid retrieval", "bm25 + dense", "sparse and dense", "keyword-based ranking to",
        "keyword-search-based", "embedding-based search", "embedding-based retrieval",
        "embedding-based ranking", "dense retrieval", "vector recall",
    ],
    "ranking_eval": [
        "ndcg", "mrr", "map@", "offline-online", "offline metrics", "online metrics",
        "a/b test", "ab test", "eval harness", "evaluation harness", "evaluation framework",
        "relevance labeling", "human judgments", "preference pairs",
    ],
    "rank_models": [
        "learning-to-rank", "learned rank", "xgboost", "lightgbm", "re-rank", "rerank",
        "ranking model", "candidate sourcing",
    ],
    "candidate_matching": [
        "candidate-jd", "candidate jd", "candidate matching", "recruiter-facing search",
        "recruiter engagement", "talent intelligence", "candidate corpus", "recruiters",
    ],
    "llm_depth": [
        "fine-tuned", "fine-tuning", "lora", "qlora", "peft", "llm-based re-ranker",
        "llm-based reranker", "openai embeddings", "bge", "e5", "mpnet", "sentence-transformer",
    ],
    "product_shipper": [
        "shipped", "built and shipped", "owned end-to-end", "rollout", "production load",
        "worked closely with pm", "product judgment", "real users", "customer-facing",
    ],
    "plain_ir": [
        "connect users with relevant", "surface relevant content", "search and discovery experience",
        "what users are looking for", "relevant matches", "large dataset", "information at scale",
    ],
}

PHRASE_WEIGHT = {
    "prod_retrieval": 5.0,
    "hybrid_retrieval": 5.0,
    "ranking_eval": 4.6,
    "rank_models": 3.8,
    "candidate_matching": 3.6,
    "llm_depth": 2.6,
    "product_shipper": 2.0,
    "plain_ir": 3.8,
}

NEGATIVE_SUBSTRINGS = [
    ("pure research environment", 6.0),
    ("academic lab", 6.0),
    ("chatgpt and a few other tools", 4.0),
    ("ai tools for productivity", 4.0),
    ("transitioning toward", 2.2),
    ("looking to grow into", 2.2),
    ("want to grow into", 2.2),
    ("still building depth", 2.2),
    ("professional experience there is limited", 2.2),
    ("technical depth in ai is limited", 4.0),
    ("production deployment was handled by the platform team", 1.5),
    ("most of my project work has been in cv", 2.0),
    ("digital transformation strategy", 2.0),
]

GOOD_SKILL_ORDER = [
    "Information Retrieval", "Information Retrieval Systems", "Ranking Systems", "Learning to Rank",
    "Search & Discovery", "Recommendation Systems", "Semantic Search", "Vector Search",
    "Embeddings", "BM25", "FAISS", "Pinecone", "Qdrant", "Milvus", "Weaviate",
    "Python", "NLP", "RAG",
]

YEAR_RE = re.compile(r"(\d{1,2}(?:\.\d)?)\+?\s+years?")


def clamp(value, low, high):
    return max(low, min(high, value))


def parse_iso_date(value):
    try:
        return date.fromisoformat(value[:10]) if value else None
    except Exception:
        return None


def candidate_text(candidate):
    profile = candidate["profile"]
    parts = [
        profile.get("current_title", ""),
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_industry", ""),
        profile.get("current_company", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.extend([
            job.get("title", ""), job.get("company", ""), job.get("industry", ""), job.get("description", "")
        ])
    parts.extend(skill.get("name", "") for skill in candidate.get("skills", []))
    return " ".join(parts)


def phrase_score(lower_text):
    score = 0.0
    hits = []
    for group, substrings in PHRASE_GROUPS.items():
        count = sum(1 for sub in substrings if sub in lower_text)
        if count:
            score += PHRASE_WEIGHT[group] * (1.0 + 0.22 * min(count - 1, 4))
            hits.append(group)

    # Controlled repetition bonus: recognizes profiles with sustained evidence,
    # while capping keyword stuffing.
    for term, weight in [
        ("retrieval", 0.35), ("ranking", 0.30), ("embedding", 0.25),
        ("recommendation", 0.25), ("search", 0.18), ("vector", 0.18),
        ("evaluation", 0.25),
    ]:
        score += min(lower_text.count(term), 8) * weight
    return score, hits


def skill_score(candidate):
    years = candidate["profile"].get("years_of_experience") or 0.0
    total = core = vector = evaluation = negative_specialty = 0.0
    expert_zero = unreasonable_duration = 0
    names = []

    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        names.append(name)
        proficiency = PROFICIENCY_WEIGHT.get(str(skill.get("proficiency", "")).lower(), 0.7)
        duration = skill.get("duration_months") or 0
        duration_multiplier = clamp(duration / 36.0, 0.25, 1.25)
        endorsement_multiplier = 1.0 + min(skill.get("endorsements") or 0, 80) / 400.0
        value = SKILL_WEIGHT.get(name, 0.0) * proficiency * duration_multiplier * endorsement_multiplier
        total += value

        if name in CORE_SKILLS:
            core += value
        if name in VECTOR_SKILLS:
            vector += value
        if name in EVALUATION_SKILLS:
            evaluation += value
        if name in NEGATIVE_SPECIALTY_SKILLS:
            negative_specialty += 0.35 * proficiency * duration_multiplier
        if proficiency >= 1.1 and duration <= 1:
            expert_zero += 1
        if years and duration > years * 12 + 24:
            unreasonable_duration += 1

    score = min(total, 38.0)
    score += min(core, 18.0) * 0.45
    score += min(vector, 10.0) * 0.25
    score += min(evaluation, 8.0) * 0.20
    return {
        "score": score,
        "core": core,
        "vector": vector,
        "negative_specialty": negative_specialty,
        "expert_zero": expert_zero,
        "unreasonable_duration": unreasonable_duration,
        "names": names,
    }


def experience_score(years):
    if years < 2:
        return -20.0
    if years < 3.5:
        return -12.0
    base = 14.0 * math.exp(-((years - 7.0) / 2.8) ** 2)
    if 5 <= years <= 9:
        base += 4.0
    elif 4 <= years < 5 or 9 < years <= 10.5:
        base += 1.0
    elif years > 12:
        base -= min((years - 12) * 2.2, 8.0)
    elif years < 4:
        base -= 4.0
    return base


def behavior_score(signals, location, country):
    score = 4.0 if signals.get("open_to_work_flag") else -2.0

    active_date = parse_iso_date(signals.get("last_active_date"))
    if active_date:
        days_inactive = (REFERENCE_DATE - active_date).days
        if days_inactive < 0:
            score -= 5.0
        elif days_inactive <= 14:
            score += 5.0
        elif days_inactive <= 45:
            score += 3.0
        elif days_inactive <= 90:
            score += 1.0
        elif days_inactive <= 180:
            score -= 3.0
        else:
            score -= 7.0

    response_rate = signals.get("recruiter_response_rate", 0.0) or 0.0
    score += 7.0 * response_rate

    response_time = signals.get("avg_response_time_hours", 999.0) or 999.0
    score += 4.0 if response_time <= 24 else 2.0 if response_time <= 72 else 0.5 if response_time <= 120 else -2.0

    notice = signals.get("notice_period_days", 90) or 0
    score += 4.0 if notice <= 30 else 1.5 if notice <= 60 else -0.5 if notice <= 90 else -3.0

    if signals.get("verified_email"):
        score += 0.7
    if signals.get("verified_phone"):
        score += 0.5
    if signals.get("linkedin_connected"):
        score += 0.8

    github = signals.get("github_activity_score", -1)
    score += min(github, 100) / 20.0 if github is not None and github >= 0 else -0.5
    score += min(signals.get("saved_by_recruiters_30d", 0) or 0, 40) / 5.0
    score += min(signals.get("profile_views_received_30d", 0) or 0, 250) / 70.0
    score += min(signals.get("applications_submitted_30d", 0) or 0, 20) / 8.0
    score += 2.0 * (signals.get("interview_completion_rate", 0.0) or 0.0)

    offer_acceptance = signals.get("offer_acceptance_rate", -1)
    if offer_acceptance is not None and offer_acceptance >= 0:
        score += 1.5 * offer_acceptance

    if any(city in location for city in PREFERRED_CITIES):
        score += 4.0
    elif any(city in location for city in WELCOME_CITIES):
        score += 2.0

    score += 2.0 if country == "India" else -3.0

    if signals.get("willing_to_relocate"):
        score += 2.5
    elif not any(city in location for city in PREFERRED_CITIES):
        score -= 1.2

    if signals.get("preferred_work_mode") in ("hybrid", "flexible"):
        score += 1.0
    return score


def product_company_score(candidate):
    profile = candidate["profile"]
    product_months = 0
    service_months = 0
    only_services = True

    for job in candidate.get("career_history", []):
        industry = job.get("industry")
        company = job.get("company")
        duration = job.get("duration_months") or 0
        if industry in PRODUCT_INDUSTRIES:
            product_months += duration
            only_services = False
        if company in SERVICE_COMPANIES or industry in {"IT Services", "Consulting"}:
            service_months += duration
        else:
            only_services = False

    score = min(product_months / 12.0, 7.0) * 1.2
    if profile.get("current_industry") in PRODUCT_INDUSTRIES:
        score += 2.5
    if only_services and service_months > 0:
        score -= 7.0
    elif service_months and product_months < 18:
        score -= 2.0
    if profile.get("current_company_size") in ("51-200", "201-500", "501-1000"):
        score += 0.8
    return score


def suspicious_penalty(candidate, lower_text, skill_features):
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    years = profile.get("years_of_experience") or 0.0
    penalty = 0.0
    flags = []

    signup = parse_iso_date(signals.get("signup_date"))
    active = parse_iso_date(signals.get("last_active_date"))
    if signup and active and active < signup:
        penalty += 35.0
        flags.append("last_active_before_signup")

    salary = signals.get("expected_salary_range_inr_lpa") or {}
    if salary.get("min") is not None and salary.get("max") is not None and salary.get("min") > salary.get("max"):
        penalty += 30.0
        flags.append("salary_min_gt_max")

    explicit_years = []
    headline_summary = (profile.get("summary", "") + " " + profile.get("headline", "")).lower()
    for match in YEAR_RE.finditer(headline_summary):
        try:
            explicit_years.append(float(match.group(1)))
        except ValueError:
            pass
    if explicit_years and max(abs(years - value) for value in explicit_years) > 4.0:
        penalty += 25.0
        flags.append("experience_text_mismatch")

    total_career_months = sum(job.get("duration_months") or 0 for job in candidate.get("career_history", []))
    if years and total_career_months and (total_career_months > years * 12 + 42 or total_career_months < years * 12 - 72):
        penalty += 6.0
        flags.append("career_duration_mismatch")

    if skill_features["expert_zero"] >= 4:
        penalty += 12.0
        flags.append("expert_zero_stuffing")
    if skill_features["unreasonable_duration"] >= 4:
        penalty += 12.0
        flags.append("unrealistic_skill_duration")

    if profile.get("current_title") in NONTECH_TITLES and skill_features["core"] > 5.0:
        penalty += 18.0
        flags.append("nontech_keyword_stuffing")

    if skill_features["negative_specialty"] > 3.2 and skill_features["core"] < 7.0:
        penalty += 7.0
        flags.append("cv_speech_dominant")

    for substring, weight in NEGATIVE_SUBSTRINGS:
        if substring in lower_text:
            penalty += weight
            flags.append(substring[:18])

    return penalty, flags


def score_candidate(candidate):
    profile = candidate["profile"]
    text = candidate_text(candidate)
    lower_text = text.lower()
    skills = skill_score(candidate)
    text_score, phrase_hits = phrase_score(lower_text)
    years = profile.get("years_of_experience") or 0.0

    score_parts = {
        "title": 22.0 * TITLE_WEIGHT.get(profile.get("current_title"), 0.0),
        "skills": skills["score"],
        "text": text_score,
        "experience": experience_score(years),
        "behavior": behavior_score(candidate["redrob_signals"], profile.get("location", ""), profile.get("country", "")),
        "product_company": product_company_score(candidate),
        "balance": 0.0,
        "penalty": 0.0,
        "flags": [],
        "skill_names": skills["names"],
        "phrase_hits": phrase_hits,
    }

    if skills["core"] < 4.0 and text_score < 12.0:
        score_parts["balance"] -= 10.0
    if "Python" not in skills["names"] and "python" not in lower_text:
        score_parts["balance"] -= 3.0
    if profile.get("current_title") in NONTECH_TITLES:
        score_parts["balance"] -= 18.0
    if profile.get("country") != "India" and not candidate["redrob_signals"].get("willing_to_relocate"):
        score_parts["balance"] -= 8.0

    penalty, flags = suspicious_penalty(candidate, lower_text, skills)
    score_parts["penalty"] = penalty
    score_parts["flags"] = flags

    total = (
        score_parts["title"]
        + score_parts["skills"]
        + score_parts["text"]
        + score_parts["experience"]
        + score_parts["behavior"]
        + score_parts["product_company"]
        + score_parts["balance"]
        - score_parts["penalty"]
    )

    # Hard-cap clear honeypots/inconsistent profiles so they cannot enter the top list.
    if penalty >= 30.0:
        total = min(total, 15.0 - penalty / 2.0)

    return total, score_parts


def build_reasoning(candidate, score_parts):
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    skill_names = score_parts["skill_names"]
    top_skills = [skill for skill in GOOD_SKILL_ORDER if skill in skill_names]

    profile_and_jobs = (
        profile.get("summary", "") + " " + " ".join(job.get("description", "") for job in candidate.get("career_history", []))
    ).lower()
    evidence = "production ML/search work"
    for phrase in [
        "candidate-jd matching", "hybrid retrieval", "embedding-based search", "semantic search",
        "ranking pipeline", "recommendation system", "learning-to-rank", "evaluation harness", "a/b test",
    ]:
        if phrase in profile_and_jobs:
            evidence = phrase
            break

    concerns = []
    notice = signals.get("notice_period_days")
    if notice is not None and notice > 60:
        concerns.append(f"{notice}d notice")
    response = signals.get("recruiter_response_rate", 0.0) or 0.0
    if response < 0.45:
        concerns.append("lower recruiter response")
    if profile.get("country") != "India":
        concerns.append("outside India")
    if score_parts["penalty"] > 0 and score_parts["flags"]:
        concerns.append("minor data-quality concern")

    skills_text = ", ".join(top_skills[:5]) if top_skills else "relevant ML/system skills"
    concern_text = f" Concern: {', '.join(concerns[:2])}." if concerns else ""

    return (
        f"{profile.get('current_title')} with {profile.get('years_of_experience')} yrs in "
        f"{profile.get('current_industry')} at {profile.get('current_company')}; evidence of {evidence} "
        f"and skills in {skills_text}. Behavioral/logistics fit: response {response:.2f}, "
        f"notice {notice}d, {profile.get('location')}.{concern_text}"
    )


def rank_candidates(candidates_path, top_n):
    heap = []
    with open(candidates_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            candidate = json.loads(line)
            raw_score, score_parts = score_candidate(candidate)
            # Heap item keeps the worst current row at heap[0]. For equal raw scores,
            # lower candidate_id is preferred by the final sort.
            item = (raw_score, candidate["candidate_id"], candidate, score_parts)
            if len(heap) < top_n:
                heapq.heappush(heap, item)
            elif raw_score > heap[0][0] or (raw_score == heap[0][0] and candidate["candidate_id"] < heap[0][1]):
                heapq.heapreplace(heap, item)
    return sorted(heap, key=lambda row: (-row[0], row[1]))


def write_submission(rows, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (raw_score, candidate_id, candidate, score_parts) in enumerate(rows, start=1):
            # Non-increasing; raw scores are sorted. 9 decimals prevents accidental ties from formatting.
            normalized_score = raw_score / 200.0
            writer.writerow([
                candidate_id,
                rank,
                f"{normalized_score:.9f}",
                build_reasoning(candidate, score_parts),
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to output")
    args = parser.parse_args()

    rows = rank_candidates(Path(args.candidates), args.top_n)
    write_submission(rows, Path(args.out))
    print(f"Wrote {len(rows)} ranked candidates to {args.out}")


if __name__ == "__main__":
    main()
