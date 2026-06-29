# Redrob Intelligent Candidate Discovery Ranker

This is a CPU-only, no-network ranker for the Redrob candidate discovery challenge.
It reads `candidates.jsonl`, scores every candidate, and produces the required top-100 CSV.

## Run

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

The implementation uses only the Python standard library. On the provided 100,000-candidate JSONL file it streams the file and keeps only a top-N heap in memory.

## Methodology

The ranker combines six recruiter-style scoring layers:

1. **Role understanding**: converts the JD into weighted evidence groups: production retrieval/search, hybrid retrieval, ranking evaluation, learning-to-rank, candidate matching, LLM depth, product shipping, and plain-language IR evidence.
2. **Skill interpretation**: scores skills by relevance, proficiency, duration, and endorsements, while requiring balance across retrieval/ranking, vector infrastructure, Python, and production ML.
3. **Career fit**: rewards 5–9 years, product-company ML/search experience, and senior hands-on AI/search titles. It downweights non-technical or pure-services profiles.
4. **Behavioral fit**: incorporates Redrob signals such as recent activity, open-to-work status, recruiter response rate, response time, notice period, GitHub activity, recruiter saves, and interview/offer behavior.
5. **Logistics**: rewards India, Pune/Noida/Delhi NCR preference, nearby Tier-1 cities, hybrid/flexible preference, and relocation willingness.
6. **Trap/honeypot defense**: caps or penalizes profiles with impossible dates, salary min > max, experience mismatches, unrealistic skill durations, keyword-stuffed non-technical profiles, and CV/speech-only profiles with little IR evidence.

The output reasoning is generated from facts present in each candidate profile and behavioral signals, avoiding unsupported claims.
