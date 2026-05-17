# Manual eval — 10 questions (Day 2)

**Date:** _fill in_  
**Model:** llama-3.3-70b-versatile (Groq)  
**Index:** `data/Faiss_Index/` · **Chunks:** `data/chunks.json`

Run the app: `streamlit run streamlit_app.py` or `python main.py ask "…"`

For each question: note if the answer is correct, if citations match, and what failed.

| # | Question | Correct? | Citations OK? | Failure mode / notes |
|---|----------|----------|---------------|----------------------|
| 1 | What is RAASB and when was it established? | ⬜ | ⬜ | |
| 2 | Which NISM certification examination must a Research Analyst pass? | ⬜ | ⬜ | |
| 3 | Can an Investment Adviser registered under SEBI (IA) Regulations 2013 also register as a Research Analyst? | ⬜ | ⬜ | |
| 4 | What are the two activities a Research Analyst is prohibited from engaging in when applying for part-time RA registration? | ⬜ | ⬜ | |
| 5 | What does an employee need to provide from their employer when applying to become a Research Analyst? | ⬜ | ⬜ | |
| 6 | What is the minimum net worth requirement for a non-individual Research Analyst? | ⬜ | ⬜ | |
| 7 | How long is the validity period of a Research Analyst registration certificate? | ⬜ | ⬜ | |
| 8 | What disclosures must a Research Analyst make in their research report? | ⬜ | ⬜ | |
| 9 | Can a Research Analyst accept gifts from a client? | ⬜ | ⬜ | |
| 10 | What is the penalty for non-compliance with SEBI RA Regulations? | ⬜ | ⬜ | |

**Summary:** _/10 correct · top failure modes:_

---

## Quick CLI checks

```bash
python main.py ask "What is RAASB and when was it established?"
python main.py ask "Which NISM certification must a Research Analyst pass?"
```
