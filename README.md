# SORETRAK Assist

A conversational assistant for the **SORETRAK** bus network in Kairouan, Tunisia. It answers questions written in French, or Arabic, from the official open GTFS tables

It is an NL2SQL system: an LLM agent writes SQL against a read-only SQLite database, and the answer is written only from the rows it returned.

## Features

- Lines serving a stop, stops of a line, timetables, next departures
- Direct rides from A to B, with travel time
- Regional lines from Kairouan (Sousse, Tunis, Mahdia, Monastir…)
- Maps and bar charts on request
- Honest refusals for what the data does not contain 

## How it works

```
message ─► Router (1 LLM call) ── intent, standalone question, places mentioned
             ├─ conversation · too vague · out of scope · off topic ─► direct reply
             └─ transport
                  ▼
           Place resolution ── stops and lines + facts about them
                  ▼
           SQL agent (ReAct, ≤ 4 tool calls) ── run_sql · resolve_place · finish
                  ├─ answered · no data ─► Synthesizer ( LLM call) ─► answer
                  └─ ambiguous · impossible · failure ─► fixed message
```

A LangGraph workflow with a single agent. **Code sets the boundaries; the LLM decides within them.**

| Component | Type | Role |
|---|---|---|
| Router | LLM chain | Classifies the message (5 intents), rewrites follow-ups as standalone questions |
| Place resolution | Code | Fuzzy matching over 192 stops and 95 lines; Arabic → Latin, *arabizi*, articles, abbreviations |
| SQL agent | LLM agent | Writes and runs SQL, corrects itself, cites the queries used as evidence |
| Synthesizer | LLM chain | Writes the answer from the evidence rows only |
| Respond | Code | Fixed messages; no technical error ever reaches the user |



## Quickstart

Python 3.10+ .

```bash
pip install -r requirements.txt
python scripts/build_db.py    # builds data/soretrak_gtfs.db from the GTFS files
cp .env.example .env          # set LLM_PROVIDER and the matching API key
streamlit run app.py
```

| Variable | Value |
|---|---|
| `LLM_PROVIDER` | `openai`  or `groq` |
| `OPENAI_API_KEY` / `GROQ_API_KEY` | key of the active provider |

Add `?debug=1` to the app URL to see the SQL and the agent trace under each answer.

## Project structure

```
app.py                 Streamlit app
soretrak/
  components/          router, sql_agent, tools, synthesizer, respond
  conversation/        main graph, conversation memory
  places/              normalization, resolution, pre-retrieved facts
  sql/                 shape guard, read-only executor
  presentation/        formatting, map, chart
  prompts/             LLM prompts (French)
  knowledge/           scope, semantic schema, verified examples
data/gtfs/             GTFS source files
data/sql/              schema and views
```
