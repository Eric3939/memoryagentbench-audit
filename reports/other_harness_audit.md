# Harness audit of other agent-memory benchmarks (LongMemEval, Mem0's LoCoMo eval, LoCoMo), 2026-10-15

This was a read-only static review. It made no API calls. The repos were cloned into `external/` (gitignored):

| Repo | Commit | Notes |
|---|---|---|
| xiaowu0162/LongMemEval | `9e0b455` (2026-05-11) | `src/{generation,evaluation,retrieval}`; benchmark data not downloaded |
| mem0ai/mem0, `evaluation/` | `aae5989e` (2025-07-08) | At HEAD `0724291`, `evaluation/` is **empty**: commit `9315e303` (2026-06-14) retired it "in favor of mem0ai/memory-benchmarks". The last populated tree was extracted to `external/mem0_eval/`. The Zep/LangMem/judge logic is unchanged since the first commit `393a4fd5` (2025-04-29), apart from judge JSON parsing (`2bb0653e`) |
| snap-research/locomo | `3eb6f2c` (2024-08-12) | `task_eval/`, `data/locomo10.json` |

The offline stub checks are in the scratchpad (`locomo_check.py`, `lme_check.py`). Each one extracts the real functions with `ast` and runs them with stub tokenizers and stub LLMs.

**Confidence** below means confidence that the code does what is described. The effect on *published* numbers is
unknown in every case, because the authors may have run code that differs from these commits.

**Bottom line.** The harness bug classes found in MAB recur in all three repos. The closest match to the MAB Mem0/Zep keying bug
is a **write/read id-coverage mismatch in Mem0's Zep baseline** (M1): it writes only conversation 0 and reads conversations 0-9. I found
no exact `context_None` vs `context_<i>` style key mismatch, and no cross-conversation leakage in any of the three repos. Mem0/RAG/Zep/LangMem all key
per conversation consistently. However, silent truncation that drops the *most recent* history (L1, C1), category handling that cannot run on the released data (C2),
substring-lenient judge parsing (L3, C3), silently dropped failures that shrink denominators (L2), unequal answer prompts across baselines (M3),
and a stale-cache keying gap (C4) are all present.

---

## M. Mem0 `evaluation/` (LoCoMo; systems: mem0, mem0-graph, zep, langmem, rag, full-context, openai)

### M1. Zep: only conversation 0 is memorized, but all 10 conversations are queried (verified statically; HIGH)
`evaluation/src/zep/add.py:63-68`
```python
for idx, item in tqdm(enumerate(self.data)):
    if idx == 0:
        self.process_conversation(run_id, item, idx)
```
`src/zep/search.py:130,193-199` searches `user_id=f"run_id_{run_id}_experiment_user_{idx}"` for **every** idx 0..9 (`Makefile`: `run-zep-add`/`run-zep-search`, run_id "1"). Users 1-9 are never created.
- **Mechanism:** this is the same write/read coverage mismatch as MAB A3 (Zep `graph_None` written, `graph_0` read). `graph.search` on a missing user either raises (it is retried 3 times, then re-raised, so the run crashes at the conversation-1 question) or returns empty results. The second case gives a context-free answer for about 90% of questions.
- The line has been present since the first commit (`393a4fd5`). The published Zep numbers therefore cannot have come from this code path unmodified.
- **Direction:** strongly negative for Zep if the failure is silent; otherwise a crash. **Cheap check:** use a free Zep key and call `graph.search(user_id="nonexistent")`. This costs no LLM spend.
- Also in the Zep adapter: both speakers go into **one** user/session with `role_type="user"` (`add.py:52-59`), and the session date is put inside the message text (`content=f"{timestamp}: ..."`) instead of the message timestamp. That conflicts with Zep's temporal-graph design, and the user/thread are never deleted, so a re-run with the same run_id hits `user.add` on an existing user. The Zep authors made similar criticisms publicly; I did not re-verify their claims beyond this code.

### M2. LangMem: the "memory" baseline also carries the raw speaker history in short-term thread state (HIGH on mechanism; MED on effect)
`src/langmem.py:69-83,113-148`: each agent is `create_react_agent(..., checkpointer=MemorySaver())`. Every message is added with
`agent.invoke({"messages":[msg]}, config={"thread_id": f"thread-{key}"})`, and every question is answered with the **same** config.
Under LangGraph's `add_messages` semantics, the thread accumulates every memorized message, the tool calls, and every earlier Q/A pair.
- **Consequences:** (a) the answer step sees that speaker's full raw transcript in context, so this is not a pure long-term memory test; (b) questions carry over within a conversation (as in MAB Letta API mode, A4); (c) latency is inflated by the growing thread.
- There is no cross-conversation leak: each conversation gets fresh `InMemoryStore` and `MemorySaver` objects in its own process.
- Secondary: `search_memory` `except` returns `t2 - t1` with `t2` unbound (`:91-93`), so any invoke error becomes an `UnboundLocalError` that aborts the pool. `speakers=set()` ordering is hash-random (`:116-126`), which affects only labels.
- **Direction:** probably positive for LangMem accuracy (extra raw context), negative for its latency. **Cheap check:** after the add loop, print `len(agent1.agent.get_state(config).values["messages"])`. This needs only embeddings, or a stubbed LLM.

### M3. Answer prompts differ across baselines (HIGH on code; MED on effect)
mem0, mem0-graph, Zep and "openai" use `ANSWER_PROMPT*` (`prompts.py:65-108`, `src/openai/predict.py:15-52`). That prompt has step-by-step temporal instructions ("convert relative time references to specific dates", "show your work", "<5-6 words").
RAG and **full-context** use a minimal prompt (`src/rag.py:200-245`). LangMem uses `ANSWER_PROMPT` on top of agent responses.
- The memory systems therefore get an explicit temporal-reasoning recipe that the full-context and RAG baselines do not. That is a confound on LoCoMo category 2 (temporal) and on the overall score.
- "show your work" can also put reasoning text into `response`. That text is scored verbatim by F1/BLEU (`evals.py:157-158`), which deflates the lexical metrics, while the judge is told to be "generous".

### M4. Judge: unpinned alias, generous rubric, contradictory format instructions (HIGH on code)
`metrics/llm_judge.py:285-301`: `model="gpt-4o-mini"` is an alias, not a snapshot (compare LongMemEval, which pins one). The rubric says "as long as it touches on the same topic ... CORRECT".
The prompt asks for "a short (one sentence) explanation ... then finish with CORRECT or WRONG" **and** "Just return the label ... in a json format", and the call uses `response_format=json_object`.
The label check is an exact `label == "CORRECT"`, so `"Correct"` scores 0. Each question gets a single judge call at temperature 0; I found no code for repeated runs.

### M5. Category 5 handling and categories (HIGH on code)
Category 5 (adversarial, 446/1986) is answered by every system. It is dropped at scoring time (`evals.py:153-155`; `llm_judge.py:333-335`) and at answer time for LangMem only (`langmem.py:142`).
The filtering is consistent within Mem0's harness, but the reported "overall" is computed over categories 1-4 (1540 questions). LoCoMo's own `analyze_aggr_acc` instead includes category 5 in the overall (`evaluation_stats.py:98-108`). Overall scores from the two harnesses are therefore not comparable.
`generate_scores.py` prints only numeric category ids. The code contains no id-to-name map, so I cannot check the paper's per-category labels from code (see C5).

### M6. Lexical F1 differs from LoCoMo's F1 (HIGH)
`metrics/utils.py:142-152` computes F1 over **sets** of lowercased, whitespace-split tokens, with no stemming, article removal or multi-answer split. LoCoMo's `f1_score` (`task_eval/evaluation.py:126-145`) uses Porter stemming, removes articles and "and", and splits category 1 answers on commas.
F1 numbers labelled "LoCoMo F1" are therefore not comparable across papers.

### M7. Unreproducible inputs (HIGH)
The "openai" baseline reads `memories/{idx}.txt` (`src/openai/predict.py:61-65`). Those files were produced outside the repo, presumably from ChatGPT memory. RAG and LangMem read `dataset/locomo10_rag.json`, whose conversion script is not in the repo. The Mem0 platform runs server-side with a project-wide `custom_instructions` setting (`memzero/add.py:128`), so the memory backend version is unpinned.

**Not found in Mem0:** the mem0 add/search keying (`f"{speaker}_{idx}"` on both paths, `add.py:165-170` and `search.py:397-398`) is consistent and does `delete_all` before adding. RAG and full-context build their index per conversation. Full-context is not truncated: the largest LoCoMo conversation is far below 128k.

---

## L. LongMemEval (`src/`)

### L1. Context truncation keeps the *oldest* history and drops the newest (verified offline; HIGH on mechanism, magnitude unknown)
`generation/run_generation.py:224-271`: retrieved or full-history chunks are **sorted by date ascending** (`:225`) and then truncated with `tokens[:max_retrieval_length]` (`:270`; the HF path uses `truncation=True`, which is right-truncation by default).
`max_retrieval_length = model_max - gen_length - 1000` (`:343`).
- **Stub check (`lme_check.py`):** with 12 monthly sessions and a 5000-word budget, the prompt kept sessions 0-3. The last date in the prompt was 2023/04 and the question date was 2023/12. The only signal is a stdout `print('Truncating ...')`.
- **Affected settings:** full-history readers whose history exceeds the window: LME_M (about 500 sessions) for any model; LME_S for phi-4 (16k) and Mistral/FILM (32k); possibly some LME_S instances on 128k models in `json` format. Large-top-k retrieval readers are affected the same way.
- **Direction:** negative, and worst for knowledge-update and temporal questions, whose evidence tends to be recent.
- **Inconsistency:** for `orig-*` retriever types the top-k cut keeps the **latest** k (`:171-172`, "keep latest"), but the token truncation then keeps the **earliest** tokens.
- **Cheap check:** count the `Truncating from` lines in existing generation logs.

### L2. Generation failures are silently dropped and the denominator shrinks (HIGH on code; size unknown)
`run_generation.py:357-378` wraps the call in `except Exception: ...; continue`. Backoff covers only `RateLimitError` (`:285`), so any other API error (5xx, timeout, BadRequest) leaves the question out of the hypothesis file.
`evaluation/evaluate_qa.py:90-130` then averages over the entries that **exist** (`np.mean(... for x in logs)`), with no check that every reference qid was answered. `print_qa_metrics.py:152-166` does the same per type.
- **Direction:** inflated if failures correlate with hard (long) instances. **Cheap check:** compare the number of hypothesis lines with 500.

### L3. Judge verdict parsed by substring (verified; LOW-MED impact)
`evaluate_qa.py:113`: `label = 'yes' in eval_response.lower()`. "No, the response does not say yes" parses as correct.
`max_tokens=10` and "Answer yes or no only" make this rare. On the positive side, the judge is pinned (`gpt-4o-2024-08-06`, `:14`), `print_qa_metrics.py:154` asserts that pin, and temperature is 0.

### L4. Retrieval metrics silently exclude questions without user-side evidence (HIGH on code; by design but easy to misread)
The corpus indexes **user turns only** (`retrieval/run_retrieval.py:206,213-215`). Answer sessions whose user turns have no `has_answer` are renamed `noans_*` (`:209-210,224`).
The averages skip abstention questions and questions with no user-side target (`:396-402`). In practice this excludes single-session-assistant questions, and the exclusion appears only in a stdout line.
A bare `except: continue` also wraps the averaging (`:405-407`). In `evaluate_retrieval_turn2session` (`eval_utils.py:32-46`), one session can appear as both `answer_x` and `noans_x`, which distorts "unique session" k (low impact). `np.asfarray` (`eval_utils.py:6`) no longer exists in NumPy 2.

### L5. `flat-turn` + `useronly=true` crashes (verified)
`run_generation.py:144` evaluates `x[-1]['role']` where `x[-1]` is a **list** of turns, which raises `TypeError` (confirmed by the stub). The default is `useronly=false`, so this only blocks that configuration. A related pattern: `flat-turn` silently drops ranked items whose ids fail to map (`:130-131`, `except: continue`).

**Not found in LME:** the gold-evidence markers do not leak into the prompt. `has_answer` is popped from retrieved turns (`:177-191`), and session ids such as `answer_*` are never rendered. There is no cross-instance state, and each question builds its own corpus. Note that `oracle-*` retriever types are no-ops that rely on the oracle input file (`:105-116`). The README's `print_qa_metrics.py gpt-4o <file>.log ...` command passes three args to a script that requires two, and the results file is named `.eval-results-<model>`, not `.log`. Both are documentation mismatches only.

---

## C. snap-research/locomo (`task_eval/`)

### C1. Truncation drops the *latest* sessions, silently (verified offline; HIGH)
`gpt_utils.py:172-204` (`get_input_context`) iterates sessions **ascending** and stops at the first session that overflows `MAX_LENGTH[model] - 50*batch_size`. All later sessions are dropped, and the log print is commented out (`:199-202`). Sessions also appear in **reverse chronological** order in the prompt.
- **Stub check:** conversation 0 (19 sessions, about 11k words) under the `gpt-3.5-turbo` 4096 budget kept only sessions 1-7 (8 May-12 Jul 2023), while the conversation runs to 22 Oct 2023.
- The 4k/8k/12k/16k gpt-3.5 settings and the hf/claude/gemini variants presumably behave the same way (not checked).
- `MAX_LENGTH['gpt-4-32k']=320000` (`:22`) looks like a typo for 32000, which would disable truncation for that model.
- **Direction:** negative for questions whose evidence is in the late sessions.

### C2. Category 5 cannot run on the released data: 444/1986 questions (22.4%) have no `answer` key (verified; HIGH)
In `data/locomo10.json`, 444 of the 446 category 5 items carry `adversarial_answer` and **no** `answer`. No other category is missing `answer`, and there are no empty-string answers.
`gpt_utils.py:248/251` builds the multiple-choice prompt from `qa['answer']`, and `evaluation.py:202` scores with `str(line['answer'])`. **Both raise `KeyError: 'answer'`** (confirmed by the stub).
- This matches the "about 23% unscorable" figure reported by ATANT (2604.10981). The mechanism in the released code is a schema mismatch between code and data (a missing key, which crashes), not an empty gold string.
- Downstream harnesses that use `.get("answer","")` (Mem0's harness, for example) get an **empty gold** for these items. They must drop category 5, or they score against "".

### C3. Category 5 scoring is substring-lenient and nondeterministic (verified)
- **Scoring:** `evaluation.py:217-221` scores 1 iff the output contains "no information available" or "not mentioned" anywhere. "It is not mentioned whether she did, but probably yes" scores 1.
- **Option order:** the order of the two options is set by unseeded `random.random()` (`gpt_utils.py:247`).
- **Mapping the model's choice back to an option:** `get_cat_5_answer` (`:128-142`) maps any 1-character output that is not 'a' to option b (e.g. 'c' maps to b). Outputs such as "a." or "(a)." are passed through raw, so they score 0 whichever option was right.

### C4. Stale embedding cache not keyed by retriever (HIGH on code; latent)
`gpt_utils.py:78-109` caches dialog embeddings as `{prefix}_dialog_{sample_id}.pkl` with **no retriever in the key**. The prediction key `{model}_{rag_mode}_top_{k}_prediction` (`evaluate_qa.py:69`) omits the retriever too, and existing predictions are skipped unless `--overwrite` is passed (`gpt_utils.py:238`).
If the retriever changes (dpr, contriever and dragon all produce 768-dim vectors), stale context embeddings are paired with new query embeddings and old predictions are reused, with no error. The shipped scripts use only `dragon`, so the shipped runs are not affected.

### C5. Category semantics are only implicit, and the comments contradict them (HIGH on code)
The code treats category 2 as temporal (date hint, `gpt_utils.py:243`) and category 3 as open-domain (`;` split, `evaluation.py:203`). The stats print order `[4,1,2,3,5]` implies 4=single-hop and 1=multi-hop.
The comment at `evaluation.py:208` lists "[2,3,4] = single-hop, temporal, open-domain", which conflicts with that. With no canonical id-to-name map, downstream papers can mislabel per-category tables. This is a plausible source of the category-label disagreements across LoCoMo papers, but I have not verified it against any specific paper.

---

## Cross-benchmark summary (bug classes from the MAB audit)

| MAB class | Mem0-LoCoMo | LongMemEval | LoCoMo |
|---|---|---|---|
| Write/read keying or coverage mismatch | **Yes: Zep idx==0 (M1)** | No | n/a (no memory systems) |
| Cross-context leakage | No (but LangMem keeps raw history and earlier Q/A in context, M2) | No | No |
| Silent truncation | No | **Yes, keeps oldest (L1)** | **Yes, drops latest sessions (C1)** |
| Scoring/format mismatch | F1 differs from LoCoMo's (M6); unequal answer prompts (M3) | Failures dropped from the denominator (L2) | Category 5 KeyError on released data (C2) |
| Substring leniency | Generous judge rubric (M4) | `'yes' in` (L3) | Category 5 "not mentioned" substring (C3) |
| Unpinned judge | **Yes** (`gpt-4o-mini` alias) | No (pinned and asserted) | n/a (F1) |
| Stale caches | No | Not found | **Yes (C4)** |
| Category miscounting | Category 5 excluded; differs from LoCoMo overall (M5) | Retrieval excludes SSA (L4) | Implicit, contradictory labels (C5) |
| Temperature | 0 everywhere | 0 | 0, but unseeded option shuffle |
