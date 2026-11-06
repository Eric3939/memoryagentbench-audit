# MemoryAgentBench harness audit (commit 5380260), 2026-10-15

This was a read-only code review. It made no API calls and did not modify the repository. The offline checks come from
`env/check_harness_audit.py`, which drives the real harness code (ConversationCreator, the AgentWrapper memorize
paths, LC truncation and post_process) on cached HF data. Its outputs are `reports/data/harness_audit_checks.{json,log}`.
To rerun it:
`HF_HUB_DISABLE_PROGRESS_BARS=1 external/mabenv/bin/python -I env/check_harness_audit.py external/MemoryAgentBench` (~10 min).

**Confidence** below means confidence that the code does what is described. The size of the effect on the
*published* numbers is uncertain in every case, because the authors may have run code that differs from this commit.
Findings are ranked by likely impact.

---

## A. Memory isolation and keying (the known bug, refined)

### A1. Mem0: memorized under `context_None`, searched under `context_<i>` (known; verified)
`initialization.py:348` `agent.send_message(chunk, memorizing=True)` passes no context_id, and `agent.py:564`
`user_id = f'context_{context_id}_{self.sub_dataset}'`. As a result, `search` (`agent.py:594`) returns 0 memories. The
answerer then sees an empty memory list plus the question, so the score is roughly a no-context score. This affects every
dataset. Confidence: high (`env/check_mem0_keying.py`).

### A2. Cognee: the keying bug is effectively a no-op, but the search is global, so contexts leak into each other (NEW)
- `agent.py:522` keys the dataset name on context_id, as Mem0 does. However, the vendored
  `cognee/modules/search/methods/search.py:36-47` runs `specific_search(...)` **without the datasets argument**.
  `datasets` only feeds a post-filter: `document_id = search_result["document_id"] if "document_id" in search_result else None` → kept if `None`.
  For the default `GRAPH_COMPLETION` (`api/v1/search/search.py:11`), each result is a plain completion string, so
  `"document_id" in str` is False and the result is always kept. For a missing dataset, `get_document_ids_for_user`
  returns `[]` (`dataset_id=None`). Neither case raises an exception or returns "No results".
- Offline check (section 6): the vendored `search()` was executed with stubs, a missing dataset and a string completion.
  The result is kept, and `datasets` never reaches `specific_search`.
- **Consequence:** retrieval runs over the global graph and vector store. That store holds **every context memorized so
  far in the run**: the cleanup at `initialization.py:211-216` runs only once per run, and every context is added under
  `..._context_None`. For a query on context *i*, the graph therefore contains contexts 0..*i* (EventQA 5 books,
  LME(S*) 5 histories, DetQA 10 novels, InfBench-Sum 100 books). Fixing the keying alone would **not** isolate Cognee.
  Single-context datasets (FC, RULER, ICL, Recsys) are unaffected.
- Direction: cross-book distractors, probably negative. EventQA books are name-substituted and may share
  character names. Size unknown. Confidence: high on the mechanism (static plus stubbed execution), medium on the
  vendored version being the one used for the paper.
- Cognee also answers with its **own** LLM call: config `gpt-4o-mini` at temperature 0.0
  (`cognee/infrastructure/llm/config.py:10,14`) with the prompt "Be as brief as possible". The harness model,
  temperature and `generation_max_length` are not applied. See C3.
- Cheap verification: one Cognee run on EventQA with 2 books. Log `get_context()` for book-1 queries and count triplets
  whose source text comes from book 0.

### A3. Zep: graph_None is written, graph_0 is read; the run probably crashes, but the server behaviour is unverified (NEW detail)
`agent.py:634-648`. During memorization, `user_None`, `thread_None` and `graph_None` are created and filled. At query
time, `graph.search(graph_id='graph_0_<ds>')` (`agent.py:673-675`) and `thread.get_user_context('thread_0_<ds>')`
(`:682`) refer to objects that were never created. None of these calls is wrapped in try/except. The Zep SDK raises
on non-2xx responses, so a 404 would **crash the run at the first query**. An empty 200 response would instead give an
answer with no context. I could not check which happens offline (`zep_cloud` is not installed and the server is
remote). On a multi-context dataset the second AgentWrapper also calls `user.add('user_None_<ds>')` again
(`self.context_id` resets to -1 at `:252`), which plausibly fails with a conflict.
**Implication:** published Zep numbers on multi-context datasets probably did not come from this code path
unmodified. Confidence: high on the id mismatch, low on the failure mode. Verify by calling `graph.search` on a
non-existent graph id with a free Zep key; this costs no LLM spend.

### A4. Other systems
- knowl, agentmemory and total-agent-memory ignore context_id, but each builds a fresh store per AgentWrapper
  (`agentmemory.py` project includes pid and time; TAM uses a temp dir per process). There is no leak. Confidence: high.
- Letta (local `chat`/`insert`): one agent per context. `load_agent()` before every query (`agent.py:443-445`) restores the
  post-memorization sqlite. There is no cross-question leak. Confidence: medium.
- **Letta API mode** (`letta_mode: api`): `save_agent` and `load_agent` skip it (`agent.py:1101,1132`), so it never resets
  between queries. Every earlier question and answer stays in the agent's history for later questions. This is a
  cross-question carry-over: probably harmless for FC and EventQA, but a confound. Confidence: high (code).
- RAG agents: a new AgentWrapper is built per context, so `self.chunks`/`self.context` are fresh. `self.context_id`
  only gates index building. Confidence: high.

## B. What reaches the model: context and chunk-store truncation

### B1. RAG chunk store silently drops the oldest chunks (NEW, verified offline)
`agent.py:718-723`:
```python
self.context_len = self.context_len + self.chunk_size
if self.context_len > self.input_length_limit:
    self.chunks = self.chunks[1:]
```
`input_length_limit` is `input_length_limit - buffer_length - generation_max_length` from the agent config. Several
configs set it to about 250-300K, so these RAG systems **index only the most recent ~250-285K tokens**. Configs
with 10M (BM25, Contriever, Qwen3, HippoRAG, Mem0) keep everything. The counter also assumes every chunk is full,
which overstates the size because sentence packing leaves chunks under-filled, so the window is smaller than the limit.

Share of context tokens kept, at chunk 4096 (512 gives similar figures). Measured by running the real memorize path:

| dataset (main tier) | text-emb-3-small/large (285K) | RAPTOR / Self-RAG (280K) | GraphRAG (249K) |
|---|---|---|---|
| RULER-QA2 421K | 0.65 | 0.64 | 0.56 |
| EventQA-full (5 books) | 0.51 | 0.51 | 0.45 |
| LME(S*) (5 contexts) | 0.77 | 0.76 | 0.67 |
| InfBench-Sum (10 books) | 0.83 | 0.83 | 0.79 |
| FC-262K | 1.00 | 1.00 | 0.90 |
| Recsys | 0.19 | 0.19 | 0.16 (MemoRAG 0.52) |
| RULER-QA1, ICL, DetQA | 1.00 | 1.00 | 1.00 |

Questions whose evidence sits in the dropped head cannot be answered from retrieval. The likely direction is
**negative** for these systems on RULER-QA2, EventQA, LME and Recsys. A rough upper bound is (share dropped) × (accuracy
otherwise), which for RULER-QA2 means up to about 15-20 points if evidence is spread evenly. This confounds within-family
comparisons, e.g. text-embedding-3 vs Contriever/Qwen3, and RAPTOR/GraphRAG vs HippoRAG. Confidence: high (code and
measurement). Verify by rescoring stored results by evidence position, or by rerunning one system with
`input_length_limit` 10M.

### B2. Long-context truncation keeps the tail (by design, but large and asymmetric)
`agent.py:312-314,349-359`. The guard `if self.input_length_limit <= self.context_max_length + 50000` is followed by
`self.context = tokenizer.decode(encoded[-self.input_length_limit:])`. The **beginning** of the context is dropped. Measured
share kept for context 0:

| dataset | gpt-4o-mini / gpt-4o (124K) | claude-3.7 (173K) | gpt-4.1-mini, gemini-2.0 |
|---|---|---|---|
| RULER-QA1 / QA2 | 0.61 / 0.29 | 0.85 / 0.40 | 1.0 / 1.0 |
| EventQA-full | 0.23 | 0.32 | 1.0 |
| LME(S*) | 0.34 | 0.48 | 1.0 |
| FC-262K (289.5K tok with templates) | 0.43 | 0.60 | 1.0 |
| InfBench-Sum book 0 | 0.31 | 0.44 | 1.0 |
| Recsys | 0.08 | 0.12 | 0.66 / 0.57 |

On FC, keeping the tail keeps the newest (highest-serial) facts. That favours conflict resolution over random
truncation, and it means FC-262K scores for 128K models partly measure the truncation rule. For InfBench-Sum, keeping
the tail drops how the book opens. Confidence: high. This is probably known to the authors but should be reported per cell.

### B3. Zep ingests only the first 9,998 characters of each chunk into the graph and 2,400 into the thread (NEW, verified)
`agent.py:659` `data=content[:9998]`; `methods/zep.py:182` `content=content[:2400]`. A 4096-token chunk is about
16-18K characters. Share of characters that reach `graph.add`: RULER 0.50-0.56, EventQA 0.56, LME 0.54, FC-262K 0.59,
ICL 0.63, DetQA 0.62, InfBench 0.54, Recsys 0.63. About 40-50% of every context is **never stored** in Zep. Direction:
negative, roughly proportional to the missing evidence. Confidence: high (the 9,998 figure is presumably an API
limit, but chunk 4096 makes the loss systematic). A fix is to split the content into ≤9,998-character episodes.

### B4. Per-system retrieval and reader settings that differ from the configs (NEW)
| system | file:line | what actually happens |
|---|---|---|
| RAPTOR | `methods/raptor.py:102,182` | `vectorstore.as_retriever()` uses the default k=4, and `run(query, k)` ignores k, so `retrieve_num: 10` has no effect. Reader is `ChatOpenAI(model_name="gpt-4o-mini")` (`:125`) with the langchain default temperature, no max_tokens, its own prompt and LLM contextual compression. Summaries are stored as `str(AIMessage)`, which includes metadata. |
| Self-RAG | `methods/self_rag.py:121` | Hard-codes `gpt-4o-mini` with `max_tokens=1000`, so the **gpt-4.1-mini Self-RAG config actually runs 4o-mini**. Re-splits chunks into 4096-**character** pieces. When the LLM says "no retrieval" it answers parametrically. |
| HippoRAG v2 | `methods/hipporag/utils/config_utils.py:175` | `qa_top_k=5`: the reader sees 5 of the 10 retrieved passages. MuSiQue CoT prompt, `max_new_tokens=2048`, temperature 0 (not the config's 0.7). LLM responses are **cached in SQLite** under `outputs/rag_retrieved/...`, so reruns replay answers. |
| GraphRAG | `methods/graph_rag.py:552,604` | Truncation bug: `decode(encoded[llm_length_limit:])` keeps the tokens *after* the first 127K and discards the first 127K, i.e. it keeps only the overflow. Only bites when the expanded context exceeds 127K tokens. Also re-splits into 4096-character pieces. |
| Mem0 | config `retrieve_num: 100` | Uses 100 memories. MemoRAG uses 3. Others use 10. |
| Cognee | see A2 | Own LLM and prompt; harness model, temperature and max_tokens ignored. |
| Zep | `methods/zep.py:102-109,114-143` | No max_tokens. Prompt says "abstain from answering". The retry decorator references the un-imported `openai` and `time`, so any API error raises NameError. |

Confidence: high (code). The effect on scores depends on the system.

## C. Scoring and parsing

The README maps AR (EventQA, RULER) and CR (FC) to `substring_exact_match`, and DetQA and ICL to `exact_match`.

### C1. DetQA `exact_match` is ~0 for answers that follow the prompted format (NEW, verified)
Each DetQA question includes a one-shot demo whose answer is `Output: {"answer":"C. Suicide", "reasoning": ...}`.
`default_post_process` (`utils/eval_other_utils.py:603-611`) compares the normalized whole output, and the parsed
version, against `"c brandt couple"`. `parse_output` looks for `Answer:`, which a JSON `"answer":` does not match, then
falls back to the first line. Offline: demo-style JSON gives EM 0 (substring 1), fenced JSON gives EM 0, and the bare
`C. The Brandt couple` gives EM 1. In addition, **5/71 golds have no space after the letter** (`"B.Hila Weber"` normalizes
to `"bhila weber"`), so the correct answer `B. Hila Weber` scores 0 on both metrics.
Consequence: DetQA EM mostly measures whether the system ignores the demo's output format. Systems with their own
"be brief" reader (Cognee, Zep, GraphRAG) are favoured over a compliant LC model. This sits on top of the n=71 sampling
noise. Confidence: high on the mechanism. If the published DetQA numbers are well above 0 for LC models, the authors
either got mostly non-JSON outputs or scored them differently. Verify by rescoring stored outputs with
JSON-aware letter extraction.

### C2. ICL `exact_match` penalizes the instructed format (README-acknowledged; system-dependent)
The template (`utils/templates.py:40-42`) says `Only output "label: {label}"`, but EM needs the bare number.
Offline: `label: 43` gives EM 0 and `43` gives EM 1. Whether a system outputs the bare number depends on its wrapper.
The LC and RAG prompts end with `label:`, which encourages the bare number. Mem0 appends
`"\n\nCurrent Time: ..."` after it (`agent.py:603`). Zep, RAPTOR, Self-RAG, HippoRAG and Cognee wrap the query in their
own prompts. **Between-system differences in TTL can therefore be format artefacts.** (If substring were used
instead, it would be inflated: `label: 17` matches gold `7`.) Confidence: high. Verify by rescoring stored outputs with
`re.search(r'\d+')`.

### C3. Substring EM rewards hedging and abstention; output caps differ by system (NEW)
- FC: `"Armenia (previously Argentina)"` scores 1 for either gold. The 10-token cap (`generation_max_length: 10`)
  stops LC, BM25, embedding RAG, Mem0, GraphRAG and the PR methods from hedging. **Cognee, Zep, RAPTOR, Letta (no cap),
  Self-RAG (1000) and HippoRAG (2048) are not capped**, so they can list old and new values and get credit.
  Direction: inflates uncapped systems on FC. Size unknown. Confidence: high on the mechanism.
- RULER-QA2 has **12/100 yes/no golds**. `"I don't know"` and `"Not mentioned in the documents"` both score 1 against
  gold `no`, because `"no"` is a substring of `"dont know"` and `"not"`. Zep's prompt explicitly asks for abstention, and
  systems with empty memory (Mem0 under A1) are likely to abstain, so abstention earns credit on about 12% of QA2.
  Confidence: high on the mechanism. Rescoring with a word-boundary match would correct it.
- `normalize_answer` removes "a" as an article, so an option label "A." disappears. No gold normalizes to empty
  (section 5). Golds of ≤3 characters: ICL all (numeric labels), LME 71/300 (judge-scored, so this doesn't matter),
  RULER-QA2 14. EventQA golds are option sentences that always appear in the question, which is harmless with the 40-token cap.
- FC golds are almost never over 10 tokens (1/100 on 262K-SH/MH), so cap-induced truncation of a correct answer is
  rare unless the model adds a preamble. Confidence: high.

## D. Configuration vs paper

- **Temperature 0.7 in every released agent config** (`configs/agent_conf/**: temperature: 0.7`). Our notes
  (`partB_engineering_notes.md` item 5, `hu2025` note) say the LC and RAG answerer runs at temperature 0. The code
  default is 0.0, but the configs override it. So the main table is presumably single runs at T=0.7, where read-time
  sampling noise is substantial. Systems that ignore the setting: Cognee 0.0, HippoRAG 0, Self-RAG and RAPTOR use their
  own values, Mem0 extraction uses 0.1. **Fix our notes.** Confidence: high (configs). Whether the paper used them is unverified.
- Chunk size: as already noted, configs use 4096 everywhere, not the paper's 512. Combined with B1, at 512 the
  `n_chunks*chunk_size` counter overstates size even more.
- Models are unpinned (`gpt-4o-mini`). Embedding RAG forces Azure (`agent.py:956` `use_azure=True` → an assert unless
  Azure env vars are set), so the published numbers came from an Azure deployment of unknown version.
- Claude LC drops the system message (`agent.py:373` `include_system=False`, and no `system=` passed). o4-mini runs
  with no max_tokens or temperature (`:333-337`). Minor.

## E. Judges (`llm_based_eval/`)

- LME: `metric_model="gpt-4o"` is **unpinned** (`longmem_qa_evaluate.py:96`), temperature 0, `max_tokens=10`, and
  `label = 'yes' in eval_response.lower()` (`:168`). That is a substring check, but with yes/no-only prompts it is
  low risk. The hypothesis file is whichever `longmemeval_s*_` file `os.walk` sees **last** (`:118-121`), so if several
  configs (k, chunk) share an output folder, the choice is arbitrary. The result cache `.eval-results-<method>-<tag>` is
  reused if it exists (`:137,192`), so a rerun with new outputs and the same tag silently returns old judgments.
  Confidence: high.
- InfBench-Sum: the judge is pinned to `gpt-4o-2024-05-13`, but it **samples** (`temperature=0.1, top_p=0.9`,
  `do_sample=True`, seed 42; `summarization_evaluate.py:493`, `model_utils_openai.py:144-160`). Entries whose
  judge JSON fails to parse are **dropped from the mean, not scored 0** (`:436-439,470-473`). If any entry is skipped
  for a missing `qa_pair_id`, `outputs[idx*3:...]` misaligns every later score (`:407-430`). Fluency is 0/1 and
  multiplies F1. Confidence: high (code). Impact is small unless parse failures are common.

## F. Caching and reruns (relevant to Part B)

- **`--force` does not rerun queries.** `should_skip_query` ignores `force` (`main.py:55-57,119`). With an existing
  results file, `--force` rebuilds memory but skips every query that was already scored, and the old outputs stay.
  The results filename omits temperature, model version and code version (`initialization.py:245-282`). **After fixing
  the keying bug, a rerun with the same config returns the buggy results unless the file is deleted.** Confidence: high.
- A resumed run rebuilds memory for a partly finished context (nothing is saved for Mem0, RAG or Cognee), so one
  context's questions can mix two builds.
- Saved-agent reuse applies only to Letta and Zep (`save_agent` creates no folder for other types, `agent.py:1097`).
  A Zep marker folder makes reruns skip memorization even if the cloud graph is gone.
- HippoRAG caches OpenIE output, the graph and LLM answers under `outputs/rag_retrieved/NV-Embed-v2/<ds>/chunksize_*/context_id_*`.
  To measure rebuild variance, this cache must be cleared. MemoRAG loads `memory.bin` if present.
  `outputs/rag_retrieved/...` JSONs are logs only and are not read back.

## G. Templates and information differences (LC vs memory agents)

- No query template uses `{answer}`, although `qa_metadata` carries it (`conversation_creator.py:221`). There is no leak.
- Templates are chosen by substring: `Structure_rag_mem0/cognee/zep` and `Agentic_memory_self_rag` contain "rag" and get
  `rag_agent` templates. Letta, knowl, agentmemory and TAM get `agentic_memory_agent` ("Search Archival Memory ...").
  The knowl docstring claims a layout "byte-identical" to BM25's, but the query template differs. The PR methods also
  receive **fact-level records parsed from the FC numbered list with serials stripped** (`agentmemory.py:57-82`), while the
  prompt still asks for the "larger serial number". This input is privileged and FC-specific, so they are not
  comparable to chunk-fed baselines. TAM can prefix each fact with its recorded time, which encodes ingestion order.
- Memorize templates contain a literal `\n` (escaped) and a wall-clock `{time_stamp}`. For LC and RAG all chunks get
  the same second, so this is harmless. For Mem0, Cognee, Zep and Letta, ingestion is slow, so timestamps increase with
  order. That is a recency cue aligned with FC serial order (matters for the order facet).
- Mem0 on LME: the question already says `Current Date: 2023/...`, and Mem0 appends `Current Time: 2026-...`
  (`agent.py:603`). Its extraction prompt also says `Today's date is <now>` (`mem0/configs/prompts.py:49`). The
  conflicting "now" likely hurts temporal-reasoning questions. Confidence: medium on the effect.
- Retrieved chunks reach the reader in similarity order, not chronological order. FC is fine because serials are in the
  text. LME temporal questions lose ordering.
- `chunk_text_into_sentences` joins sentences with spaces, flattening newlines, and can emit a chunk larger than
  `chunk_size` (a single long sentence) or an empty first chunk. All systems are affected the same way.

## Suggested cheap verifications (no or minimal API)
1. Rescore any stored outputs: DetQA with JSON-aware letter extraction, ICL with `\d+` extraction, RULER yes/no with
   word boundaries, and FC with a first-entity-only match (C1-C3). No API.
2. B1/B3 numbers come from `env/check_harness_audit.py`. To quantify the score effect, rerun text-embedding-3-small on
   RULER-QA2 with `input_length_limit: 10000000` (about $1).
3. Zep missing-graph behaviour: one `graph.search` call on a non-existent id (free tier).
4. Cognee leak: EventQA with 2 books, logging retrieved triplet provenance (a few cents).
