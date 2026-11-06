# Sources reporting their own numbers on MemoryAgentBench (MAB) datasets

Compiled 2026-10-15. Claim rows: papers/search/mab_claims.csv (268 rows, 41 sources with rows). Found via the Semantic Scholar citation list (252 arXiv full texts scanned), an arXiv API search (MemoryAgentBench / FactConsolidation / EventQA), and the GitHub PRs and issues of HUST-AI-HYZ/MemoryAgentBench. Sources noted as 'no own MAB numbers' were triaged and have no rows.

## Batch 0 (known examples, GitHub PRs and issues, and TransMem/KVMem)
- **2608.12888** When Your Agent Opens the Chat App: Agent-Controlled Search over Raw Chat Logs Rivals Structured Memory (ReFind) (2026) — https://arxiv.org/abs/2608.12888
  Lexical agentic search over raw chat logs. Claims the top 6-task MAB mean (58.2 vs HippoRAG-2 53.2), GPT-4o-mini, with all baselines copied from Hu et al.
  The MAB suite is one run with no uncertainty. 5-run std is given only for the LongMemEval-S/M subsets (GPT-5-mini, n=50/15). EventQA and LME margins are under 1 point.
- **2610.09193** Bookkeeping, Composition, or Unreachable Gold? Reading MemoryAgentBench's Conflict-Resolution Scores (2026) — https://arxiv.org/abs/2610.09193
  Validity audit: a frozen last-write rule resolver scores 80.25% (642/800) on FactConsolidation, and at 262K exceeds every published system (88/36 vs GPT-5-mini 78/28). 67 golds cannot be reached under last-write.
  Reports Clopper-Pearson CIs and a 4-list clustered bootstrap (80.25 [67.25, 92.75]). Its BM25 re-run has 3 repeats (kappa 0.90, 35-37 of 800 items flip). Explicitly "not a leaderboard".
- **2608.15008** Harness the Memory: A Holistic Evaluation of Memory Substrates in Memory Agents (2026) — https://arxiv.org/abs/2608.15008
  Evaluates 11 memory substrates on LoCoMo, LME-S and MAB LRU/TTL/CR with Qwen3-8B, Qwen3-32B-AWQ and Gemma-4 backbones. Claims "no single substrate wins".
  The MAB best-vs-runner-up gaps are 2-10 points. Point estimates only. Says gaps of 0.05 or more are "unlikely to reverse" but cites n only for the non-MAB sets.
- **2609.32658** Contract Memory Compiler: Resolve, Then Traverse (CMC) (2026) — https://arxiv.org/abs/2609.32658
  Resolves updates, then traverses relations. Claims SOTA FC multi-hop (78.25% over 800-Q pooled MH; 61.0 at 262K vs published GPT-5-mini 28). Also gives a full same-reader MAB grid (HY4-preview).
  Published baselines are copied as "reference", and same-reader baselines are re-run. Single run, no CIs. Several same-reader gaps are small (MCC 86.2 vs 86.4; FC-SH 97.25 vs 97.50).
- **2607.29032** TransMem: Transforming Hidden States into Memory for LLMs (2026) — https://arxiv.org/abs/2607.29032
  Parametric hidden-state memory on Qwen3-4B. MAB avg 40.00 vs delta-Mem 38.85 (+1.15). It loses on AR and LRU.
  Single fixed seed (42). Subset sizes not stated. No uncertainty.
- **2609.04852** KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU (2026) — https://arxiv.org/abs/2609.04852
  KV-cache virtualization. On MAB histories over 256K, overall 40.99 vs Compact+RAG 34.80 (Qwen3.x-27B, same budget, all baselines re-run).
  Single run, no uncertainty, n not stated.
- **gh-MAB-PR23** GitHub PR #23 "Add knowl and agentmemory as memory methods" (Aug 2026) — https://github.com/HUST-AI-HYZ/MemoryAgentBench/pull/23
  Vendor (knowl maintainer) adds two write-time-resolving methods. FC-SH 6K/262K: knowl 95/89, agentmemory 83/79 (GPT-4o-mini reader, temp 0.7).
  "Single runs ... directional, not publication-grade". The PR is closed.
- **gh-MAB-PR25** GitHub PR #25 "Add total-agent-memory as a memory method" (Sep 2026) — https://github.com/HUST-AI-HYZ/MemoryAgentBench/pull/25
  TAM (vendor-maintained) on FC-SH/MH 6K/262K vs BM25 re-run here (e.g. 85 vs 46 at FC-SH 262K). Temp 0.7.
  Reports a 3-run range for FC-SH (80-84 at 6K): run-to-run spread of about 4 points at n=100.
- **gh-MAB-issue18** GitHub issue #18 "Typed conflict resolution (Mnemos) achieves 12% on FC-MH" (Mar 2026) — https://github.com/HUST-AI-HYZ/MemoryAgentBench/issues/18
  Mnemos vs a naive same-stack baseline on all 8 FC cells (GPT-4.1-mini). MH avg 12 vs 5. Large SH losses at 32K and above.
  Single run, no uncertainty. The 262K MH cells tie at 2 vs 2.
- **gh-MAB-issue20** GitHub issue #20 "review request for narrow public transfer result on multi-hop conflict" (Jul 2026) — https://github.com/HUST-AI-HYZ/MemoryAgentBench/issues/20
  Anonymous compiler-style typed-path system. FC-MH 220/400 -> 291/400, and 262K 27/100 -> 44/100. Backbone not stated.
  Self-described as "not an official MemoryAgentBench score".
- **gh-MAB-issue15** GitHub issue #15 "Question about reproducing Mem0 + LongMemEval(S*) results" (Dec 2025) — https://github.com/HUST-AI-HYZ/MemoryAgentBench/issues/15
  An independent re-run of Mem0 with the official config gets 15.3% on LME(S*) vs the published 36.0.
  This is reproduction evidence, not a claim: a 20-point discrepancy from config/SDK drift.

## Batch 1
- **2606.01435** Reliable Post-Retrieval Assembly for Agent Memory: Separating Evidence Extraction from Policy Execution (2026) — https://arxiv.org/abs/2606.01435
  Task-specialized extract-then-execute pipeline for FactConsolidation; 82%/93% FC-SH and 27%/41% FC-MH at 262K (gpt-4o-mini/gpt-4o), compared against MAB v3 published 262K rows (copied).
  Single run per cell, but Wilson CIs and paired McNemar for the controlled re-run comparison; explicitly scopes the claim as task-level.
- **2606.04555** Temporal Order Matters for Agentic Memory: Segment Trees for Long-Horizon Agents (2026) — https://arxiv.org/abs/2606.04555
  SegTreeMem; uses only MAB's LongMemEval(S*) split ("LongMemEval-MAB", 300 Q), LLM-judge accuracy; all baselines re-run.
  Reports 95% CIs per cell (question-level); CIs of SegTreeMem and HippoRAG overlap (0.591-0.699 vs 0.504-0.616).
- **2608.07429** TEPA: Revoking Stale Memories for Conflict-Robust Language Agents (2026) — https://arxiv.org/abs/2608.07429
  Revocable memory; MAB FC-SH 6K over 3 query-order seeds (300 Q), TEPA ties last-write-wins at 0.890; MH-6K and SH-262K reported as failure boundaries.
  CIs/permutation tests reported for drift benchmarks, not for the MAB rows.
- **2609.34545** Remember Before You're Asked: MemDream for Self-Probing Memory Evolution (2026) — https://arxiv.org/abs/2609.34545
  Self-probing memory evolution; full 9-task MAB grid with GPT-4.1 and Llama-3.1-8B readers, all baselines re-run; claims +9.1 overall over O-Mem.
  Bootstrap CIs are given only for LoCoMo; MAB rows are single deterministic runs.
- **2609.33268** LSTMem: Hierarchical Long Short-Term Online Memory for Large Language Models (2026) — https://arxiv.org/abs/2609.33268
  Parametric online memory; MAB family scores (AR/TTL/LRU/SF) on 3 small backbones; baselines copied from the delta-Mem paper (Lei et al. 2026b, arXiv 2605.12357), averages recomputed.
  Notes a 1.2-point difference between two same-seed retrainings, larger than its +1.00 SF gain on Qwen3-4B.
- **2609.39765** MemCodex: Self-Programming Hierarchical Memory for Language Agents (2026) — https://arxiv.org/abs/2609.39765
  Evolved memory programs evaluated via its MemArena harness; a single aggregate MAB score (LLM-judge, GPT-4o-mini) on an 80% test split; baselines re-run.
  Search-based results averaged over 3 seeds; seed SD <=0.8 stated.
- **2610.08586** MINDSET: Energy-based Schema Evolution for Long Conversational Agent Memory (2026) — https://arxiv.org/abs/2610.08586
  Schema-based memory controller; MAB-150 subsample (125 ICL + 25 Recsys) with gpt-oss-120b; near-zero ICL scores.
  Claims "highest observed" on MAB but concedes difference is not significant; Recsys n=25.
- **2610.12124** Use and Disuse: Intent-Structured Experience Consolidation for Memory and Learning in LLM Agents (2026) — https://arxiv.org/abs/2610.12124
  Hippocam; FC-MH at 6K/32K/64K (100 Q each) and DetectiveQA (71 Q) vs Letta, A-Mem, mem0, HiAgent, ACE, all re-run on DeepSeek-V4.1-Flash.
  No uncertainty; DetQA lead over Letta is 3.5 pts on 71 questions.
- **2610.15715** History Replays Itself: Unleashing Long-Context Capability in Linear Attention (2026) — https://arxiv.org/abs/2610.15715
  NoteBank sparse episodic memory for linear-attention models; 2,100-question MAB QA subset (FC, EventQA, RULER QA) macro-avg; compares to the same backbone plain.
  Qwen3.5-9B gain is +1.1 pts; no uncertainty.
- **2610.11920** Event-Centric Memory with Query-Aware Graph Augmentation for Long-Term Conversational Agents (2026) — https://arxiv.org/abs/2610.11920
  QGMem; FC-SH/FC-MH (N=400 each) scored by LLM judge (not SubEM), baselines re-run on Qwen3-8B; claims SOTA.
  Reports +- SD over 10 judge repeats and bootstrap CIs on coverage; FC-SH lead over Mem0 is 2.18 pts.

## Batch 2
- **2509.25911** Mem-alpha: Learning Memory Construction via Reinforcement Learning (2025) — https://arxiv.org/abs/2509.25911
  RL-trained Qwen3-4B memory manager (core/episodic/semantic memory), Qwen3-32B answerer; MAB used as OOD test (9 datasets, 112 instances: RULER-QA1/2, LME, 5 ICL, InfBench-Sum).
  Re-runs Long-Context/RAG-Top2/MemAgent/MEM1; avg 0.592 vs RAG-Top2 0.502; single run, no uncertainty; "state-of-the-art" framing.
- **2601.08435** Fine-Mem: Fine-Grained Feedback Alignment for Long-Horizon Memory Management (2026) — https://arxiv.org/abs/2601.08435
  Step-level reward RL for memory manager on the Mem-alpha MAB subset; avg 0.664 vs Mem-alpha 0.592 (+7.2).
  LC/RAG/MemAgent/MEM1/Mem-alpha rows identical to Mem-alpha paper (copied); A-Mem, LightMem new. No uncertainty.
- **2605.14175** Grounded Continuation: A Linear-Time Runtime Verifier for LLM Conversations (2026) — https://arxiv.org/abs/2605.14175
  Symbolic dependency-map verifier evaluated on all 800 FactConsolidation Qs (pooled 4 lengths) across 5 QA models vs re-run fact-level RAG and LLM-only.
  FC-SH 0.94-0.98, FC-MH 0.64-0.74; exact McNemar tests and bootstrap intervals; compares to published GPT-4o-mini BM25 0.48/0.03.
- **2605.15710** SMMBench (2026) — https://arxiv.org/abs/2605.15710
  New multimodal memory benchmark; only lists MAB in a related-benchmark table. No own MAB numbers.
- **2606.04315** Exploring Cross-Scenario Generality of Agentic Memory Systems (AutoMEM) (2026) — https://arxiv.org/abs/2606.04315
  Evaluates 9 memory systems on a 200-question MAB subset (~50 per AR/TTL/LRU/CR) with Qwen3-32B and LLM-judge scoring; no MAB SOTA claim.
  Notable: reports a "Variance" (sampling-noise floor) row, 1.8-3.0 pp on MAB columns.
- **2606.06054** Beyond Similarity: Trustworthy Memory Search for Personal AI Agents (MemGate) (2026) — https://arxiv.org/abs/2606.06054
  Safety gate for memory retrieval; claims MAB AR-overall gains of ~1-1.4 pts for 4 agents x 2 backbones (n unclear, no uncertainty).
  GPT-4o-mini "Mem0" MAB row exactly matches MAB Table 3 Mem0 AR values (likely copied).
- **2606.10677** Infini Memory: Maintainable Topic Documents for Long-Term LLM Agent Memory (2026) — https://arxiv.org/abs/2606.10677
  Topic-document memory + agentic retrieval; full MAB with gpt-5-mini, gpt-5 binary LLM judge; baselines re-run via official integrations.
  Claims 64.7 overall vs HippoRAG-v2 45.5 (+19.2); single run, no uncertainty; FC length unspecified.
- **2606.15903** Control-Plane Placement Shapes Forgetting (ForgetEval) (2026) — https://arxiv.org/abs/2606.15903
  Appendix J cross-eval on FC 6K/32K (400 Qs) using top-10 retrieval substring scoring (not official), six systems.
  FC-SH saturates at 100%; FC-MH Mem0 37 vs 28 (6K), 19 vs 17 (32K); no uncertainty.
- **2606.25161** TRUSTMEM: Learning Trustworthy Memory Consolidation for LLM Agents (2026) — https://arxiv.org/abs/2606.25161
  Verifier-guided preference RL (Qwen3-4B) on Mem-alpha MAB subset; avg 65.7 vs Mem-alpha 59.2; claims SOTA.
  LC/RAG/Mem-alpha rows copied from Mem-alpha paper; loses on 4 of 9 datasets; single run, no uncertainty.
- **2607.05844** StateFuse: Deterministic Conflict-Preserving Memory for Multi-Agent Systems (2026) — https://arxiv.org/abs/2607.05844
  CRDT-based conflict-preserving memory; 282-question conflict-bearing MAB CR slice; all methods tie (64.9% cons., 97.5% aggressive).
  Deterministic; no performance claim beyond conflict recall.
- **2607.19096** Supra Cognitive Modes: A Routed Architecture for Agent Memory (2026) — https://arxiv.org/abs/2607.19096
  Routed retrieval architecture (Claude Sonnet 4.5 synthesis); MAB v3 full (n=3,671) 61.49% over 2 runs (61.72/61.26) vs production comparator 52.22, Mem0 v2 OSS 34.03.
  Explicitly disclaims significance; baseline raw rows unavailable; unusually careful reporting.
- **2607.20553** CMI-Mem: Generalizable Long-Term Memory Management via CMI-Augmented RL (2026) — https://arxiv.org/abs/2607.20553
  RL memory manager (4B/8B); MAB excluding FC-MH, LLM-judge (Qwen3.7-Max) replacing SubEM; CMI-Mem-8B 56.7 vs MemBuilder-RL-8B 51.7.
  Baselines re-run; single run, no uncertainty.

## Batch 3
- **2607.27834** MemTxn: A Transaction Boundary for Source-Supported Updates and Complete-State Recovery in Agent Memory (2026) — https://arxiv.org/abs/2607.27834
  Governance layer controlling version visibility; FactConsolidation (800 Q) F1 vs 6 re-run memory systems across 5 answer models.
  Claims best average F1 in every model block; 95% CIs only for a matched top-8 control vs Dense; per-cell margins at 262K/MH are 0.4–2.5 pts.
- **2608.04746** Caching for the Future: Scrub Jay Episodic Memory Principles for Agent Memory Systems (2026) — https://arxiv.org/abs/2608.04746
  ScrubJay-MEM; EventQA-64K (500 Q) with llama3.1:8b, re-run baselines, best F1 by +2.66 over Mem0; single run, no CI.
  With qwen3:30b Contriever beats it (78.68 vs 72.82); FC results relegated to appendix as unfavourable.
- **2608.12428** MindMemOS: A Portable and Self-Evolving Memory Operating Layer for AI Agents (2026) — https://arxiv.org/abs/2608.12428
  "Dreaming" consolidation; FC SH/MH averaged over 4 lengths vs Mem0/MemoRAG/HippoRAG-v2 copied from MAB.
  The copied baseline values match MAB's 262K-only cells, not 4-length averages (apples-to-oranges); no runs/CI.
- **2609.12436** LifeFuse-Mem: Lifecycle-Aware State Fusion Against Temporary Overwriting for Long-Term Memory (2026) — https://arxiv.org/abs/2609.12436
  LoRA-memory adapter; MAB (3,671 Q, 14 datasets) as compatibility check vs LoRA-Mem/Titans/δ-Mem on Qwen3-4B and SmolLM3-3B.
  Overall margins 0.41 and 0.08 pts; authors themselves call MAB results "mixed".
- **2609.25054** Memory of Memory / Provenant Memory (P-Mem) (2026) — https://arxiv.org/abs/2609.25054
  Provenance-ledger memory; FactConsolidation at 6K/64K/262K (n=100 per cell) vs re-run Mem0, HippoRAG-v2, Graphiti, MemoryOS, flat stores, Qwen3-4B.
  Notes informally that an 11 vs 8 gap is "within noise"; no CIs or repeats.
- **2605.12357** δ-mem: Efficient Online Memory for Large Language Models (2026) — https://arxiv.org/abs/2605.12357
  Online state memory for LLM; MAB category averages (AR/TTL/LRU/SF-SH only) on Qwen3-4B/8B vs re-run memory baselines.
  Claims strongest performance; n per category unclear; loses on LRU and SF to backbone/Context2LoRA.
- **2606.06448** Agent Memory: Characterization and System Implications of Stateful Long-Horizon Workloads (2026) — https://arxiv.org/abs/2606.06448
  Systems/cost characterization of memory systems on MAB; macro-averaged accuracy: BM25 55.8% best, HippoRAG v2 47.4%, GraphRAG 47.0%, Mem0 26.8%.
  Mostly figures; no uncertainty.
- **2609.10263** What Should an Agent Forget? Separating What Is Stored from What Is Used (2026) — https://arxiv.org/abs/2609.10263
  RD-Forget curation; MAB-FC = 100 Q from FC-MH 6K only; vs ACE/ReasoningBank across 4 models.
  Same config re-run in a separate batch gives 74 vs 77 (3-pt batch-to-batch difference on n=100).
- **2605.18421** EvoMemBench: Benchmarking Agent Memory from a Self-Evolving Perspective (2026) — https://arxiv.org/abs/2605.18421
  New benchmark whose InEp-Know split re-uses MAB AR (2000 Q) and SF (800 Q) data; evaluates 12 systems.
  Re-run Mem0 scores (LME 52, RULER qa2 68) far exceed MAB-published Mem0 (36, 32); backbones unclear.
- **2609.16073** The Immutable Past (2026) — https://arxiv.org/abs/2609.16073 — no own MAB numbers (says FactConsolidation unsuitable; uses synthetic data).
- **2602.13967** Neuromem (2026) — https://arxiv.org/abs/2602.13967 — uses MAB Selective-Forgetting re-streamed; results only in figures, no tabulated MAB numbers.
- **2608.19652** Can Agent Memory Systems Track Evolving State? (2026) — https://arxiv.org/abs/2608.19652 — qualitative critique of MAB only; no numbers.
- **2604.22085** Memanto (2026) — https://arxiv.org/abs/2604.22085 — cites MAB only; no MAB numbers.
- **2606.24775** Are We Ready For An Agent-Native Memory System? (2026) — https://arxiv.org/abs/2606.24775 — MAB(LongMemEval) shown only in a figure; no tabulated numbers.
- **2602.16313** MemoryArena (2026) — https://arxiv.org/abs/2602.16313 — MAB authors' follow-up; cites MAB in related work only, no MAB numbers.
- **2608.13883** MemoryLake on MemoryArena (2026) — https://arxiv.org/abs/2608.13883 — MemoryArena only; no MAB numbers.
