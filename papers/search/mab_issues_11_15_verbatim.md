# MemoryAgentBench issue #11 (fetched 2026-10-15 via api.github.com)
## Empty memory in Mem0 (NiNicoder, 2025-11-18T15:32:23Z)

Hi, thanks for your good work. I have a question while reproducing mem0 on the Conflict Resolution benchmark: the memory remains empty after processing each chunk, with almost no information being extracted. Is this due to prompt design issues, or is this expected behavior? 

### comment by HUST-AI-HYZ (2025-11-19T04:30:22Z)

I suspect this might be related to the performance of mem0 itself. I've tried various methods, such as improving the prompt design, but mem0's memory is often empty. I've found that mem0 only performs relatively well in memory retrieval when processing information like longmemeval, which is more focused on the history of human-computer dialogue. However, the type of input text/tasks of our designed benchmark are not limited to the history of human-computer dialogue.

# MemoryAgentBench issue #15 (fetched 2026-10-15 via api.github.com)
## Question about reproducing Mem0 + LongMemEval(S*) results (jisuozhao, 2025-12-03T10:33:31Z)

Hello,

First of all, great work! Thank you for your efforts.

---

I have a question regarding reproducing the Mem0 + LongMemEval(S*) results. I am currently using the MemoryAgentBench's latest main branch (b763f7e).

The command I ran is:

```
python MemoryAgentBench/main.py --agent_config "MemoryAgentBench/configs/agent_conf/RAG_Agents/gpt-4o-mini/Structure_rag_gpt-4o-mini-mem0.yaml" --dataset_config "MemoryAgentBench/configs/data_conf/Accurate_Retrieval/LongMemEval/Longmemeval_s_star.yaml"
```

I did not modify `Structure_rag_gpt-4o-mini-mem0.yaml` or `Longmemeval_s_star.yaml`.

---

The results I obtained are:
```
Accuracy: 0.1533
        single-session-user: 0.2 (45)
        temporal-reasoning: 0.2 (75)
        knowledge-update: 0.2 (45)
        multi-session: 0.0933 (75)
        single-session-preference: 0.0333 (30)
        single-session-assistant: 0.1667 (30)
```

In the paper, the reported accuracy is 36.0. Is this the same value as the Accuracy shown above?

<img width="1250" height="682" alt="Image" src="https://github.com/user-attachments/assets/3bb4e371-1ab1-4a31-8833-acd47a93056c" />

Would you be able to share the results you obtained, as well as the version of the Mem0 open-source SDK and any Mem0-related configurations you used? I would greatly appreciate it!

Thank you very much.

### comment by HUST-AI-HYZ (2025-12-04T08:13:16Z)

Hello,

Thank you for your hard work and for sharing your results. 

I have a couple of observations and questions regarding the Mem0 evaluation process that might relate to the discrepancies you're seeing:

- Memory Extraction and Memory Pool Instability in Mem0:
When I run evaluations with Mem0, I frequently encounter issues where the necessary memory cannot be properly extracted and saved to the memory pool. I wanted to check if you also observed that during your evaluation, the printed memory chunks processed by the memory pool were often empty?
My understanding is that this inherent characteristic—the instability in extracting and pooling memory—significantly contributes to the inconsistent performance of Mem0.

- Robustness of Mem0's Memory Construction:
I've experimented with a different approach (which I manually commented out in my local Mem0 implementation) where the memorized needed information is directly sent to gpt-4o-mini, which returns an output. We then wrap this output and information into a chunk and send it back for Mem0 to process. This seemed to alleviate some of the performance issues with Mem0.

However, my overall impression is that Mem0's memory construction is not robust on our Benchmark. It often requires slight, delicate adjustments to its configuration or internal process before it can work reasonably well.

### comment by jisuozhao (2025-12-05T02:17:22Z)

Yes, I also frequently encounter cases where Mem0 fails to add any memory.

At first, I tried reproducing your `FactConsolidation` dataset, but Mem0 was almost unable to generate any memories on that dataset.

Then I switched to the `LongMemEval` dataset, and during evaluation, I again observed multiple instances where no memory was created.

May I confirm whether the modification you mentioned refers to the commented logic in `agent.py,` specifically in `AgentWrapper::_handle_mem0_agent()`?

<img width="2048" height="993" alt="Image" src="https://github.com/user-attachments/assets/0f4bafc7-ee0b-48a0-a4f7-f7208f25511d" />

### comment by HUST-AI-HYZ (2025-12-05T06:53:52Z)

Yes. I think the Mem0 performs not well when the assistant's message is short. 

