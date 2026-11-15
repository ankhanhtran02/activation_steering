"""Budget cut 1: draw the 18 tries with vLLM (same weights, bf16, same prompt token ids, T = 1, top-p 1, no
top-k, at most 16 new tokens, per-request seed = hash(model, qid, try)). Used only when the budget guard applies
the cut and `import vllm` works; any error falls back to Hugging Face generation (vesecs.pipeline.c_tries).
Not exercised by the CPU smoke test."""
import gc

import torch


def vllm_tries(pl, eng, prompts, qids, n, try_offset=1):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from vesecs.io import stable_int
    spec = eng.spec
    llm = LLM(model=spec["hf_id"], revision=spec["revision"], dtype="bfloat16", seed=int(pl.cfg["seed"]),
              gpu_memory_utilization=float(pl.cfg["budget"]["vllm_gpu_memory_utilization"]))
    try:
        reqs, params = [], []
        for p, q in zip(prompts, qids):
            for j in range(n):
                reqs.append(TokensPrompt(prompt_token_ids=list(p)))
                params.append(SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=eng.max_new,
                                             seed=stable_int(eng.name, q, try_offset + j), stop_token_ids=list(eng.fmt.stop_ids),
                                             skip_special_tokens=True))
        outs = llm.generate(reqs, params, use_tqdm=False)
        stop = set(eng.fmt.stop_ids)
        flat = []
        for o in outs:
            c = o.outputs[0]
            ids = [t for t in c.token_ids if t not in stop]
            flat.append((eng.decode(ids), ids, c.finish_reason == "length"))
        return [flat[i * n:(i + 1) * n] for i in range(len(prompts))]
    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
