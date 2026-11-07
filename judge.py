"""Final F9: local LLM judge over every distinct test-split output (phase D of main.py).

Run 2's grader was never validated (no human or judge label was filled in), and spot checks found
plausibly wrong "wrong" labels (ambiguous titles, too-narrow gold aliases) and missed abstentions. The
final run therefore labels every distinct (question, output) of every PopQA test condition (seed 0 and
the seed arms), EntityQuestions and TriviaQA with a pinned local judge (config.JUDGE; not one of the
models under test) and attaches the result to each output as extra["judge_states"]. The registered
automatic labels stay primary; every verdict is recomputed under the judge labels (report.py,
verdicts.py), and the disagreements are written for a human tie-break (report.write_judge_disagreements).

Outputs that need no judgement keep a deterministic label: answers withdrawn by a gate and the exact
IDK reference forms are abstentions; degenerate outputs are wrong (as in the registered rules). If the
judge's reply cannot be parsed, the automatic label is kept and counted.
"""
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import model_utils as mu
import stats as ST
from audit import JUDGE_SYSTEM, JUDGE_TEMPLATE, LABEL_TO_STATE, parse_label
from config import IDK_VARIANTS, JUDGE, JUDGE_BATCH_SIZE, JUDGE_MAX_NEW_TOKENS, MODELS
from scoring import A_, STATE_NAMES, W_, WITHDRAWN_PREFIX, is_degenerate

FORCED_ABSTAIN = ("always_abstain",)


def output_groups(r):
    """[(split, items, {label: CondOutput})] for every test-split output of one model runner."""
    groups = [("test", r.items["test"], {"unsteered": r.uout["test"], **r.test_out, **r.ablation_out})]
    for s, outs in r.seed_out.items():
        groups.append(("test", r.items["test"], {f"{n}@seed{s}": o for n, o in outs.items()}))
    for split, store in (("eq", r.eq_out), ("tqa", r.tqa_out)):
        if store:
            groups.append((split, r.items[split], {"unsteered": r.uout[split], **store["full"]}))
    return groups


def deterministic_state(text):
    if text.startswith(WITHDRAWN_PREFIX) or text.strip() in IDK_VARIANTS:
        return A_
    if is_degenerate(text):
        return W_
    return None


class BatchedJudge:
    def __init__(self, spec):
        if spec["hf_id"] in {m["hf_id"] for m in MODELS}:
            raise ValueError("the judge must not be one of the models under test")
        self.tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"], local_files_only=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=spec["revision"], dtype=mu.DTYPE,
                                                          local_files_only=True)
        self.model.to(mu.DEVICE).train(False)
        self.model.requires_grad_(False)
        eos = self.tok.eos_token_id
        self.gc = GenerationConfig(do_sample=False, max_new_tokens=JUDGE_MAX_NEW_TOKENS, temperature=None, top_p=None,
                                   top_k=None, repetition_penalty=1.0, eos_token_id=eos, pad_token_id=self.tok.pad_token_id)

    def prompt_ids(self, question, aliases, output):
        msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_TEMPLATE.format(question=question, aliases=" | ".join(aliases),
                                                                  output=output)}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, add_special_tokens=False).input_ids

    @torch.no_grad()
    def label(self, triples):
        out = []
        for sl in mu.batch_slices(len(triples), JUDGE_BATCH_SIZE):
            ids, attn, T = mu.left_pad([self.prompt_ids(*t) for t in triples[sl]], self.tok.pad_token_id)
            gen = self.model.generate(input_ids=ids, attention_mask=attn, generation_config=self.gc)
            out += [parse_label(self.tok.decode(row, skip_special_tokens=True)) for row in gen[:, T:].tolist()]
        return out

    def close(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_judge(runners, timer=None):
    t0 = time.time()
    keys, triples, plan = {}, [], []
    for r in runners:
        for split, items, outs in output_groups(r):
            for name, o in outs.items():
                refs = []
                for it, text in zip(items, o.texts):
                    s = deterministic_state(text)
                    if s is not None:
                        refs.append(("fixed", s))
                        continue
                    key = (it.qid, " ".join(text.split()))
                    if key not in keys:
                        keys[key] = len(triples)
                        triples.append((it.question, it.aliases, text))
                    refs.append(("judge", keys[key]))
                plan.append((name, o, refs))
    print(f"JUDGE: model={JUDGE['hf_id']}@{JUDGE['revision'][:8]} distinct_outputs={len(triples)}", flush=True)
    judge = BatchedJudge(JUDGE)
    try:
        order = sorted(range(len(triples)), key=lambda i: len(triples[i][0]) + len(triples[i][2]))   # less padding
        labels = [None] * len(triples)
        for i, lab in zip(order, judge.label([triples[i] for i in order])):
            labels[i] = lab
    finally:
        judge.close()
    n_unparsed = sum(lab is None for lab in labels)
    for name, o, refs in plan:
        st = o.states.copy()
        for i, (kind, v) in enumerate(refs):
            if kind == "fixed":
                st[i] = v
            elif labels[v] is not None:
                st[i] = LABEL_TO_STATE[labels[v]]
        if name.split("@")[0] in FORCED_ABSTAIN:
            st[:] = A_
        o.extra["judge_states"] = st
    agreement = {}
    for r in runners:
        u = r.uout["test"]
        a = [STATE_NAMES[int(x)] for x in u.states]
        j = [STATE_NAMES[int(x)] for x in u.extra["judge_states"]]
        k = ST.cohen_kappa(a, j)
        n_cells = {f"{x}->{y}": int(sum(1 for p, q in zip(a, j) if p == x and q == y))
                   for x in ("correct", "abstained", "wrong") for y in ("correct", "abstained", "wrong")}
        agreement[r.name] = {"unsteered_test_kappa_auto_vs_judge": k["kappa"], "agreement": k["agreement"],
                             "n": k["n"], "auto_to_judge": n_cells}
        print(f"JUDGE_AGREEMENT model={r.name} unsteered_test kappa={k['kappa']:.3f} agreement={k['agreement']:.3f} "
              f"auto_wrong->judge_correct={n_cells['wrong->correct']} auto_correct->judge_wrong={n_cells['correct->wrong']} "
              f"auto_wrong->judge_abstained={n_cells['wrong->abstained']}", flush=True)
    info = {"status": "ok", "model": dict(JUDGE), "n_distinct_outputs": len(triples), "n_unparsed": int(n_unparsed),
            "seconds": round(time.time() - t0, 1), "agreement": agreement}
    print(f"JUDGE: done distinct_outputs={len(triples)} unparsed={n_unparsed} seconds={info['seconds']:.0f}", flush=True)
    return info
