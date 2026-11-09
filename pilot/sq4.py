"""SQ4: can training teach the model to use hidden information? (run only if SQ2 is go / borderline)

On one model, two LoRA adapters are trained on the same questions (folds 0-2) to answer when their
labels say so and to reply "I don't know." otherwise:
  real-label       : answer iff the main answer is correct;
  confidence-label : answer iff the confidence (margin) is above a threshold set so that both label
                     sets answer exactly the same number of questions; it can only learn the rule.
Both use the model's own main answer as the answer target, the same initialisation, data order and
steps. On the held-out folds 3-4 each adapter is compared with the confidence rule at the same answer
rate (the unsteered main answers, keeping the most confident ones). Go if the real-label adapter's
surplus over the rule beats the confidence-label adapter's.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

import model_utils as mu
from data import qid_hash
from pilot import config as PC
from pilot.metrics import bootstrap_ci, transition_counts
from scoring import A_, C_, abstention_score, classify, match_coverage


class LoRALinear(torch.nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        dev = base.weight.device
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.t()) @ self.B.t() * self.scale).to(x.dtype)


class LoRA:
    def __init__(self, sess, seed):
        self.sess, self.mods = sess, []
        torch.manual_seed(seed)
        for layer in sess.layers:
            for name in PC.LORA_TARGETS:
                base = getattr(layer.self_attn, name)
                m = LoRALinear(base, PC.LORA_RANK, PC.LORA_ALPHA)
                setattr(layer.self_attn, name, m)
                self.mods.append((layer.self_attn, name, base, m))

    def params(self):
        return [p for *_, m in self.mods for p in (m.A, m.B)]

    def detach(self):
        for parent, name, base, _ in self.mods:
            setattr(parent, name, base)
        self.mods = []


def train_adapter(sess, data, seed, epochs):
    """data: list of (prompt ids, target ids incl. end-of-turn). Returns the attached LoRA and losses."""
    lora = LoRA(sess, PC.LORA_SEED)
    opt = torch.optim.AdamW(lora.params(), lr=PC.LORA_LR, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    steps = epochs * math.ceil(len(data) / PC.LORA_BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(steps // 10, 1)))
    losses = []
    model = sess.model
    for _ in range(epochs):
        order = rng.permutation(len(data))
        for b in range(0, len(order), PC.LORA_BATCH):
            batch = [data[i] for i in order[b:b + PC.LORA_BATCH]]
            seqs = [p + t for p, t in batch]
            ids, attn, T = mu.left_pad(seqs, sess.fmt.pad_id)
            labels = torch.full_like(ids, -100)
            for i, (p, t) in enumerate(batch):
                labels[i, T - len(t):] = torch.tensor(t, device=ids.device)
            with torch.enable_grad():
                out = model(input_ids=ids, attention_mask=attn, position_ids=mu.position_ids_from_mask(attn), use_cache=False)
                logits = out.logits[:, :-1].float()
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(lora.params(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            del out, logits, loss
    return lora, losses


def run_sq4(sess, pool, sh, epochs):
    folds = np.array([it.extra["fold"] for it in pool])
    tr = np.flatnonzero(np.isin(folds, PC.SQ4_TRAIN_FOLDS))
    te = np.flatnonzero(~np.isin(folds, PC.SQ4_TRAIN_FOLDS))
    states, margin = sh["states"], sh["margin"]
    answered = states != A_
    real_flag = states[tr] == C_
    k = int(real_flag.sum())
    ans_tr = tr[answered[tr]]
    conf_flag = np.zeros(len(tr), dtype=bool)
    if k:
        top = ans_tr[np.argsort(-margin[ans_tr], kind="stable")[:k]]
        conf_flag = np.isin(tr, top)
    prompts = [sess.prompt(it.question) for it in pool]
    eot = sess.fmt.eot_id

    def target(i, ans):
        return list(sh["token_ids"][i]) + [eot] if ans else list(sess.idk_cont)

    data_real = [(prompts[i], target(i, f)) for i, f in zip(tr, real_flag)]
    data_conf = [(prompts[i], target(i, f)) for i, f in zip(tr, conf_flag)]
    out = {"n_train": int(len(tr)), "n_eval": int(len(te)), "train_answer_targets": k,
           "label_overlap": float(np.mean(real_flag == conf_flag)) if len(tr) else float("nan"), "epochs": epochs}

    items_te = [pool[i] for i in te]
    base_st = states[te]
    tb = np.array([qid_hash(pool[i].qid) for i in te])
    results, st_by = {}, {}
    for name, data in (("real_label", data_real), ("confidence_label", data_conf)):
        lora, losses = train_adapter(sess, data, PC.LORA_SEED + 1, epochs)
        try:
            g = sess.greedy([prompts[i] for i in te])
        finally:
            lora.detach()
        st = np.array([classify(t, it.aliases)[0] for t, it in zip(g.texts, items_te)], dtype=np.int8)
        st_by[name] = st
        kk = int((st != A_).sum())
        rule = match_coverage(base_st, margin[te], tb, kk)
        results[name] = {"score_S": abstention_score(st), "coverage": kk / len(te) if len(te) else float("nan"),
                         "accuracy_answered": float(np.mean(st[st != A_] == C_)) if kk else float("nan"),
                         "rule_score_same_coverage": abstention_score(rule) if rule is not None else float("nan"),
                         "surplus_vs_rule": (abstention_score(st) - abstention_score(rule)) if rule is not None else float("nan"),
                         "loss_first": losses[0] if losses else float("nan"), "loss_last": float(np.mean(losses[-10:])) if losses else float("nan"),
                         "transitions_vs_unsteered": transition_counts(base_st, st)}
    out["adapters"] = results
    out["unsteered_score_S"] = abstention_score(base_st)
    sr, sc = st_by["real_label"], st_by["confidence_label"]
    kr, kc = int((sr != A_).sum()), int((sc != A_).sum())

    def stat(idx):
        rr = match_coverage(base_st[idx], margin[te][idx], tb[idx], int((sr[idx] != A_).sum()))
        rc = match_coverage(base_st[idx], margin[te][idx], tb[idx], int((sc[idx] != A_).sum()))
        if rr is None or rc is None:
            return float("nan")
        return (abstention_score(sr[idx]) - abstention_score(rr)) - (abstention_score(sc[idx]) - abstention_score(rc))

    out["delta_surplus"] = results["real_label"]["surplus_vs_rule"] - results["confidence_label"]["surplus_vs_rule"]
    out["delta_surplus_ci"] = bootstrap_ci(stat, len(te))
    out["coverages"] = {"real_label": kr, "confidence_label": kc}
    return out, {"eval_idx": te, "states": st_by}
