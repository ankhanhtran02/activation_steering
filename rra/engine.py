"""Model operations: the VES/ECS engine (vesecs/engine.py, on top of SQ2's pilot/session.py) unchanged for the two
main models, plus
  - Gemma-2 loading for the optional generality stage G (eager attention so that Gemma-2's attention soft-capping
    is applied; Gemma-2 has no system role, so the SQ2 system prompt is put at the start of the user turn);
  - the failure policy for GPU memory errors (spec section 9): retry with half the batch size, up to 3 times,
    then stop the run. Model loading is retried after freeing memory and waiting (last run: a CUDA context
    could not be created right after start-up).
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import model_utils as mu
from model_utils import ChatFormatter, decoder_layers
from pilot import config as PC
from pilot.session import Session
from vesecs.engine import Engine as VesEngine

# Gemma-2's own chat template (used only if a tokenizer ships without one, e.g. the smoke test's tiny model)
GEMMA_TEMPLATE = ("{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'assistant') %}"
                  "{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}"
                  "{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}"
                  "{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}")


class RunStopped(Exception):
    """The failure policy stops the run (a check could not run, or a model could not be loaded)."""


def is_oom(e):
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class GemmaFormatter(ChatFormatter):
    """SQ2's prompt for a model without a system role: system prompt + blank line + user text in the user turn."""

    def _render(self, messages, add_generation_prompt):
        msgs = [dict(m) for m in messages]
        if msgs and msgs[0]["role"] == "system":
            sys_text = msgs.pop(0)["content"]
            if msgs and msgs[0]["role"] == "user":
                msgs[0]["content"] = sys_text + "\n\n" + msgs[0]["content"]
        if not getattr(self.tok, "chat_template", None):
            self.tok.chat_template = GEMMA_TEMPLATE
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)


def _fill_dims(spec):
    """Smoke test only: a tiny model's dimensions are read from its config."""
    if all(int(spec.get(k) or 0) > 0 for k in ("n_layers", "d_model", "n_heads", "head_dim")):
        return spec
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(spec["hf_id"], revision=spec["revision"], local_files_only=True)
    spec.update({"n_layers": c.num_hidden_layers, "d_model": c.hidden_size, "n_heads": c.num_attention_heads,
                 "head_dim": getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)})
    return spec


def load_gemma(spec):
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"], local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=spec["revision"], dtype=mu.DTYPE,
                                                 attn_implementation=spec.get("attn_implementation", "eager"),
                                                 local_files_only=True)
    model.to(mu.DEVICE).train(False)
    model.requires_grad_(False)
    assert len(model.model.layers) == spec["n_layers"], "unexpected layer count"
    assert model.config.hidden_size == spec["d_model"], "unexpected d_model"
    fmt = GemmaFormatter(tok, spec["name"])
    model.generation_config = GenerationConfig(bos_token_id=tok.bos_token_id, eos_token_id=fmt.stop_ids,
                                               pad_token_id=tok.pad_token_id)
    return model, tok, fmt


class GemmaSession(Session):
    def __init__(self, spec, revision):           # Session.__init__ with the Gemma loader
        self.spec = spec
        self.name = spec["name"]
        self.model, self.tok, self.fmt = load_gemma(spec)
        self.layers = decoder_layers(self.model)
        self.n_layers = len(self.layers)
        assert self.model.config.num_attention_heads == spec["n_heads"], "unexpected head count"
        self.plan = PC.layer_plan(spec)
        self.idk_cont = list(self.fmt.idk_ids) + [self.fmt.eot_id]
        self.sampler_mode = None


class Engine(VesEngine):
    """vesecs.engine.Engine with a Gemma-2 session when spec['family'] == 'gemma2'."""

    def __init__(self, spec, cfg):
        if spec.get("family") != "gemma2":
            super().__init__(spec, cfg)
            return
        spec = _fill_dims(spec)
        self.spec, self.cfg, self.name = spec, cfg, spec["name"]
        t = time.time()
        self.sess = GemmaSession(spec, spec["revision"])
        self.load_seconds = time.time() - t
        self.model, self.tok, self.fmt = self.sess.model, self.sess.tok, self.sess.fmt
        self.layers = self.sess.layers
        self.n_layers = len(self.layers)
        self.d = int(spec["d_model"])
        self.dev = mu.DEVICE
        dec = cfg["decoding"]
        self.bs = int(dec["batch_size"])
        self.max_new = int(dec["max_new_tokens"])
        self.eot = self.fmt.eot_id
        self.stop = set(self.fmt.stop_ids)
        self.idk_cont = self.sess.idk_cont

    # ------------------------------------------------------------ RRA operations
    def first_token_ban(self, prompts, a1_token_ids):
        """A2 = B7 round 1: greedy decoding with A1's first token banned at the first answer step only (at most
        max_new_tokens). Rows are processed in the given order with the engine's batch size, as B7 did."""
        ban = [[int(t[0])] if len(t) else [] for t in a1_token_ids]
        return self.generate(prompts, ban=ban)


def load_engine(spec, cfg, log=print):
    """Engine(spec) with the load retry: on a GPU memory error, free memory, wait and try again (waits from the
    config); after the last retry the run stops."""
    waits = list(cfg["checks"]["load_retry_wait_s"])
    for attempt in range(len(waits) + 1):
        try:
            return Engine(spec, cfg)
        except Exception as e:
            if not is_oom(e):
                raise
            free_memory()
            if attempt == len(waits):
                raise RunStopped(f"loading {spec['name']} failed with a GPU memory error after {len(waits)} retries: {e}")
            log(f"OOM_RETRY: loading {spec['name']} ({type(e).__name__}); waiting {waits[attempt]} s, retry "
                f"{attempt + 1}/{len(waits)}")
            time.sleep(float(waits[attempt]))


def with_oom_retry(eng, fn, name, retries, log=print):
    """fn() with the failure policy: on a GPU memory error, halve eng.bs and retry, up to `retries` times; then
    RunStopped. The batch size is restored afterwards. Returns (result, batch size used)."""
    bs0 = eng.bs
    try:
        for attempt in range(retries + 1):
            try:
                return fn(), eng.bs
            except Exception as e:
                if not is_oom(e):
                    raise
                free_memory()
                if attempt == retries:
                    raise RunStopped(f"{name}: GPU memory error after {retries} retries with halved batch sizes "
                                     f"(last batch size {eng.bs}): {e}")
                eng.bs = max(1, eng.bs // 2)
                log(f"OOM_RETRY: {name} ({type(e).__name__}); batch size -> {eng.bs} (retry {attempt + 1}/{retries})")
    finally:
        eng.bs = bs0


def snapshot_ok(spec):
    """True if the model's weights are in the local Hugging Face cache at its pinned revision."""
    from huggingface_hub import snapshot_download
    try:
        snap = Path(snapshot_download(spec["hf_id"], revision=spec["revision"], local_files_only=True,
                                      allow_patterns=PC.MODEL_FILE_PATTERNS))
    except Exception:
        return False
    idx = snap / "model.safetensors.index.json"
    shards = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else \
        [q.name for q in snap.glob("*.safetensors")]
    return bool(shards) and all((snap / x).exists() for x in shards)


def as_np(x):
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
