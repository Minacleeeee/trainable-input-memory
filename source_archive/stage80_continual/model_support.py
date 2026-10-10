"""
Stage 79B dual-adapter model management.

Handles:
  * Loading the frozen SmolLM2-1.7B base.
  * Attaching fast (q/v r=32) and slow (full-layer r=6) adapters.
  * Training with BOTH adapters in the forward pass (only target gets grads)
    -- the critical Stage 78 fix.
  * Single-adapter configs (A, B, D, D') use the same plumbing.
  * Saving / loading adapter weight directories between cycles.
  * Resetting the fast adapter to a fresh zero-init.

The base model is NEVER merged into after training; adapter weights are saved
to disk and re-attached next cycle. This keeps fast (volatile) and slow
(consolidated) knowledge cleanly separated.
"""
import os
import sys
import time
import random
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_STAGE72 = os.path.normpath(os.path.join(_HERE, "..", "stage72_gpu"))
for _p in (_STAGE72,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from directml_patch import patch_causal_mask, make_gpu_compatible, get_directml_device  # noqa: E402

sys.path.insert(0, _HERE)
from stage79a_scorer import strict_score  # noqa: E402

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MODEL_REVISION = None

# --------------------------------------------------------------------------- #
# Architecture configs
# --------------------------------------------------------------------------- #
FAST_NAME = "fast"
SLOW_NAME = "slow"
SINGLE_NAME = "single"

FAST_TARGETS = ["q_proj", "v_proj"]
SLOW_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

BATCH_SIZE = 2
MAX_LENGTH = 64
GRAD_CLIP = 1.0


def fast_config():
    return LoraConfig(r=32, lora_alpha=64, target_modules=FAST_TARGETS,
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")


def slow_config():
    return LoraConfig(r=6, lora_alpha=12, target_modules=SLOW_TARGETS,
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")


def single_config(arch):
    """Return LoraConfig for single-adapter configs A/B/D/D'.

    arch: 'A' -> q/v r=32
          'B' -> full-layer r=6
          'D' -> q/v r=64
          'Dp' -> q/v r=66
    """
    if arch == "A":
        return LoraConfig(r=32, lora_alpha=64, target_modules=FAST_TARGETS,
                          lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    if arch == "B":
        return LoraConfig(r=6, lora_alpha=12, target_modules=SLOW_TARGETS,
                          lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    if arch == "D":
        return LoraConfig(r=64, lora_alpha=128, target_modules=FAST_TARGETS,
                          lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    if arch == "Dp":
        return LoraConfig(r=66, lora_alpha=132, target_modules=FAST_TARGETS,
                          lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    raise ValueError(arch)


# --------------------------------------------------------------------------- #
# Tokenizer / base model
# --------------------------------------------------------------------------- #
def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION,
                                       local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(device):
    """Load pristine base model on GPU, frozen, GPU-patched."""
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, local_files_only=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True)
    m.to(device)
    m = make_gpu_compatible(m)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# --------------------------------------------------------------------------- #
# Chat batching (reused from stage79a_model.py)
# --------------------------------------------------------------------------- #
def format_chat(tokenizer, question, answer=None):
    messages = [{"role": "user", "content": question}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=(answer is None))


def make_chat_batch(tokenizer, items, device, max_length=MAX_LENGTH):
    """items: list[(question, answer)]. Returns input_ids, labels (answer-only)."""
    all_input_ids, all_labels = [], []
    for q, a in items:
        prompt_text = format_chat(tokenizer, q, answer=None)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prefix_len = len(prompt_ids)
        full_text = format_chat(tokenizer, q, a)
        full_ids = tokenizer(full_text, truncation=True,
                             max_length=max_length)["input_ids"]
        labels = [-100] * prefix_len + full_ids[prefix_len:]
        all_input_ids.append(full_ids)
        all_labels.append(labels)
    max_len = max(len(ids) for ids in all_input_ids)
    padded_ids, padded_labels = [], []
    for ids, labs in zip(all_input_ids, all_labels):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
        padded_labels.append(labs + [-100] * pad_len)
    input_ids = torch.tensor(padded_ids, dtype=torch.long).to(device)
    labels = torch.tensor(padded_labels, dtype=torch.long).to(device)
    return input_ids, labels


def generate_response(model, tok, prompt, max_new_tokens=30):
    text = format_chat(tok, prompt, answer=None)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip().lower()


# --------------------------------------------------------------------------- #
# Dual-adapter attachment
# --------------------------------------------------------------------------- #
def attach_dual_adapters(base_model):
    """Attach fresh fast + slow adapters to a base model. Returns PeftModel."""
    model = get_peft_model(base_model, fast_config(), adapter_name=FAST_NAME)
    model.add_adapter(SLOW_NAME, slow_config())
    return model


def attach_single_adapter(base_model, arch):
    """Attach a single fresh adapter. Returns PeftModel."""
    model = get_peft_model(base_model, single_config(arch),
                           adapter_name=SINGLE_NAME)
    return model


def load_saved_adapters(base_model, adapter_dir):
    """Load saved adapter weights onto a fresh base, preserving names.

    PEFT multi-adapter save layout: adapter_dir/<name>/adapter_config.json.
    A single-adapter save may place adapter_config.json at the top level.
    Each subdir is loaded with adapter_name=<subdir name> so that fast/slow/
    single naming survives the reload (needed by handoff.delete_adapter).
    """
    subs = []
    top_cfg = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(top_cfg):
        subs.append(("default", adapter_dir))
    else:
        for entry in sorted(os.listdir(adapter_dir)):
            sub = os.path.join(adapter_dir, entry)
            if os.path.isdir(sub) and os.path.exists(
                    os.path.join(sub, "adapter_config.json")):
                subs.append((entry, sub))
    if not subs:
        raise FileNotFoundError(f"No adapters found in {adapter_dir}")
    first_name, first_path = subs[0]
    model = PeftModel.from_pretrained(base_model, first_path,
                                      adapter_name=first_name)
    for name, path in subs[1:]:
        model.load_adapter(path, adapter_name=name)
    return model


# --------------------------------------------------------------------------- #
# Training mode switches
# --------------------------------------------------------------------------- #
def set_train_target(model, target):
    """Activate adapters in forward; set requires_grad so only `target` trains.

    target: FAST_NAME, SLOW_NAME, or SINGLE_NAME.
    For dual-adapter configs, the other adapter stays in forward (frozen).
    For single-adapter configs, just activate the single adapter.
    """
    if target == SINGLE_NAME:
        model.base_model.set_adapter([SINGLE_NAME], inference_mode=False)
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
        model.train()
        return

    # Dual: both adapters in forward, freeze non-target.
    other = SLOW_NAME if target == FAST_NAME else FAST_NAME
    model.base_model.set_adapter([FAST_NAME, SLOW_NAME], inference_mode=False)
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)
            continue
        if other in n:
            p.requires_grad_(False)
        elif target in n:
            p.requires_grad_(True)
    model.train()


def set_eval_adapters(model, adapters=None):
    """Set model to eval mode and activate the given adapters.

    adapters: list of adapter names, or None for "all currently available".
    """
    if adapters is not None:
        model.base_model.set_adapter(adapters, inference_mode=True)
    model.eval()


# --------------------------------------------------------------------------- #
# Save / load adapter weights
# --------------------------------------------------------------------------- #
def save_adapters(model, path, selected=None):
    """Save adapter weights to disk. Deletes path first.

    Must move to CPU first: DirectML tensors don't support untyped_storage()
    introspection that safetensors needs.
    """
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    model = model.cpu()
    model.save_pretrained(path, selected_adapters=selected if selected else None)
    return path


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def _train_once_b(model, tokenizer, items, epochs, lr, device, tag, shuffle, batch_size):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return [0.0], 0
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.0, foreach=False)
    epoch_losses = []
    steps = 0
    for ep in range(epochs):
        order = list(range(len(items)))
        if shuffle:
            random.shuffle(order)
        ep_loss = 0.0
        ep_steps = 0
        for i in range(0, len(order), batch_size):
            idxs = order[i:i + batch_size]
            batch = [items[j] for j in idxs]
            input_ids, labels = make_chat_batch(tokenizer, batch, device)
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            opt.zero_grad()
            lv = loss.item()
            if lv != lv or lv in (float("inf"), float("-inf")):
                raise FloatingPointError(f"NaN/Inf loss in {tag} ep{ep+1}")
            ep_loss += lv
            steps += 1
            ep_steps += 1
        avg = ep_loss / max(ep_steps, 1)
        epoch_losses.append(avg)
        print(f"      [{tag}] ep{ep+1}/{epochs} loss={avg:.4f}", flush=True)
    return epoch_losses, steps


def train_items(model, tokenizer, items, epochs, lr, device, tag="",
                shuffle=True, batch_size=BATCH_SIZE):
    """Fresh AdamW each call. Returns (avg_loss, n_updates).

    Numerical guard: retry the phase ONCE with a fresh shuffle if any epoch
    loss > 1.0 or the final-epoch loss > 0.5; raise if it recurs.
    """
    if not items:
        return 0.0, 0
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return 0.0, 0
    final_ep = max_ep = float("nan")
    for attempt in range(2):
        epoch_losses, steps = _train_once_b(
            model, tokenizer, items, epochs, lr, device, tag, shuffle, batch_size)
        final_ep = epoch_losses[-1]
        max_ep = max(epoch_losses)
        if final_ep <= 0.5:
            return sum(epoch_losses) / max(len(epoch_losses), 1), steps
        print(f"      [guard] {tag} unstable (final={final_ep:.3f}, "
              f"max={max_ep:.3f}); retry {attempt+1}", flush=True)
    raise RuntimeError(
        f"{tag}: unstable after retry (final={final_ep:.3f}, max={max_ep:.3f})")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def eval_items(model, tokenizer, items, known_answers=None, max_new_tokens=30):
    """Evaluate a list of {eval_q, answer, id, ...}. Returns list of result dicts."""
    results = []
    for it in items:
        resp = generate_response(model, tokenizer, it["eval_q"],
                                 max_new_tokens=max_new_tokens)
        ok, reason = strict_score(resp, it["answer"], known_answers=known_answers)
        results.append({
            "id": it["id"], "type": it["type"],
            "eval_q": it["eval_q"], "expected": it["answer"],
            "response": resp, "correct": ok, "reason": reason,
        })
    return results


def eval_accuracy(results):
    if not results:
        return 0.0
    return sum(1 for r in results if r["correct"]) / len(results)
