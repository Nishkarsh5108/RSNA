"""Report -> per-finding assertion probabilities with a local instruction LLM.

Method (v3, "forced-choice assertion scoring"):
  * The full report (no character truncation) is placed in a chat prompt that
    defines the 12 competition findings and five assertion categories:
        A PRESENT | B POSSIBLE | C NOT MENTIONED | D ABSENT | E PRIOR ONLY
  * The assistant turn is teacher-forced to the fixed layout
        "ACL injury: X\nMCL injury: X\n..."
    and at every X the next-token distribution restricted to the five letter
    tokens is recorded. The argmax letter is fed back before the next item, so
    the model sees its own previous answers exactly as in greedy decoding.
  * Nothing is parsed from free text, so there is no JSON failure mode and no
    0.5 fallback. `letter_mass` (probability mass the model put on the five
    letters before renormalisation) is stored as a format-adherence check; rows
    below `--min-letter-mass` are flagged, not silently accepted.

Outputs (append-only CSV, resumable, one row per study):
  StudyInstanceUID, status, error, n_prompt_tokens, report_truncated,
  {target}__pA..pE, {target}__letter_mass, {target}__argmax, run metadata.

The reports never leave this machine.
"""
import argparse
import hashlib
import json
import os
import time
import traceback
import unicodedata

import numpy as np
import pandas as pd
import torch
from transformers import DynamicCache

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]

# Item labels used in the forced assistant layout (order == TARGETS).
ITEM_KEYS = ["ACL injury", "MCL injury", "Medial meniscus tear", "Lateral meniscus tear",
             "Medial tibiofemoral OA", "Lateral tibiofemoral OA", "Patellofemoral OA",
             "Joint effusion", "Synovitis", "Baker's cyst", "Bone contusion", "Fracture"]
LETTERS = ["A", "B", "C", "D", "E"]
PROMPT_VERSION = "assert-v3.1"

# 95% of the Greek reports spell mu with the MICRO SIGN and Delta with the
# INCREMENT sign (PDF/Word conversion artefact); map them back to Greek letters.
GREEK_FIXES = {"µ": "μ", "∆": "Δ", "Ω": "Ω"}


def normalize_report(text):
    text = unicodedata.normalize("NFC", str(text))
    n_greek = sum(1 for c in text if "Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿")
    n_latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if n_greek > 0.3 * max(n_latin, 1):
        for a, b in GREEK_FIXES.items():
            text = text.replace(a, b)
    return text

SYSTEM_PROMPT = """You are a meticulous musculoskeletal radiologist. You read knee MRI reports written in any language (English, Spanish, Turkish, Croatian/Bosnian/Serbian, Greek, German, Bulgarian, Dutch, French, and others) and state, for one finding at a time, what the report says about it.

Answer every item with exactly one letter:
A = PRESENT: the report states the finding is currently present (any grade or size, including partial, small, mild, or chronic but still present).
B = POSSIBLE: the report calls it possible, suspected, probable, equivocal or "cannot be excluded", or describes changes that only partly meet the definition below.
C = NOT MENTIONED: nothing in the report addresses this finding or the structure it concerns.
D = ABSENT: the report states it is absent, normal or intact, or describes the relevant structure as normal.
E = PRIOR ONLY: the finding is only historical (operated, repaired, reconstructed, healed or resolved) and is not currently present.

Reading rules:
- Use the findings and the impression/conclusion. Clinical history, indication or the clinical question are NOT evidence: "rule out meniscal tear", "trauma", "contusion of the knee" or "suspected ACL tear?" written as the reason for the exam does not make a finding present.
- If findings and impression disagree, prefer the impression/conclusion.
- A statement that a structure is normal/intact counts as D for findings of that structure.
- Inside the findings or impression, "R/O X", "rule out X", "DD X", "to exclude X" or "X?" means X is suspected (B), not excluded.
- Each report describes one knee. "Medial" and "lateral" refer to the compartments of that knee.

Finding definitions:
1. ACL injury: current tear of the anterior cruciate ligament (complete, partial, high- or low-grade), rupture, sprain, avulsion, or a torn ACL graft. Mucoid/myxoid degeneration, ganglion cyst or thickening without a tear = B at most. Intact graft after reconstruction = E.
2. MCL injury: sprain of any grade (including grade 1 periligamentous edema), partial or complete tear, or avulsion of the medial collateral ligament. Old post-traumatic thickening/ossification only (e.g. Pellegrini-Stieda) = E.
3. Medial meniscus tear: any tear of the medial meniscus (horizontal, vertical, longitudinal, radial, oblique, complex, flap, bucket-handle, root tear, meniscocapsular separation), grade 3 signal reaching the articular surface (also when called "grade III degeneration" or "grade III meniscopathy"), maceration or fragmentation. Grade 1-2 intrasubstance signal/degeneration, extrusion alone or a discoid meniscus without tear = D. Partial meniscectomy without a new tear = E.
4. Lateral meniscus tear: same rules for the lateral meniscus.
5. Medial tibiofemoral OA: osteoarthritis (arthrosis, gonarthrosis, degenerative arthropathy) of the medial femorotibial compartment, or advanced cartilage loss there (grade 3-4 chondromalacia/chondropathy, full-thickness defects, denuded bone), especially with osteophytes, subchondral sclerosis/cysts/edema or joint-space narrowing. Only mild (grade 1-2) chondromalacia = B. Generalized/tricompartmental/femorotibial OA counts for both medial and lateral compartments.
6. Lateral tibiofemoral OA: same rules for the lateral femorotibial compartment.
7. Patellofemoral OA: osteoarthritis of the patellofemoral joint, or advanced cartilage loss of the patella or trochlea (grade 3-4, full-thickness ulcers), especially with osteophytes or subchondral changes. Only mild (grade 1-2) chondromalacia patellae = B.
8. Joint effusion: increased intra-articular fluid of any amount (small, mild, moderate, large), hemarthrosis or lipohemarthrosis. Fluid described as physiological or normal = D. Fluid only in a Baker's cyst or a bursa does not count.
9. Synovitis: inflammation of the joint lining: synovitis, synovial thickening, proliferation or hypertrophy, effusion-synovitis, Hoffa synovitis, PVNS. Hoffa fat-pad edema or impingement without the words synovitis/synovial thickening = B. Plica alone = D unless thickened/inflamed.
10. Baker's cyst: popliteal (Baker's) cyst / distended gastrocnemius-semimembranosus bursa, including ruptured or leaking cysts. Ganglion cysts, parameniscal cysts and other bursae do not count.
11. Bone contusion: bone bruise / contusion / traumatic bone-marrow edema. Bone-marrow edema that is clearly non-traumatic (degenerative subchondral edema, insufficiency) = B. Red-marrow reconversion/hyperplasia, enchondroma, bone cysts or a normal marrow signal are not contusion (D).
12. Fracture: any current fracture of the knee bones, including osteochondral, impaction, avulsion (e.g. Segond), stress or subchondral insufficiency fracture. Old healed fracture = E. Microtrabecular injury described only as bone bruise = B.

Common terms: tear = rotura/desgarro (es), yırtık (tr), ruptura/puknuće (hr), ρήξη (el), Riss/Ruptur (de), руптура/разкъсване (bg), scheur (nl), rupture/fissure/déchirure (fr). Effusion = derrame (es), sıvı artışı/efüzyon (tr), izljev (hr), συλλογή υγρού (el), Erguss (de), излив (bg), hydrops/vocht (nl), épanchement (fr). Bone marrow edema/bruise = edema óseo/contusión ósea, kemik iliği ödemi/kontüzyon, edem kosti/kontuzija, οίδημα μυελού/οστική θλάση, Knochenmarködem/bone bruise, костномозъчен едем/контузия, botoedeem/bone bruise, œdème osseux/contusion osseuse."""

USER_TEMPLATE = """Knee MRI report:
<<<
{report}
>>>

Answer all 12 items with one letter each (A, B, C, D or E), in this exact layout:
{layout}"""


def build_layout():
    return "\n".join(f"{k}: <letter>" for k in ITEM_KEYS)


def load_model(model_id, quant, cache_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    kw = {"dtype": torch.bfloat16, "device_map": "cuda:0", "cache_dir": cache_dir}
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    elif quant == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    return tok, model


def single_token_id(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not a single token: {ids}")
    return ids[0]


class Scorer:
    """Scores reports. With `prefix_cache=True` the constant part of the prompt
    (system message + start of the user turn) is encoded once; each batch only
    prefills the report-specific tokens, padded between the cached prefix and
    the report ("middle padding") with explicit RoPE position ids."""

    SENTINEL = "⁣@@REPORT@@⁣"

    def __init__(self, tok, model, max_report_tokens=6000, prefix_cache=True):
        self.tok, self.model = tok, model
        self.max_report_tokens = max_report_tokens
        self.letter_ids = [single_token_id(tok, " " + L) for L in LETTERS]
        # forced segments: first item directly after the assistant header, later
        # items after a newline that follows the previous answer letter.
        self.first_ids = tok.encode(ITEM_KEYS[0] + ":", add_special_tokens=False)
        self.next_ids = [tok.encode("\n" + k + ":", add_special_tokens=False) for k in ITEM_KEYS[1:]]
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(report=self.SENTINEL, layout=build_layout())}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pre, post = text.split(self.SENTINEL)
        self.prefix_ids = tok.encode(pre, add_special_tokens=False)
        self.post_ids = tok.encode(post, add_special_tokens=False) + self.first_ids
        self.prefix_cache = prefix_cache
        self.prefix_kv = None
        if prefix_cache:
            with torch.no_grad():
                out = model(input_ids=torch.tensor([self.prefix_ids], device=model.device),
                            use_cache=True, logits_to_keep=1)
            self.prefix_kv = [(lay.keys, lay.values) for lay in out.past_key_values.layers]

    def prompt_ids(self, report):
        """Report-specific token ids (everything after the shared prefix)."""
        report = normalize_report(report)
        rep_ids = self.tok.encode(report, add_special_tokens=False)
        truncated = len(rep_ids) > self.max_report_tokens
        if truncated:  # never silent: flagged in the output
            rep_ids = rep_ids[: self.max_report_tokens]
        return rep_ids + self.post_ids, truncated

    @torch.no_grad()
    def score_batch(self, prompts):
        """prompts: list of report-specific token-id lists (see prompt_ids).
        Returns (B, 12, 5) renormalised letter probabilities and (B, 12) letter mass."""
        dev = self.model.device
        B = len(prompts)
        if not self.prefix_cache:  # full prompts, plain left padding
            prompts = [self.prefix_ids + p for p in prompts]
        L = max(len(p) for p in prompts)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        att = torch.zeros((B, L), dtype=torch.long)
        for i, p in enumerate(prompts):  # left padding (after the cached prefix, if any)
            ids[i, L - len(p):] = torch.tensor(p)
            att[i, L - len(p):] = 1
        ids, att = ids.to(dev), att.to(dev)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        past = None
        if self.prefix_cache:
            P = len(self.prefix_ids)
            past = DynamicCache(config=self.model.config)
            for li, (k, v) in enumerate(self.prefix_kv):
                past.update(k.expand(B, -1, -1, -1).contiguous(), v.expand(B, -1, -1, -1).contiguous(), li)
            att = torch.cat([torch.ones((B, P), dtype=att.dtype, device=dev), att], dim=1)
            pos = pos + P
        out = self.model(input_ids=ids, attention_mask=att, position_ids=pos,
                         past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()
        probs_all = np.zeros((B, len(ITEM_KEYS), len(LETTERS)), dtype=np.float32)
        mass_all = np.zeros((B, len(ITEM_KEYS)), dtype=np.float32)
        next_pos = pos[:, -1] + 1
        for k in range(len(ITEM_KEYS)):
            p_full = torch.softmax(logits, dim=-1)
            p_let = p_full[:, self.letter_ids]
            mass = p_let.sum(-1)
            p_norm = p_let / mass.clamp(min=1e-12).unsqueeze(-1)
            probs_all[:, k, :] = p_norm.cpu().numpy()
            mass_all[:, k] = mass.cpu().numpy()
            if k == len(ITEM_KEYS) - 1:
                break
            choice = torch.tensor(self.letter_ids, device=dev)[p_norm.argmax(-1)]
            seg = torch.tensor(self.next_ids[k], device=dev).unsqueeze(0).expand(B, -1)
            step_ids = torch.cat([choice.unsqueeze(1), seg], dim=1)
            n_new = step_ids.shape[1]
            att = torch.cat([att, torch.ones((B, n_new), dtype=att.dtype, device=dev)], dim=1)
            step_pos = next_pos.unsqueeze(1) + torch.arange(n_new, device=dev).unsqueeze(0)
            out = self.model(input_ids=step_ids, attention_mask=att, position_ids=step_pos,
                             past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            next_pos = next_pos + n_new
        del past, out
        return probs_all, mass_all


def make_batches(lengths, max_batch_tokens, max_batch_size):
    order = np.argsort(lengths)[::-1]  # longest first: early OOM if any
    batches, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and (new_max * (len(cur) + 1) > max_batch_tokens or len(cur) >= max_batch_size):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = lengths[i]
        cur.append(int(i))
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="D:/Coding/kaggle/RSNA/data/Raw_dicom/train.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--quant", default="nf4", choices=["nf4", "int8", "none"])
    ap.add_argument("--cache-dir", default="D:/rsna_work/hf_cache/hub")
    ap.add_argument("--uids", default="", help="optional file with StudyInstanceUIDs to score")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-batch-tokens", type=int, default=12000)
    ap.add_argument("--max-batch-size", type=int, default=16)
    ap.add_argument("--max-report-tokens", type=int, default=6000)
    ap.add_argument("--min-letter-mass", type=float, default=0.5)
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--reverse-items", action="store_true",
                    help="answer the 12 items in reverse order (order-sensitivity / test-time averaging)")
    args = ap.parse_args()
    global ITEM_KEYS
    order = list(range(len(TARGETS)))[::-1] if args.reverse_items else list(range(len(TARGETS)))
    ITEM_KEYS = [ITEM_KEYS[i] for i in order]  # forced layout order; outputs are mapped back below

    df = pd.read_csv(args.train)
    if args.uids:
        keep = set(pd.read_csv(args.uids).StudyInstanceUID)
        df = df[df.StudyInstanceUID.isin(keep)]
    if args.limit:
        df = df.head(args.limit)
    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out, usecols=["StudyInstanceUID", "status"])
        done = set(prev.loc[prev.status == "ok", "StudyInstanceUID"])
    todo = df[~df.StudyInstanceUID.isin(done)].reset_index(drop=True)
    print(f"studies={len(df)} already_ok={len(done)} todo={len(todo)}", flush=True)
    if todo.empty:
        return

    tok, model = load_model(args.model, args.quant, args.cache_dir)
    try:
        from huggingface_hub import scan_cache_dir  # noqa: F401
        revision = getattr(model.config, "_commit_hash", None)
    except Exception:
        revision = None
    scorer = Scorer(tok, model, args.max_report_tokens, prefix_cache=not args.no_prefix_cache)
    prompt_hash = hashlib.sha1((SYSTEM_PROMPT + USER_TEMPLATE + PROMPT_VERSION).encode()).hexdigest()[:12]
    meta = {"model": args.model, "model_revision": revision, "quant": args.quant,
            "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash,
            "prefix_cache": not args.no_prefix_cache, "item_order": "reversed" if args.reverse_items else "canonical"}
    print("run meta:", json.dumps(meta), flush=True)

    enc = [scorer.prompt_ids(str(r)) for r in todo.Report]
    lengths = np.array([len(e[0]) + len(scorer.prefix_ids) for e in enc])  # full context length
    print(f"prompt tokens: median={np.median(lengths):.0f} max={lengths.max()} "
          f"total={lengths.sum()}", flush=True)
    batches = make_batches(lengths, args.max_batch_tokens, args.max_batch_size)
    t0, n_done = time.time(), 0
    for bi, idx in enumerate(batches):
        rows = []
        try:
            probs, mass = scorer.score_batch([enc[i][0] for i in idx])
            for j, i in enumerate(idx):
                r = {"StudyInstanceUID": todo.StudyInstanceUID[i], "status": "ok", "error": "",
                     "n_prompt_tokens": int(lengths[i]), "report_truncated": bool(enc[i][1])}
                low = []
                for k, ti in enumerate(order):
                    t = TARGETS[ti]
                    for li, L in enumerate(LETTERS):
                        r[f"{t}__p{L}"] = round(float(probs[j, k, li]), 5)
                    r[f"{t}__letter_mass"] = round(float(mass[j, k]), 4)
                    r[f"{t}__argmax"] = LETTERS[int(probs[j, k].argmax())]
                    if mass[j, k] < args.min_letter_mass:
                        low.append(t)
                if low:
                    r["status"] = "low_letter_mass"
                    r["error"] = "letter mass < threshold for: " + "|".join(low)
                r.update(meta)
                rows.append(r)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for i in idx:  # retry one by one; record failures explicitly
                try:
                    probs, mass = scorer.score_batch([enc[i][0]])
                    r = {"StudyInstanceUID": todo.StudyInstanceUID[i], "status": "ok", "error": "retried_single",
                         "n_prompt_tokens": int(lengths[i]), "report_truncated": bool(enc[i][1])}
                    for k, ti in enumerate(order):
                        t = TARGETS[ti]
                        for li, L in enumerate(LETTERS):
                            r[f"{t}__p{L}"] = round(float(probs[0, k, li]), 5)
                        r[f"{t}__letter_mass"] = round(float(mass[0, k]), 4)
                        r[f"{t}__argmax"] = LETTERS[int(probs[0, k].argmax())]
                    r.update(meta)
                except Exception as ex:
                    torch.cuda.empty_cache()
                    r = {"StudyInstanceUID": todo.StudyInstanceUID[i], "status": "failed",
                         "error": f"{type(ex).__name__}: {str(ex)[:200]}", "n_prompt_tokens": int(lengths[i])}
                    r.update(meta)
                rows.append(r)
        except Exception as ex:
            for i in idx:
                r = {"StudyInstanceUID": todo.StudyInstanceUID[i], "status": "failed",
                     "error": f"{type(ex).__name__}: {str(ex)[:200]}", "n_prompt_tokens": int(lengths[i])}
                r.update(meta)
                rows.append(r)
            traceback.print_exc()
        out = pd.DataFrame(rows)
        out.to_csv(args.out, mode="a", index=False, header=not os.path.exists(args.out))
        n_done += len(idx)
        if bi % 10 == 0 or bi == len(batches) - 1:
            el = time.time() - t0
            print(f"batch {bi+1}/{len(batches)} studies {n_done}/{len(todo)} "
                  f"{n_done/el:.2f} studies/s eta {(len(todo)-n_done)/(n_done/el)/60:.1f} min "
                  f"vram_peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
