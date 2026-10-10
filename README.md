# Trainable Input Memories and Skill Binding

A bounded empirical study of a frozen, previously adapted SmolLM2-1.7B-Instruct reader.

**Author:** Shenghe Li (Minacle). **Status:** English preprint and evidence release; not peer-reviewed.

## What changed in the model's behavior?

A new command, `qezorvank`, asks for four input words in the order third, first, fourth, second. We trained one 2,048-dimensional input embedding (8 KiB FP32) while freezing the reader and older memory rows.

| Condition | Before: transfer | After: transfer | After: withheld combinations | Old responses changed / 21,348 | Previously correct old responses lost |
|---|---:|---:|---:|---:|---:|
| Input-only, world A | 0/192 | 155/192 (80.73%) | 190/192 | 0 | 0 |
| Input + shared residual, world A | 0/192 | 145/192 (75.52%) | 187/192 | 1,788 | 92 |
| Input-only, world B | 0/192 | 171/192 (89.06%) | 191/192 | 0 | 0 |

Transfer uses 16 words excluded from this command's 64-word teaching vocabulary, not words necessarily absent from pretraining. Worlds A and B have disjoint vocabularies. The reader had prior three-word reordering training. This is limited vocabulary transfer and skill binding, not proof of a newly invented general algorithm.

Example (world A):

- Input: `qezorvank: chisel rye silver barley`
- Target: `silver chisel barley rye`
- Before: `rye silver barley chisel`
- After: `silver chisel barley rye`

Failure (world B):

- Input: `qezorvank: anchor parsley thyme basil`
- Target: `thyme anchor basil parsley`
- After: `thymus parsley basil anchor`

Additional completed observations: 10,240 synthetic associations occupy 80 MiB of input tensors and yield 10,182 canonical / 9,963 withheld-wording successes. Adding 512 isolated associations preserves all 448 recorded skill replies. A deployment-only FP16 export halves association-tensor storage to 40 MiB while preserving 21,348 replies.

## Read the paper

- [English PDF](paper/main.pdf)
- [LaTeX source](paper/main.tex)
- [Scientific limitations and critical review](docs/CRITICAL_REVIEW.md)

## Verify the evidence without a GPU

Requires Python 3.10 or newer and no third-party packages. Do not use Python's `-O` flag: checks use assertions.

```sh
python tools/verify_results.py
```

The verifier reports byte-exact archived sources separately from sources whose original hash is recovered using the recorded newline layout in `data/source_newlines.json`. This accounts for Git LF/CRLF conversion, including mixed historical line endings. It still requires the original full-byte hash and does not ignore other code changes or rewrite historical hashes.

The verifier reads saved records, recomputes scores, checks paired responses, validates vocabulary splits and exposure balance, reconstructs conservative acquisitions, and verifies export/source hashes. It makes **zero model calls**. A successful run ends with `PASS`.

## Files

- `data/*.json.gz`: six evidence exports; gzip-compressed JSON, not model binaries.
- `data/provenance.json`: original and exported hashes, transformations, and code inventory.
- `data/reader_lineage.json`: prior adapter/skill exposure.
- `data/reply_examples.json`: first-success/first-failure examples per condition.
- `source_archive/`: 76 historical Python source snapshots and local dependencies.
- `tools/`: portable evidence verification.
- `docs/`: reproduction boundary, release instructions, and AI assistance.
- Separate [`model_assets.zip` release asset](https://github.com/Minacleeeee/trainable-input-memory/releases/download/v1.0.0/model_assets.zip): input parameters, lexical metadata, shared module, and maintenance adapter. The base weights are not bundled.

## Reproduction boundary

This release reproduces **reported counts from saved evidence**. It is not a verified one-command training/inference distribution. Historical scripts assume prepared files and a Windows/DirectML environment. Do not point them at an active experiment or treat archived source as an automatic launch recipe. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

There is no textual-skill baseline, no cross-model test, no long-project-context experiment, and no evidence that the extra supervised phase has an independent consolidation effect. The study uses one training seed and two vocabulary worlds. Shared A changes both parameterization and learning rates. No new neural experiments were run to prepare this release.

## Authorship and AI assistance

Shenghe Li is the human author. OpenAI Codex assisted with implementation, evidence checking, writing, and packaging. ByteDance Seed2.1 (Doubao) contributed to earlier project work according to the author; the exact serving variant is not established. AI tools are acknowledged, not listed as human authors. See [AI_CONTRIBUTIONS.md](docs/AI_CONTRIBUTIONS.md).

## License

Original release code, synthetic evidence, and parameter contributions are provided under Apache-2.0; the English manuscript is CC BY 4.0. Upstream dependencies retain their own terms. See LICENSE, PAPER_LICENSE.txt, and THIRD_PARTY_NOTICES.md. The base model is obtained separately from its pinned upstream repository.
