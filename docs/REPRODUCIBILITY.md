# Reproducibility and provenance

## Validated for this release

The standard-library verifier recomputes primary scores from stored question/target/response records; checks paired IDs/questions/targets; validates 64 training words versus 16 transfer words, disjoint vocabulary worlds, 960 distinct queries and balanced training exposure; reconstructs the conservative 9,089 and 447 association gains; checks old-response and deployment equality; and verifies evidence and archived-code hashes. Verification does not invoke the historical auditor or import model code.

All parameter/adapter files were checked against their historical recorded SHA-256 values before export. Their current release hashes are in `data/model_asset_manifest.json` and in the separate model archive. Inference configuration paths are not assumed unchanged: all exported `inference_config.json` files explicitly relocate `frozen_adapter_path` to `../maintenance`, interpreted relative to that checkpoint directory. Those configurations have new hashes. Historical code may resolve paths relative to the current directory; a caller must resolve paths before use.

## Not validated for this release

No fresh model inference, neural gradient verification, full rerun, clean-machine dependency installation, or turnkey training entry point was tested. Do not equate PASS from the evidence verifier with fresh neural reproduction. The historical independent audits are included as records, not freshly executed certificates. Saved strings were stripped and lowercased by the historical decoder before the scorer collapsed whitespace and stripped leading/trailing period/exclamation/space characters. Raw equality means equality at that stored-string boundary.

## Model lineage and runtime

Base: HuggingFaceTB/SmolLM2-1.7B-Instruct, revision `31b70e2e869a7173562077fd711b654946d38674`. Original vocabulary 49,152; hidden width 2,048. Obtain the base separately. The frozen maintenance adapter is included in model assets at `maintenance/single/`. Its weights hash is `3cc7e8055da82cba4332f557850aa9892fe075da9a69b6b7f0476ce8e5010897`.

Historical execution used Windows/DirectML, Transformers and PEFT, with an FP16 reader and FP32 trainable input rows. A PyTorch CPU rerun is not asserted numerically equivalent. The deployment record specifically distinguishes DirectML conversion from an earlier failed CPU-conversion attempt. Use source manifests and recorded configs for the actual original conditions; no guessed dependency lockfile is provided.

## Source archive

The 76 Python files include files pinned by result manifests and their locally resolvable import dependencies. `data/provenance.json` records whether each file was historically pinned. Source files are preserved unchanged. They are research snapshots, and some historical descriptions or path assumptions refer to previous stages. They are not automatically safe or complete to launch outside the original prepared environment.

## Export transformations

Original result/audit hashes are retained. JSON evidence is reserialized as ASCII JSON and gzip-compressed; machine-local path prefixes are replaced with neutral placeholders. Saved answer strings are preserved. New compressed hashes are separate from original hashes. Lexical inventories, input parameter files, shared-module parameters and maintenance weights preserve original bytes. No base-model weights are redistributed.

## To reproduce paper compilation

From `paper/`, run `pdflatex main.tex` twice using a normal TeX Live installation. References and tables are inline; there are no external figures, BibTeX files, custom classes or private fonts. `arxiv_source.zip` contains only the required source. PDF inspection and a clean source-only compilation were performed locally. The hosted arXiv compiler still must be checked during submission.
