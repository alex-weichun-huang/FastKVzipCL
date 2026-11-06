# Evaluation selection only

Base: teammate commit `91b73e54de5a5f92442847d5d8df2e8287893cea`, used by our
experiments. The separate full-pipeline commit `1575920d4587d8bf8b574f206908a7a2e998c9eb`
is not included here. No training, feature generation, metrics, gate code or
model-weight loading; only public dataset loading and a pinned tokenizer.

Integrated main: `5367b15053cfa70fc4d6018f3f294c24348ae11a`. Its retrieval-hint
and `STEPS.md` additions do not change the dataset loaders or evaluator selection
code. The selection configuration, ID/hash lock and executable selection code
remain unchanged from the completed-run handoff.

From the checkout root, in an appropriate CPU allocation with the existing
`datasets` and `transformers` environment:

```bash
python evaluation_selection/select_samples.py --output-dir "$SELECTION_OUTPUT_DIR"
```

`config.json` pins dataset/config/split/revisions and tokenizer revision.
`ids.json` contains only source rows/IDs, original context SHA256, token counts,
ordered question IDs and the locked SCBench split. No text, answers or results.
The command reloads these exact samples; missing/changed IDs, hashes, token
counts or question order fail instead of choosing replacements or truncating.
Run it again to verify/reload complete outputs without downloading again.

|Final evaluation|Contexts|Questions|
|---|---:|---:|
|SQuAD validation|100|1,037|
|MRCR train (held out from fitting)|100|100|
|SCBench English Summarization test subset|14|70|

Original selection rules: SQuAD first100 eligible context groups in source order,
all questions/all references; MRCR first100 eligible distinct flattened contexts,
actual Qwen context length<=128000. The locked MRCR subset is two-needle,
16252–32824 tokens, rows100–199; that start row resulted from filtering, not a
`--start 100` default. SCBench has70 source contexts: locked49 train/7 validation/
14 test, original seed20261028. No split is regenerated; training and validation
hashes are excluded from all final selections. Smoke/debug selections are omitted.
There is no additional SQuAD/SCBench context cutoff or implicit truncation in this
replay. SCBench saved context-token lengths are checked on the14 test contexts.

**These are our completed-run selections, not guaranteed teammate defaults.**
At the integrated main above, `prefill/data/load.py::load_squad` uses TRAIN with an off-by-one
context stopping condition, and `load_scbench` uses Jang-Hyun/SCBench-preprocessed.
We use official SQuAD validation and microsoft/SCBench/scbench_summary with the
locked held-out subset. MRCR shares the formatter and128000-token filter, but
equivalence also depends on dataset/tokenizer revisions, ordering and evaluator
arguments. Matching the formatter/filter does not establish identical selected
MRCR samples under unpinned defaults. This delivery does not replace their evaluator or repeat training-source
overlap audits; it replays the previously chosen samples.

Output: three dataset JSONL selection manifests (including public context,
questions and references), an ID-only SCBench split file, and an atomic completion
marker with checksums. Store these generated files outside git. There are no
predictions, metrics or feature tensors. The question IDs/order and context hashes
are verified against the completed run; full source re-download/tokenization is
not part of the lightweight handoff tests.

Selection tests: `python -m unittest discover -s evaluation_selection -p 'test_*.py'`.
Read-only comparison with a completed run:
`python evaluation_selection/select_samples.py --verify-existing "$EXISTING_RUN"`.
