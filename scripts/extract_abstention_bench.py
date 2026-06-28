"""One-off extraction of AbstentionBench -> data/processed/abstention_bench.jsonl.

AbstentionBench is script-based and ONLY supports datasets<=3.6.0. Run this in a
SEPARATE throwaway environment so it cannot disturb the inference env:

    python -m venv .venv_ab
    .venv_ab\\Scripts\\activate            # Windows
    pip install "datasets==3.6.0" gdown pandas torch pydantic jsonlines requests wget numpy
    python scripts/extract_abstention_bench.py \
        --out data/processed/abstention_bench.jsonl --max-samples 1000

It writes one JSON object per line in the canonical schema consumed by
aq.data.abstention_bench.load:

    {"id", "question", "reference_answers", "should_abstain", "category",
     "metadata_json"}

Notes
-----
* License: CC-BY-NC-4.0 (research/non-commercial). Upstream sources carry their
  own licenses (see the dataset README).
* A balanced subset is taken (answerable vs should-abstain) for a manageable run
  on a single machine; --max-samples bounds the total.
* The category is derived from metadata when available, else from should_abstain.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _cached_module_dirs():
    from pathlib import Path  # noqa: PLC0415

    base = (Path.home() / ".cache/huggingface/modules/datasets_modules/datasets"
            / "facebook--AbstentionBench")
    return list(base.glob("*/"))


def _patch_windows_zip_bug() -> None:
    """Fix a Windows-incompatible zip-member path in AbstentionBench's data.py.

    The upstream base loader does ``zip_ref.extract(os.path.join('raw_data',
    name))``; on Windows os.path.join yields backslashes but zip members always
    use '/', raising KeyError. Rewrite to an explicit posix path. This single
    fix unblocks every gdrive-zip sub-source (musique, kuq, etc.). Idempotent.
    """
    bad = "zip_ref.extract(os.path.join('raw_data', self.file_name), self.data_dir)"
    good = "zip_ref.extract('raw_data/' + self.file_name.replace('\\\\', '/'), self.data_dir)"
    patched = 0
    for d in _cached_module_dirs():
        f = d / "data.py"
        if not f.exists():
            continue
        txt = f.read_text(encoding="utf-8")
        if bad in txt:
            f.write_text(txt.replace(bad, good), encoding="utf-8")
            patched += 1
    print(f"[patch] fixed Windows zip-path bug in {patched} cached data.py file(s)")


def _patch_skip_gpqa() -> None:
    """Comment out the gated GPQA entry in the cached AbstentionBench.py."""
    needle = '"gpqa_abstain": GPQA(),'
    patched = 0
    for d in _cached_module_dirs():
        f = d / "AbstentionBench.py"
        if not f.exists():
            continue
        txt = f.read_text(encoding="utf-8")
        if needle in txt:
            f.write_text(txt.replace(needle, f"# [skip-gated] {needle}"), encoding="utf-8")
            patched += 1
    print(f"[skip-gated] patched {patched} cached AbstentionBench.py file(s) to drop GPQA")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/abstention_bench.jsonl")
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-gated", action="store_true",
                    help="drop the gated GPQA sub-source (yields AbstentionBench-minus-GPQA, "
                         "19/20 sources) so no HF gate acceptance is needed")
    args = ap.parse_args()

    # Some sub-sources download via `wget`/urllib, which on Windows venvs can't
    # find a CA store -> CERTIFICATE_VERIFY_FAILED. Point urllib at certifi's CA
    # bundle (still fully verifies TLS; does NOT disable verification).
    import os as _os  # noqa: PLC0415

    try:
        import certifi  # noqa: PLC0415

        _os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        _os.environ.setdefault("SSL_CERT_DIR", _os.path.dirname(certifi.where()))
    except Exception:
        pass

    import datasets  # noqa: PLC0415

    _patch_windows_zip_bug()  # always: upstream loader is Windows-broken
    if args.skip_gated:
        _patch_skip_gpqa()

    if not datasets.__version__.startswith("3."):
        print(f"WARNING: datasets {datasets.__version__} detected; AbstentionBench "
              f"needs <=3.6.0 (script-based). Loading will likely fail.")

    ds = datasets.load_dataset("facebook/AbstentionBench", trust_remote_code=True)
    # AbstentionBench's as_dataset(split=None) returns a single concatenated
    # Dataset, not a DatasetDict.
    if isinstance(ds, datasets.Dataset):
        split = ds
    elif args.split in ds:
        split = ds[args.split]
    else:
        split = ds[list(ds.keys())[0]]

    rows = []
    for i, ex in enumerate(split):
        meta_raw = ex.get("metadata_json", "")
        category = "answerable"
        if ex.get("should_abstain"):
            category = "unanswerable"
            try:
                md = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
                # AbstentionBench encodes scenario in metadata; surface it if present
                scenario = md.get("abstention_scenario") or md.get("scenario")
                category = str(scenario or "unanswerable")
            except Exception:
                pass
        rows.append({
            "id": f"ab-{i}",
            "question": ex["question"],
            "reference_answers": ex.get("reference_answers"),
            "should_abstain": bool(ex.get("should_abstain")),
            "category": category,
            "metadata_json": meta_raw if isinstance(meta_raw, str) else json.dumps(meta_raw),
        })

    # Balanced subset: split by should_abstain, interleave up to max-samples.
    rng = random.Random(args.seed)
    abst = [r for r in rows if r["should_abstain"]]
    ans = [r for r in rows if not r["should_abstain"]]
    rng.shuffle(abst)
    rng.shuffle(ans)
    half = args.max_samples // 2
    selected = abst[:half] + ans[: args.max_samples - len(abst[:half])]
    rng.shuffle(selected)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} samples "
          f"({sum(r['should_abstain'] for r in selected)} should-abstain) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
