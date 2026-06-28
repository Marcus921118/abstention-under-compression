#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校準感知混精度量化 — 加速版 (RTX 5060 8GB / WSL)
==============================================================================
和 mixed_precision_calib.py 同邏輯、同輸出, 但修掉三個拖慢的點:
  1. 每個 seed 只 tokenize 一次, 把 batch 預先編碼並搬上 GPU, 之後 29 次評估重用
     (原版每次評估都重切 1500 題, 145 次 = 大量 CPU 白工)。
  2. 逐層 profiling 的權重備份「留在 GPU、只換參考」, 不再 CPU 來回搬
     (單層備份很小; 1.5B 在 8GB 綽綽有餘)。
  3. 拿掉每層的 empty_cache()/gc.collect() (原本每層都做, 很慢且不需要)。
  4. 加每層進度顯示, 不會再「好幾分鐘沒輸出像當機」。
  5. profiling 用較少題 (--profile-n, 預設 600) 排層, frontier 才用滿 (--n)。

跑 (你的 GPU):
  ./myenv/bin/python3 experiments/mixed_precision_calib_fast.py \
      --model Qwen/Qwen2.5-1.5B-Instruct --n 1500 --profile-n 600 --seeds 5
先冒煙測 (1 分鐘確認能跑完):
  ./myenv/bin/python3 experiments/mixed_precision_calib_fast.py --smoke
8GB 跑 3B 若 OOM: 加 --cpu-backup (備份改放 CPU, 慢一點但省 VRAM)。
"""
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path
import numpy as np

# ---- torch-free 指標 (與原版相同) ----
def auroc(score, label):
    score = np.asarray(score, float); label = np.asarray(label, int)
    n_pos = int(label.sum()); n_neg = int((label == 0).sum())
    if n_pos == 0 or n_neg == 0: return float("nan")
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = (csum - cnt + csum + 1) / 2.0
    ranks = avg[inv]
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def calib_metrics(conf, correct, tau=0.8, n_bins=15):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float); n = len(conf)
    overconf = float(((conf > tau) & (correct == 0)).mean())
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() > 0: ece += abs(correct[m].mean() - conf[m].mean()) * m.sum() / n
    return {"acc": float(correct.mean()), "overconf": overconf, "ece": float(ece),
            "brier": float(np.mean((conf - correct) ** 2)), "auroc": auroc(conf, correct)}

def bootstrap_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

def paired_perm_test(a, b, n_perm=10000, seed=0):
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    if len(d) == 0 or np.allclose(d, 0): return (float(d.mean()) if len(d) else float("nan"), 1.0)
    rng = np.random.default_rng(seed); obs = d.mean(); cnt = 0
    for _ in range(n_perm):
        s = rng.integers(0, 2, len(d)) * 2 - 1
        if abs((s * d).mean()) >= abs(obs) - 1e-12: cnt += 1
    return (float(obs), (cnt + 1) / (n_perm + 1))

def kendall_tau(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); n = len(a); c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            c += s > 0; d += s < 0
    return (c - d) / (0.5 * n * (n - 1)) if n > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=1500, help="frontier 評估題數")
    ap.add_argument("--profile-n", type=int, default=600, help="逐層排層用題數 (較少即可)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--bits", type=int, default=4); ap.add_argument("--gs", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.8); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--frac-grid", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--cpu-backup", action="store_true", help="備份放 CPU (省 VRAM, 較慢)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke: args.n = 150; args.profile_n = 100; args.seeds = 2

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, **k): return x

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    print(f"[init] device={dev} dtype={dtype} model={args.model}")
    tok = AutoTokenizer.from_pretrained(args.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(dev).eval()
    layers = model.model.layers; L = len(layers)
    print(f"[init] #layers={L}  batch={args.batch}")
    ds = load_dataset("cais/mmlu", "all", split="test")
    opt_ids = [tok(" " + x, add_special_tokens=False).input_ids[-1] for x in ["A", "B", "C", "D"]]
    optt = torch.tensor(opt_ids, device=dev)

    def fmt(e):
        c = e["choices"]
        return f"Question: {e['question']}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer:"

    def encode(seed, n):
        """關鍵加速: 只 tokenize 一次, batch 預先搬上 GPU, 之後重用。"""
        rng = np.random.RandomState(seed); idx = rng.permutation(len(ds))[:n]
        exs = [ds[int(i)] for i in idx]; batches = []
        for s in range(0, len(exs), args.batch):
            b = exs[s:s + args.batch]
            enc = tok([fmt(e) for e in b], return_tensors="pt", padding=True).to(dev)
            ans = torch.tensor([e["answer"] for e in b], device=dev)
            batches.append((enc, ans))
        return batches

    @torch.no_grad()
    def evaluate(batches):
        conf, corr = [], []
        for enc, ans in batches:
            lg = model(**enc).logits[:, -1, :].index_select(1, optt)
            p = torch.softmax(lg.float(), dim=-1); mx, pred = p.max(dim=-1)
            conf.append(mx.cpu().numpy()); corr.append((pred == ans).cpu().numpy().astype(int))
        if dev == "cuda": torch.cuda.synchronize()   # 讓 CUDA 錯誤即時浮現在正確位置, 不要非同步累積
        return np.concatenate(conf), np.concatenate(corr)

    def fq(W):
        dt = W.dtype; W = W.float()          # 在 fp32 做量化, 避免 fp16 邊角數值問題
        qmax = 2 ** (args.bits - 1) - 1
        if W.dim() != 2:
            s = W.abs().amax() / qmax + 1e-12; return (torch.clamp(torch.round(W / s), -qmax - 1, qmax) * s).to(dt)
        o, i = W.shape; g = args.gs if i % args.gs == 0 else i
        Wg = W.reshape(o, i // g, g); s = Wg.abs().amax(2, keepdim=True) / qmax + 1e-12
        return (torch.clamp(torch.round(Wg / s), -qmax - 1, qmax) * s).reshape(o, i).contiguous().to(dt)

    def quant(idxs):
        """備份只換參考、留在 GPU (除非 --cpu-backup); 不 clone、不 empty_cache。"""
        bk = {}
        for k in idxs:
            for n, m in layers[k].named_modules():
                if isinstance(m, torch.nn.Linear):
                    bk[(k, n)] = m.weight.data.to("cpu") if args.cpu_backup else m.weight.data
                    m.weight.data = fq(m.weight.data)
        return bk

    def restore(bk):
        for (k, n), w in bk.items():
            for nn_, m in layers[k].named_modules():
                if nn_ == n and isinstance(m, torch.nn.Linear):
                    m.weight.data = w.to(dev) if args.cpu_backup else w

    outdir = Path("results") / ("mpcalib_" + args.model.split("/")[-1].replace(".", "").lower())
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- 命題一: 逐層敏感度 ----
    print("\n=== 命題一: 逐層敏感度 (跨 seed) ===")
    dacc = np.zeros((args.seeds, L)); dover = np.zeros((args.seeds, L)); base_rows = []
    for si in range(args.seeds):
        batches = encode(1000 + si, args.profile_n)              # ← 一次編碼
        base = calib_metrics(*evaluate(batches), tau=args.tau); base_rows.append(base)
        print(f" seed {si}: baseline acc={base['acc']:.3f} overconf={base['overconf']:.3f} ece={base['ece']:.3f}")
        for k in tqdm(range(L), desc=f"  seed{si} 逐層", leave=False):
            bk = quant([k]); m = calib_metrics(*evaluate(batches), tau=args.tau); restore(bk)
            dacc[si, k] = base["acc"] - m["acc"]; dover[si, k] = m["overconf"] - base["overconf"]
        print(f"   seed {si} 完成 ({time.time()-t0:.0f}s)")
        del batches; gc.collect()
        if dev == "cuda": torch.cuda.empty_cache()   # 每個 seed 清一次, 避免顯存碎片累積
    dacc_m, dover_m = dacc.mean(0), dover.mean(0)
    rs = [np.corrcoef(dacc[s], dover[s])[0, 1] for s in range(args.seeds)]
    r_pool = float(np.corrcoef(dacc_m, dover_m)[0, 1])
    taus = [kendall_tau(dover[a], dover[b]) for a in range(args.seeds) for b in range(a + 1, args.seeds)]
    print(f"[命題一] r pooled={r_pool:.3f} | per-seed mean={np.mean(rs):.3f} CI={bootstrap_ci(rs)}")
    print(f"[穩定度] calib 排名跨 seed Kendall tau={np.nanmean(taus):.3f}")

    # ---- 命題二: 準確度匹配前緣 ----
    print("\n=== 命題二: 準確度匹配前緣 ===")
    acc_rank = np.argsort(dacc_m)[::-1]; cal_rank = np.argsort(dover_m)[::-1]
    eb = encode(7777, args.n); base_eval = calib_metrics(*evaluate(eb), tau=args.tau)
    fr = []
    for frac in [float(x) for x in args.frac_grid.split(",")]:
        keep = int(round(frac * L))
        for strat, rank in [("acc", acc_rank), ("calib", cal_rank)]:
            fp = set(rank[:keep].tolist()); qs = [k for k in range(L) if k not in fp]
            bk = quant(qs); m = calib_metrics(*evaluate(eb), tau=args.tau); restore(bk)
            fr.append({"strategy": strat, "frac_fp": frac, **m})
            print(f" frac_fp={frac:.2f} {strat:5s}: acc={m['acc']:.3f} overconf={m['overconf']:.3f} ece={m['ece']:.3f}")
    diff, p = paired_perm_test([r["overconf"] for r in fr if r["strategy"] == "acc"],
                               [r["overconf"] for r in fr if r["strategy"] == "calib"])
    print(f"[命題二] mean(overconf_acc - overconf_calib)={diff:+.4f} p={p:.3f}")

    import csv
    with open(outdir / "profile.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "dAcc", "dOverconf"])
        for k in range(L): w.writerow([k, f"{dacc_m[k]:.5f}", f"{dover_m[k]:.5f}"])
    with open(outdir / "frontier.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fr[0].keys())); w.writeheader(); w.writerows(fr)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(5.2, 4.4)); plt.scatter(dacc_m, dover_m, c="#2980b9")
        for k in range(L): plt.annotate(str(k), (dacc_m[k], dover_m[k]), fontsize=6, alpha=.6)
        plt.axhline(0, color="grey", lw=.6); plt.axvline(0, color="grey", lw=.6)
        plt.xlabel("ΔAcc"); plt.ylabel("ΔOverconf"); plt.title(f"orthogonality r={r_pool:.3f}")
        plt.tight_layout(); plt.savefig(outdir / "fig1_orthogonality.png", dpi=140); plt.close()
        plt.figure(figsize=(5.6, 4.4))
        for strat, col, mk in [("acc", "#e67e22", "o-"), ("calib", "#2980b9", "s-")]:
            xs = [r["acc"] for r in fr if r["strategy"] == strat]; ys = [r["overconf"] for r in fr if r["strategy"] == strat]
            o = np.argsort(xs); plt.plot(np.array(xs)[o], np.array(ys)[o], mk, color=col,
                                         label="Acc-Aware" if strat == "acc" else "Calib-Aware")
        plt.scatter([base_eval["acc"]], [base_eval["overconf"]], c="green", marker="*", s=160, label="FP16", zorder=5)
        plt.xlabel("accuracy"); plt.ylabel(f"overconf (τ={args.tau})"); plt.title("accuracy-matched frontier")
        plt.legend(); plt.tight_layout(); plt.savefig(outdir / "fig2_frontier.png", dpi=140); plt.close()
        print(f"[圖] 存於 {outdir}")
    except Exception as e:
        print("[圖] 略過:", e)

    json.dump({"model": args.model, "n": args.n, "profile_n": args.profile_n, "seeds": args.seeds,
               "bits": args.bits, "r_pooled": r_pool, "kendall_tau": float(np.nanmean(taus)),
               "prop2_diff": diff, "prop2_p": p, "frontier": fr, "seconds": time.time() - t0},
              open(outdir / "summary.json", "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\n完成, 總耗時 {time.time()-t0:.0f}s。摘要: {outdir}/summary.json")


if __name__ == "__main__":
    main()
