#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校準感知混精度量化 — 進階前導實驗 (GPU / RTX 5060 8GB / WSL)
==============================================================================
升級自 0.5B CPU 版 (N=30)。原版的三個盲點:
  (1) N=30 -> 過度自信率粒度只有 1/30, 命題二其實是 3/30 vs 4/30 的雜訊。
  (2) 單一 seed, 沒有信賴區間, 沒有顯著性。
  (3) Acc-Aware 與 Calib-Aware 在「不同準確度」下相比 -> Calib 過度自信較低,
      可能只是因為它是比較弱、整體比較不自信的模型 (混淆), 而非校準真的較好。

本版修正:
  1. N 拉大 (預設 1500) + 多個資料分片 seed (預設 5) -> 過度自信率帶 95% CI。
  2. 命題一(正交性): 逐層 ΔAcc vs ΔOverconf 的相關係數 r, 跨 seed 給 mean±CI。
  3. 命題二改成「準確度匹配前緣 (accuracy-matched frontier)」:
     掃描保留全精度的層數 k, 對 Acc-Aware / Calib-Aware 各畫出
     (accuracy, over_confident_wrong_rate) 的 Pareto 前緣。
     若 Calib-Aware 在『相同準確度』下過度自信更低 -> 排除上面 (3) 的混淆,
     才算真正支持命題二。
  4. 逐層敏感度排名的「跨 seed 穩定度」(Kendall's tau) -> 方法能否泛化的關鍵。
  5. ECE / Brier / overconf / AUROC 多指標並陳, bootstrap CI + 跨 seed 配對檢定。
  6. 記憶體安全 (fp16, 備份權重移到 CPU, 每層後 empty_cache, 小 batch) 適配 8GB。

用法 (WSL, 你的 RTX 5060):
  python experiments/mixed_precision_calib.py \
      --model Qwen/Qwen2.5-1.5B-Instruct --n 1500 --seeds 5 --bits 4 --gs 128
快速冒煙測試:
  python experiments/mixed_precision_calib.py --model Qwen/Qwen2.5-0.5B-Instruct --n 200 --seeds 2 --smoke
輸出: results/mpcalib_<model>/{profile.csv, frontier.csv, summary.json, *.png}

注意: 本腳本用「群組式假量化 (fake-quant)」做逐層配置研究 (因為混精度需要逐層控制,
真實 bnb/gguf 是整模型一次量化)。確定 Calib-Aware 配置後, 真實棄答任務的頭條數字
(abstention_f1 等) 請用你 aq 框架的 squad2 / abstention_bench 管線驗證 (見檔末整合說明)。
"""
from __future__ import annotations
import argparse, gc, json, os
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------------
# A. 與 torch 無關的指標 / 統計 (可單獨單元測試)
# ----------------------------------------------------------------------------
def auroc(score, label):
    """label=1 為正類 (此處 correct=1)。純 numpy rank-based AUROC。"""
    score = np.asarray(score, float); label = np.asarray(label, int)
    n_pos = int(label.sum()); n_neg = int((label == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float); ranks[order] = np.arange(1, len(score) + 1)
    # tie correction
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    auc = (ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def calib_metrics(conf, correct, tau=0.8, n_bins=15):
    """conf in [0,1], correct in {0,1}。回傳 acc / overconf / ece / brier / auroc。"""
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    n = len(conf); acc = float(correct.mean())
    overconf = float(((conf > tau) & (correct == 0)).mean())
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() > 0:
            ece += abs(correct[m].mean() - conf[m].mean()) * m.sum() / n
    brier = float(np.mean((conf - correct) ** 2))
    return {"acc": acc, "overconf": overconf, "ece": float(ece),
            "brier": brier, "auroc": auroc(conf, correct)}


def bootstrap_ci(x, n_boot=2000, seed=0, stat=np.mean):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def paired_perm_test(a, b, n_perm=10000, seed=0):
    """跨 seed 配對置換檢定: H0 = 兩策略 overconf 無差異。回傳 (mean_diff, p)。"""
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    if len(d) == 0 or np.allclose(d, 0):
        return (float(d.mean()) if len(d) else float("nan"), 1.0)
    rng = np.random.default_rng(seed); obs = d.mean(); cnt = 0
    for _ in range(n_perm):
        s = rng.integers(0, 2, len(d)) * 2 - 1
        if abs((s * d).mean()) >= abs(obs) - 1e-12:
            cnt += 1
    return (float(obs), (cnt + 1) / (n_perm + 1))


def kendall_tau(a, b):
    """兩個排名的一致度 (層敏感度跨 seed 是否穩定)。純 numpy。"""
    a = np.asarray(a, float); b = np.asarray(b, float); n = len(a); c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0: c += 1
            elif s < 0: d += 1
    return (c - d) / (0.5 * n * (n - 1)) if n > 1 else float("nan")


# ----------------------------------------------------------------------------
# B. torch / 模型相關 (在你的 GPU 上跑; 沙箱無 GPU 不執行此段)
# ----------------------------------------------------------------------------
def build_pipeline(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    print(f"[init] device={dev} dtype={dtype} model={args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(dev).eval()
    layers = model.model.layers
    print(f"[init] #transformer layers = {len(layers)}")

    ds = load_dataset("cais/mmlu", "all", split="test")
    opt_letters = ["A", "B", "C", "D"]
    opt_ids = [tok(" " + L, add_special_tokens=False).input_ids[-1] for L in opt_letters]

    def fmt(e):
        c = e["choices"]
        return (f"Question: {e['question']}\nA. {c[0]}\nB. {c[1]}\n"
                f"C. {c[2]}\nD. {c[3]}\nAnswer:")

    def sample_idx(seed, n):
        rng = np.random.RandomState(seed)
        return rng.permutation(len(ds))[:n]

    @torch.no_grad()
    def evaluate(idx):
        """回傳 (conf[], correct[])。conf = 選項 softmax 後的最大機率。"""
        conf_all, corr_all = [], []
        exs = [ds[int(i)] for i in idx]
        for s in range(0, len(exs), args.batch):
            batch = exs[s:s + args.batch]
            enc = tok([fmt(e) for e in batch], return_tensors="pt", padding=True).to(dev)
            logits = model(**enc).logits[:, -1, :][:, opt_ids]
            probs = torch.softmax(logits.float(), dim=-1)
            mx, pred = probs.max(dim=-1)
            for j, e in enumerate(batch):
                conf_all.append(float(mx[j])); corr_all.append(int(pred[j] == e["answer"]))
        return np.array(conf_all), np.array(corr_all, int)

    return torch, model, layers, evaluate, sample_idx


def fake_quant(W, bits, gs):
    """群組式對稱假量化 (per-output-channel, group size gs)。"""
    import torch
    qmax = 2 ** (bits - 1) - 1
    if W.dim() != 2:
        s = W.abs().amax() / qmax + 1e-12
        return torch.clamp(torch.round(W / s), -qmax - 1, qmax) * s
    o, i = W.shape
    g = gs if i % gs == 0 else i
    Wg = W.reshape(o, i // g, g)
    s = Wg.abs().amax(2, keepdim=True) / qmax + 1e-12
    return (torch.clamp(torch.round(Wg / s), -qmax - 1, qmax) * s).reshape(o, i)


def quant_layers(layers, idxs, bits, gs):
    """把 idxs 這些層的 Linear 權重換成假量化; 備份移到 CPU 省 VRAM。回傳備份。"""
    import torch
    backup = {}
    for k in idxs:
        for name, m in layers[k].named_modules():
            if isinstance(m, torch.nn.Linear):
                backup[(k, name)] = m.weight.detach().to("cpu", copy=True)
                m.weight.data = fake_quant(m.weight.data, bits, gs)
    return backup


def restore_layers(layers, backup, dev):
    import torch
    for (k, name), w in backup.items():
        for n, m in layers[k].named_modules():
            if n == name and isinstance(m, torch.nn.Linear):
                m.weight.data = w.to(dev)
    backup.clear()
    if dev == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


# ----------------------------------------------------------------------------
# C. 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=1500, help="每個 seed 的評估題數")
    ap.add_argument("--seeds", type=int, default=5, help="資料分片數 (給 CI)")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--gs", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--frac-grid", default="0,0.25,0.5,0.75,1.0",
                    help="frontier 掃描: 保留全精度的層比例")
    ap.add_argument("--smoke", action="store_true", help="冒煙測試: 縮小規模")
    args = ap.parse_args()
    if args.smoke:
        args.n = min(args.n, 200); args.seeds = min(args.seeds, 2)

    outdir = Path("results") / ("mpcalib_" + args.model.split("/")[-1].replace(".", "").lower())
    outdir.mkdir(parents=True, exist_ok=True)

    torch, model, layers, evaluate, sample_idx = build_pipeline(args)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L = len(layers)

    # ---- 命題一: 逐層敏感度 (每個 seed 一份資料分片) ----
    print("\n=== 命題一: 逐層敏感度 (跨 seed) ===")
    dacc = np.zeros((args.seeds, L)); dover = np.zeros((args.seeds, L))
    base_rows = []
    for si in range(args.seeds):
        idx = sample_idx(1000 + si, args.n)
        base = calib_metrics(*evaluate(idx), tau=args.tau)
        base_rows.append(base)
        print(f" seed {si}: baseline acc={base['acc']:.3f} overconf={base['overconf']:.3f} ece={base['ece']:.3f}")
        for k in range(L):
            bk = quant_layers(layers, [k], args.bits, args.gs)
            m = calib_metrics(*evaluate(idx), tau=args.tau)
            restore_layers(layers, bk, dev)
            dacc[si, k] = base["acc"] - m["acc"]          # 正值=量化後準確度掉多少
            dover[si, k] = m["overconf"] - base["overconf"]  # 正值=量化後過度自信升多少
    dacc_mean = dacc.mean(0); dover_mean = dover.mean(0)

    # 正交性 r: 每個 seed 一個 r, 取 mean±CI
    rs = [np.corrcoef(dacc[si], dover[si])[0, 1] for si in range(args.seeds)]
    r_pooled = float(np.corrcoef(dacc_mean, dover_mean)[0, 1])
    r_ci = bootstrap_ci(rs) if args.seeds >= 3 else (float("nan"), float("nan"))
    # 排名跨 seed 穩定度 (calib 敏感度)
    taus = []
    for a in range(args.seeds):
        for b in range(a + 1, args.seeds):
            taus.append(kendall_tau(dover[a], dover[b]))
    print(f"[命題一] r(ΔAcc,ΔOverconf) pooled={r_pooled:.3f} | per-seed mean={np.mean(rs):.3f} "
          f"95%CI=({r_ci[0]:.3f},{r_ci[1]:.3f})")
    print(f"[穩定度] calib 敏感度排名 跨 seed Kendall tau 平均={np.nanmean(taus):.3f} "
          f"(接近1=穩定可泛化; 接近0=排名是雜訊)")

    # 存逐層 profile
    import csv
    with open(outdir / "profile.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "dAcc_mean", "dOverconf_mean"])
        for k in range(L):
            w.writerow([k, f"{dacc_mean[k]:.5f}", f"{dover_mean[k]:.5f}"])

    # ---- 命題二: 準確度匹配前緣 (Acc-Aware vs Calib-Aware) ----
    print("\n=== 命題二: 準確度匹配前緣 ===")
    acc_rank = np.argsort(dacc_mean)[::-1]   # 對準確度最敏感的層 (Acc-Aware 優先保 FP)
    cal_rank = np.argsort(dover_mean)[::-1]  # 對校準最敏感的層   (Calib-Aware 優先保 FP)
    fracs = [float(x) for x in args.frac_grid.split(",")]
    eval_idx = sample_idx(7777, args.n)      # 固定的前緣評估集
    base_eval = calib_metrics(*evaluate(eval_idx), tau=args.tau)
    frontier = []
    for frac in fracs:
        keep = int(round(frac * L))           # 保留 FP 的層數
        for strat, rank in [("acc", acc_rank), ("calib", cal_rank)]:
            keep_fp = set(rank[:keep].tolist())
            quant_set = [k for k in range(L) if k not in keep_fp]
            bk = quant_layers(layers, quant_set, args.bits, args.gs)
            m = calib_metrics(*evaluate(eval_idx), tau=args.tau)
            restore_layers(layers, bk, dev)
            frontier.append({"strategy": strat, "frac_fp": frac, "n_quant": len(quant_set),
                             **m})
            print(f" frac_fp={frac:.2f} {strat:5s}: acc={m['acc']:.3f} "
                  f"overconf={m['overconf']:.3f} ece={m['ece']:.3f}")
    with open(outdir / "frontier.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(frontier[0].keys())); w.writeheader()
        w.writerows(frontier)

    # 配對檢定: 在每個 frac 上, calib vs acc 的 overconf 差 (跨 frac 當配對樣本)
    accv = [r["overconf"] for r in frontier if r["strategy"] == "acc"]
    calv = [r["overconf"] for r in frontier if r["strategy"] == "calib"]
    diff, p = paired_perm_test(accv, calv)
    print(f"[命題二] 跨 frac: mean(overconf_acc - overconf_calib)={diff:+.4f} p={p:.3f} "
          f"(>0 且顯著 = Calib-Aware 過度自信較低)")

    # ---- 圖 ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 圖1: 逐層 ΔAcc vs ΔOverconf 散點 (正交性)
        plt.figure(figsize=(5.2, 4.6))
        plt.scatter(dacc_mean, dover_mean, c="#2980b9")
        for k in range(L):
            plt.annotate(str(k), (dacc_mean[k], dover_mean[k]), fontsize=6, alpha=.6)
        plt.axhline(0, color="grey", lw=.6); plt.axvline(0, color="grey", lw=.6)
        plt.xlabel("Accuracy drop when layer quantized (ΔAcc)")
        plt.ylabel("Overconfidence rise (ΔOverconf)")
        plt.title(f"Layer sensitivity orthogonality  r={r_pooled:.3f}")
        plt.tight_layout(); plt.savefig(outdir / "fig1_orthogonality.png", dpi=140); plt.close()
        # 圖2: 準確度匹配前緣
        plt.figure(figsize=(5.6, 4.6))
        for strat, col, mk in [("acc", "#e67e22", "o"), ("calib", "#2980b9", "s")]:
            xs = [r["acc"] for r in frontier if r["strategy"] == strat]
            ys = [r["overconf"] for r in frontier if r["strategy"] == strat]
            order = np.argsort(xs)
            plt.plot(np.array(xs)[order], np.array(ys)[order], mk + "-", color=col,
                     label=("Acc-Aware" if strat == "acc" else "Calib-Aware"))
        plt.scatter([base_eval["acc"]], [base_eval["overconf"]], c="green", marker="*",
                    s=160, label="FP16 baseline", zorder=5)
        plt.xlabel("Accuracy on answered"); plt.ylabel(f"Over-confident wrong rate (τ={args.tau})")
        plt.title("Accuracy-matched frontier (lower-right is better)")
        plt.legend(); plt.tight_layout(); plt.savefig(outdir / "fig2_frontier.png", dpi=140); plt.close()
        print(f"[圖] 已存 {outdir}/fig1_orthogonality.png, fig2_frontier.png")
    except Exception as e:
        print("[圖] 略過繪圖:", e)

    summary = {
        "model": args.model, "n": args.n, "seeds": args.seeds, "bits": args.bits,
        "gs": args.gs, "tau": args.tau, "n_layers": L,
        "baseline_mean": {k: float(np.mean([b[k] for b in base_rows])) for k in base_rows[0]},
        "prop1_r_pooled": r_pooled, "prop1_r_per_seed_mean": float(np.mean(rs)),
        "prop1_r_ci": r_ci, "rank_stability_kendall_tau": float(np.nanmean(taus)),
        "prop2_overconf_diff_acc_minus_calib": diff, "prop2_p": p,
        "frontier": frontier,
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\n完成。摘要: {outdir}/summary.json")
    print("解讀: 命題一看 r 是否≈0 且 CI 不跨大範圍; 穩定度看 tau 是否>0.3;")
    print("      命題二看 fig2 — Calib-Aware 曲線是否在相同 accuracy 下 overconf 更低。")


if __name__ == "__main__":
    main()
