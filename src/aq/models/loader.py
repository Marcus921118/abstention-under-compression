"""Model loading: FP16 baseline and bitsandbytes-quantized variants.

Heavy deps (torch/transformers) are imported lazily so the package still
imports on the metrics-only env (Python 3.14). GPTQ/AWQ live in
``aq.models.quantize`` and arrive in Stage 3; G2 uses FP16 + bnb (NF4 / int8),
which quantize at load time and need no calibration set.

Contract (workflow §0.6): exactly one model resident in VRAM at a time; the
caller releases it via ``release_model``.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any


@contextlib.contextmanager
def _without_gptqmodel():
    """Temporarily block gptqmodel import to avoid CUDA JIT compilation."""
    saved = {k: sys.modules.pop(k)
             for k in list(sys.modules) if k.startswith("gptqmodel")}

    class _Blocker:
        def find_module(self, name, path=None):
            if name == "gptqmodel" or name.startswith("gptqmodel."):
                return self
            return None
        def load_module(self, name):
            raise ImportError(f"{name} blocked (_without_gptqmodel)")

    blocker = _Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


if TYPE_CHECKING:  # pydantic 只在型別檢查時需要
    # 模組層 import RunConfig 會把 pydantic 變成硬相依,牴觸本檔開頭
    # 「heavy deps 延遲載入,讓 package 在 metrics-only 環境仍可 import」的設計。
    # build_skip_modules 是純函式且無 torch/pydantic 相依,單元測試(鐵則 8)
    # 必須能在數學層 venv 直接跑。`from __future__ import annotations` 已生效,
    # 所以 cfg: RunConfig 這類註記在執行期不會被求值。
    from ..config import RunConfig


def _resolve_dtype(name: str):
    import torch  # noqa: PLC0415

    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(name, torch.float16)


# ── skip-module 建構(純函式,可單測)────────────────────────────────────

#: Llama/Qwen 系列的預設 nn.Linear 型別。**Phi-3 不是這組**(它把 gate 與 up
#: 融合成 `gate_up_proj`),所以需要逐型別選擇時必須用
#: `cfg.quant.extra["module_types"]` 明確傳入該架構的實際型別。
MODULE_TYPES_LLAMA_LIKE = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


def build_skip_modules(
    *,
    skip_layers: list[int] | None = None,
    only_module_types: list[str] | None = None,
    arch_module_types: list[str] | None = None,
) -> list[str] | None:
    """Build the `llm_int8_skip_modules` list for bnb.

    兩種互斥的選擇模式:

    * ``skip_layers``  —— 逐**塊**(decoder block)保留 fp16。既有行為,
      regex 帶負向前瞻避免 `model.layers.1` 前綴命中 10–19(LESSONS 2026-07-17)。
    * ``only_module_types`` —— 逐**模組型別**量化:只量化清單裡的型別,
      其餘全部保留 fp16。

    ⚠️ **方向是反的,這是最容易出錯的地方。**
    假量化(`module_granularity_profile.py`)的語義是「只量化型別 X」;
    bnb 的 ``llm_int8_skip_modules`` 語義是「**除清單外全部量化**」。
    所以要複製假量化的對比,必須把「其餘型別」全部列進 skip ——
    也就是傳補集。本函式負責這個轉換,呼叫端只需說要量化哪些型別。

    transformers 的 `should_convert_module` 對每個 key 做三種比對,其中
    ``name.endswith(key)`` 讓純型別名(如 ``"up_proj"``)可以直接命中
    ``model.layers.3.mlp.up_proj``。但這也代表**型別名互為後綴時會誤傷**:
    在 Phi-3 上 ``"gate_up_proj".endswith("up_proj")`` 為真。因此本函式在
    偵測到後綴衝突時直接 raise,不讓它靜默過關。

    Returns None 表示不傳 skip 清單(交回 transformers 自動判斷)。
    """
    if skip_layers and only_module_types:
        raise ValueError(
            "skip_layers 與 only_module_types 不可同時使用:前者逐塊、後者逐型別,"
            "混用時 bnb 的實際行為難以逐 module 驗證。要組合請先寫新的凍結判準。"
        )

    if skip_layers:
        # lm_head 必須自己補回:傳了明確 skip 清單會繞過 transformers 的
        # get_keys_to_not_convert,tied-embedding 模型(如 Qwen2.5)的 lm_head
        # 被量化會讓 bnb 4-bit forward 掛掉。
        return [rf"model\.layers\.{i}(?!\d)" for i in skip_layers] + ["lm_head"]

    if only_module_types:
        types = list(arch_module_types or MODULE_TYPES_LLAMA_LIKE)
        unknown = [t for t in only_module_types if t not in types]
        if unknown:
            raise ValueError(
                f"only_module_types 含不在架構型別清單內的項目 {unknown};"
                f"該架構型別為 {types}。用 quant.extra['module_types'] 指定正確清單。"
            )
        complement = [t for t in types if t not in only_module_types]
        # 後綴衝突:被 skip 的型別若是目標型別的後綴(或反之),endswith 比對
        # 會誤傷。Phi-3 的 gate_up_proj / up_proj 就是這種情形。
        for a in complement:
            for b in only_module_types:
                if a.endswith(b) or b.endswith(a):
                    raise ValueError(
                        f"型別名後綴衝突:skip 清單的 '{a}' 與目標 '{b}' 互為後綴,"
                        f"transformers 的 endswith 比對會誤傷。此架構不支援用型別名"
                        f"做選擇,需改用完整 module 名。"
                    )
        return complement + ["lm_head"]

    return None


def hqq_quantize_in_place(model, *, only_module_types, arch_module_types=None,
                          nbits=4, group_size=64, axis=1, compute_dtype=None,
                          device="cuda"):
    """就地把指定型別的 nn.Linear 換成 HQQLinear。

    ⚠️ **本函式是 DIRECT 語義:傳入的是「要量化的型別」,不是補集。**
    與 bnb 路徑的 ``build_skip_modules``(COMPLEMENT)方向相反,**不可混用**。
    這裡刻意不呼叫 build_skip_modules,就是為了避免那個方向陷阱。

    WHY NOT transformers' HqqConfig(2026-08-02 實測):
    transformers 5.14.1 的 `HqqConfig` 類別存在,但載入路徑未實作 ——
    `quantizers/base.py:290 get_quantize_ops()` 直接拋
    `NotImplementedError: QuantizationMethod.HQQ is not available yet`。
    故繞過整合層,直接用 hqq 套件自己的 API 就地改寫。
    (API 表面存在 ≠ 已實作;探查腳本只檢查了簽章,這是它的缺陷,
     已記入 LESSONS 2026-08-02。)

    lm_head 一律不動:tied-embedding 模型量化 lm_head 會壞掉,
    且與 bnb 臂的處理保持一致(該臂也把 lm_head 留在 fp16)。
    """
    import torch  # noqa: PLC0415
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear  # noqa: PLC0415

    types = list(arch_module_types or MODULE_TYPES_LLAMA_LIKE)
    unknown = [t for t in only_module_types if t not in types]
    if unknown:
        raise ValueError(f"only_module_types 含未知型別 {unknown};架構型別為 {types}")

    qcfg = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=axis)
    compute_dtype = compute_dtype or torch.bfloat16

    targets = []
    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "lm_head":
            continue
        if leaf in only_module_types and type(mod).__name__ == "Linear":
            targets.append(name)

    if not targets:
        raise RuntimeError(
            f"找不到任何要量化的 module(型別 {only_module_types})。"
            f"檢查架構的實際 module 命名,不要讓它靜默跑成全 fp16。")

    for name in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        lin = getattr(parent, leaf)
        setattr(parent, leaf, HQQLinear(
            lin, quant_config=qcfg, compute_dtype=compute_dtype,
            device=device, del_orig=True))

    return len(targets)


def compute_dtype_hqq(cfg: RunConfig):
    """HQQ 的計算 dtype;與 bnb 臂沿用同一個 extra 鍵,預設 bfloat16。"""
    return _resolve_dtype(cfg.quant.extra.get("compute_dtype", "bfloat16"))


def load_model(cfg: RunConfig) -> tuple[Any, Any]:
    """Load (model, tokenizer) for the FP16 or bnb-quantized variant in cfg.

    Supported in G2: quant.method in {"fp16", "bnb_nf4"}.
      - fp16            -> dtype float16 on CUDA
      - bnb_nf4 bits=4  -> 4-bit NF4 (double-quant per extra)
      - bnb_nf4 bits=8  -> 8-bit int8
    GPTQ/AWQ raise NotImplementedError until Stage 3.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    hf_id = cfg.model.hf_id
    method = cfg.quant.method

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if method == "fp16":
        # Use model-native dtype: most models use float16, but some (e.g. Gemma-3)
        # were trained in bfloat16 and produce degenerate (empty) output in float16
        # due to logit collapse. cfg.model.dtype defaults to "float16".
        base_dtype = _resolve_dtype(cfg.model.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=base_dtype, device_map="cuda"
        )
    elif method in ("bnb_nf4", "bnb_fp4"):
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        compute_dtype = _resolve_dtype(cfg.quant.extra.get("compute_dtype", "bfloat16"))
        # skip_layers: list[int] of decoder-layer indices to keep unquantized
        # (real bnb equivalent of generate_h4_records.py's fake_quant layer
        # mask -- lets calib-aware/acc-aware layer selection run on real NF4).
        # granularity note (docs/agents/KNOWLEDGE.md "粒度術語"): each index
        # here is a decoder BLOCK (model.layers[i]) -- skipping index i skips
        # every nn.Linear inside that block, i.e. this is block-wise (逐塊),
        # not module-wise/per-Linear selection.
        # transformers' should_convert_module (quantizers/quantizers_utils.py)
        # treats each skip string as a REGEX matched via re.match anchored at
        # the START of the full dotted module name -- via three OR'd checks:
        #   re.match(f"{key}\.", name) or re.match(f"{key}", name) or name.endswith(key)
        # The *second* check has no trailing boundary, so a plain
        # "model.layers.1" prefix-matches "model.layers.10".."model.layers.19"
        # too (found via autopsy_skip_layers.py 2026-07-17: acc-aware's
        # skip_layers=[1,13,18] silently left layers 10-19 unquantized as
        # well -- 11 fp16 layers instead of the intended 3, voiding that
        # run). Fix: append a negative lookahead so the layer index can't be
        # a prefix of a longer one (works for both regex checks since both
        # re.match on the same key string; irrelevant to the third,
        # non-regex `.endswith` check).
        skip_layers = cfg.quant.extra.get("skip_layers") or []
        only_types = cfg.quant.extra.get("only_module_types") or None
        arch_types = cfg.quant.extra.get("module_types") or None
        # Passing an explicit llm_int8_skip_modules list bypasses
        # transformers' automatic get_keys_to_not_convert (tied weights,
        # output embedding, last module) -- must re-add "lm_head" ourselves
        # or a tied-embedding model (e.g. Qwen2.5) gets its lm_head quantized
        # too, which crashes bnb's 4-bit forward (module.weight.shape[1]==1
        # assertion) since Params4bit is incompatible with the tied-weight
        # sharing path.
        skip_modules = build_skip_modules(
            skip_layers=skip_layers, only_module_types=only_types,
            arch_module_types=arch_types,
        )
        if cfg.quant.bits == 4:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4" if method == "bnb_fp4" else "nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=bool(cfg.quant.extra.get("double_quant", True)),
                llm_int8_skip_modules=skip_modules,
            )
        elif cfg.quant.bits == 8:
            bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip_modules)
        else:
            raise ValueError(f"bnb_nf4 supports bits in {{4,8}}, got {cfg.quant.bits}")
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb, device_map={"": 0}
        )
    elif method == "hqq":
        # 2026-08-02 量化器軸:HQQ 為第二個 round-to-nearest 實作,用來檢驗
        # #19 的 down_proj 解離是不是 bnb 這個特定實作的產物。
        #
        # 不走 transformers 的 HqqConfig —— 該版(5.14.1)載入路徑未實作,
        # 詳見 hqq_quantize_in_place 的 docstring。改為 fp16 載入後就地改寫。
        #
        # ⚠️ 這條路是 DIRECT 語義(指定要量化的型別),與 bnb 的補集語義相反。
        only_types = cfg.quant.extra.get("only_module_types")
        arch_types = cfg.quant.extra.get("module_types") or None
        if not only_types:
            raise ValueError(
                "hqq 路徑必須指定 quant.extra['only_module_types']。"
                "要量化全部型別請明確列出七個型別,不要靠預設值 —— "
                "沉默的預設值正是 #19 那次差點出錯的地方。")
        compute_dtype = compute_dtype_hqq(cfg)
        # ⚠️ 底模 dtype 必須等於 HQQ 的 compute_dtype,否則未量化模組(fp16)
        # 與 HQQLinear 反量化後的權重(bfloat16)在 matmul 對不上:
        #   RuntimeError: expected mat1 and mat2 to have the same dtype,
        #                 but got c10::Half != c10::BFloat16
        # 判準 §3 已寫死 compute dtype = bfloat16(與 bnb 臂相同),
        # 故此處跟著 compute_dtype 走,不寫死 float16。
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, dtype=compute_dtype, device_map={"": 0})
        n = hqq_quantize_in_place(
            model, only_module_types=only_types, arch_module_types=arch_types,
            nbits=cfg.quant.bits, group_size=cfg.quant.group_size or 64,
            axis=cfg.quant.extra.get("axis", 1),
            compute_dtype=compute_dtype, device="cuda")
        print(f"[hqq] 就地量化 {n} 個 module(型別 {only_types}, "
              f"nbits={cfg.quant.bits}, gs={cfg.quant.group_size or 64})")
    elif method == "gptq":
        hf_id_quant = cfg.quant.extra.get("hf_id")
        if not hf_id_quant:
            raise ValueError(
                "gptq: 必須在 quant.extra['hf_id'] 指定預量化 HF repo。")
        from gptqmodel import GPTQModel, BACKEND
        backend = getattr(BACKEND, "TORCH", BACKEND.TRITON)
        print(f"[gptq] GPTQModel.load {hf_id_quant} "
              f"(backend={backend.name}, CPU→CUDA)")
        gptq_wrapper = GPTQModel.load(
            hf_id_quant, backend=backend, device_map={"": "cpu"})
        model = gptq_wrapper.model.to("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif method == "awq":
        hf_id_quant = cfg.quant.extra.get("hf_id")
        if not hf_id_quant:
            raise ValueError(
                "awq: 必須在 quant.extra['hf_id'] 指定預量化 HF repo。")
        from awq import AutoAWQForCausalLM
        awq_wrapper = AutoAWQForCausalLM.from_quantized(
            hf_id_quant, fuse_layers=False)
        model = awq_wrapper.model
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif method == "autoround":
        hf_id_quant = cfg.quant.extra.get("hf_id")
        if not hf_id_quant:
            raise ValueError(
                "autoround: 必須在 quant.extra['hf_id'] 指定預量化模型路徑。"
                " 先跑 experiments/prequantize_autoround.py 產生。")
        model_path = hf_id_quant
        if not Path(model_path).is_absolute():
            candidate = Path(__file__).resolve().parent.parent.parent / model_path
            if candidate.exists():
                model_path = str(candidate)
        from gptqmodel import GPTQModel, BACKEND
        backend = getattr(BACKEND, "TORCH", BACKEND.TRITON)
        print(f"[autoround] GPTQModel.load {model_path} "
              f"(backend={backend.name}, CPU→CUDA)")
        gptq_wrapper = GPTQModel.load(
            model_path, backend=backend, device_map={"": "cpu"})
        model = gptq_wrapper.model.to("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif method == "rtn":
        hf_id_quant = cfg.quant.extra.get("hf_id")
        if not hf_id_quant:
            raise ValueError(
                "rtn: 必須在 quant.extra['hf_id'] 指定 RTN 預量化模型路徑。"
                " 先跑 experiments/prequantize_rtn.py 產生。")
        model_path = hf_id_quant
        if not Path(model_path).is_absolute():
            candidate = Path(__file__).resolve().parent.parent.parent / model_path
            if candidate.exists():
                model_path = str(candidate)
        from gptqmodel import GPTQModel, BACKEND
        backend = getattr(BACKEND, "TORCH", BACKEND.TRITON)
        print(f"[rtn] GPTQModel.load {model_path} "
              f"(backend={backend.name}, CPU→CUDA)")
        gptq_wrapper = GPTQModel.load(
            model_path, backend=backend, device_map={"": "cpu"})
        model = gptq_wrapper.model.to("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif method == "spqr":
        hf_id_quant = cfg.quant.extra.get("hf_id")
        if not hf_id_quant:
            raise ValueError(
                "spqr: 必須在 quant.extra['hf_id'] 指定預量化模型路徑。"
                " 先跑 experiments/prequantize_spqr.py 產生。")
        model_path = hf_id_quant
        if not Path(model_path).is_absolute():
            candidate = Path(__file__).resolve().parent.parent.parent / model_path
            if candidate.exists():
                model_path = str(candidate)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=_resolve_dtype(cfg.model.dtype),
            device_map={"": 0})
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f"unknown quant method: {method}")

    model.eval()
    return model, tokenizer


def release_model(model: Any) -> None:
    """Free a model from VRAM (del + empty_cache). Safe if torch absent."""
    try:
        import gc

        import torch  # noqa: PLC0415

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
