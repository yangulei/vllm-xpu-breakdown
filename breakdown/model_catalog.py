# SPDX-License-Identifier: Apache-2.0
"""Model catalog — registry of target models for XPU profiling and analysis.

Each entry contains metadata about a model: HuggingFace ID, precision targets,
model type, architecture, ownership, priority, and enablement status.

Usage:
    from breakdown.model_catalog import CATALOG, get_models_by_type, get_model

    llms = get_models_by_type("LLM")
    entry = get_model("DeepSeek-R1")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelEntry:
    """A model in the XPU support catalog."""
    name: str
    hf_id: str | None               # Primary HuggingFace model ID (if known)
    hf_ids: list[str] = field(default_factory=list)  # All known HF IDs/variants
    precision: list[str] = field(default_factory=list)  # Target precisions
    model_type: str = "LLM"         # LLM, MLLM, T2I, T2V, T2I_I2I, T2V_I2V,
                                    # Audio, Embedding, Reranker, Segmentation, MTP
    architecture: str | None = None  # HuggingFace architecture class name
    family: str | None = None        # Architecture family (Llama, Qwen3, etc.)
    owner: str = ""                  # Team/person responsible
    focus: str = ""                  # Technical focus area
    priority: str = ""               # H (high), M (medium), L (low)
    in_cri_plan: bool = False        # Whether in R&D CRI plan
    status: str = "planned"          # planned, in_progress, supported, blocked
    notes: str = ""                  # Additional notes
    vllm_supported: bool = True      # Whether vLLM can load this model type

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hf_id": self.hf_id,
            "hf_ids": self.hf_ids,
            "precision": self.precision,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "family": self.family,
            "owner": self.owner,
            "focus": self.focus,
            "priority": self.priority,
            "in_cri_plan": self.in_cri_plan,
            "status": self.status,
            "notes": self.notes,
            "vllm_supported": self.vllm_supported,
        }


# ===================================================================
# LLM Models
# ===================================================================

_LLM_MODELS: list[ModelEntry] = [
    ModelEntry(
        name="GLM-5.1",
        hf_id="THUDM/GLM-5.1",
        hf_ids=["THUDM/GLM-5.1"],
        precision=["FP8"],
        model_type="LLM",
        architecture="Glm4ForCausalLM",
        family="GLM4",
        priority="H",
        in_cri_plan=True,
    ),
    ModelEntry(
        name="Step-3.5-Flash",
        hf_id=None,
        precision=["FP8"],
        model_type="LLM",
        architecture="StepForCausalLM",
        family="Step",
        priority="H",
        in_cri_plan=True,
    ),
    ModelEntry(
        name="DeepSeek-V4",
        hf_id="deepseek-ai/DeepSeek-V4",
        hf_ids=["deepseek-ai/DeepSeek-V4"],
        precision=["FP8"],
        model_type="LLM",
        architecture="DeepseekV3ForCausalLM",
        family="DeepSeekV3",
        priority="H",
        in_cri_plan=True,
        notes="Expected to use DeepSeekV3 architecture with MLA",
    ),
    ModelEntry(
        name="DeepSeek-R1",
        hf_id="deepseek-ai/DeepSeek-R1",
        hf_ids=["deepseek-ai/DeepSeek-R1",
                "deepseek-ai/DeepSeek-R1-0528"],
        precision=["FP8"],
        model_type="LLM",
        architecture="DeepseekV3ForCausalLM",
        family="DeepSeekV3",
        priority="H",
        in_cri_plan=True,
        notes="Reasoning model using DeepSeekV3 architecture with MLA",
    ),
    ModelEntry(
        name="Qwen3.6/3.5",
        hf_id="Qwen/Qwen3.5",
        hf_ids=["Qwen/Qwen3.5"],
        precision=["BF16", "FP8"],
        model_type="LLM",
        architecture="Qwen3ForCausalLM",
        family="Qwen3",
        owner="Ling, Jing1 / Huanxing",
        focus="FLA",
        priority="H",
        in_cri_plan=True,
    ),
    ModelEntry(
        name="Kimi-K2.5/K2.6",
        hf_id="moonshotai/Kimi-K2-Instruct",
        hf_ids=["moonshotai/Kimi-K2-Instruct"],
        precision=["INT4"],
        model_type="LLM",
        architecture="Kimi2ForCausalLM",
        family="Kimi",
        owner="Jin, Youzhi",
        focus="INT4 Weight Only",
        priority="H",
        in_cri_plan=True,
    ),
    ModelEntry(
        name="MiniMax-M2.5/M2.7",
        hf_id="MiniMaxAI/MiniMax-M1-80B",
        hf_ids=["MiniMaxAI/MiniMax-M1-80B",
                "MiniMaxAI/MiniMax-M1-40k"],
        precision=["FP8"],
        model_type="LLM",
        architecture="MiniMaxM1ForCausalLM",
        family="MiniMax",
        owner="Luo, Focus",
        priority="H",
        in_cri_plan=True,
        notes="Lightning attention architecture",
    ),
    ModelEntry(
        name="Mimo-V2/V2.5",
        hf_id="XiaomiMiMo/MiMo-7B-RL",
        hf_ids=["XiaomiMiMo/MiMo-7B-RL"],
        precision=["FP8"],
        model_type="LLM",
        architecture="MiMoForCausalLM",
        family="Mimo",
        owner="Yuan, Tian",
        focus="Sink Attn",
        priority="H",
        in_cri_plan=True,
    ),
    ModelEntry(
        name="DeepSeek-V3.2 / GLM5",
        hf_id="deepseek-ai/DeepSeek-V3-0324",
        hf_ids=["deepseek-ai/DeepSeek-V3-0324",
                "THUDM/glm-4-9b-chat"],
        precision=["FP8"],
        model_type="LLM",
        architecture="DeepseekV3ForCausalLM",
        family="DeepSeekV3",
        owner="Chen, Wenbin",
        focus="MLA / DSA",
        priority="H",
        in_cri_plan=True,
    ),
    ModelEntry(
        name="Qwen3-30B-A3B",
        hf_id="Qwen/Qwen3-30B-A3B",
        hf_ids=["Qwen/Qwen3-30B-A3B"],
        precision=["BF16", "FP8"],
        model_type="LLM",
        architecture="Qwen3MoeForCausalLM",
        family="Qwen3Moe",
        owner="Huanxing",
        focus="MOE - Group GEMM",
        priority="H",
        in_cri_plan=True,
    ),
    ModelEntry(
        name="Qwen3-235B",
        hf_id="Qwen/Qwen3-235B-A22B",
        hf_ids=["Qwen/Qwen3-235B-A22B"],
        precision=["BF16", "FP8"],
        model_type="LLM",
        architecture="Qwen3MoeForCausalLM",
        family="Qwen3Moe",
        owner="Huanxing",
        focus="MOE - Group GEMM",
        priority="H",
        in_cri_plan=True,
        notes="235B may need multi-node; CRI plan status TBD",
    ),
    ModelEntry(
        name="Qwen3-32B",
        hf_id="Qwen/Qwen3-32B",
        hf_ids=["Qwen/Qwen3-32B"],
        precision=["BF16", "FP8"],
        model_type="LLM",
        architecture="Qwen3ForCausalLM",
        family="Qwen3",
        owner="Ma, Junpo",
        priority="H",
        in_cri_plan=True,
    ),
    ModelEntry(
        name="Hunyuan3",
        hf_id="tencent/Hunyuan-A13B-Instruct",
        hf_ids=["tencent/Hunyuan-A13B-Instruct"],
        precision=["BF16", "FP8"],
        model_type="LLM",
        architecture="HunYuanMoEV1ForCausalLM",
        family="Hunyuan",
        priority="H",
        status="planned",
        notes="Not in CRI plan",
    ),
    ModelEntry(
        name="Qwen2.5-VL",
        hf_id="Qwen/Qwen2.5-VL-72B-Instruct",
        hf_ids=["Qwen/Qwen2.5-VL-72B-Instruct",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "Qwen/Qwen2.5-VL-3B-Instruct"],
        precision=["BF16", "FP8"],
        model_type="MLLM",
        architecture="Qwen2_5_VLForConditionalGeneration",
        family="Qwen2VL",
        priority="H",
    ),
    ModelEntry(
        name="GLM-4.7",
        hf_id="THUDM/glm-4-9b-chat",
        hf_ids=["THUDM/glm-4-9b-chat",
                "THUDM/glm-4-9b"],
        precision=["BF16", "FP8"],
        model_type="LLM",
        architecture="ChatGLMForConditionalGeneration",
        family="GLM4",
        owner="Yao, KeFei",
    ),
    ModelEntry(
        name="LongCat-Flash-Thinking-2601",
        hf_id=None,
        precision=["BF16", "FP8"],
        model_type="LLM",
        architecture="LlamaForCausalLM",
        family="Llama",
        notes="Likely Llama-based architecture",
    ),
    ModelEntry(
        name="Youtu-LLM-2B",
        hf_id=None,
        precision=["BF16"],
        model_type="LLM",
    ),
    ModelEntry(
        name="Qwen3-30B-A3B-GPTQ-Int4",
        hf_id="Qwen/Qwen3-30B-A3B-GPTQ-Int4",
        hf_ids=["Qwen/Qwen3-30B-A3B-GPTQ-Int4"],
        precision=["GPTQ-INT4"],
        model_type="LLM",
        architecture="Qwen3MoeForCausalLM",
        family="Qwen3Moe",
        owner="Cheng, Yanfei",
        focus="GPTQ-INT4",
    ),
    ModelEntry(
        name="Hunyuan Dense & MOE",
        hf_id="tencent/Hunyuan-A13B-Instruct",
        hf_ids=["tencent/Hunyuan-A13B-Instruct"],
        precision=["BF16", "FP8"],
        model_type="LLM",
        architecture="HunYuanMoEV1ForCausalLM",
        family="Hunyuan",
    ),
    ModelEntry(
        name="Seed-OSS",
        hf_id="ByteDance-Seed/Seed-Coder-8B-Reasoning",
        hf_ids=["ByteDance-Seed/Seed-Coder-8B-Reasoning"],
        precision=["BF16"],
        model_type="LLM",
        architecture="LlamaForCausalLM",
        family="Llama",
        notes="Seed models typically use Llama-based architecture",
    ),
    ModelEntry(
        name="Keye-VL",
        hf_id=None,
        precision=["BF16"],
        model_type="MLLM",
        notes="Keye vision-language model",
    ),
    ModelEntry(
        name="InternVL V2 & V3.5",
        hf_id="OpenGVLab/InternVL2-8B",
        hf_ids=["OpenGVLab/InternVL2-8B",
                "OpenGVLab/InternVL2-26B",
                "OpenGVLab/InternVL3-8B"],
        precision=["BF16", "FP8"],
        model_type="MLLM",
        architecture="InternVLChatModel",
        family="InternVL",
    ),
    ModelEntry(
        name="gpt-oss-20b",
        hf_id=None,
        precision=["FP4"],
        model_type="LLM",
        owner="Huanxing",
    ),
]

# ===================================================================
# MLLM (Multimodal LLM) Models
# ===================================================================

_MLLM_MODELS: list[ModelEntry] = [
    ModelEntry(
        name="STEP3-VL-10B",
        hf_id=None,
        precision=["BF16", "FP8"],
        model_type="MLLM",
        owner="Yao, KeFei",
    ),
    ModelEntry(
        name="Qwen3-VL-30B-A3B",
        hf_id="Qwen/Qwen3-VL-30B-A3B",
        hf_ids=["Qwen/Qwen3-VL-30B-A3B"],
        precision=["BF16", "FP8"],
        model_type="MLLM",
        architecture="Qwen3VLForConditionalGeneration",
        family="Qwen3VL",
        owner="Voas, Tanner",
    ),
    ModelEntry(
        name="Qwen2.5-VL-7B",
        hf_id="Qwen/Qwen2.5-VL-7B-Instruct",
        hf_ids=["Qwen/Qwen2.5-VL-7B-Instruct"],
        precision=["BF16"],
        model_type="MLLM",
        architecture="Qwen2_5_VLForConditionalGeneration",
        family="Qwen2VL",
        owner="Voas, Tanner",
    ),
    ModelEntry(
        name="Qwen3-Omni-30B-A3B-Instruct",
        hf_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        hf_ids=["Qwen/Qwen3-Omni-30B-A3B-Instruct"],
        precision=["BF16"],
        model_type="MLLM",
        architecture="Qwen3OmniForConditionalGeneration",
        family="Qwen3Omni",
        owner="Miao, Avery",
    ),
    ModelEntry(
        name="GLM-OCR",
        hf_id=None,
        precision=["BF16"],
        model_type="MLLM",
        notes="GLM-based OCR model",
    ),
    ModelEntry(
        name="DeepSeek-OCR-2",
        hf_id=None,
        precision=["BF16"],
        model_type="MLLM",
        owner="He, Junyan",
    ),
    ModelEntry(
        name="PaddleOCR-VL-1.5",
        hf_id=None,
        precision=["BF16"],
        model_type="MLLM",
        vllm_supported=False,
        notes="PaddlePaddle ecosystem; may not be vLLM-compatible",
    ),
    ModelEntry(
        name="dots.ocr",
        hf_id=None,
        precision=["BF16"],
        model_type="MLLM",
        owner="Yuan, Tian",
    ),
    ModelEntry(
        name="Qwen3-ASR",
        hf_id=None,
        precision=["BF16"],
        model_type="MLLM",
        notes="Audio speech recognition with Qwen3 backbone",
    ),
    ModelEntry(
        name="Ernie4.5-VL",
        hf_id=None,
        precision=["BF16"],
        model_type="MLLM",
        vllm_supported=False,
        notes="Baidu Ernie; PaddlePaddle ecosystem",
    ),
]

# ===================================================================
# Text-to-Image / Image-to-Image Models
# ===================================================================

_T2I_MODELS: list[ModelEntry] = [
    ModelEntry(
        name="Z-Image",
        hf_id=None,
        precision=["BF16"],
        model_type="T2I",
        owner="Li, Bo O",
        priority="H",
        in_cri_plan=True,
        vllm_supported=False,
        notes="Diffusion-based; uses diffusers pipeline",
    ),
    ModelEntry(
        name="Qwen-Image Family",
        hf_id=None,
        precision=["BF16"],
        model_type="T2I",
        owner="Han, Yingjie",
        priority="H",
        in_cri_plan=True,
        vllm_supported=False,
    ),
    ModelEntry(
        name="FLUX V1 & V2",
        hf_id="black-forest-labs/FLUX.1-dev",
        hf_ids=["black-forest-labs/FLUX.1-dev",
                "black-forest-labs/FLUX.1-schnell"],
        precision=["BF16"],
        model_type="T2I",
        architecture="FluxTransformer2DModel",
        family="Flux",
        owner="Yuan, Tian",
        priority="H",
        in_cri_plan=True,
        vllm_supported=False,
        notes="DiT-based diffusion; uses diffusers pipeline",
    ),
    ModelEntry(
        name="HunyuanImage3.0",
        hf_id="tencent/HunyuanDiT-v1.2-Diffusers",
        hf_ids=["tencent/HunyuanDiT-v1.2-Diffusers"],
        precision=["BF16"],
        model_type="T2I",
        architecture="HunyuanDiT2DModel",
        family="HunyuanDiT",
        priority="H",
        in_cri_plan=True,
        vllm_supported=False,
    ),
    ModelEntry(
        name="HiDream-O1-Image",
        hf_id="HiDream-ai/HiDream-I1-Dev",
        hf_ids=["HiDream-ai/HiDream-I1-Dev"],
        precision=["BF16"],
        model_type="T2I",
        priority="H",
        vllm_supported=False,
    ),
    ModelEntry(
        name="GLM-Image",
        hf_id=None,
        precision=["BF16"],
        model_type="T2I",
        vllm_supported=False,
    ),
]

# ===================================================================
# Text-to-Video / Image-to-Video Models
# ===================================================================

_T2V_MODELS: list[ModelEntry] = [
    ModelEntry(
        name="Stable Video Diffusion",
        hf_id="stabilityai/stable-video-diffusion-img2vid-xt",
        hf_ids=["stabilityai/stable-video-diffusion-img2vid-xt"],
        precision=["BF16"],
        model_type="T2V",
        architecture="UNetSpatioTemporalConditionModel",
        family="SVD",
        owner="Miao, Jincheng",
        vllm_supported=False,
    ),
    ModelEntry(
        name="CogVideoX",
        hf_id="THUDM/CogVideoX-5b",
        hf_ids=["THUDM/CogVideoX-5b",
                "THUDM/CogVideoX-2b"],
        precision=["BF16"],
        model_type="T2V",
        architecture="CogVideoXTransformer3DModel",
        family="CogVideoX",
        vllm_supported=False,
    ),
    ModelEntry(
        name="Wan2.2",
        hf_id="Wan-AI/Wan2.2-T2V-14B",
        hf_ids=["Wan-AI/Wan2.2-T2V-14B",
                "Wan-AI/Wan2.2-I2V-14B"],
        precision=["BF16"],
        model_type="T2V",
        architecture="WanTransformer3DModel",
        family="Wan",
        owner="Ba, Mengkejiergeli",
        focus="CP & Ulysses, FA2/3",
        priority="H",
        in_cri_plan=True,
        vllm_supported=False,
    ),
    ModelEntry(
        name="LTX-2.3",
        hf_id="Lightricks/LTX-Video",
        hf_ids=["Lightricks/LTX-Video"],
        precision=["BF16"],
        model_type="T2V",
        architecture="LTXVideoTransformer3DModel",
        family="LTX",
        priority="H",
        in_cri_plan=True,
        vllm_supported=False,
    ),
    ModelEntry(
        name="HunyuanVideo V1.5",
        hf_id="tencent/HunyuanVideo",
        hf_ids=["tencent/HunyuanVideo"],
        precision=["BF16"],
        model_type="T2V",
        architecture="HunyuanVideoTransformer3DModel",
        family="HunyuanVideo",
        owner="Han, Yingjie",
        vllm_supported=False,
    ),
    ModelEntry(
        name="Hy3 Preview",
        hf_id="tencent/Hy3-preview",
        hf_ids=["tencent/Hy3-preview"],
        precision=["BF16"],
        model_type="T2V",
        architecture="HunyuanVideoTransformer3DModel",
        family="HunyuanVideo",
        vllm_supported=False,
        notes="Hunyuan Video 3.0 preview",
    ),
    ModelEntry(
        name="StepVideo",
        hf_id=None,
        precision=["BF16"],
        model_type="T2V",
        vllm_supported=False,
    ),
]

# ===================================================================
# Audio / TTS Models
# ===================================================================

_AUDIO_MODELS: list[ModelEntry] = [
    ModelEntry(
        name="Diffrhythm",
        hf_id="ASLP-lab/DiffRhythm-base",
        hf_ids=["ASLP-lab/DiffRhythm-base"],
        precision=["BF16", "FP16"],
        model_type="Audio",
        owner="Zhang, Yanqiu",
        vllm_supported=False,
    ),
    ModelEntry(
        name="CosyVoice3",
        hf_id="FunAudioLLM/CosyVoice2-0.5B",
        hf_ids=["FunAudioLLM/CosyVoice2-0.5B"],
        precision=["BF16", "FP16"],
        model_type="Audio",
        owner="Meng, Chen",
        vllm_supported=False,
    ),
    ModelEntry(
        name="SongGeneration",
        hf_id=None,
        precision=["BF16", "FP16"],
        model_type="Audio",
        owner="Xiang, haihao",
        vllm_supported=False,
    ),
    ModelEntry(
        name="InspireMusic",
        hf_id="FunAudioLLM/InspireMusic-Base",
        hf_ids=["FunAudioLLM/InspireMusic-Base"],
        precision=["BF16", "FP16"],
        model_type="Audio",
        owner="Meng, Chen",
        vllm_supported=False,
    ),
    ModelEntry(
        name="Qwen3-TTS",
        hf_id=None,
        precision=["BF16", "FP16"],
        model_type="Audio",
        owner="Yuan, Tian",
        vllm_supported=False,
    ),
    ModelEntry(
        name="MuseTalk",
        hf_id="TMElyralab/MuseTalk",
        hf_ids=["TMElyralab/MuseTalk"],
        precision=["BF16", "FP16"],
        model_type="Audio",
        vllm_supported=False,
    ),
    ModelEntry(
        name="VoxCPM",
        hf_id=None,
        precision=["BF16", "FP16"],
        model_type="Audio",
        vllm_supported=False,
    ),
    ModelEntry(
        name="SenseVoice",
        hf_id="FunAudioLLM/SenseVoiceSmall",
        hf_ids=["FunAudioLLM/SenseVoiceSmall"],
        precision=["BF16", "FP16"],
        model_type="Audio",
        vllm_supported=False,
    ),
    ModelEntry(
        name="ACE-Step1.5",
        hf_id="ACE-Step/ACE-Step-v1-3.5B",
        hf_ids=["ACE-Step/ACE-Step-v1-3.5B"],
        precision=["BF16", "FP16"],
        model_type="Audio",
        owner="Wang, Hanying",
        vllm_supported=False,
    ),
    ModelEntry(
        name="Step-Video (Audio/Video)",
        hf_id=None,
        precision=["BF16"],
        model_type="T2V",
        vllm_supported=False,
    ),
    ModelEntry(
        name="Open-Sora",
        hf_id="hpcai-tech/Open-Sora",
        hf_ids=["hpcai-tech/Open-Sora"],
        precision=["BF16"],
        model_type="T2V",
        vllm_supported=False,
    ),
    ModelEntry(
        name="InfiniteTalk",
        hf_id=None,
        precision=["BF16"],
        model_type="T2V",
        owner="Jin, Youzhi",
        priority="Low",
        vllm_supported=False,
    ),
]

# ===================================================================
# Image Segmentation Models
# ===================================================================

_SEGMENTATION_MODELS: list[ModelEntry] = [
    ModelEntry(
        name="SAM V2 & V3",
        hf_id="facebook/sam2-hiera-large",
        hf_ids=["facebook/sam2-hiera-large",
                "facebook/sam2-hiera-small"],
        precision=["BF16", "FP16"],
        model_type="Segmentation",
        owner="Liu, Heyuan",
        priority="Low",
        vllm_supported=False,
    ),
]

# ===================================================================
# Embedding / Reranker Models
# ===================================================================

_EMBEDDING_MODELS: list[ModelEntry] = [
    ModelEntry(
        name="Embedding - BGE",
        hf_id="BAAI/bge-large-en-v1.5",
        hf_ids=["BAAI/bge-large-en-v1.5",
                "BAAI/bge-m3",
                "BAAI/bge-base-en-v1.5"],
        precision=["BF16", "FP16"],
        model_type="Embedding",
        architecture="XLMRobertaModel",
        family="RoBERTa",
        owner="Wang, Huiqi",
    ),
    ModelEntry(
        name="Embedding - Qwen3",
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        hf_ids=["Qwen/Qwen3-Embedding-0.6B"],
        precision=["BF16", "FP16"],
        model_type="Embedding",
        architecture="Qwen3ForCausalLM",
        family="Qwen3",
    ),
    ModelEntry(
        name="Reranker - RoBERTa",
        hf_id="BAAI/bge-reranker-v2-m3",
        hf_ids=["BAAI/bge-reranker-v2-m3",
                "BAAI/bge-reranker-large"],
        precision=["BF16", "FP16"],
        model_type="Reranker",
        architecture="XLMRobertaForSequenceClassification",
        family="RoBERTa",
    ),
    ModelEntry(
        name="Reranker - Qwen3",
        hf_id="Qwen/Qwen3-Reranker-0.6B",
        hf_ids=["Qwen/Qwen3-Reranker-0.6B"],
        precision=["BF16", "FP16"],
        model_type="Reranker",
        architecture="Qwen3ForSequenceClassification",
        family="Qwen3",
    ),
]

# ===================================================================
# MTP (Multi-Token Prediction) Models
# ===================================================================

_MTP_MODELS: list[ModelEntry] = [
    ModelEntry(
        name="GLM4.x MTP",
        hf_id=None,
        precision=["BF16", "FP8"],
        model_type="MTP",
        architecture="Glm4ForCausalLM",
        family="GLM4",
        owner="Yu, Jiankang",
        notes="Multi-Token Prediction variant",
    ),
    ModelEntry(
        name="DeepSeek MTP",
        hf_id=None,
        precision=["BF16", "FP8"],
        model_type="MTP",
        architecture="DeepseekV3ForCausalLM",
        family="DeepSeekV3",
        owner="Yu, Jiankang",
        notes="Multi-Token Prediction variant",
    ),
    ModelEntry(
        name="Qwen3.5 MTP",
        hf_id=None,
        precision=["BF16", "FP8"],
        model_type="MTP",
        architecture="Qwen3ForCausalLM",
        family="Qwen3",
        owner="Yu, Jiankang",
        notes="Multi-Token Prediction variant",
    ),
]


# ===================================================================
# Full catalog
# ===================================================================

CATALOG: list[ModelEntry] = (
    _LLM_MODELS
    + _MLLM_MODELS
    + _T2I_MODELS
    + _T2V_MODELS
    + _AUDIO_MODELS
    + _SEGMENTATION_MODELS
    + _EMBEDDING_MODELS
    + _MTP_MODELS
)

# Model type categories for grouping
MODEL_TYPES: dict[str, str] = {
    "LLM": "Large Language Models",
    "MLLM": "Multimodal LLMs (Vision-Language)",
    "T2I": "Text-to-Image",
    "T2V": "Text-to-Video / Image-to-Video",
    "Audio": "Audio / TTS",
    "Segmentation": "Image Segmentation",
    "Embedding": "Embedding Models",
    "Reranker": "Reranker Models",
    "MTP": "Multi-Token Prediction",
}


# ===================================================================
# Lookup helpers
# ===================================================================

def get_models_by_type(model_type: str) -> list[ModelEntry]:
    """Return all models of a given type."""
    return [m for m in CATALOG if m.model_type == model_type]


def get_models_by_priority(priority: str) -> list[ModelEntry]:
    """Return all models with a given priority level."""
    return [m for m in CATALOG if m.priority.upper() == priority.upper()]


def get_high_priority_models() -> list[ModelEntry]:
    """Return all high-priority models."""
    return get_models_by_priority("H")


def get_vllm_models() -> list[ModelEntry]:
    """Return all models that vLLM can load (LLM + MLLM + Embedding)."""
    return [m for m in CATALOG if m.vllm_supported]


def get_model(name: str) -> ModelEntry | None:
    """Find a model by name (case-insensitive partial match)."""
    name_lower = name.lower()
    for m in CATALOG:
        if m.name.lower() == name_lower:
            return m
    for m in CATALOG:
        if name_lower in m.name.lower():
            return m
    return None


def get_model_by_hf_id(hf_id: str) -> ModelEntry | None:
    """Find a model by HuggingFace ID."""
    for m in CATALOG:
        if m.hf_id == hf_id:
            return m
        if hf_id in m.hf_ids:
            return m
    return None


def catalog_summary() -> dict[str, Any]:
    """Return summary stats for the full catalog."""
    by_type: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    vllm_count = 0
    for m in CATALOG:
        by_type[m.model_type] = by_type.get(m.model_type, 0) + 1
        if m.priority:
            by_priority[m.priority] = by_priority.get(m.priority, 0) + 1
        if m.vllm_supported:
            vllm_count += 1
    return {
        "total": len(CATALOG),
        "by_type": by_type,
        "by_priority": by_priority,
        "vllm_supported": vllm_count,
    }
