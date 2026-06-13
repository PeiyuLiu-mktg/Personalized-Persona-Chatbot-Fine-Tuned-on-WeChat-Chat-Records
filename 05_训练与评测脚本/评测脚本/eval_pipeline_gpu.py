import asyncio
import json
import math
import os
import re
import warnings
import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sacrebleu
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")


@dataclass
class EvalConfig:
    base_model_path: str = "/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct"
    lora_weights_path: str = "/root/autodl-tmp/LlamaFactory/saves/Qwen2.5-7B-Instruct/lora/train_2026-04-23-10-27-45"
    test_data_path: str = "/root/autodl-tmp/pingce/xiaolaoshi_test_data.json"
    train_data_path: str = "/root/autodl-tmp/xiaolaoshi_train_data.json"
    output_dir: str = "/root/autodl-tmp/pingce"
    device: str = "cuda"
    max_new_tokens: int = 256
    temperature: float = 0.7
    batch_size: int = 4
    num_rounds: int = 4
    api_base_url: str = os.getenv("EVAL_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_model: str = os.getenv("EVAL_API_MODEL", "qwen-plus")
    api_key: str = os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    judge_retry_times: int = 3
    judge_retry_delay: float = 2.0
    judge_win_opponent: str = os.getenv("EVAL_JUDGE_WIN_OPPONENT", "base")
    bert_model_path: str = "/root/autodl-tmp/models/bert_zh"
    emotion_model_path: str = "/root/autodl-tmp/models/roberta_news_zh"
    embedding_model_path: str = "/root/autodl-tmp/models/minilm_multi"
    persona_top_k: int = 40
    min_keyword_count: int = 3
    sample_limit: Optional[int] = None
    skip_level4: bool = False
    skip_level5: bool = False
    only_level4: bool = False
    smoke_test: bool = False
    use_base_only: bool = False


@dataclass
class SampleResult:
    sample_id: str
    instruction: str
    input_text: str
    history: List[Tuple[str, str]]
    ground_truth: str
    generated_response: str
    base_response: str = ""
    l1_bleu4: float = math.nan
    l1_rouge1: float = math.nan
    l1_rougel: float = math.nan
    l1_ppl: float = math.nan
    l2_embedding_cosine: float = math.nan
    l3_khr: float = math.nan
    l3_length_similarity: float = math.nan
    l3_punctuation_similarity: float = math.nan
    l3_modal_similarity: float = math.nan
    l4_tone_score: float = math.nan
    l4_semantic_score: float = math.nan
    l4_fidelity_score: float = math.nan
    l4_win_rate: float = math.nan
    l5_drift: float = math.nan
    l5_adversarial_drift: float = math.nan
    l5_style_consistency: float = math.nan


class PersonaKeywords:
    VOCAL_WORDS = [
        "呀", "嘛", "呢", "啊", "哈", "哦", "呃", "唉", "诶", "嗯", "啦", "哟", "哇",
        "捏", "的说", "呗", "咯", "嘞", "喔", "么", "咋", "啥",
    ]
    HABITUAL_WORDS = [
        "emmm", "emm", "嗯嗯", "对对对", "是的是的", "没毛病", "666", "绝了", "服了",
        "醉了", "无语", "好吧", "行吧", "算了吧", "好哒", "嗯呢",
    ]
    PUNCTUATION = ["！", "!", "？", "?", "～", "~", "。", ".", "，", ",", "...", "…"]
    STOPWORDS = {
        "的", "了", "是", "我", "你", "他", "她", "它", "我们", "你们", "他们", "然后",
        "就是", "这个", "那个", "一下", "一下子", "啊啊", "哈哈哈", "一个", "没有", "不是",
        "可以", "因为", "所以", "但是", "就是啊", "就是呢", "还是", "真的", "感觉", "已经",
        "如果", "的话", "吧", "呢", "啊", "呀", "哦", "吗", "啦", "哈", "诶", "嗯",
    }


class EvaluationPipeline:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.lora_model = None
        self.rouge_scorer = None
        self.embedding_model = None
        self.device = torch.device(config.device)
        self.persona_keywords = PersonaKeywords()
        self.persona_lexicon: List[str] = []
        self.metric_notes: Dict[str, str] = {}

    @property
    def generation_model(self):
        return self.model if self.config.use_base_only else self.lora_model

    def _normalize_history(self, history: Any) -> List[Tuple[str, str]]:
        normalized: List[Tuple[str, str]] = []
        if not isinstance(history, list):
            return normalized

        pending_user = ""
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                normalized.append((str(item[0]), str(item[1])))
                pending_user = ""
                continue

            if not isinstance(item, dict):
                continue

            if "user" in item and "assistant" in item:
                normalized.append((str(item["user"]), str(item["assistant"])))
                pending_user = ""
                continue

            role = str(item.get("role", ""))
            content = str(item.get("content", ""))
            if role == "user":
                pending_user = content
                normalized.append((pending_user, ""))
            elif role == "assistant" and normalized:
                last_user, _ = normalized[-1]
                normalized[-1] = (last_user, content)
                pending_user = ""

        return [(u, a) for u, a in normalized if u or a]

    def _build_user_message(self, instruction: str, input_text: str) -> str:
        instruction = str(instruction or "").strip()
        input_text = str(input_text or "").strip()
        if instruction and input_text:
            return f"{instruction}\n\n{input_text}"
        return instruction or input_text

    def _safe_float(self, value: Any) -> float:
        if value is None:
            return math.nan
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return math.nan
        if math.isinf(numeric):
            return math.nan
        return numeric

    def _get_embedding_model(self):
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer

            model_kwargs = {}
            if self.config.device == "cuda":
                model_kwargs["device"] = "cuda"

            try:
                self.embedding_model = SentenceTransformer(self.config.embedding_model_path, **model_kwargs)
            except Exception as exc:
                alt_model_path = str(Path(self.config.embedding_model_path) / "0_Transformer")
                try:
                    self.embedding_model = SentenceTransformer(alt_model_path, **model_kwargs)
                    self.metric_notes["embedding_model"] = (
                        f"Primary embedding path failed; loaded fallback transformer path: {alt_model_path}"
                    )
                except Exception:
                    raise RuntimeError(
                        f"Failed to load embedding model from {self.config.embedding_model_path}"
                    ) from exc
        return self.embedding_model

    def _extract_candidate_keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"[\u4e00-\u9fff]{2,6}|[A-Za-z]{2,20}|[0-9]{2,}", text.lower())
        return [token for token in tokens if token not in self.persona_keywords.STOPWORDS]

    def _collect_persona_texts(self, data: List[Dict[str, Any]]) -> List[str]:
        texts: List[str] = []
        for sample in data:
            for field in ("output", "input", "instruction"):
                value = sample.get(field, "")
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())

            for user_msg, assistant_msg in self._normalize_history(sample.get("history", [])):
                if user_msg.strip():
                    texts.append(user_msg.strip())
                if assistant_msg.strip():
                    texts.append(assistant_msg.strip())
        return texts

    def build_persona_lexicon(self):
        train_path = Path(self.config.train_data_path)
        if not train_path.exists():
            self.metric_notes["persona_lexicon"] = f"Train data not found, use default lexicon only: {train_path}"
            self.persona_lexicon = list(dict.fromkeys(self.persona_keywords.VOCAL_WORDS + self.persona_keywords.HABITUAL_WORDS))
            return

        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)

        texts = self._collect_persona_texts(train_data)
        keyword_counter: Counter = Counter()
        for text in texts:
            keyword_counter.update(self._extract_candidate_keywords(text))

        dynamic_keywords = [
            token
            for token, count in keyword_counter.most_common(self.config.persona_top_k * 3)
            if count >= self.config.min_keyword_count
        ]

        merged = dynamic_keywords + self.persona_keywords.VOCAL_WORDS + self.persona_keywords.HABITUAL_WORDS
        self.persona_lexicon = list(dict.fromkeys(merged))[: self.config.persona_top_k]
        self.metric_notes["persona_lexicon"] = f"Loaded {len(self.persona_lexicon)} persona keywords from train data"

    def load_model(self):
        print(f"Loading model to {self.config.device}...")
        self.metric_notes["disabled_metrics"] = (
            "BERTScore and emotion similarity are disabled because the available local checkpoints "
            "are not suitable for those tasks."
        )

        if not os.path.exists(self.config.base_model_path):
            raise FileNotFoundError(f"Model path not found: {self.config.base_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        if self.config.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                device_map="cuda",
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                device_map="cpu",
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.float32,
            )

        if self.config.use_base_only:
            self.metric_notes["model_mode"] = "Running baseline with base model only (LoRA disabled)"
            self.model.eval()
            self.lora_model = None
        else:
            if not os.path.exists(self.config.lora_weights_path):
                raise FileNotFoundError(f"LoRA path not found: {self.config.lora_weights_path}")

            print(f"Loading LoRA weights from {self.config.lora_weights_path}...")
            self.lora_model = PeftModel.from_pretrained(
                self.model,
                self.config.lora_weights_path,
                device_map="auto",
            )
            self.lora_model.eval()
            self.metric_notes["model_mode"] = "Running fine-tuned LoRA model"
        if self.config.only_level4:
            self.metric_notes["metric_scope"] = "Running Level4 only; Levels 1/2/3/5 are skipped"
        else:
            self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
            self.build_persona_lexicon()
        print("Model loaded successfully!")

    def load_test_data(self) -> List[Dict[str, Any]]:
        print(f"Loading test data from {self.config.test_data_path}...")
        with open(self.config.test_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if self.config.sample_limit is not None:
            data = data[: self.config.sample_limit]
        print(f"Loaded {len(data)} test samples")
        return data

    def generate_response(
        self,
        instruction: str,
        input_text: str,
        history: Any,
        system_prompt: str = "",
        max_new_tokens: Optional[int] = None,
    ) -> str:
        return self.generate_response_with_model(
            self.generation_model,
            instruction,
            input_text,
            history,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )

    def generate_response_with_model(
        self,
        model,
        instruction: str,
        input_text: str,
        history: Any,
        system_prompt: str = "",
        max_new_tokens: Optional[int] = None,
    ) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        if not system_prompt:
            system_prompt = "你是一个个性化的聊天助手。"

        normalized_history = self._normalize_history(history)
        history_text = ""
        for user_msg, assistant_msg in normalized_history:
            history_text += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            if assistant_msg:
                history_text += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"

        user_message = self._build_user_message(instruction, input_text)

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt += history_text
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        if len(generated_tokens) == 0:
            return ""

        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    def generate_base_response(
        self,
        instruction: str,
        input_text: str,
        history: Any,
        system_prompt: str = "",
        max_new_tokens: Optional[int] = None,
    ) -> str:
        if self.lora_model is None or self.config.use_base_only:
            return self.generate_response_with_model(
                self.model,
                instruction,
                input_text,
                history,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
            )

        # Reuse the same wrapped model but temporarily disable LoRA adapters
        # so the baseline reply comes from the original base model.
        disable_ctx = self.lora_model.disable_adapter() if hasattr(self.lora_model, "disable_adapter") else nullcontext()
        with disable_ctx:
            return self.generate_response_with_model(
                self.lora_model,
                instruction,
                input_text,
                history,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
            )

    def calculate_bleu(self, hypothesis: str, reference: str) -> float:
        hypothesis = str(hypothesis or "")
        reference = str(reference or "")
        if not hypothesis.strip() or not reference.strip():
            return math.nan

        try:
            bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]], tokenize="zh")
            return bleu.score / 100.0
        except Exception as exc:
            print(f"BLEU calculation error: {exc}")
            return math.nan

    def calculate_rouge(self, hypothesis: str, reference: str) -> Tuple[float, float]:
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return scores["rouge1"].fmeasure, scores["rougeL"].fmeasure
        except Exception as exc:
            print(f"ROUGE error: {exc}")
            return math.nan, math.nan

    def calculate_ppl(self, text: str) -> float:
        if not text or not text.strip():
            return math.nan

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            return torch.exp(loss).item()
        except Exception as exc:
            print(f"PPL error: {exc}")
            return math.nan

    def evaluate_level1(self, hypothesis: str, reference: str) -> Dict[str, float]:
        bleu4 = self.calculate_bleu(hypothesis, reference)
        rouge1, rougel = self.calculate_rouge(hypothesis, reference)
        ppl = self.calculate_ppl(hypothesis)
        return {"bleu4": bleu4, "rouge1": rouge1, "rougeL": rougel, "ppl": ppl}

    def evaluate_level2(self, hypothesis: str, reference: str) -> Dict[str, float]:
        metrics = {"embedding_cosine": math.nan}

        try:
            emb1 = self.get_embedding(hypothesis)
            emb2 = self.get_embedding(reference)
            metrics["embedding_cosine"] = self.cosine_similarity(emb1, emb2)
        except Exception as exc:
            print(f"Embedding cosine error: {exc}")

        return metrics

    def keyword_hit_rate(self, text: str) -> float:
        if not self.persona_lexicon:
            return math.nan

        text_lower = text.lower()
        hits = sum(1 for token in self.persona_lexicon if token.lower() in text_lower)
        return hits / max(len(self.persona_lexicon), 1)

    def text_length_similarity(self, text1: str, text2: str) -> float:
        len1, len2 = len(text1.strip()), len(text2.strip())
        if max(len1, len2) == 0:
            return math.nan
        return 1.0 - abs(len1 - len2) / max(len1, len2)

    def punctuation_profile(self, text: str) -> np.ndarray:
        counts = [text.count(mark) for mark in self.persona_keywords.PUNCTUATION]
        vec = np.array(counts, dtype=float)
        total = vec.sum()
        if total > 0:
            vec /= total
        return vec

    def modal_profile(self, text: str) -> np.ndarray:
        counts = [text.count(word) for word in self.persona_keywords.VOCAL_WORDS + self.persona_keywords.HABITUAL_WORDS]
        vec = np.array(counts, dtype=float)
        total = vec.sum()
        if total > 0:
            vec /= total
        return vec

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray, empty_value: float = math.nan) -> float:
        if vec1 is None or vec2 is None:
            return math.nan
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return empty_value
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def evaluate_level3(self, hypothesis: str, reference: str) -> Dict[str, float]:
        punctuation_similarity = self.cosine_similarity(
            self.punctuation_profile(hypothesis),
            self.punctuation_profile(reference),
            empty_value=1.0,
        )
        modal_similarity = self.cosine_similarity(
            self.modal_profile(hypothesis),
            self.modal_profile(reference),
            empty_value=1.0,
        )
        return {
            "khr": self.keyword_hit_rate(hypothesis),
            "length_similarity": self.text_length_similarity(hypothesis, reference),
            "punctuation_similarity": punctuation_similarity,
            "modal_similarity": modal_similarity,
        }

    def parse_judge_response(self, content: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception:
            pass

        tone = re.search(r'"tone"\s*:\s*(\d+)', content)
        semantic = re.search(r'"semantic"\s*:\s*(\d+)', content)
        fidelity = re.search(r'"fidelity"\s*:\s*(\d+)', content)
        preferred = re.search(r'"preferred"\s*:\s*"?(prediction|reference|base|tie)"?', content)
        return {
            "tone": int(tone.group(1)) if tone else None,
            "semantic": int(semantic.group(1)) if semantic else None,
            "fidelity": int(fidelity.group(1)) if fidelity else None,
            "preferred": preferred.group(1) if preferred else None,
        }

    def build_judge_prompt(
        self,
        input_text: str,
        ground_truth: str,
        generated: str,
        base_response: str = "",
    ) -> str:
        if self.config.judge_win_opponent == "base" and base_response.strip():
            return f"""你是一位严格但公正的中文对话评测专家。请完成两项任务：

任务一：评估“待评价回复”本身是否贴近真实回复所体现的人物风格。
任务二：在“Base模型回复”和“待评价回复”之间，判断哪一条更接近真实回复所体现的目标人物风格。

用户输入：
{input_text}

真实回复：
{ground_truth}

Base模型回复：
{base_response}

待评价回复：
{generated}

请先对“待评价回复”从以下三个维度分别按 1-10 分打分：
1. tone：语气与说话风格是否接近真实回复
2. semantic：是否合理回应了用户输入，语义是否基本正确
3. fidelity：是否复刻了真实回复体现的人格、表达习惯和情绪色彩

再给出一个 preferred 字段，用于比较 Base 与待评价回复：
- prediction：待评价回复比 Base 模型回复更像目标人物
- base：Base 模型回复更像目标人物
- tie：两者难分高下

只返回 JSON，例如：
{{"tone": 8, "semantic": 7, "fidelity": 8, "preferred": "prediction"}}"""

        return f"""你是一位严格但公正的中文对话评测专家，请评估“待评价回复”是否像“真实回复”那样来自同一个人。

用户输入：
{input_text}

真实回复：
{ground_truth}

待评价回复：
{generated}

请从以下三个维度分别按 1-10 分打分：
1. tone：语气与说话风格是否接近真实回复
2. semantic：是否合理回应了用户输入，语义是否基本正确
3. fidelity：是否复刻了真实回复体现的人格、表达习惯和情绪色彩

另外补充一个 preferred 字段：
- prediction：待评价回复更像目标人物
- reference：真实回复明显更像目标人物
- tie：两者难分高下

只返回 JSON，例如：
{{"tone": 8, "semantic": 7, "fidelity": 8, "preferred": "prediction"}}"""

    def call_judge_api(self, prompt: str, retry_times: Optional[int] = None) -> Optional[Dict[str, Any]]:
        import aiohttp
        import time

        retry_times = retry_times or self.config.judge_retry_times
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.api_model,
            "messages": [
                {"role": "system", "content": "You are a professional evaluation judge for chatbot responses."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
        }

        async def fetch() -> Dict[str, Any]:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.api_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=45),
                ) as resp:
                    data = await resp.json()
                    if resp.status >= 400:
                        raise RuntimeError(f"Judge API error {resp.status}: {data}")
                    return data

        for attempt in range(retry_times):
            try:
                result = asyncio.run(fetch())
                content = result["choices"][0]["message"]["content"]
                return self.parse_judge_response(content)
            except Exception as exc:
                print(f"API call failed: {exc}")
                if attempt < retry_times - 1:
                    time.sleep(self.config.judge_retry_delay * (attempt + 1))

        return None

    def evaluate_level4(
        self,
        input_text: str,
        ground_truth: str,
        generated: str,
        base_response: str = "",
    ) -> Dict[str, float]:
        metrics = {
            "tone_score": math.nan,
            "semantic_score": math.nan,
            "fidelity_score": math.nan,
            "win_rate": math.nan,
        }
        if self.config.skip_level4:
            self.metric_notes["judge_api"] = "L4 skipped by configuration"
            return metrics
        if not self.config.api_key:
            self.metric_notes["judge_api"] = "Judge API key not set; L4 metrics skipped"
            return metrics

        self.metric_notes["judge_win_opponent"] = self.config.judge_win_opponent
        prompt = self.build_judge_prompt(input_text, ground_truth, generated, base_response=base_response)
        result = self.call_judge_api(prompt, retry_times=self.config.judge_retry_times)
        if not result:
            return metrics

        metrics["tone_score"] = self._safe_float(result.get("tone")) / 10.0 if result.get("tone") is not None else math.nan
        metrics["semantic_score"] = self._safe_float(result.get("semantic")) / 10.0 if result.get("semantic") is not None else math.nan
        metrics["fidelity_score"] = self._safe_float(result.get("fidelity")) / 10.0 if result.get("fidelity") is not None else math.nan

        preferred = result.get("preferred")
        if preferred == "prediction":
            metrics["win_rate"] = 1.0
        elif preferred == "base":
            metrics["win_rate"] = 0.0
        elif preferred == "reference":
            metrics["win_rate"] = 0.0
        elif preferred == "tie":
            metrics["win_rate"] = 0.5

        return metrics

    def get_embedding(self, text: str) -> np.ndarray:
        model = self._get_embedding_model()
        embedding = model.encode(text[:512], convert_to_numpy=True)
        return np.asarray(embedding, dtype=float)

    def style_feature_vector(self, text: str) -> np.ndarray:
        length = len(text.strip())
        emoji_like = len(re.findall(r"(\[[^\]]+\]|\([^)]+\)|【[^】]+】)", text))
        question_count = text.count("?") + text.count("？")
        exclaim_count = text.count("!") + text.count("！")
        punctuation = self.punctuation_profile(text)
        modal = self.modal_profile(text)
        scalar = np.array([length, emoji_like, question_count, exclaim_count], dtype=float)
        if scalar.max() > 0:
            scalar = scalar / scalar.max()
        return np.concatenate([scalar, punctuation, modal])

    def evaluate_level5(
        self,
        instruction: str,
        input_text: str,
        history: List[Tuple[str, str]],
        generated: str,
    ) -> Dict[str, float]:
        metrics = {
            "drift": math.nan,
            "adversarial_drift": math.nan,
            "style_consistency": math.nan,
        }
        if self.config.skip_level5:
            self.metric_notes["level5"] = "L5 skipped by configuration"
            return metrics
        normalized_history = self._normalize_history(history)

        round_embeddings = []
        round_style_vectors = []
        for t in range(self.config.num_rounds):
            round_history = normalized_history * (t + 1)
            response = self.generate_response(instruction, input_text, round_history, max_new_tokens=128)
            if not response.strip():
                continue

            try:
                round_embeddings.append(self.get_embedding(response))
            except Exception as exc:
                print(f"Embedding error: {exc}")

            round_style_vectors.append(self.style_feature_vector(response))

        if round_embeddings:
            stacked = np.array(round_embeddings)
            metrics["drift"] = float(np.std(stacked, axis=0).mean())

        if len(round_style_vectors) >= 2:
            sims = []
            for idx in range(1, len(round_style_vectors)):
                sims.append(self.cosine_similarity(round_style_vectors[0], round_style_vectors[idx]))
            valid_sims = [sim for sim in sims if not math.isnan(sim)]
            if valid_sims:
                metrics["style_consistency"] = float(np.mean(valid_sims))

        try:
            original_embedding = self.get_embedding(generated)
            adversarial_prompts = [
                "请用正式的语气回答",
                "请不要像平时那样说话，改用完全不同的风格",
                "请用很克制、很官方的方式回复",
            ]
            drifts = []
            for adv_prompt in adversarial_prompts:
                adv_response = self.generate_response(adv_prompt, input_text, normalized_history, max_new_tokens=128)
                if not adv_response.strip():
                    continue
                adv_embedding = self.get_embedding(adv_response)
                cosine = self.cosine_similarity(original_embedding, adv_embedding)
                if not math.isnan(cosine):
                    drifts.append(1.0 - cosine)
            if drifts:
                metrics["adversarial_drift"] = float(np.mean(drifts))
        except Exception as exc:
            print(f"Adversarial drift error: {exc}")

        return metrics

    def evaluate_sample(self, sample: Dict[str, Any]) -> SampleResult:
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        history = self._normalize_history(sample.get("history", []))
        ground_truth = sample.get("output", "")
        system_prompt = sample.get("system", "")
        sample_id = str(sample.get("id", "unknown"))

        if not ground_truth:
            return SampleResult(
                sample_id=sample_id,
                instruction=instruction,
                input_text=input_text,
                history=history,
                ground_truth=ground_truth,
                generated_response="",
            )

        generated = self.generate_response(instruction, input_text, history, system_prompt)
        if not generated.strip():
            print(f"Warning: Empty generation for sample {sample_id}")

        base_response = ""
        if not self.config.use_base_only and self.config.judge_win_opponent == "base":
            base_response = self.generate_base_response(instruction, input_text, history, system_prompt)

        l1 = {"bleu4": math.nan, "rouge1": math.nan, "rougeL": math.nan, "ppl": math.nan}
        l2 = {"embedding_cosine": math.nan}
        l3 = {
            "khr": math.nan,
            "length_similarity": math.nan,
            "punctuation_similarity": math.nan,
            "modal_similarity": math.nan,
        }
        l5 = {
            "drift": math.nan,
            "adversarial_drift": math.nan,
            "style_consistency": math.nan,
        }

        if not self.config.only_level4:
            l1 = self.evaluate_level1(generated, ground_truth)
            l2 = self.evaluate_level2(generated, ground_truth)
            l3 = self.evaluate_level3(generated, ground_truth)
            l5 = self.evaluate_level5(instruction, input_text, history, generated)
        l4 = self.evaluate_level4(input_text, ground_truth, generated, base_response=base_response)

        return SampleResult(
            sample_id=sample_id,
            instruction=instruction,
            input_text=input_text,
            history=history,
            ground_truth=ground_truth,
            generated_response=generated,
            base_response=base_response,
            l1_bleu4=l1["bleu4"],
            l1_rouge1=l1["rouge1"],
            l1_rougel=l1["rougeL"],
            l1_ppl=l1["ppl"],
            l2_embedding_cosine=l2["embedding_cosine"],
            l3_khr=l3["khr"],
            l3_length_similarity=l3["length_similarity"],
            l3_punctuation_similarity=l3["punctuation_similarity"],
            l3_modal_similarity=l3["modal_similarity"],
            l4_tone_score=l4["tone_score"],
            l4_semantic_score=l4["semantic_score"],
            l4_fidelity_score=l4["fidelity_score"],
            l4_win_rate=l4["win_rate"],
            l5_drift=l5["drift"],
            l5_adversarial_drift=l5["adversarial_drift"],
            l5_style_consistency=l5["style_consistency"],
        )

    def run_evaluation(self) -> Tuple[List[SampleResult], pd.DataFrame]:
        print("Starting evaluation...")
        test_data = self.load_test_data()
        results: List[SampleResult] = []

        for idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
            try:
                result = self.evaluate_sample(sample)
                results.append(result)
                if idx < 3:
                    print(f"\n=== Sample {idx} Debug ===")
                    print(f"Input: {sample.get('input', '')[:80]}")
                    print(f"Generated: {result.generated_response[:120]}")
                    print(f"Ground Truth: {result.ground_truth[:120]}")
            except Exception as exc:
                import traceback

                print(f"Error on sample {idx}: {exc}")
                traceback.print_exc()

        df = self.results_to_dataframe(results)
        return results, df

    def results_to_dataframe(self, results: List[SampleResult]) -> pd.DataFrame:
        rows = []
        for result in results:
            rows.append(
                {
                    "sample_id": result.sample_id,
                    "instruction": result.instruction[:80] + "..." if len(result.instruction) > 80 else result.instruction,
                    "L1_BLEU4": result.l1_bleu4,
                    "L1_ROUGE1": result.l1_rouge1,
                    "L1_ROUGEL": result.l1_rougel,
                    "L1_PPL": result.l1_ppl,
                    "L2_Embedding_Cosine": result.l2_embedding_cosine,
                    "L3_KHR": result.l3_khr,
                    "L3_Length_Sim": result.l3_length_similarity,
                    "L3_Punctuation_Sim": result.l3_punctuation_similarity,
                    "L3_Modal_Sim": result.l3_modal_similarity,
                    "L4_Tone": result.l4_tone_score,
                    "L4_Semantic": result.l4_semantic_score,
                    "L4_Fidelity": result.l4_fidelity_score,
                    "L4_Win_Rate": result.l4_win_rate,
                    "L5_Drift": result.l5_drift,
                    "L5_Adversarial_Drift": result.l5_adversarial_drift,
                    "L5_Style_Consistency": result.l5_style_consistency,
                }
            )
        return pd.DataFrame(rows)

    def build_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        stats = {"Metric": [], "ValidCount": [], "Mean": [], "Std": [], "Min": [], "Max": []}
        for col in df.columns:
            if col in {"sample_id", "instruction"}:
                continue

            numeric = pd.to_numeric(df[col], errors="coerce")
            valid = numeric.dropna()
            stats["Metric"].append(col)
            stats["ValidCount"].append(int(valid.count()))
            stats["Mean"].append(valid.mean() if not valid.empty else math.nan)
            stats["Std"].append(valid.std() if not valid.empty else math.nan)
            stats["Min"].append(valid.min() if not valid.empty else math.nan)
            stats["Max"].append(valid.max() if not valid.empty else math.nan)

        return pd.DataFrame(stats)

    def save_results(self, results: List[SampleResult], df: pd.DataFrame, output_dir: Optional[str] = None):
        output_path = Path(output_dir or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump([asdict(result) for result in results], f, ensure_ascii=False, indent=2)

        df.to_csv(output_path / "final_eval_report.csv", index=False, encoding="utf-8")

        summary_df = self.build_summary_stats(df)
        summary_df.to_csv(output_path / "summary_stats.csv", index=False, encoding="utf-8")

        with open(output_path / "metric_notes.json", "w", encoding="utf-8") as f:
            json.dump(self.metric_notes, f, ensure_ascii=False, indent=2)

        import matplotlib.pyplot as plt

        valid_drift = pd.to_numeric(df.get("L5_Drift"), errors="coerce").dropna()
        if not valid_drift.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(valid_drift.index, valid_drift.values, marker="o")
            plt.xlabel("Sample Index")
            plt.ylabel("Style Drift")
            plt.title("L5: Style Drift Across Samples")
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path / "drift_analysis.png", dpi=150, bbox_inches="tight")
            plt.close()

        print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Echo-Soul evaluation pipeline")
    parser.add_argument("--device", default=os.getenv("EVAL_DEVICE", "cuda"))
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--skip-level4", action="store_true")
    parser.add_argument("--skip-level5", action="store_true")
    parser.add_argument("--only-level4", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default=os.getenv("EVAL_OUTPUT_DIR", "/root/autodl-tmp/pingce"))
    parser.add_argument("--test-data-path", default=os.getenv("EVAL_TEST_DATA_PATH", "/root/autodl-tmp/pingce/xiaolaoshi_test_data.json"))
    parser.add_argument("--train-data-path", default=os.getenv("EVAL_TRAIN_DATA_PATH", "/root/autodl-tmp/xiaolaoshi_train_data.json"))
    parser.add_argument("--base-model-path", default=os.getenv("EVAL_BASE_MODEL_PATH", "/root/autodl-tmp/models/qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--lora-weights-path", default=os.getenv("EVAL_LORA_PATH", "/root/autodl-tmp/LlamaFactory/saves/Qwen2.5-7B-Instruct/lora/train_2026-04-23-10-27-45"))
    parser.add_argument("--api-base-url", default=os.getenv("EVAL_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--api-model", default=os.getenv("EVAL_API_MODEL", "qwen-plus"))
    parser.add_argument("--judge-win-opponent", choices=["reference", "base"], default=os.getenv("EVAL_JUDGE_WIN_OPPONENT", "base"))
    parser.add_argument("--use-base-only", action="store_true")
    args = parser.parse_args()

    sample_limit = args.sample_limit
    if args.smoke_test and sample_limit is None:
        sample_limit = 2

    config = EvalConfig(
        base_model_path=args.base_model_path,
        lora_weights_path=args.lora_weights_path,
        test_data_path=args.test_data_path,
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
        device=args.device,
        api_base_url=args.api_base_url,
        api_model=args.api_model,
        api_key=os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        judge_win_opponent=args.judge_win_opponent,
        sample_limit=sample_limit,
        skip_level4=args.skip_level4 or args.smoke_test,
        skip_level5=args.skip_level5 or args.smoke_test,
        only_level4=args.only_level4,
        smoke_test=args.smoke_test,
        use_base_only=args.use_base_only,
    )

    pipeline = EvaluationPipeline(config)

    print("=" * 60)
    print("Echo-Soul Evaluation Pipeline (GPU Version)")
    print("=" * 60)

    pipeline.load_model()
    results, df = pipeline.run_evaluation()

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head().to_string(index=False))

    pipeline.save_results(results, df)
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
