from __future__ import annotations

import re
import threading
import time

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.prompt_template import build_prompt

logger = get_logger(__name__)


class TextToSqlService:
    """Generate SQL from natural language and a BIRD-rendered schema."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._tokenizer = None
        self._model = None
        self._model_ready = False
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        if self.settings.debug_mode:
            return True
        return self._model_ready

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def warmup(self) -> None:
        if self.settings.debug_mode:
            self._model_ready = True
            return
        if not self.settings.model_warmup_on_startup:
            return
        self._load_model()
        self._run_warmup_inference()

    def _load_model(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                logger.info("loading_model", model=self.settings.model_name)
                self._tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)

                model_kwargs: dict = {"device_map": "auto"}
                if self.settings.model_quantization == "4bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                elif self.settings.model_quantization == "8bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.settings.model_name,
                    **model_kwargs,
                )
                self._model_ready = True
                logger.info("model_loaded", model=self.settings.model_name)
            except Exception as exc:
                self._load_error = str(exc)
                logger.exception("model_load_failed", error=str(exc))
                raise

    def _run_warmup_inference(self) -> None:
        self.generate(
            question="How many rows?",
            schema_context="Database: demo\n\nTable: users\n  - id (integer)",
            sql_dialect="sqlite",
        )

    def generate(self, question: str, schema_context: str, sql_dialect: str) -> str:
        if self.settings.debug_mode:
            return self._debug_sql(question, schema_context)

        self._load_model()
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(self._load_error or "Model is not loaded")

        prompt = build_prompt(schema_context, question)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        start = time.perf_counter()
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.settings.max_new_tokens,
            do_sample=False,
        )
        elapsed = time.perf_counter() - start
        logger.info("inference_complete", elapsed_seconds=round(elapsed, 3))

        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return self._extract_sql(generated)

    def _debug_sql(self, question: str, schema_context: str) -> str:
        question_lower = question.lower()
        table_names = re.findall(r"^Table: (\w+)", schema_context, re.MULTILINE)

        if "how many" in question_lower and table_names:
            table = table_names[0]
            if "client" in question_lower:
                for name in table_names:
                    if "client" in name.lower():
                        table = name
                        break
            return f'SELECT COUNT(*) AS count FROM "{table}";'

        if table_names:
            return f'SELECT * FROM "{table_names[0]}" LIMIT 5;'

        return "SELECT 1 AS result;"

    def _extract_sql(self, text: str) -> str:
        cleaned = text.strip()

        fence_match = re.search(
            r"```(?:sql)?\s*(.*?)```",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fence_match:
            cleaned = fence_match.group(1).strip()

        if "### SQL" in cleaned:
            cleaned = cleaned.split("### SQL", 1)[-1].strip()

        select_match = re.search(
            r"(SELECT\b.*?)(?:;|\Z)",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if select_match:
            sql = select_match.group(1).strip()
            return sql if sql.endswith(";") else f"{sql};"

        return cleaned.strip().rstrip(";") + ";"
