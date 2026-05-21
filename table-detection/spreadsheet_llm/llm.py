import json
import time
from dataclasses import dataclass
from typing import List, Optional

import torch


# ============================================================
#          PROMPT TEMPLATES (verbatim from Appendix L)
# ============================================================

PROMPT_VANILLA_DETECTION = (
    "Given an input that is a string denoting data of cells in a spreadsheet. "
    "The input spreadsheet includes many pairs, and each pair consists of a "
    "cell address and the text in that cell with a ',' in between, like "
    "'A1,Year'. Cells are separated by '|' like 'A1,Year|A2,Profit'. The text "
    "can be empty so the cell data is like 'A1, |A2,Profit'. The cells are "
    "organized in row-major order. Now you should tell me the range of the "
    "table in a format like A2:D5, and the range of the table should only "
    "CONTAIN HEADER REGION and the data region, DON'T include the title or "
    "comments. Note that there can be more than one table in the string, so "
    "you should return all the RANGE, LIKE ['range': 'A1:F9', 'range': "
    "'A12:F18']. DON'T ADD OTHER WORDS OR EXPLANATION."
)

PROMPT_COMPRESSOR_DETECTION = (
    "Given an input that is a string denoting data of cells in an Excel "
    "spreadsheet. The input spreadsheet contains many tuples, describing the "
    "cells with content in the spreadsheet. Each tuple consists of two "
    "elements separated by a '|': the cell content and the cell "
    "address/region, like (Year|A1), ( |A1) or (IntNum|A1:B3). The content "
    "in some cells such as '#,##0'/'d-mmm-yy'/'H:mm:ss',etc., represents the "
    "CELL DATA FORMATS of Excel. The content in some cells such as "
    "'IntNum'/'DateData'/'EmailData',etc., represents a category of data "
    "with the same format and similar semantics. For example, 'IntNum' "
    "represents integer type data, and 'ScientificNum' represents scientific "
    "notation type data. 'A1:B3' represents a region in a spreadsheet, from "
    "the first row to the third row and from column A to column B. Some "
    "cells with empty content in the spreadsheet are not entered. Now you "
    "should tell me the range of the table in a format like A2:D5, and the "
    "range of the table should only CONTAIN HEADER REGION and the data "
    "region. DON'T include the title or comments. Note that there can be "
    "more than one table in a string, so you should return all the RANGE. "
    "DON'T ADD OTHER WORDS OR EXPLANATION."
)

PROMPT_COMPRESSOR_DETECTION_M1M2 = (
    "Given an input that is a string denoting data of cells in an Excel "
    "spreadsheet. The input spreadsheet contains many tuples, describing the "
    "cells with content in the spreadsheet. Each tuple consists of two "
    "elements separated by a '|': the cell content and the cell "
    "address/region, like (Year|A1), ( |A1) or (2024|A1:B3). When the same "
    "value appears in multiple adjacent cells, they are merged into a single "
    "tuple with a range address, like (100|A2:A10). 'A1:B3' represents a "
    "region in a spreadsheet, from the first row to the third row and from "
    "column A to column B. Some cells with empty content in the spreadsheet "
    "are not entered. Now you should tell me the range of the table in a "
    "format like A2:D5, and the range of the table should only CONTAIN "
    "HEADER REGION and the data region. DON'T include the title or comments. "
    "Note that there can be more than one table in a string, so you should "
    "return all the RANGE, LIKE ['range': 'A1:F9', 'range': 'A12:F18']. "
    "DON'T ADD OTHER WORDS OR EXPLANATION."
)

PROMPT_QA_STAGE1 = (
    "Given an input that is a string denoting data of cells in a table. The "
    "input table contains many tuples, describing the cells with content in "
    "the spreadsheet. Each tuple consists of two elements separated by a "
    "'|': the cell content and the cell address/region, like (Year|A1), "
    "( |A1) or (IntNum|A1:B3). The content in some cells such as "
    "'#,##0'/'d-mmm-yy'/'H:mm:ss',etc., represents the CELL DATA FORMATS of "
    "Excel. The content in some cells such as "
    "'IntNum'/'DateData'/'EmailData',etc., represents a category of data "
    "with the same format and similar semantics. For example, 'IntNum' "
    "represents integer type data, and 'ScientificNum' represents scientific "
    "notation type data. 'A1:B3' represents a region in a spreadsheet, from "
    "the first row to the third row and from column A to column B. Some "
    "cells with empty content in the spreadsheet are not entered. How many "
    "tables are there in the spreadsheet? Below is a question about one "
    "certain table in this spreadsheet. I need you to determine in which "
    "table the answer to the following question can be found, and return "
    "the RANGE of the ONE table you choose, LIKE ['range': 'A1:F9']. DON'T "
    "ADD OTHER WORDS OR EXPLANATION."
)

PROMPT_QA_STAGE2 = (
    "Given an input that is a string denoting data of cells in a table and "
    "a question about this table. The answer to the question can be found "
    "in the table. The input table includes many pairs, and each pair "
    "consists of a cell address and the text in that cell with a ',' in "
    "between, like 'A1,Year'. Cells are separated by '|' like "
    "'A1,Year|A2,Profit'. The text can be empty so the cell data is like "
    "'A1, |A2,Profit'. The cells are organized in row-major order. The "
    "answer to the input question is contained in the input table and can "
    "be represented by cell address. I need you to find the cell address of "
    "the answer in the given table based on the given question description, "
    "and return the cell ADDRESS of the answer like '[B3]' or "
    "'[SUM(A2:A10)]'. DON'T ADD ANY OTHER WORDS."
)


# ============================================================
#                    CALL RECORD
# ============================================================

@dataclass
class LLMCallRecord:
    elapsed_sec: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response: str
    instruction_preview: str
    input_tokens_estimated: int
    label: str = ''


# ============================================================
#              LOCAL LLM CLIENT
# ============================================================

class LLMClient:
    """Local inference wrapper for Mistral-7B-Instruct-v0.2 with optional PEFT adapter.

    Parameters
    ----------
    base_model_name:
        HuggingFace model ID, default mistralai/Mistral-7B-Instruct-v0.2.
    adapter_path:
        Path to a saved PEFT LoRA adapter directory (from finetune.py).
        If None, the base model is used as-is (zero-shot).
    device_map:
        Passed to from_pretrained. 'auto' distributes across all available GPUs.
    """

    DEFAULT_PARAMS = dict(
        max_new_tokens=300,
        do_sample=False,        # greedy = temperature 0 (Appendix G)
        repetition_penalty=1.0, # frequency_penalty=0, presence_penalty=0
    )

    def __init__(
        self,
        base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.history: List[LLMCallRecord] = []

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
        )

        if adapter_path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()  # fuse weights for faster inference

        model.eval()
        self.model = model

    # ----------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # ----------------------------------------------------------

    def complete(
        self,
        instruction: str,
        input_text: str,
        *,
        use_system_role: bool = True,  # kept for API parity, ignored (Mistral has no system role)
        label: str = '',
        **overrides,
    ) -> LLMCallRecord:
        """Run one inference pass.

        Combines instruction + input into a single user message using
        Mistral-Instruct's chat template, matching the training format
        used in finetune.py.
        """
        params = {**self.DEFAULT_PARAMS, **overrides}

        # Mistral-Instruct-v0.2 has no system role; instruction goes into user message
        user_content = instruction + "\nINPUT:\n" + input_text
        messages = [{"role": "user", "content": user_content}]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=params["max_new_tokens"],
                do_sample=params["do_sample"],
                repetition_penalty=params["repetition_penalty"],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        new_tokens = outputs[0][prompt_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion_len = len(new_tokens)

        record = LLMCallRecord(
            elapsed_sec=elapsed,
            model=self.base_model_name
                  + (f" +adapter={self.adapter_path}" if self.adapter_path else ""),
            prompt_tokens=prompt_len,
            completion_tokens=completion_len,
            total_tokens=prompt_len + completion_len,
            response=response,
            instruction_preview=instruction[:120].replace('\n', ' '),
            input_tokens_estimated=self.count_tokens(input_text),
            label=label,
        )
        self.history.append(record)
        return record

    # ----------------------------------------------------------

    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.history)

    def total_time(self) -> float:
        return sum(r.elapsed_sec for r in self.history)

    def history_as_dicts(self):
        return [r.__dict__ for r in self.history]

    def save_history(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history_as_dicts(), f, indent=2, default=str)
