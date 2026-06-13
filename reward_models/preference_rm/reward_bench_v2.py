from transformers import AutoTokenizer
from datasets import Dataset, load_dataset


def format_record(prompt, response):
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def prepare_reward_bench_dataset(
    tokenizer: AutoTokenizer, subset: str = None, max_length: int = 1024
):

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("allenai/reward-bench-2", split="test")
    if subset:
        ds = ds.filter(lambda ex: ex["subset"] == subset)

    records = []
    for row in ds:
        prompt = row["prompt"]
        chosen = row["chosen"][-1]
        rejects = row["rejected"]

        chosen_record = format_record(prompt, chosen)
        chosen_tokenized = tokenizer.apply_chat_template(
            chosen_record,
            tokenize=True,
            max_length=max_length,
            truncation=True,
            add_generation_prompt=False,
            return_dict=True,
        )

        for r in rejects:
            rejected_record = format_record(prompt, r)
            reject_tokenized = tokenizer.apply_chat_template(
                rejected_record,
                tokenize=True,
                max_length=max_length,
                truncation=True,
                add_generation_prompt=False,
                return_dict=True,
            )

            records.append(
                {
                    "chosen_ids": chosen_tokenized["input_ids"],
                    "chosen_mask": chosen_tokenized["attention_mask"],
                    "rejected_ids": reject_tokenized["input_ids"],
                    "rejected_mask": reject_tokenized["attention_mask"],
                }
            )

    return Dataset.from_list(records)
