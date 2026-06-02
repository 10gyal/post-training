from dataclasses import dataclass


@dataclass
class Config:
    pretrained_lm: str = "HuggingFaceTB/SmolLM2-135M"
    ift_dataset: str = "HuggingFaceH4/no_robots"
    # the pretrained lm does not come with a chat template
    chat_template_tokenizer: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dataset_split: str = "train"
    batch_size: int = 4
