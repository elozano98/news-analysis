from typing import Dict, Optional

from transformers import TextClassificationPipeline


class NewsPipeline(TextClassificationPipeline):
    def __init__(self, emojis: Dict[str, str], **kwargs) -> None:
        self.emojis = emojis
        super().__init__(**kwargs)

    def __call__(self, headline: str, content: Optional[str]) -> str:
        if content:
            text = f" {self.tokenizer.sep_token} ".join([headline, content])
        else:
            text = headline
        prediction = super().__call__(text, padding=True, truncation=True)[0]
        return {**prediction, "emoji": self.emojis[prediction["label"]]}
