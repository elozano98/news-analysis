from typing import Dict, Optional, Union

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    BertTokenizer,
    AutoTokenizer,
    TokenClassificationPipeline,
)

from pipeline import NewsPipeline

CATEGORY_EMOJIS = {
    "Automobile": "๐",
    "Entertainment": "๐ฟ",
    "Politics": "โ๏ธ",
    "Science": "๐งช",
    "Sports": "๐",
    "Technology": "๐ป",
    "World": "๐",
}
FAKE_EMOJIS = {"Fake": "๐ป", "Real": "๐"}
CLICKBAIT_EMOJIS = {"Clickbait": "๐ฃ", "Normal": "โ"}


class NewsAnalyzer:
    def __init__(
        self,
        category_model_name: str,
        fake_model_name: str,
        clickbait_model_name: str,
        ner_model_name: str,
    ) -> None:
        self.category_pipe = NewsPipeline(
            model=AutoModelForSequenceClassification.from_pretrained(
                category_model_name
            ),
            tokenizer=BertTokenizer.from_pretrained(category_model_name),
            emojis=CATEGORY_EMOJIS,
        )
        self.fake_pipe = NewsPipeline(
            model=AutoModelForSequenceClassification.from_pretrained(fake_model_name),
            tokenizer=BertTokenizer.from_pretrained(fake_model_name),
            emojis=FAKE_EMOJIS,
        )
        self.clickbait_pipe = NewsPipeline(
            model=AutoModelForSequenceClassification.from_pretrained(
                clickbait_model_name
            ),
            tokenizer=BertTokenizer.from_pretrained(clickbait_model_name),
            emojis=CLICKBAIT_EMOJIS,
        )
        self.ner_pipe = TokenClassificationPipeline(
            model=AutoModelForTokenClassification.from_pretrained(ner_model_name),
            tokenizer=AutoTokenizer.from_pretrained(ner_model_name),
            aggregation_strategy="simple",
        )

    def __call__(
        self, headline: str, content: Optional[str] = None
    ) -> Dict[str, Union[str, float]]:
        return {
            "category": self.category_pipe(headline=headline, content=content),
            "fake": self.fake_pipe(headline=headline, content=content),
            "clickbait": self.clickbait_pipe(headline=headline, content=None),
            "ner": {
                "headline": self.ner_pipe(headline),
                "content": self.ner_pipe(content) if content else None,
            },
        }
