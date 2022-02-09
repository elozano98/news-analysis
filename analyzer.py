from typing import Dict, Optional, Union

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
)

from pipeline import NewsPipeline

CATEGORY_EMOJIS = {
    "Automobile": "🚗",
    "Entertainment": "🍿",
    "Politics": "⚖️",
    "Science": "🧪",
    "Sports": "🏀",
    "Technology": "💻",
    "World": "🌍",
}
FAKE_EMOJIS = {"Fake": "👻", "Real": "👍"}
CLICKBAIT_EMOJIS = {"Clickbait": "🎣", "Normal": "✅"}


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
            tokenizer=AutoTokenizer.from_pretrained(category_model_name),
            emojis=CATEGORY_EMOJIS,
        )
        self.fake_pipe = NewsPipeline(
            model=AutoModelForSequenceClassification.from_pretrained(fake_model_name),
            tokenizer=AutoTokenizer.from_pretrained(fake_model_name),
            emojis=FAKE_EMOJIS,
        )
        self.clickbait_pipe = NewsPipeline(
            model=AutoModelForSequenceClassification.from_pretrained(
                clickbait_model_name
            ),
            tokenizer=AutoTokenizer.from_pretrained(clickbait_model_name),
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


if __name__ == "__main__":
    analyzer = NewsAnalyzer(
        category_model_name="elozano/news-category",
        fake_model_name="elozano/news-fake",
        clickbait_model_name="elozano/news-clickbait",
        ner_model_name="dslim/bert-base-NER",
    )
    prediction = analyzer(headline="Lakers Won!")
    print(prediction)
