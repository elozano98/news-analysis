from typing import Dict, List, Tuple, Union

import streamlit as st
from annotated_text import annotated_text

from analyzer import NewsAnalyzer

ENTITY_COLOR = {
    "PER": "#b2ffff",
    "LOC": "#ffffb2",
    "ORG": "#adfbaf",
    "MISC": "#ffb2b2",
}


def run() -> None:
    analyzer = NewsAnalyzer(
        category_model_name="elozano/news-category",
        fake_model_name="elozano/news-fake",
        clickbait_model_name="elozano/news-clickbait",
        ner_model_name="dslim/bert-base-NER",
    )
    st.title("📰 News Analyzer")
    headline = st.text_input("Headline:")
    content = st.text_area("Content:", height=200)
    if headline == "":
        st.error("Please, provide a headline.")
    else:
        if content == "":
            st.warning(
                "Please, provide both headline and content to achieve better results."
            )
        button = st.button("Analyze")
        if button:
            predictions = analyzer(headline=headline, content=content)
            col1, _, col2 = st.columns([2, 1, 4])

            with col1:
                st.subheader("Analysis:")
                category_prediction = predictions["category"]
                st.markdown(
                    f"{category_prediction['emoji']} **Category**: {category_prediction['label']}"
                )
                clickbait_prediction = predictions["clickbait"]
                st.markdown(
                    f"{clickbait_prediction['emoji']} **Clickbait**: {'Yes' if clickbait_prediction['label'] == 'Clickbait' else 'No'}"
                )
                fake_prediction = predictions["fake"]
                st.markdown(
                    f"{fake_prediction['emoji']} **Fake**: {'Yes' if fake_prediction['label'] == 'Fake' else 'No'}"
                )

            with col2:
                st.subheader("Headline:")
                annotated_text(
                    *parse_entities(headline, predictions["ner"]["headline"])
                )
                st.subheader("Content:")
                if content:
                    annotated_text(
                        *parse_entities(content, predictions["ner"]["content"])
                    )
                else:
                    st.error("Content not provided.")


def parse_entities(
    text: str, entities: Dict[str, Union[str, int]]
) -> List[Union[str, Tuple[str, str]]]:
    start = 0
    parsed_text = []
    for entity in entities:
        parsed_text.append(text[start : entity["start"]])
        parsed_text.append(
            (
                entity["word"],
                entity["entity_group"],
                ENTITY_COLOR[entity["entity_group"]],
            )
        )
        start = entity["end"]
    parsed_text.append(text[start:])
    return parsed_text


if __name__ == "__main__":
    run()
