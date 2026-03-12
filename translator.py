from deep_translator import GoogleTranslator

SUPPORTED_LANGS = {
    "en": "english",
    "ta": "tamil"
}

def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang == "en":
        return text

    translator = GoogleTranslator(
        source=source_lang,
        target="en"
    )
    return translator.translate(text)
