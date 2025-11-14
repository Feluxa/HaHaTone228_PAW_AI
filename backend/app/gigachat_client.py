import os
from pathlib import Path

from dotenv import load_dotenv
from gigachat import GigaChat


# Загружаем .env, который лежит в папке backend
BASE_DIR = Path(__file__).resolve().parents[1]  # .../backend
load_dotenv(BASE_DIR / ".env")


def ask_gigachat(prompt: str) -> str:
    """
    Простой вызов GigaChat: на вход текст, на выход текстовый ответ.
    """
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    verify_ssl = os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "False").lower() == "true"

    if not credentials:
        raise RuntimeError("Не задан GIGACHAT_CREDENTIALS в .env")

    # самый простой вариант: используем Authorization Key из .env
    with GigaChat(
        credentials=credentials,
        verify_ssl_certs=verify_ssl,
    ) as giga:
        response = giga.chat(prompt)

    # Берём текст ответа из первого варианта
    return response.choices[0].message.content
