import os
import re

import PyPDF2


def clean_text(text: str) -> str:
    """
    Entfernt überflüssige Whitespaces, Tabs, etc. und normalisiert den Text grob.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_text_from_pdf(pdf_path: str) -> str | None:
    """
    Extrahiert den Text aus einer PDF-Datei.
    """
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            all_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)
            return "\n".join(all_text)
    except Exception as e:
        print(f"Fehler beim Lesen der PDF: {e}")
        return None


def extract_text_from_txt(file_path: str) -> str | None:
    """
    Liest eine einfache .txt-Datei ein und gibt ihren Inhalt zurück.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Fehler beim Lesen der Textdatei: {e}")
        return None


def extract_text(file_path: str, model_size: str = "base") -> str:
    """
    Vereinheitlichte Funktion: Erkennt den Dateityp und ruft den passenden Extraktionsprozess auf.
    Danach wird der Text bereinigt (clean_text).
    Gibt den fertigen Text zurück (oder einen leeren String bei Fehlern).
    """
    if not os.path.exists(file_path):
        print("Datei existiert nicht.")
        return ""

    file_ext = os.path.splitext(file_path)[1].lower()

    raw_text = ""

    if file_ext == ".pdf":
        raw_text = extract_text_from_pdf(file_path) or ""
    elif file_ext == ".txt":
        raw_text = extract_text_from_txt(file_path) or ""
    else:
        print(f"Dateiformat '{file_ext}' wird nicht unterstützt.")
        return ""

    cleaned = clean_text(raw_text)
    return cleaned


if __name__ == "__main__":
    test_file_path = r"data\documents\Data_Governance_A_Guide_----_(Chapter_1_Overview_and Importance_of Data_Governance).pdf"

    extracted_text = extract_text(test_file_path, model_size="base")
    if extracted_text:
        print("Extrahierter und bereinigter Text:\n")
        print(extracted_text[:500])  # Nur ein Ausschnitt
    else:
        print("Kein Text extrahiert.")
