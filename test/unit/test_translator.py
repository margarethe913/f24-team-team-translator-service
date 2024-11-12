from src.translator import query_llm_robust
from unittest.mock import patch

def test_llm_normal_response():
    is_english, translated_content = query_llm_robust("Der frühe Vogel fängt den Wurm.")
    assert is_english == False
    assert translated_content == "The early bird catches the worm."

def test_llm_normal_eng_response():
    is_english, translated_content = query_llm_robust("Hello everyone!")
    assert is_english == True
    assert translated_content ==  "Hello everyone!"

@patch("src.translator.client.chat.completions.create")
def test_llm_gibberish_response(mock_create):
    mock_create.return_value = type(
        "MockResponse",
        (),
        {"choices": [type("MockChoice", (), {"message": type("MockMessage", (), {"content": "aswerfads aasdf"})})()]}
    )()
    
    is_english, translated_content = query_llm_robust("Bonjour tout le monde")
    assert is_english == False
    assert translated_content == ""

@patch("src.translator.client.chat.completions.create")
def test_malformed_no_tuple(mock_create):
    mock_create.return_value = type(
        "MockResponse",
        (),
        {"choices": [type("MockChoice", (), {"message": type("MockMessage", (), {"content": "This response has no tuple format."})})()]}
    )()
    
    is_english, translated_content = query_llm_robust("Bonjour tout le monde")
    assert is_english == False
    assert translated_content == ""

@patch("src.translator.client.chat.completions.create")
def test_non_string_english_content(mock_create):
    mock_create.return_value = type(
        "MockResponse",
        (),
        {"choices": [type("MockChoice", (), {"message": type("MockMessage", (), {"content": "True, 12@@%3"})})()]}
    )()
    is_english, translated_content = query_llm_robust("Hello!")
    assert is_english == True
    assert translated_content == ""

@patch("src.translator.client.chat.completions.create")
def test_non_string_translation_content(mock_create):
    mock_create.return_value = type(
        "MockResponse",
        (),
        {"choices": [type("MockChoice", (), {"message": type("MockMessage", (), {"content": "False, !@#$%"})})()]}
    )()
    is_english, translated_content = query_llm_robust("Buenos días a todos!")
    assert is_english == False
    assert translated_content == ""

@patch("src.translator.client.chat.completions.create")
def test_completely_unexpected_format(mock_create):
    mock_create.return_value = type(
        "MockResponse",
        (),
        {"choices": [type("MockChoice", (), {"message": type("MockMessage", (), {"content": "Unexpected format"})})()]}
    )()
    is_english, translated_content = query_llm_robust("こんにちは")
    assert is_english == False
    assert translated_content == ""

@patch("src.translator.client.chat.completions.create")
def test_api_call_exception(mock_create):
    mock_create.side_effect = Exception("Simulated API call error")
    is_english, translated_content = query_llm_robust("Bonjour tout le monde")
    assert is_english == False
    assert translated_content == ""