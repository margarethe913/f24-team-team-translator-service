from src.translator import translate_content
from unittest.mock import patch
import openai


def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message"

def test_llm_normal_response():
    is_english, translated_content = translate_content("Der frühe Vogel fängt den Wurm.")
    assert is_english == False
    assert translated_content == "The early bird catches the worm."

@patch.object(openai.Completion, 'create')
def test_llm_gibberish_response(mocker):
    mocker.return_value.choices[0].message.content = "Unexpected format"
    
    is_english, translated_content = translate_content("asdkjfhasd kjhasd kjashdfk jhasdkjfh")
    
    assert is_english == False
    assert translated_content == ""
    
# Test for a valid English response
@patch.object(openai.Completion, 'create')
def test_valid_english_response(mocker):
    mocker.return_value.choices[0].message.content = "True, 'Hello everyone!'"
    is_english, translated_content = translate_content("Hello everyone!")
    assert is_english == True
    assert translated_content ==  "Hello everyone!"

# Test for a valid translation response
@patch.object(openai.Completion, 'create')
def test_valid_translation_response(mocker):
    mocker.return_value.choices[0].message.content = "False, 'Good morning to all'"
    is_english, translated_content = translate_content("Buenos días a todos")
    assert is_english == False
    assert translated_content ==  "Good morning to all"

# Test for a response in a malformed format (no tuple structure)
@patch.object(openai.Completion, 'create')
def test_malformed_no_tuple(mocker):
    mocker.return_value.choices[0].message.content = "This response has no tuple format."
    is_english, translated_content = translate_content("Bonjour tout le monde")
    assert is_english == False
    assert translated_content ==  ""

# Test for a response with non-string content in the English section
@patch.object(openai.Completion, 'create')
def test_non_string_english_content(mocker):
    mocker.return_value.choices[0].message.content = "True, 12@@%3"
    is_english, translated_content = translate_content("Hello!")
    assert is_english == True
    assert translated_content ==  ""

# Test for a response with non-string content in the translation section
@patch.object(openai.Completion, 'create')
def test_non_string_translation_content(mocker):
    mocker.return_value.choices[0].message.content = "False, !@#$%"
    is_english, translated_content = translate_content("Buenos días a todos!")
    assert is_english == False
    assert translated_content ==  ""

# Test for an unexpected format that does not start with (True, or (False,
@patch.object(openai.Completion, 'create')
def test_completely_unexpected_format(mocker):
    mocker.return_value.choices[0].message.content = "Unexpected format"
    is_english, translated_content = translate_content("こんにちは")
    assert is_english == False
    assert translated_content ==  ""


# Test for exception handling when an error occurs in the API call
@patch.object(openai.Completion, 'create')
def test_api_call_exception(mocker):
    mocker.side_effect = Exception("Simulated API call error")
    is_english, translated_content = translate_content("Bonjour tout le monde")
    assert is_english == False
    assert translated_content ==  ""
