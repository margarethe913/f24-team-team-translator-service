import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("LLM_KEY"),  # Fetch API key from environment
    api_version="2024-02-15-preview",
    azure_endpoint="https://p4-team-team.openai.azure.com/"
)

def query_llm_robust(post: str) -> tuple[bool, str]:
    context = (
        "You're an expert translator proficient in identifying and translating various languages. "
        "If the text provided is in English, respond with True, \"<text>\". "
        "If it's in another language, translate it into English and respond with False, \"<translation>\". "
        "If the text is unintelligible or malformed, respond with False, \"\"."
    )

    try:
      response = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[
              {
                  "role": "system",
                  "content": context
              },
              {
                  "role": "user",
                  "content": post
              }
          ]
      )

      response_text = response.choices[0].message.content.strip()

      if "," not in response_text:
            print("Warning: Unexpected response format from LLM. Missing comma separator.")
            return (False, "")

      first_part, _, second_part = response_text.partition(",")

      if first_part.strip() == "True":
          english_text = second_part.strip()[1:-1]  # Assuming quotes are included in the response
          if isinstance(english_text, str) and any(char.isalpha() for char in english_text):
              return (True, english_text)
          else:
              print("Warning: Expected string content for English text but got a different type or invalid content.")
              return (True, "") 

      elif first_part.strip() == "False":
          translation = second_part.strip()[1:-1]  # Assuming quotes are included in the response
          if isinstance(translation, str) and any(char.isalpha() for char in translation):
              return (False, translation)
          else:
              print("Warning: Expected string content for translation but got a different type or invalid content.")
              return (False, "") 
      else:
          print("Warning: Unexpected response format from LLM.")
          return (False, "")

    except Exception as e:
        print(f"Error in query_llm_robust: {e}. Defaulting to safe output.")
        return (False, "")