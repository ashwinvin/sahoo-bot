import dspy
import os

LLM_API_KEY = os.getenv("GEMINI_KEY")
model_api = dspy.LM("gemini/gemini-2.5-flash", api_key=LLM_API_KEY, max_tokens=8000)
dspy.settings.configure(lm=model_api)
