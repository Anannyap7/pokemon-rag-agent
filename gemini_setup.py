'''Tested and working!'''

from google import genai
import os

client = genai.Client(api_key = "AIzaSyBoHHAlJ6su_Qb9-2hqv4mOxiLjbIQ5qxk")
# Or use .env files

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works in a 100 words"
)

print(response.text)