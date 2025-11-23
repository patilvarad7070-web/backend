from dotenv import load_dotenv
import os

print("Loading .env...")
load_dotenv()

print("GOOGLE_API_KEY =", os.getenv("GOOGLE_API_KEY"))
print("SECRET_KEY =", os.getenv("SECRET_KEY"))
