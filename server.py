# server.py
# ================================================================
# Aura bespoke backend — FINAL PRODUCTION VERSION
# - Async FastAPI + Motor (MongoDB)
# - JWT auth (PyJWT), bcrypt via passlib
# - Gemini 2.5 Flash for AI + image analysis
# - Color science (RGB → HEX → LAB)
# ================================================================

import os
import logging
import uuid
import base64
import json
import asyncio
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException, Depends, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, EmailStr

from passlib.context import CryptContext
import jwt

import google.generativeai as genai

from motor.motor_asyncio import AsyncIOMotorClient
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# -------------------------------
# Load env + Config
# -------------------------------
ROOT_DIR = Path(__file__).parent.resolve()
load_dotenv(ROOT_DIR / ".env")

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ Missing Gemini API key")

genai.configure(api_key=GEMINI_API_KEY)

JWT_SECRET = os.environ.get("JWT_SECRET_KEY", "default_secret_key")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7))

CORS_ORIGINS = [
    "https://aura-beauty-boutique.com",
    "https://www.aura-beauty-boutique.com",
    "https://app.aura-beauty-boutique.com",
    "https://api.aura-beauty-boutique.com",
]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

app = FastAPI(title="Aura Backend")
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aura-backend")

# -------------------------------
# MongoDB
# -------------------------------
client = None
db = None

async def connect_to_mongo():
    global client, db
    try:
        client = AsyncIOMotorClient(os.getenv("MONGO_URL"), serverSelectionTimeoutMS=3000)
        await client.server_info()
        db = client[os.getenv("DB_NAME")]
        logger.info(f"✅ Connected to MongoDB")
    except Exception as e:
        logger.error(f"Mongo connection failed: {e}")
        await asyncio.sleep(2)
        await connect_to_mongo()

async def close_mongo():
    global client
    if client:
        client.close()
        logger.info("🛑 MongoDB connection closed")

@app.on_event("startup")
async def on_startup():
    await connect_to_mongo()

@app.on_event("shutdown")
async def on_shutdown():
    await close_mongo()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# -------------------------------
# Models
# -------------------------------
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    email: str
    full_name: str
    skin_tone: Optional[str] = None
    hair_color: Optional[str] = None
    created_at: datetime

class UserProfile(BaseModel):
    skin_tone: Optional[str] = None
    hair_color: Optional[str] = None

class Shade(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    user_id: str
    name: str
    rgb: dict
    lab: Optional[dict]
    hex_color: str
    source: str
    created_at: datetime

class ShadeCreate(BaseModel):
    name: str
    rgb: dict
    lab: Optional[dict]
    hex_color: str
    source: str = "manual"

class AIShadeRequest(BaseModel):
    image_base64: str
    analysis_type: str  # "reference_look" / "event_look"

# -------------------------------
# Utility Functions
# -------------------------------
def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"

def rgb_to_lab(r, g, b):
    r, g, b = [x / 255 for x in (r, g, b)]
    def lin(c): return ((c + 0.055)/1.055)**2.4 if c > 0.04045 else c/12.92
    r, g, b = lin(r), lin(g), lin(b)
    x = r*0.4124 + g*0.3576 + b*0.1805
    y = r*0.2126 + g*0.7152 + b*0.0722
    z = r*0.0193 + g*0.1192 + b*0.9505
    x /= 0.95047; z /= 1.08883
    def f(t): return t**(1/3) if t > 0.008856 else 7.787*t + 16/116
    L = 116*f(y) - 16
    a = 500*(f(x)-f(y))
    b = 200*(f(y)-f(z))
    return {"l": round(L, 2), "a": round(a, 2), "b": round(b, 2)}

def extract_dominant_color(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((150, 150))
    pixels = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=4, n_init=10)
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels)
    dom = kmeans.cluster_centers_[counts.argmax()]
    r, g, b = map(int, dom)
    return {"r": r, "g": g, "b": b}

def hash_pw(p): 
    p = p.encode()[:72].decode(errors="ignore")
    return pwd_context.hash(p)

def verify_pw(p, h):
    p = p.encode()[:72].decode(errors="ignore")
    return pwd_context.verify(p, h)

def create_token(data):
    exp = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    data.update({"exp": exp})
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_user(auth: HTTPAuthorizationCredentials = Depends(security)):
    token = auth.credentials
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        uid = data.get("sub")
        u = await db.users.find_one({"id": uid}, {"_id": 0})
        if not u:
            raise HTTPException(401, "User not found")
        u["created_at"] = datetime.fromisoformat(u["created_at"])
        return User(**u)
    except Exception:
        raise HTTPException(401, "Invalid token")

# -------------------------------
# Auth Routes
# -------------------------------
@api_router.post("/auth/register")
async def register(user_data: UserRegister):
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(400, "Email already exists")

    user_id = str(uuid.uuid4())
    hashed = hash_pw(user_data.password)

    doc = {
        "id": user_id,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "password_hash": hashed,
        "created_at": datetime.utcnow().isoformat()
    }

    await db.users.insert_one(doc)
    token = create_token({"sub": user_id})

    return {"token": token, "user": doc}


@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email})
    if not user:
        raise HTTPException(401, "Invalid credentials")

    if not verify_pw(credentials.password, user["password_hash"]):
        raise HTTPException(401, "Invalid credentials")

    token = create_token({"sub": user["id"]})
    user.pop("password_hash", None)
    return {"token": token, "user": user}


@api_router.get("/auth/me")
async def me(user: User = Depends(get_user)):
    return user


@api_router.put("/auth/profile")
async def update_profile(profile: UserProfile, user: User = Depends(get_user)):
    update = {}
    if profile.skin_tone: update["skin_tone"] = profile.skin_tone
    if profile.hair_color: update["hair_color"] = profile.hair_color

    if update:
        await db.users.update_one({"id": user.id}, {"$set": update})

    return {"message": "Profile updated"}


# -------------------------------
# AI HANDLERS — LATEST GEMINI 2.5 FLASH
# -------------------------------

def gemini_image_call(model_name, prompt, image_bytes):
    """Correct Gemini 2.5 Flash image input format."""
    model = genai.GenerativeModel(model_name)

    response = model.generate_content(
        [
            {"mime_type": "image/jpeg", "data": image_bytes},
            prompt
        ]
    )

    return response.text

# -------------------------------
# AI: Reference Look (RGB Extraction)
# -------------------------------
async def run_reference_look(image_bytes):
    prompt = """
Analyze the image. 
If a face with lips is present → return ONLY the lipstick RGB worn on lips.
If no face/lips → return the dominant lipstick/swatch RGB.

Return JSON only:
{
  "r": num,
  "g": num,
  "b": num
}
"""
    try:
        text = gemini_image_call("gemini-2.5-flash", prompt, image_bytes)
        data = json.loads(text)
        return data
    except Exception:
        return extract_dominant_color(image_bytes)

# -------------------------------
# AI: Event Look (Face + Outfit + Shade Suggestions)
# -------------------------------
async def run_event_look(image_bytes):
    prompt = """
Analyze the person AND their outfit from the same image.
Detect:
- skin_tone
- undertone
- hair_color
- eye_color
- outfit_color

Then based on BOTH face + outfit, recommend EXACTLY 3 lipstick shades.

Return strict JSON:
{
  "skin_tone": "",
  "undertone": "",
  "hair_color": "",
  "eye_color": "",
  "outfit_color": "",
  "best_shades": ["", "", ""]
}
"""
    text = gemini_image_call("gemini-2.5-flash", prompt, image_bytes)
    return json.loads(text)

# -------------------------------
# AI Endpoint
# -------------------------------
@api_router.post("/analyze/ai-shade")
async def ai_shade(req: AIShadeRequest, user: User = Depends(get_user)):
    image_bytes = base64.b64decode(req.image_base64)

    if req.analysis_type == "reference_look":
        rgb = await run_reference_look(image_bytes)
        lab = rgb_to_lab(rgb["r"], rgb["g"], rgb["b"])
        hex_color = rgb_to_hex(rgb["r"], rgb["g"], rgb["b"])
        return {
            "rgb": rgb,
            "lab_values": lab,
            "hex_color": hex_color,
            "source": "gemini"
        }

    elif req.analysis_type == "event_look":
        data = await run_event_look(image_bytes)
        return {
            "analysis": data,
            "source": "gemini"
        }

    else:
        raise HTTPException(400, "Invalid analysis_type")

# -------------------------------
# Register Router
# -------------------------------
app.include_router(api_router)