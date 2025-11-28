# server.py
# ================================================================
# Aura bespoke backend — corrected & ready to run
# - FastAPI + Motor (MongoDB)
# - JWT auth (PyJWT), bcrypt via passlib
# - Gemini 2.5 Flash image analysis
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
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException, Depends, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, EmailStr

from passlib.context import CryptContext
import jwt

import google.generativeai as genai

from motor.motor_asyncio import AsyncIOMotorClient
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from bson import ObjectId

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
        logger.info("✅ Connected to MongoDB")
    except Exception as e:
        logger.error("Mongo connection failed: %s", e)
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

class AIShadeRequest(BaseModel):
    image_base64: str
    analysis_type: str  # "reference_look" / "event_look"

# -------------------------------
# Utility Functions (color & security)
# -------------------------------
def rgb_to_hex(r, g, b):
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

def rgb_to_lab(r, g, b):
    r_n, g_n, b_n = [x / 255.0 for x in (r, g, b)]
    def linearize(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
    r_l, g_l, b_l = [linearize(c) for c in (r_n, g_n, b_n)]
    x = r_l * 0.4124 + g_l * 0.3576 + b_l * 0.1805
    y = r_l * 0.2126 + g_l * 0.7152 + b_l * 0.0722
    z = r_l * 0.0193 + g_l * 0.1192 + b_l * 0.9505
    x /= 0.95047; z /= 1.08883
    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t) + (16.0 / 116.0)
    L = max(0.0, (116.0 * f(y)) - 16.0)
    a = 500.0 * (f(x) - f(y))
    b_val = 200.0 * (f(y) - f(z))
    return {"l": round(L, 2), "a": round(a, 2), "b": round(b_val, 2)}

def extract_dominant_color(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((150, 150))
    pixels = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels)
    dominant = kmeans.cluster_centers_[counts.argmax()]
    r, g, b = [int(round(x)) for x in dominant]
    return {"r": r, "g": g, "b": b}

def hash_pw(password: str) -> str:
    pw_bytes = password.encode("utf-8")[:72]
    safe_pw = pw_bytes.decode("utf-8", errors="ignore")
    return pwd_context.hash(safe_pw)

def verify_pw(plain: str, hashed: str) -> bool:
    pw_bytes = plain.encode("utf-8")[:72]
    safe_pw = pw_bytes.decode("utf-8", errors="ignore")
    return pwd_context.verify(safe_pw, hashed)

def create_token(data: dict) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = data.copy()
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    token = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

# -------------------------------
# Auth helper: get current user (safe)
# -------------------------------
async def get_user(auth: HTTPAuthorizationCredentials = Depends(security)) -> User:
    token = auth.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        if db is None:
            raise HTTPException(status_code=500, detail="Database not initialized")
        user_doc = await db.users.find_one({"id": user_id})
        if not user_doc:
            raise HTTPException(status_code=401, detail="User not found")

        # Remove Mongo's _id if present (ObjectId not JSON serializable)
        user_doc.pop("_id", None)

        # Normalize created_at
        if isinstance(user_doc.get("created_at"), str):
            try:
                user_doc["created_at"] = datetime.fromisoformat(user_doc["created_at"])
            except Exception:
                user_doc["created_at"] = datetime.utcnow()

        user_doc.pop("password_hash", None)
        return User(**user_doc)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception as e:
        logger.debug("get_user failure: %s", e)
        raise HTTPException(status_code=401, detail="Invalid token")

# -------------------------------
# Auth Routes
# -------------------------------
@api_router.post("/auth/register")
async def register(user_data: UserRegister):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")

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

    # Return safe user object (no password_hash, no _id)
    safe_user = {k: v for k, v in doc.items() if k != "password_hash"}
    return {"token": token, "user": safe_user}

@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    user_doc = await db.users.find_one({"email": credentials.email})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_pw(credentials.password, user_doc.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Remove sensitive / non-serializable fields
    user_doc.pop("password_hash", None)
    user_doc.pop("_id", None)

    token = create_token({"sub": user_doc["id"]})

    # convert created_at string to datetime for returned model if needed
    if isinstance(user_doc.get("created_at"), str):
        try:
            user_doc["created_at"] = datetime.fromisoformat(user_doc["created_at"])
        except Exception:
            user_doc["created_at"] = datetime.utcnow()

    return {"token": token, "user": user_doc}

@api_router.get("/auth/me")
async def get_me(current_user: User = Depends(get_user)):
    return current_user

@api_router.put("/auth/profile")
async def update_profile(profile: UserProfile, current_user: User = Depends(get_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    update = {}
    if profile.skin_tone is not None:
        update["skin_tone"] = profile.skin_tone
    if profile.hair_color is not None:
        update["hair_color"] = profile.hair_color
    if update:
        await db.users.update_one({"id": current_user.id}, {"$set": update})
    return {"message": "Profile updated successfully"}

# -------------------------------
# AI: Gemini image call (defensive)
# -------------------------------
def gemini_image_call(model_name: str, prompt: str, image_bytes: bytes):
    try:
        model = getattr(genai, "GenerativeModel")(model_name)
        response = model.generate_content(
            [
                {"mime_type": "image/jpeg", "data": image_bytes},
                prompt
            ]
        )
        # Try a few ways to extract text safely
        if hasattr(response, "text"):
            return response.text
        if hasattr(response, "candidates") and response.candidates:
            try:
                return response.candidates[0].content.parts[0].text
            except Exception:
                try:
                    return getattr(response.candidates[0], "text", str(response))
                except Exception:
                    return str(response)
        return str(response)
    except Exception as e:
        logger.warning("Gemini image call failed: %s", e)
        raise

# -------------------------------
# AI: Reference Look (RGB extraction)
# -------------------------------
async def run_reference_look(image_bytes: bytes):
    prompt = """
Analyze the image. If a face with lips is present → return ONLY the lipstick RGB worn on lips.
If no face/lips → return the dominant lipstick/swatch RGB.

Return JSON only like:
{ "r": 123, "g": 45, "b": 67 }
"""
    try:
        text = gemini_image_call("gemini-2.5-flash", prompt, image_bytes)
        data = json.loads(text)
        return data
    except Exception as e:
        logger.info("Gemini reference_look failed, falling back to local extraction: %s", e)
        return extract_dominant_color(image_bytes)

# -------------------------------
# AI: Event Look (face + outfit + 3 shades)
# -------------------------------
async def run_event_look(image_bytes: bytes):
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
  "best_shades": ["Shade1","Shade2","Shade3"]
}
"""
    text = gemini_image_call("gemini-2.5-flash", prompt, image_bytes)
    try:
        return json.loads(text)
    except Exception as e:
        logger.info("Gemini event_look parse failed: %s", e)
        raise HTTPException(status_code=502, detail="AI analysis failed")

# -------------------------------
# AI Endpoint
# -------------------------------
@api_router.post("/analyze/ai-shade")
async def ai_shade(req: AIShadeRequest, current_user: User = Depends(get_user)):
    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")

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
        return {"analysis": data, "source": "gemini"}
    else:
        raise HTTPException(status_code=400, detail="Invalid analysis_type")

# -------------------------------
# Register router
# -------------------------------
app.include_router(api_router)