# server.py
# ================================================================
# Aura bespoke backend — corrected & ready to run
# - Async FastAPI + Motor (MongoDB)
# - JWT auth (PyJWT), bcrypt via passlib
# - Color analysis (PIL + numpy + sklearn KMeans)
# - Optional LLM/vision integration (guarded)
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
from jwt.exceptions import InvalidTokenError
import google.generativeai as genai
from motor.motor_asyncio import AsyncIOMotorClient
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# Optional emergentintegrations import — guarded
try:
    from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent  # optional
    HAVE_EMERGENT = True
except Exception:
    HAVE_EMERGENT = False

# -------------------------------
# Config & environment
# -------------------------------
ROOT_DIR = Path(__file__).parent.resolve()
load_dotenv(ROOT_DIR / ".env")

JWT_SECRET = os.environ.get("JWT_SECRET_KEY", "default_secret_key")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_EXP_MINUTES = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7))  # default 1 week
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")


# -------------------------------
# Security setup
# -------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# -------------------------------
# FastAPI app + router
# -------------------------------
app = FastAPI(title="Aura Backend")
api_router = APIRouter(prefix="/api")

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "aura_beauty")
client = None
db = None

async def connect_to_mongo():
    global client, db
    try:
        client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=2000)
        await client.server_info()  # Force connection test
        db = client[DB_NAME]
        print("✅ Connected to MongoDB at", MONGO_URL)
    except Exception as e:
        print("❌ MongoDB connection failed:", e)
        print("Retrying in 2 seconds…")
        await asyncio.sleep(2)
        await connect_to_mongo()  # auto retry

async def close_mongo():
    if client:
        client.close()
        print("🛑 MongoDB connection closed")

@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("aura-backend")
# -------------------------------
# DB client
# -------------------------------
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# -------------------------------
# Pydantic models
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
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    full_name: str
    skin_tone: Optional[str] = None
    hair_color: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserProfile(BaseModel):
    skin_tone: Optional[str] = None
    hair_color: Optional[str] = None

class Shade(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str
    rgb: dict  # {"r": int, "g": int, "b": int}
    lab: Optional[dict] = None
    hex_color: str
    source: str  # "camera","ai","manual"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ShadeCreate(BaseModel):
    name: str
    rgb: dict
    lab: Optional[dict] = None
    hex_color: str
    source: str = "manual"

class DeviceSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    device_id: str = "LipstickDispenser"
    status: str  # connected/disconnected/dispensing
    connected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_shade_dispensed: Optional[dict] = None

class ColorAnalysisResponse(BaseModel):
    dominant_color: dict
    lab_values: dict
    hex_color: str
    suggested_name: str

class AIShadeRequest(BaseModel):
    image_base64: str
    analysis_type: str  # e.g. "reference_look" / "event_look" / "default"

class DispenseRequest(BaseModel):
    shade_id: str

# -------------------------------
# Utility functions: color math
# -------------------------------
def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))

def rgb_to_lab(r, g, b):
    # sRGB to CIE-Lab conversion (approx)
    r_n, g_n, b_n = [x / 255.0 for x in (r, g, b)]

    def linearize(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

    r_l, g_l, b_l = [linearize(c) for c in (r_n, g_n, b_n)]
    x = r_l * 0.4124 + g_l * 0.3576 + b_l * 0.1805
    y = r_l * 0.2126 + g_l * 0.7152 + b_l * 0.0722
    z = r_l * 0.0193 + g_l * 0.1192 + b_l * 0.9505

    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t) + (16.0 / 116.0)

    l = max(0.0, (116.0 * f(y)) - 16.0)
    a = 500.0 * (f(x) - f(y))
    b_val = 200.0 * (f(y) - f(z))
    return {"l": round(l, 2), "a": round(a, 2), "b": round(b_val, 2)}

def extract_dominant_color(image_data: bytes):
    try:
        img = Image.open(BytesIO(image_data)).convert("RGB")
        img = img.resize((150, 150))
        pixels = np.array(img).reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        counts = np.bincount(kmeans.labels_)
        dominant = kmeans.cluster_centers_[counts.argmax()]
        r, g, b = [int(round(x)) for x in dominant]
        return {"r": r, "g": g, "b": b}
    except Exception as e:
        logger.exception("extract_dominant_color failed")
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

def generate_shade_name(rgb: dict) -> str:
    r, g, b = rgb["r"], rgb["g"], rgb["b"]
    # basic heuristics
    if r > 200 and g < 120 and b < 120:
        base = "Ruby"
    elif r > 150 and g < 100 and b < 100:
        base = "Crimson"
    elif r > 200 and g > 100 and b < 110:
        base = "Coral"
    elif r > 150 and g > 80 and b > 100:
        base = "Mauve"
    elif r < 80 and g < 60 and b < 60:
        base = "Deep Wine"
    else:
        base = "Custom"
    lightness = (r + g + b) / 3
    if lightness > 200:
        desc = "Light"
    elif lightness > 120:
        desc = "Medium"
    else:
        desc = "Deep"
    return f"{desc} {base}"

# -------------------------------
# Auth helpers
# -------------------------------
def hash_password(password: str) -> str:
    # Ensure password not longer than 72 bytes & encoded
    password = password.encode("utf-8")[:72]
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    plain = plain.encode("utf-8")[:72]   # required by bcrypt
    return pwd_context.verify(plain, hashed)

ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour

def create_access_token(data: dict) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = data.copy()
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        user_doc = await db.users.find_one({"id": user_id}, {"_id": 0})
        if not user_doc:
            raise HTTPException(status_code=401, detail="User not found")
        # ensure created_at is datetime
        if isinstance(user_doc.get("created_at"), str):
            user_doc["created_at"] = datetime.fromisoformat(user_doc["created_at"])
        # remove password_hash before building model
        user_doc.pop("password_hash", None)
        return User(**user_doc)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except (jwt.InvalidTokenError, InvalidTokenError) as e:
        raise HTTPException(status_code=401, detail="Invalid token")

# -------------------------------
# Routes
# -------------------------------
@api_router.get("/")
async def root():
    return {"message": "Aura Bespoke Beauty API"}

# Register
@api_router.post("/auth/register")
async def register(user_data: UserRegister):
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=user_data.email, full_name=user_data.full_name)
    user_dict = user.model_dump()
    user_dict["password_hash"] = hash_password(user_data.password)
    user_dict["created_at"] = user_dict["created_at"].isoformat()
    await db.users.insert_one(user_dict)
    token = create_access_token({"sub": user.id})
    # hide password_hash when returning
    user_resp = user.model_dump()
    return {"token": token, "user": user_resp}

# Login
@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    user_doc = await db.users.find_one({"email": credentials.email})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(credentials.password, user_doc.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user_doc["id"]})
    # sanitize user_doc
    user_doc.pop("password_hash", None)
    user_doc.pop("_id", None)
    if isinstance(user_doc.get("created_at"), str):
        user_doc["created_at"] = datetime.fromisoformat(user_doc["created_at"])
    return {"token": token, "user": User(**user_doc).model_dump()}

@api_router.get("/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

@api_router.put("/auth/profile")
async def update_profile(profile: UserProfile, current_user: User = Depends(get_current_user)):
    update = {}
    if profile.skin_tone is not None:
        update["skin_tone"] = profile.skin_tone
    if profile.hair_color is not None:
        update["hair_color"] = profile.hair_color
    if update:
        await db.users.update_one({"id": current_user.id}, {"$set": update})
    return {"message": "Profile updated successfully"}

# Shades
@api_router.get("/shades", response_model=List[Shade])
async def get_shades(current_user: User = Depends(get_current_user)):
    shades = await db.shades.find({"user_id": current_user.id}, {"_id": 0}).to_list(1000)
    # convert created_at strings to datetime if necessary
    for s in shades:
        if isinstance(s.get("created_at"), str):
            s["created_at"] = datetime.fromisoformat(s["created_at"])
    return shades

@api_router.post("/shades", response_model=Shade)
async def create_shade(shade_data: ShadeCreate, current_user: User = Depends(get_current_user)):
    shade = Shade(
        user_id=current_user.id,
        name=shade_data.name,
        rgb=shade_data.rgb,
        lab=shade_data.lab,
        hex_color=shade_data.hex_color,
        source=shade_data.source
    )
    shade_dict = shade.model_dump()
    shade_dict["created_at"] = shade_dict["created_at"].isoformat()
    await db.shades.insert_one(shade_dict)
    return shade

@api_router.delete("/shades/{shade_id}")
async def delete_shade(shade_id: str, current_user: User = Depends(get_current_user)):
    result = await db.shades.delete_one({"id": shade_id, "user_id": current_user.id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Shade not found")
    return {"message": "Shade deleted successfully"}

# Color analysis (image upload)
@api_router.post("/analyze/color", response_model=ColorAnalysisResponse)
async def analyze_color(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    contents = await file.read()
    rgb = extract_dominant_color(contents)
    lab = rgb_to_lab(rgb["r"], rgb["g"], rgb["b"])
    hex_color = rgb_to_hex(rgb["r"], rgb["g"], rgb["b"])
    suggested_name = generate_shade_name(rgb)
    return ColorAnalysisResponse(
        dominant_color=rgb,
        lab_values=lab,
        hex_color=hex_color,
        suggested_name=suggested_name
    )

@api_router.post("/analyze/ai-shade")
async def analyze_ai_shade(request: AIShadeRequest, current_user: User = Depends(get_current_user)):
    try:
        image_bytes = base64.b64decode(request.image_base64)

        # Gemini vision model
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Send request to Gemini
        prompt = (
            "You are an expert makeup artist and color scientist.\n"
            "Analyze the uploaded image and identify the exact lipstick color on the lips.\n"
            "Consider realistic cosmetic tones, lighting, skin tone, hair color and visual context.\n"
            "DO NOT describe the image.\n"
            "DO NOT add explanations.\n\n"
            "OUTPUT RULES (VERY IMPORTANT):\n"
            "Return ONLY the lipstick RGB value in strict format:\n"
            "R:### G:### B:###\n"
            "No other text, no sentences, no punctuation, no comments.\n"
            "Example output: R:201 G:60 B:85\n"
        )

        response = model.generate_content(
            [
                {"role":"user","parts":[
                prompt,
                {"mime_type": "image/jpeg", "data": image_bytes}
               ]}
            ]
        )

        # Extract RGB from Gemini response
        import re
        text = response.candidates[0].content.parts[0].text.strip()
        match = re.search(r"R[: ]?(\d+)\s*G[: ]?(\d+)\s*B[: ]?(\d+)", text)

        if match:
            rgb = {
                "r": int(match.group(1)),
                "g": int(match.group(2)),
                "b": int(match.group(3)),
            }
        else:
            # fallback dominant color
            rgb = extract_dominant_color(image_bytes)

        # Convert values
        lab = rgb_to_lab(rgb["r"], rgb["g"], rgb["b"])
        hex_color = rgb_to_hex(rgb["r"], rgb["g"], rgb["b"])
        suggested_name = generate_shade_name(rgb)

        return {
            "dominant_color": rgb,
            "lab_values": lab,
            "hex_color": hex_color,
            "suggested_name": suggested_name,
            "ai_description": text,
            "analysis_type": request.analysis_type
        }

    except Exception as e:
        logger.exception("AI shade analysis failed")
        raise HTTPException(status_code=500, detail=f"AI Shade processing error: {str(e)}")


# Device endpoints
@api_router.post("/device/connect")
async def connect_device(current_user: User = Depends(get_current_user)):
    existing = await db.device_sessions.find_one({"user_id": current_user.id, "status": "connected"})
    if existing:
        return {"message": "Device already connected", "device_id": existing["device_id"]}
    session = DeviceSession(user_id=current_user.id, status="connected")
    sess = session.model_dump()
    sess["connected_at"] = sess["connected_at"].isoformat()
    await db.device_sessions.insert_one(sess)
    return {"message": "Device connected successfully", "device_id": session.device_id}

@api_router.post("/device/disconnect")
async def disconnect_device(current_user: User = Depends(get_current_user)):
    await db.device_sessions.update_many({"user_id": current_user.id, "status": "connected"}, {"$set": {"status": "disconnected"}})
    return {"message": "Device disconnected successfully"}

@api_router.get("/device/status")
async def device_status(current_user: User = Depends(get_current_user)):
    session = await db.device_sessions.find_one({"user_id": current_user.id, "status": "connected"}, {"_id": 0})
    if not session:
        return {"connected": False}
    return {"connected": True, "device_id": session["device_id"], "status": session["status"]}

@api_router.post("/device/dispense")
async def dispense_shade(req: DispenseRequest, current_user: User = Depends(get_current_user)):
    session = await db.device_sessions.find_one({"user_id": current_user.id, "status": "connected"})
    if not session:
        raise HTTPException(status_code=400, detail="Device not connected")
    shade = await db.shades.find_one({"id": req.shade_id, "user_id": current_user.id}, {"_id": 0})
    if not shade:
        raise HTTPException(status_code=404, detail="Shade not found")
    # mark dispensing
    await db.device_sessions.update_one({"id": session["id"]}, {"$set": {"status": "dispensing", "last_shade_dispensed": shade}})
    # simulate
    await asyncio.sleep(1.5)
    await db.device_sessions.update_one({"id": session["id"]}, {"$set": {"status": "connected"}})
    mix = {
        "cyan": round(255 - shade["rgb"]["r"], 2),
        "magenta": round(255 - shade["rgb"]["g"], 2),
        "yellow": round(255 - shade["rgb"]["b"], 2),
        "black": round((255 - max(shade["rgb"]["r"], shade["rgb"]["g"], shade["rgb"]["b"])) * 0.3, 2)
    }
    return {"message": "Shade dispensed successfully", "shade": shade, "mix_formula": mix}

# Register router and run
app.include_router(api_router)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()