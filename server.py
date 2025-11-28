# service.py
# ================================================================
# Aura bespoke backend — unified final service
# - FastAPI + Motor (MongoDB)
# - JWT auth (PyJWT), passlib bcrypt
# - Gemini image analysis with robust fallback to local extraction
# - Color analysis (PIL + numpy + sklearn KMeans)
# - Shades + Device endpoints (compatible with frontend)
# ================================================================

import os
import uuid
import base64
import json
import asyncio
import logging
from io import BytesIO
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Any

from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException, Depends, File, UploadFile, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from passlib.context import CryptContext
import jwt

from motor.motor_asyncio import AsyncIOMotorClient
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# Optional: google generative ai (Gemini)
try:
    import google.generativeai as genai
    HAVE_GENAI = True
except Exception:
    HAVE_GENAI = False

# -------------------------------
# Load env
# -------------------------------
ROOT_DIR = Path(__file__).parent.resolve()
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "aura_beauty")
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "default_secret_key")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXP_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7))
CORS_ORIGINS = [
    "https://aura-beauty-boutique.com",
    "https://www.aura-beauty-boutique.com",
    "https://app.aura-beauty-boutique.com",
    "https://api.aura-beauty-boutique.com",
]

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if HAVE_GENAI and not GEMINI_API_KEY:
    # Do not crash if missing; we will fallback to local extraction when called
    logging.warning("GENAI present but GEMINI API key not configured. AI calls will failover to local extraction.")
if HAVE_GENAI and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# -------------------------------
# App setup
# -------------------------------
app = FastAPI(title="Aura Backend")
api = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aura-backend")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# -------------------------------
# Database
# -------------------------------
client: Optional[AsyncIOMotorClient] = None
db = None

async def connect_to_mongo():
    global client, db
    try:
        client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=3000)
        await client.server_info()
        db = client[DB_NAME]
        logger.info("✅ Connected to MongoDB")
    except Exception as e:
        logger.error("Mongo connection failed: %s", e)
        # retry with backoff
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
    id: str
    email: str
    full_name: str
    skin_tone: Optional[str] = None
    hair_color: Optional[str] = None
    created_at: datetime

class UserProfile(BaseModel):
    skin_tone: Optional[str] = None
    hair_color: Optional[str] = None

class ShadeCreate(BaseModel):
    name: str
    rgb: dict
    lab: Optional[dict] = None
    hex_color: str
    source: str = "manual"

class AIShadeRequest(BaseModel):
    image_base64: Optional[str] = None
    analysis_type: str  # "reference_look" | "event_look"

# -------------------------------
# Utilities: security + tokens
# -------------------------------
def hash_password(password: str) -> str:
    # limit to 72 bytes for bcrypt safety
    pw = password.encode("utf-8")[:72].decode("utf-8", errors="ignore")
    return pwd_context.hash(pw)

def verify_password(plain: str, hashed: str) -> bool:
    pw = plain.encode("utf-8")[:72].decode("utf-8", errors="ignore")
    return pwd_context.verify(pw, hashed)

def create_token(data: dict) -> str:
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXP_MINUTES)
    payload = data.copy()
    payload.update({"exp": int(expire.timestamp()), "iat": int(datetime.utcnow().timestamp())})
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

async def get_current_user(auth: HTTPAuthorizationCredentials = Depends(security)) -> User:
    token = auth.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        if db is None:
            raise HTTPException(status_code=500, detail="Database not initialized")
        user_doc = await db.users.find_one({"id": user_id})
        if not user_doc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        user_doc.pop("_id", None)
        # normalize created_at
        if isinstance(user_doc.get("created_at"), str):
            try:
                user_doc["created_at"] = datetime.fromisoformat(user_doc["created_at"])
            except Exception:
                user_doc["created_at"] = datetime.utcnow()
        user_doc.pop("password_hash", None)
        return User(**user_doc)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except Exception as e:
        logger.debug("get_current_user failed: %s", e)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# -------------------------------
# Color utils
# -------------------------------
def rgb_to_hex(r, g, b) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))

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

def extract_dominant_color(image_bytes: bytes) -> dict:
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        # small guard: if image too small, resize up
        w, h = img.size
        if w < 50 or h < 50:
            img = img.resize((max(150, w*3), max(150, h*3)))
        img = img.resize((150, 150))
        pixels = np.array(img).reshape(-1, 3)
        # if too few pixels, just average
        if pixels.shape[0] < 10:
            avg = pixels.mean(axis=0)
            r, g, b = [int(round(x)) for x in avg]
            return {"r": r, "g": g, "b": b}
        k = min(5, len(pixels))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        counts = np.bincount(labels)
        dominant = kmeans.cluster_centers_[counts.argmax()]
        r, g, b = [int(round(x)) for x in dominant]
        return {"r": r, "g": g, "b": b}
    except Exception as e:
        logger.exception("extract_dominant_color failed")
        # fallback to a neutral gray
        return {"r": 128, "g": 128, "b": 128}

# -------------------------------
# Gemini helper (defensive)
# -------------------------------
def gemini_image_call(model_name: str, prompt: str, image_bytes: bytes) -> str:
    """
    Return string result from Gemini model. Raises Exception on hard fail.
    This function is defensive: it attempts to extract text from various shapes of response.
    """
    if not HAVE_GENAI:
        raise RuntimeError("Generative AI SDK not installed")
    try:
        model = getattr(genai, "GenerativeModel")(model_name)
        response = model.generate_content(
            [
                {"mime_type": "image/jpeg", "data": image_bytes},
                prompt
            ]
        )
        # try multiple extraction strategies
        if hasattr(response, "text"):
            return response.text
        if hasattr(response, "candidates") and response.candidates:
            try:
                part = response.candidates[0]
                # many SDK variants: try .content.parts[0].text
                if hasattr(part, "content") and hasattr(part.content, "parts"):
                    return part.content.parts[0].text
                if hasattr(part, "text"):
                    return part.text
                return str(part)
            except Exception:
                return str(response)
        return str(response)
    except Exception as e:
        logger.warning("Gemini image call failed: %s", e)
        raise

# -------------------------------
# AI analysis workflows
# -------------------------------
async def run_reference_look(image_bytes: bytes) -> dict:
    """
    Try Gemini first (if available), otherwise fallback to local dominant color.
    Expect JSON: { "r": int, "g": int, "b": int }
    """
    prompt = """
You are an image analyzer. If the image contains a face and lips, return ONLY the RGB color of the lipstick on the lips.
If no face/lips present, return the dominant lipstick/swatch RGB.
Return JSON only in the exact form:
{ "r": 123, "g": 45, "b": 67 }
"""
    # attempt Gemini if possible
    if HAVE_GENAI and GEMINI_API_KEY:
        try:
            text = gemini_image_call("gemini-2.5-flash", prompt, image_bytes)
            # try to parse JSON strictly
            data = json.loads(text)
            if all(k in data for k in ("r", "g", "b")):
                return {"r": int(data["r"]), "g": int(data["g"]), "b": int(data["b"])}
            # try to extract digits with fallback
            import re
            m = re.search(r"(\d{1,3}).*?(\d{1,3}).*?(\d{1,3})", text)
            if m:
                return {"r": int(m.group(1)), "g": int(m.group(2)), "b": int(m.group(3))}
            logger.info("Gemini returned unexpected data for reference_look, falling back")
        except Exception as e:
            logger.info("Gemini reference_look failed, falling back to local extraction: %s", e)
    # local fallback
    return extract_dominant_color(image_bytes)

async def run_event_look(image_bytes: bytes) -> dict:
    """
    Try Gemini to extract detailed person+outfit attributes and 3 shade recommendations.
    Expect strict JSON containing fields listed in prompt.
    """
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
    if HAVE_GENAI and GEMINI_API_KEY:
        try:
            text = gemini_image_call("gemini-2.5-flash", prompt, image_bytes)
            data = json.loads(text)
            return data
        except Exception as e:
            logger.info("Gemini event_look failed or returned unparsable JSON, falling back: %s", e)
            # we still attempt a graceful fallback with limited info
    # fallback: limited local response
    dom = extract_dominant_color(image_bytes)
    hexc = rgb_to_hex(dom["r"], dom["g"], dom["b"])
    return {
        "skin_tone": "unknown",
        "undertone": "unknown",
        "hair_color": "unknown",
        "eye_color": "unknown",
        "outfit_color": hexc,
        "best_shades": [hexc, hexc, hexc]
    }

# -------------------------------
# API: Auth
# -------------------------------
@api.post("/auth/register")
async def register(user: UserRegister):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    existing = await db.users.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")
    user_id = str(uuid.uuid4())
    doc = {
        "id": user_id,
        "email": user.email,
        "full_name": user.full_name,
        "password_hash": hash_password(user.password),
        "created_at": datetime.utcnow().isoformat()
    }
    await db.users.insert_one(doc)
    token = create_token({"sub": user_id})
    safe = {k: v for k, v in doc.items() if k != "password_hash"}
    return {"token": token, "user": safe}

@api.post("/auth/login")
async def login(credentials: UserLogin):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    user_doc = await db.users.find_one({"email": credentials.email})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(credentials.password, user_doc.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user_doc.pop("password_hash", None)
    user_doc.pop("_id", None)
    if isinstance(user_doc.get("created_at"), str):
        try:
            user_doc["created_at"] = datetime.fromisoformat(user_doc["created_at"])
        except Exception:
            user_doc["created_at"] = datetime.utcnow()
    token = create_token({"sub": user_doc["id"]})
    return {"token": token, "user": user_doc}

@api.get("/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

@api.put("/auth/profile")
async def update_profile(profile: UserProfile, current_user: User = Depends(get_current_user)):
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
# Shades endpoints
# -------------------------------
@api.get("/shades")
async def get_shades(current_user: User = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    shades = await db.shades.find({"user_id": current_user.id}, {"_id": 0}).to_list(length=1000)
    # normalize created_at
    for s in shades:
        if isinstance(s.get("created_at"), str):
            try:
                s["created_at"] = datetime.fromisoformat(s["created_at"])
            except Exception:
                s["created_at"] = datetime.utcnow()
    return {"count": len(shades), "shades": shades}

@api.post("/shades")
async def create_shade(shade: ShadeCreate, current_user: User = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    shade_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    doc = {
        "id": shade_id,
        "user_id": current_user.id,
        "name": shade.name,
        "rgb": shade.rgb,
        "lab": shade.lab,
        "hex_color": shade.hex_color,
        "source": shade.source,
        "created_at": now
    }
    await db.shades.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api.delete("/shades/{shade_id}")
async def delete_shade(shade_id: str, current_user: User = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    res = await db.shades.delete_one({"id": shade_id, "user_id": current_user.id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Shade not found")
    return {"message": "Shade deleted successfully"}

# -------------------------------
# Devices endpoints
# -------------------------------
@api.post("/device/connect")
async def connect_device(current_user: User = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    existing = await db.device_sessions.find_one({"user_id": current_user.id, "status": "connected"})
    if existing:
        return {"message": "Device already connected", "device_id": existing["device_id"]}
    session_id = str(uuid.uuid4())
    doc = {
        "id": session_id,
        "user_id": current_user.id,
        "device_id": "LipstickDispenser",
        "status": "connected",
        "connected_at": datetime.utcnow().isoformat(),
        "last_shade_dispensed": None
    }
    await db.device_sessions.insert_one(doc)
    return {"message": "Device connected successfully", "device_id": doc["device_id"]}

@api.post("/device/disconnect")
async def disconnect_device(current_user: User = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    await db.device_sessions.update_many({"user_id": current_user.id, "status": "connected"}, {"$set": {"status": "disconnected"}})
    return {"message": "Device disconnected successfully"}

@api.get("/device/status")
async def device_status(current_user: User = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    session = await db.device_sessions.find_one({"user_id": current_user.id, "status": "connected"}, {"_id": 0})
    if not session:
        return {"connected": False}
    return {"connected": True, "device_id": session["device_id"], "status": session["status"]}

@api.post("/device/dispense")
async def dispense_shade(payload: dict, current_user: User = Depends(get_current_user)):
    """
    Expects JSON body { "shade_id": "<id>" }
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    shade_id = payload.get("shade_id")
    if not shade_id:
        raise HTTPException(status_code=400, detail="Missing shade_id")
    session = await db.device_sessions.find_one({"user_id": current_user.id, "status": "connected"})
    if not session:
        raise HTTPException(status_code=400, detail="Device not connected")
    shade = await db.shades.find_one({"id": shade_id, "user_id": current_user.id}, {"_id": 0})
    if not shade:
        raise HTTPException(status_code=404, detail="Shade not found")
    await db.device_sessions.update_one({"id": session["id"]}, {"$set": {"status": "dispensing", "last_shade_dispensed": shade}})
    # simulate dispensing
    await asyncio.sleep(1.2)
    await db.device_sessions.update_one({"id": session["id"]}, {"$set": {"status": "connected"}})
    mix = {
        "cyan": round(255 - shade["rgb"]["r"], 2),
        "magenta": round(255 - shade["rgb"]["g"], 2),
        "yellow": round(255 - shade["rgb"]["b"], 2),
        "black": round((255 - max(shade["rgb"]["r"], shade["rgb"]["g"], shade["rgb"]["b"])) * 0.3, 2)
    }
    return {"message": "Shade dispensed successfully", "shade": shade, "mix_formula": mix}

# -------------------------------
# Color analyze: supports UploadFile (multipart/form-data) or JSON {image_base64}
# -------------------------------
@api.post("/analyze/color")
async def analyze_color(request: Request, file: Optional[UploadFile] = File(None), current_user: User = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")

    image_bytes = None
    # If file provided (multipart)
    if file is not None:
        try:
            image_bytes = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed reading file: {str(e)}")
    else:
        # try JSON body
        try:
            body = await request.json()
            img_b64 = body.get("image_base64") or body.get("image")
            if not img_b64:
                raise HTTPException(status_code=400, detail="No image provided")
            image_bytes = base64.b64decode(img_b64)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request body or base64 image: {str(e)}")

    rgb = extract_dominant_color(image_bytes)
    lab = rgb_to_lab(rgb["r"], rgb["g"], rgb["b"])
    hex_color = rgb_to_hex(rgb["r"], rgb["g"], rgb["b"])

    return {
        "rgb": rgb,
        "lab_values": lab,
        "hex_color": hex_color,
        "source": "local"
    }

# -------------------------------
# AI Shade analyze: expects JSON { image_base64, analysis_type }
# -------------------------------
@api.post("/analyze/ai-shade")
async def analyze_ai_shade(req: AIShadeRequest, current_user: User = Depends(get_current_user)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    if not req.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

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
            "source": "ai" if HAVE_GENAI and GEMINI_API_KEY else "local"
        }

    elif req.analysis_type == "event_look":
        data = await run_event_look(image_bytes)
        return {"analysis": data, "source": "ai" if HAVE_GENAI and GEMINI_API_KEY else "local"}

    else:
        raise HTTPException(status_code=400, detail="Invalid analysis_type")

# -------------------------------
# Liveness root (optional health)
# -------------------------------
@api.get("/")
async def root():
    return {"message": "Aura Backend is running"}

# Attach router
app.include_router(api)

# Clean shutdown handler
@app.on_event("shutdown")
async def shutdown_hook():
    if client:
        client.close()