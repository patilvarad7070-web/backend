# ================================================================
# Aura bespoke backend — unified final service (2025-11-28)
# - FastAPI + Motor (MongoDB)
# - JWT auth (PyJWT), passlib bcrypt
# - Gemini image analysis with fallback to local extraction
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, HTTPException, Depends, File, UploadFile, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, ConfigDict
from passlib.context import CryptContext
import jwt

from motor.motor_asyncio import AsyncIOMotorClient
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# Optional: google generative ai
try:
    import google.generativeai as genai
    HAVE_GENAI = True
except Exception:
    HAVE_GENAI = False

# -------------------------------
# Load ENV
# -------------------------------
ROOT_DIR = Path(__file__).parent.resolve()
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "aura_beauty")

JWT_SECRET = os.getenv("JWT_SECRET_KEY", "default_secret_key")
JWT_ALGORITHM = "HS256"
JWT_EXP_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7))

CORS_ORIGINS = [
    "https://aura-beauty-boutique.com",
    "https://www.aura-beauty-boutique.com",
    "https://app.aura-beauty-boutique.com",
    "https://api.aura-beauty-boutique.com",
]

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if HAVE_GENAI and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
elif HAVE_GENAI:
    logging.warning("Gemini SDK installed but API key missing. AI will fallback to local.")

# -------------------------------
# FastAPI setup
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
)

# -------------------------------
# MongoDB
# -------------------------------
client = None
db = None

async def connect_to_mongo():
    global client, db
    try:
        client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=3000)
        await client.server_info()
        db = client[DB_NAME]
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.error(f"MongoDB failed: {e}")
        await asyncio.sleep(2)
        await connect_to_mongo()

@app.on_event("startup")
async def startup():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown():
    if client:
        client.close()

# -------------------------------
# Pydantic Models
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
    created_at: datetime
    skin_tone: Optional[str] = None
    hair_color: Optional[str] = None

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
    image_base64: str
    analysis_type: str  # "reference_look" | "event_look"

# -------------------------------
# Auth Helpers
# -------------------------------
def hash_password(password: str) -> str:
    return pwd_context.hash(password[:72])

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain[:72], hashed)

def create_token(data: dict) -> str:
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXP_MINUTES)
    payload = {**data, "exp": int(expire.timestamp()), "iat": int(datetime.utcnow().timestamp())}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token if isinstance(token, str) else token.decode()

async def get_current_user(auth: HTTPAuthorizationCredentials = Depends(security)) -> User:
    token = auth.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        uid = payload.get("sub")
        user_doc = await db.users.find_one({"id": uid})
        if not user_doc:
            raise HTTPException(401, "User not found")

        user_doc.pop("_id", None)
        user_doc.pop("password_hash", None)

        if isinstance(user_doc.get("created_at"), str):
            user_doc["created_at"] = datetime.fromisoformat(user_doc["created_at"])

        return User(**user_doc)
    except Exception:
        raise HTTPException(401, "Invalid or expired token")

# -------------------------------
# Color Utilities
# -------------------------------
def rgb_to_hex(r, g, b): return "#{:02x}{:02x}{:02x}".format(r, g, b)

def rgb_to_lab(r, g, b):
    r_n, g_n, b_n = [x / 255 for x in (r, g, b)]
    def f(c): return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
    R, G, B = f(r_n), f(g_n), f(b_n)
    X = R*0.4124 + G*0.3576 + B*0.1805
    Y = R*0.2126 + G*0.7152 + B*0.0722
    Z = R*0.0193 + G*0.1192 + B*0.9505
    X/=0.95047; Z/=1.08883
    def fl(t): return t**(1/3) if t>0.008856 else 7.787*t+0.13793
    L=116*fl(Y)-16; a=500*(fl(X)-fl(Y)); b=200*(fl(Y)-fl(Z))
    return {"l":round(L,2),"a":round(a,2),"b":round(b,2)}

def extract_dominant_color(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((150,150))
    pixels = np.array(img).reshape(-1,3)
    k = min(5, len(pixels))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(pixels)
    c = kmeans.cluster_centers_[np.bincount(kmeans.labels_).argmax()]
    r,g,b = [int(x) for x in c]
    return {"r":r,"g":g,"b":b}

# -------------------------------
# Gemini Helper
# -------------------------------
def gemini_image_call(model, prompt, img_bytes):
    if not (HAVE_GENAI and GEMINI_API_KEY):
        raise RuntimeError("Gemini unavailable")

    try:
        m = genai.GenerativeModel(model)
        resp = m.generate_content([
            {"mime_type":"image/jpeg","data":img_bytes},
            prompt
        ])
        if hasattr(resp, "text"):
            return resp.text
        return str(resp)
    except Exception as e:
        logger.warning("Gemini failed: %s", e)
        raise

# -------------------------------
# AI Logic
# -------------------------------
async def run_reference_look(img):
    prompt = """
Return lipstick RGB only as:
{ "r":123, "g":45, "b":67 }
"""
    if HAVE_GENAI and GEMINI_API_KEY:
        try:
            txt = gemini_image_call("gemini-2.5-flash", prompt, img)
            data = json.loads(txt)
            if all(k in data for k in ("r","g","b")):
                return {k:int(data[k]) for k in ("r","g","b")}
        except Exception:
            pass

    return extract_dominant_color(img)

async def run_event_look(img):
    prompt = """
Analyze face + outfit. Return:
{
 "skin_tone":"",
 "undertone":"",
 "hair_color":"",
 "eye_color":"",
 "outfit_color":"",
 "best_shades":["A","B","C"]
}
"""
    if HAVE_GENAI and GEMINI_API_KEY:
        try:
            txt = gemini_image_call("gemini-2.5-flash", prompt, img)
            return json.loads(txt)
        except:
            pass

    dom = extract_dominant_color(img)
    hex_c = rgb_to_hex(dom["r"],dom["g"],dom["b"])
    return {
        "skin_tone":"unknown",
        "undertone":"unknown",
        "hair_color":"unknown",
        "eye_color":"unknown",
        "outfit_color":hex_c,
        "best_shades":[hex_c,hex_c,hex_c]
    }

# -------------------------------
# Auth Routes
# -------------------------------
@api.post("/auth/register")
async def register(u: UserRegister):
    if await db.users.find_one({"email": u.email}):
        raise HTTPException(400, "Email exists")

    uid = str(uuid.uuid4())
    doc = {
        "id": uid,
        "email": u.email,
        "full_name": u.full_name,
        "password_hash": hash_password(u.password),
        "created_at": datetime.utcnow().isoformat()
    }
    await db.users.insert_one(doc)
    doc.pop("password_hash")
    return {"token": create_token({"sub": uid}), "user": doc}

@api.post("/auth/login")
async def login(u: UserLogin):
    user = await db.users.find_one({"email": u.email})
    if not user or not verify_password(u.password, user["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    user.pop("password_hash")
    user.pop("_id")
    return {"token": create_token({"sub": user["id"]}), "user": user}

@api.get("/auth/me")
async def me(user: User = Depends(get_current_user)): return user

@api.put("/auth/profile")
async def update_profile(p:UserProfile, user:User=Depends(get_current_user)):
    updates={k:v for k,v in p.dict().items() if v is not None}
    if updates:
        await db.users.update_one({"id":user.id},{"$set":updates})
    return {"message":"Profile updated"}

# -------------------------------
# Shades Routes
# -------------------------------
@api.get("/shades")
async def get_shades(user:User=Depends(get_current_user)):
    shades=await db.shades.find({"user_id":user.id},{"_id":0}).to_list(999)
    return {"count":len(shades),"shades":shades}

@api.post("/shades")
async def create_shade(s:ShadeCreate,user:User=Depends(get_current_user)):
    sid=str(uuid.uuid4())
    doc={"id":sid,"user_id":user.id,**s.dict(),"created_at":datetime.utcnow().isoformat()}
    await db.shades.insert_one(doc)
    doc.pop("_id",None)
    return doc

@api.delete("/shades/{sid}")
async def delete_shade(sid:str,user:User=Depends(get_current_user)):
    r=await db.shades.delete_one({"id":sid,"user_id":user.id})
    if r.deleted_count==0:
        raise HTTPException(404,"Shade not found")
    return {"message":"Deleted"}

# -------------------------------
# Device Routes
# -------------------------------
@api.post("/device/connect")
async def connect_device(user:User=Depends(get_current_user)):
    existing=await db.device_sessions.find_one({"user_id":user.id,"status":"connected"})
    if existing:
        return {"message":"Already connected","device_id":existing["device_id"]}
    sid=str(uuid.uuid4())
    doc={"id":sid,"user_id":user.id,"device_id":"LipstickDispenser",
         "status":"connected","connected_at":datetime.utcnow().isoformat(),
         "last_shade_dispensed":None}
    await db.device_sessions.insert_one(doc)
    return {"message":"Device connected","device_id":"LipstickDispenser"}

@api.post("/device/disconnect")
async def disconnect_device(user:User=Depends(get_current_user)):
    await db.device_sessions.update_many({"user_id":user.id},{"$set":{"status":"disconnected"}})
    return {"message":"Disconnected"}

@api.get("/device/status")
async def device_status(user:User=Depends(get_current_user)):
    s=await db.device_sessions.find_one({"user_id":user.id,"status":"connected"},{"_id":0})
    return {"connected":bool(s),**(s or {})}

@api.post("/device/dispense")
async def dispense(data:dict,user:User=Depends(get_current_user)):
    sid=data.get("shade_id")
    if not sid: raise HTTPException(400,"Missing shade_id")
    session=await db.device_sessions.find_one({"user_id":user.id,"status":"connected"})
    if not session: raise HTTPException(400,"Device not connected")
    shade=await db.shades.find_one({"id":sid,"user_id":user.id},{"_id":0})
    if not shade: raise HTTPException(404,"Shade not found")

    await db.device_sessions.update_one({"id":session["id"]},{"$set":{"status":"dispensing"}})
    await asyncio.sleep(1.2)
    await db.device_sessions.update_one({"id":session["id"]},{"$set":{"status":"connected"}})

    mix={
        "cyan":255-shade["rgb"]["r"],
        "magenta":255-shade["rgb"]["g"],
        "yellow":255-shade["rgb"]["b"],
        "black":round((255-max(shade["rgb"].values()))*0.3,2)
    }
    return {"message":"Dispensed","shade":shade,"mix_formula":mix}

# -------------------------------
# Color Analyze
# -------------------------------
@api.post("/analyze/color")
async def analyze_color(request:Request,file:Optional[UploadFile]=File(None),
                        user:User=Depends(get_current_user)):

    if file:
        img=await file.read()
    else:
        body=await request.json()
        img=base64.b64decode(body.get("image_base64") or body.get("image"))

    rgb=extract_dominant_color(img)
    lab=rgb_to_lab(rgb["r"],rgb["g"],rgb["b"])
    hex_c=rgb_to_hex(rgb["r"],rgb["g"],rgb["b"])

    return {
        "dominant_color":rgb,
        "lab_values":lab,
        "hex_color":hex_c,
        "source":"local"
    }

# -------------------------------
# AI Shade (reference + event look)
# -------------------------------
@api.post("/analyze/ai-shade")
async def analyze_ai(req:AIShadeRequest,user:User=Depends(get_current_user)):
    img=base64.b64decode(req.image_base64)

    if req.analysis_type=="reference_look":
        rgb=await run_reference_look(img)
        lab=rgb_to_lab(rgb["r"],rgb["g"],rgb["b"])
        hex_c=rgb_to_hex(rgb["r"],rgb["g"],rgb["b"])
        return {
            "dominant_color":rgb,
            "lab_values":lab,
            "hex_color":hex_c,
            "source":"ai" if (HAVE_GENAI and GEMINI_API_KEY) else "local"
        }

    # -------------------------------
    # EVENT LOOK
    # -------------------------------
    if req.analysis_type == "event_look":
        data = await run_event_look(img)

        # use extracted dominant color for preview
        dom = extract_dominant_color(img)

        return {
            "dominant_color": {"r": dom["r"], "g": dom["g"], "b": dom["b"]},
            "lab_values": rgb_to_lab(dom["r"], dom["g"], dom["b"]),
            "hex_color": rgb_to_hex(dom["r"], dom["g"], dom["b"]),
            "ai_description": f"Recommended shades: {', '.join(data['best_shades'])}",
            "analysis": data,
            "source": "ai" if (HAVE_GENAI and GEMINI_API_KEY) else "local"
        }

    # -------------------------------
    # INVALID TYPE
    # -------------------------------
    raise HTTPException(400, "Invalid analysis_type")

# -------------------------------
# Root
# -------------------------------
@api.get("/")
async def root():
    return {"message":"Aura Backend Running"}

app.include_router(api)