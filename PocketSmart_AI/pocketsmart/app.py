"""
app.py
------
PocketSmart AI: Your Smart Budget & Recommendation Assistant

FastAPI backend that:
- Serves the HTML/CSS frontend (Jinja2 templates)
- Handles user registration/login/session via JWT
- Exposes /home-budget, /party-budget, /jewelry-budget endpoints that call
  Gemini 1.5 Flash (via gemini_utils.py) to generate curated, budget-aware
  recommendations with shopping links to Indian e-commerce platforms
- Keeps a simple in-memory recommendation history per user

NOTE: This project uses simple in-memory dictionaries (active_sessions,
users_db, user_recommendations) to keep the reference implementation easy
to read and run locally. Swap these for a real database (Postgres, Mongo,
etc.) before deploying to production.
"""

import os
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from fastapi import (
    FastAPI, HTTPException, Depends, File, UploadFile, Form,
    Request, status, Cookie
)
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

from models import (
    RegisterUser, UserInDB, Token, UserSession,
    HomeBudgetInput, PartyBudgetInput, JewelryBudgetInput,
    RecommendationHistoryItem,
)
from gemini_utils import (
    configure_gemini,
    get_home_recommendations,
    get_party_recommendations,
    get_jewelry_recommendations,
)

# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError(
        "No Google API key found in environment variables. "
        "Please set GOOGLE_API_KEY in your .env file."
    )
configure_gemini(API_KEY)

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# ---------------------------------------------------------------------------
# FastAPI app init
# ---------------------------------------------------------------------------
app = FastAPI(title="PocketSmart: AI Budget Planner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/uploads", exist_ok=True)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# ---------------------------------------------------------------------------
# In-memory "database". Replace with a real DB for production use.
# ---------------------------------------------------------------------------
users_db: Dict[str, UserInDB] = {}
active_sessions: Dict[str, UserSession] = {}
blacklisted_tokens: set = set()
user_recommendations: Dict[str, List[RecommendationHistoryItem]] = {}


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def authenticate_user(db: Dict[str, UserInDB], username: str, password: str):
    user = db.get(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_token(request: Request) -> Optional[str]:
    """Pull the JWT from the Authorization header or from the access_token
    cookie set at login."""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ")[1]
    return request.cookies.get("access_token")


async def get_current_user(request: Request, token: Optional[str] = Depends(get_token)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token or token in blacklisted_tokens:
        raise credentials_exception
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = users_db.get(username)
    if user is None:
        raise credentials_exception

    if username in active_sessions:
        active_sessions[username].last_activity = datetime.utcnow()

    return user


async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    return current_user


def save_to_history(username: str, recommendation_type: str, input_data: dict, result: dict):
    """Save a trimmed-down summary + full result to the in-memory history."""
    if username not in user_recommendations:
        user_recommendations[username] = []

    if recommendation_type == "home":
        input_summary = {
            "budget": input_data.get("total_budget"),
            "rooms": [
                r for r, has in [
                    ("Living Room", input_data.get("has_living_room")),
                    ("Kitchen", input_data.get("has_kitchen")),
                    ("Bedroom", input_data.get("has_bedroom")),
                ] if has
            ],
            "lights": input_data.get("num_lights"),
            "fans": input_data.get("num_fans"),
            "furniture": input_data.get("num_furniture"),
        }
        result_summary = {"remaining_budget": result.get("remaining_budget")}
    elif recommendation_type == "party":
        input_summary = {
            "budget": input_data.get("total_budget"),
            "party_type": input_data.get("party_type"),
            "guests": input_data.get("num_guests"),
            "needs": [
                n for n, has in [
                    ("Catering", input_data.get("needs_catering")),
                    ("Decoration", input_data.get("needs_decoration")),
                    ("Entertainment", input_data.get("needs_entertainment")),
                ] if has
            ],
        }
        result_summary = {"remaining_budget": result.get("remaining_budget")}
    else:  # jewelry
        input_summary = {
            "budget": input_data.get("total_budget"),
            "occasion": input_data.get("occasion"),
            "with_outfit_image": bool(input_data.get("image")),
        }
        result_summary = {"remaining_budget": result.get("remaining_budget")}

    item = RecommendationHistoryItem(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        recommendation_type=recommendation_type,
        input_summary=input_summary,
        result_summary=result_summary,
        full_result=result,
    )
    user_recommendations[username].append(item)


def save_upload_file(upload_file: UploadFile) -> str:
    """Persist an uploaded outfit image to static/uploads and return its path."""
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    safe_name = f"{timestamp}_{upload_file.filename}".replace(" ", "_")
    dest_path = os.path.join("static", "uploads", safe_name)
    with open(dest_path, "wb") as buffer:
        buffer.write(upload_file.file.read())
    return dest_path


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.post("/register")
async def register(user: RegisterUser):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    users_db[user.username] = UserInDB(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
    )
    return {"message": "User registered successfully"}


def _issue_login_response(username: str) -> JSONResponse:
    """Shared logic: create a JWT, register the session, and set the cookie."""
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )

    existing_user_data = {}
    if username in active_sessions:
        existing_user_data = active_sessions[username].user_data
        blacklisted_tokens.add(active_sessions[username].token)

    active_sessions[username] = UserSession(
        username=username,
        login_time=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        token=access_token,
        user_data=existing_user_data,
    )

    response = JSONResponse(content={"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
    )
    return response


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Standard OAuth2-compatible login endpoint (used by Swagger UI /docs
    and any API client)."""
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _issue_login_response(user.username)


@app.post("/login")
async def login_form(username: str = Form(...), password: str = Form(...)):
    """Convenience endpoint so the login.html form can POST simple form
    fields directly, without needing the full OAuth2 multipart payload."""
    user = authenticate_user(users_db, username, password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
        )
    return _issue_login_response(user.username)


@app.post("/logout")
async def logout(request: Request):
    """Logout user by blacklisting their token and clearing session"""
    token = await get_token(request)

    if token:
        blacklisted_tokens.add(token)
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username and username in active_sessions:
                del active_sessions[username]
        except JWTError:
            pass

    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response


@app.get("/session-info")
async def get_session_info(current_user: UserInDB = Depends(get_current_active_user)):
    if current_user.username in active_sessions:
        session = active_sessions[current_user.username]
        return {
            "username": session.username,
            "login_time": session.login_time,
            "last_activity": session.last_activity,
            "session_duration": (datetime.utcnow() - session.login_time).total_seconds() // 60,
            "user_data": session.user_data,
        }
    raise HTTPException(status_code=404, detail="No active session found")


@app.post("/session-data")
async def update_session_data(
    data: Dict[str, Any],
    current_user: UserInDB = Depends(get_current_active_user),
):
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data.update(data)
        active_sessions[current_user.username].last_activity = datetime.utcnow()
        return {"message": "Session data updated", "data": active_sessions[current_user.username].user_data}
    raise HTTPException(status_code=404, detail="No active session found")


# ---------------------------------------------------------------------------
# Page routes (HTML)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    try:
        token = await get_token(request)
        if token:
            user = await get_current_user(request, token)
            if user:
                return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    except Exception:
        pass
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    try:
        token = await get_token(request)
        if token:
            user = await get_current_user(request, token)
            if user:
                return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    except Exception:
        pass
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    history = sorted(
        user_recommendations.get(current_user.username, []),
        key=lambda x: x.timestamp,
        reverse=True,
    )[:3]
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "user": current_user, "recent_history": history},
    )


@app.get("/home-planner", response_class=HTMLResponse)
async def home_planner(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    """Home budget planner page"""
    return templates.TemplateResponse("home_planner.html", {"request": request, "user": current_user})


@app.get("/party-planner", response_class=HTMLResponse)
async def party_planner(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    """Party budget planner page"""
    return templates.TemplateResponse("party_planner.html", {"request": request, "user": current_user})


@app.get("/jewelry-planner", response_class=HTMLResponse)
async def jewelry_planner(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    """Jewelry budget planner page"""
    return templates.TemplateResponse("jewelry_planner.html", {"request": request, "user": current_user})


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    """History page to view past recommendations"""
    return templates.TemplateResponse("history.html", {"request": request, "user": current_user})


# ---------------------------------------------------------------------------
# Planner API routes
# ---------------------------------------------------------------------------
@app.post("/home-budget")
async def plan_home_budget(
    budget_input: HomeBudgetInput,
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Generate home budget recommendations"""
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_home_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
            "requirements": {
                "lights": budget_input.num_lights,
                "fans": budget_input.num_fans,
                "furniture": budget_input.num_furniture,
                "dining_tables": budget_input.num_dining_tables,
            },
        }

    try:
        result = get_home_recommendations(budget_input)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    save_to_history(
        username=current_user.username,
        recommendation_type="home",
        input_data=budget_input.dict(),
        result=result,
    )
    return result


@app.post("/party-budget")
async def plan_party_budget(
    budget_input: PartyBudgetInput,
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Generate party budget recommendations"""
    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_party_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
            "party_type": budget_input.party_type,
            "guests": budget_input.num_guests,
        }

    try:
        result = get_party_recommendations(budget_input)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    save_to_history(
        username=current_user.username,
        recommendation_type="party",
        input_data=budget_input.dict(),
        result=result,
    )
    return result


@app.post("/jewelry-budget")
async def plan_jewelry_budget(
    total_budget: float = Form(...),
    occasion: str = Form(...),
    preferences: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    request: Request = None,
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Generate jewelry budget recommendations with optional outfit image"""
    budget_input = JewelryBudgetInput(
        total_budget=total_budget,
        occasion=occasion,
        preferences=preferences,
    )

    image_path = None
    if image and image.filename:
        image_path = save_upload_file(image)

    if current_user.username in active_sessions:
        active_sessions[current_user.username].user_data["last_jewelry_budget"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "budget": budget_input.total_budget,
            "occasion": budget_input.occasion,
            "has_image": image_path is not None,
        }

    try:
        result = get_jewelry_recommendations(budget_input, image_path)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    input_data = budget_input.dict()
    if image_path:
        input_data["image"] = image.filename

    save_to_history(
        username=current_user.username,
        recommendation_type="jewelry",
        input_data=input_data,
        result=result,
    )
    return result


@app.get("/recommendation-history")
async def get_recommendation_history(
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Get the user's recommendation history"""
    if current_user.username not in user_recommendations:
        return {"history": []}

    history = sorted(
        user_recommendations[current_user.username],
        key=lambda x: x.timestamp,
        reverse=True,
    )

    history_data = [
        {
            "id": item.id,
            "timestamp": item.timestamp,
            "type": item.recommendation_type,
            "input": item.input_summary,
            "summary": item.result_summary,
        }
        for item in history
    ]
    return {"history": history_data}


@app.get("/recommendation-details/{recommendation_id}")
async def get_recommendation_details(
    recommendation_id: str,
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user),
):
    """Get the full details of a specific recommendation"""
    if current_user.username not in user_recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found")

    for item in user_recommendations[current_user.username]:
        if item.id == recommendation_id:
            return {
                "id": item.id,
                "timestamp": item.timestamp,
                "type": item.recommendation_type,
                "input": item.input_summary,
                "full_result": item.full_result,
            }

    raise HTTPException(status_code=404, detail="Recommendation not found")


# ---------------------------------------------------------------------------
# Startup: background task to clean up idle sessions
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def setup_session_cleanup():
    """Background task to clean up expired sessions"""
    async def cleanup_expired_sessions():
        while True:
            current_time = datetime.utcnow()
            expired = [
                username for username, session in active_sessions.items()
                if (current_time - session.last_activity).total_seconds() > 1800  # 30 min
            ]
            for username in expired:
                print(f"Removing expired session for {username}")
                del active_sessions[username]
            await asyncio.sleep(300)  # check every 5 minutes

    asyncio.create_task(cleanup_expired_sessions())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("Starting PocketSmart: AI Budget Planner...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
