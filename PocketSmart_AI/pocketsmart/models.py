"""
models.py
---------
Pydantic input/output schemas and lightweight in-memory data structures
used across PocketSmart AI (users, sessions, and recommendation history).
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# Auth models
# ---------------------------------------------------------------------------
class RegisterUser(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str


class UserInDB(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class UserSession(BaseModel):
    username: str
    login_time: datetime
    last_activity: datetime
    token: str
    user_data: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Planner input models
# ---------------------------------------------------------------------------
class HomeBudgetInput(BaseModel):
    total_budget: float
    num_lights: int = 0
    num_fans: int = 0
    num_furniture: int = 0
    num_dining_tables: int = 0
    has_living_room: bool = False
    has_kitchen: bool = False
    has_bedroom: bool = False
    additional_requirements: Optional[str] = None


class PartyBudgetInput(BaseModel):
    total_budget: float
    num_guests: int
    party_type: str
    venue_type: Optional[str] = None
    needs_catering: bool = True
    needs_decoration: bool = True
    needs_entertainment: bool = False
    additional_requirements: Optional[str] = None


class JewelryBudgetInput(BaseModel):
    total_budget: float
    occasion: str
    preferences: Optional[str] = None


# ---------------------------------------------------------------------------
# History record (stored in-memory per user)
# ---------------------------------------------------------------------------
class RecommendationHistoryItem(BaseModel):
    id: str
    timestamp: str
    recommendation_type: str  # "home" | "party" | "jewelry"
    input_summary: Dict[str, Any]
    result_summary: Dict[str, Any]
    full_result: Dict[str, Any]
