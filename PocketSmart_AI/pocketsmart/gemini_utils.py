"""
gemini_utils.py
----------------
Core AI utility layer for PocketSmart AI.

Handles:
- Prompt construction for the Home, Party, and Jewelry planners
- Communication with Google's Gemini model (text + optional image)
- Parsing/validating the JSON returned by Gemini
- Attaching shopping search links for supported Indian e-commerce platforms
"""

import json
import re
import urllib.parse
from typing import Optional, Dict, Any

import google.generativeai as genai
from PIL import Image

from models import HomeBudgetInput, PartyBudgetInput, JewelryBudgetInput

# ---------------------------------------------------------------------------
# Gemini model handle. `configure_gemini()` is called once at app startup
# from app.py after the API key has been loaded from the environment.
# ---------------------------------------------------------------------------
_model = None
MODEL_NAME = "gemini-1.5-flash"


def configure_gemini(api_key: str, model_name: str = MODEL_NAME):
    """Initialise the Gemini client. Must be called once before any
    recommendation function is used."""
    global _model
    genai.configure(api_key=api_key)
    _model = genai.GenerativeModel(model_name)
    return _model


def _get_model():
    if _model is None:
        raise RuntimeError(
            "Gemini has not been configured. Call configure_gemini(api_key) first."
        )
    return _model


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """Gemini sometimes wraps JSON in ```json fences or adds preamble text.
    This strips that away and returns a parsed dict."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = cleaned.strip()

    # Fallback: find the first { ... last } block if there is stray text
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-resort fallback so the API never hard-crashes on bad JSON
        return {
            "error": "Could not parse AI response as JSON",
            "raw_response": text,
        }


def _shopping_links(search_terms: str, platforms: Dict[str, str]) -> Dict[str, str]:
    """Build search-result URLs for a set of platform -> url-template pairs."""
    if not search_terms:
        return {}
    q = urllib.parse.quote_plus(search_terms)
    return {name: template.format(q=q) for name, template in platforms.items()}


# ---------------------------------------------------------------------------
# Platform URL templates used across planners
# ---------------------------------------------------------------------------
HOME_PLATFORMS = {
    "amazon": "https://www.amazon.in/s?k={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "ikea": "https://www.ikea.com/in/en/search/?q={q}",
    "myntra": "https://www.myntra.com/search?q={q}",
    "ajio": "https://www.ajio.com/search/?text={q}",
}

PARTY_PLATFORM_MAP = {
    "venue": {
        "google": "https://www.google.com/search?q={q}",
        "booking": "https://www.booking.com/search.html?ss={q}",
        "makemytrip": "https://www.makemytrip.com/hotels/hotel-listing/?searchText={q}",
        "oyorooms": "https://www.oyorooms.com/search/?location={q}",
        "nobroker": "https://www.nobroker.in/property/search?searchTerm={q}",
    },
    "catering": {
        "swiggy": "https://www.swiggy.com/search?query={q}",
        "zomato": "https://www.zomato.com/search?q={q}",
        "bigbasket": "https://www.bigbasket.com/ps/?q={q}",
    },
    "food": {
        "swiggy": "https://www.swiggy.com/search?query={q}",
        "zomato": "https://www.zomato.com/search?q={q}",
        "bigbasket": "https://www.bigbasket.com/ps/?q={q}",
        "amazon": "https://www.amazon.in/s?k={q}",
        "flipkart": "https://www.flipkart.com/search?q={q}",
    },
    "drinks": {
        "swiggy": "https://www.swiggy.com/search?query={q}",
        "zomato": "https://www.zomato.com/search?q={q}",
        "bigbasket": "https://www.bigbasket.com/ps/?q={q}",
    },
    "decoration": {
        "amazon": "https://www.amazon.in/s?k={q}",
        "flipkart": "https://www.flipkart.com/search?q={q}",
        "meesho": "https://www.meesho.com/search?q={q}",
        "myntra": "https://www.myntra.com/search?q={q}",
    },
    "entertainment": {
        "amazon": "https://www.amazon.in/s?k={q}",
        "flipkart": "https://www.flipkart.com/search?q={q}",
        "bookmyshow": "https://in.bookmyshow.com/search?q={q}",
    },
    "gifts": {
        "amazon": "https://www.amazon.in/s?k={q}",
        "flipkart": "https://www.flipkart.com/search?q={q}",
        "meesho": "https://www.meesho.com/search?q={q}",
    },
    "photography": {
        "google": "https://www.google.com/search?q={q}",
        "amazon": "https://www.amazon.in/s?k={q}",
        "flipkart": "https://www.flipkart.com/search?q={q}",
    },
    "music": {
        "amazon": "https://www.amazon.in/s?k={q}",
        "flipkart": "https://www.flipkart.com/search?q={q}",
        "bookmyshow": "https://in.bookmyshow.com/search?q={q}",
    },
    "games": {
        "amazon": "https://www.amazon.in/s?k={q}",
        "flipkart": "https://www.flipkart.com/search?q={q}",
    },
    "accessories": {
        "amazon": "https://www.amazon.in/s?k={q}",
        "flipkart": "https://www.flipkart.com/search?q={q}",
        "meesho": "https://www.meesho.com/search?q={q}",
    },
    "transportation": {
        "makemytrip": "https://www.makemytrip.com/hotels/hotel-listing/?searchText={q}",
        "google": "https://www.google.com/search?q={q}",
    },
    "return_gifts": {
        "amazon": "https://www.amazon.in/s?k={q}",
        "flipkart": "https://www.flipkart.com/search?q={q}",
        "meesho": "https://www.meesho.com/search?q={q}",
    },
}
PARTY_DEFAULT_PLATFORMS = {
    "amazon": "https://www.amazon.in/s?k={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "google": "https://www.google.com/search?q={q}",
}

JEWELRY_PLATFORMS = {
    "amazon": "https://www.amazon.in/s?k={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "bluestone": "https://www.bluestone.com/search.html?query={q}",
    "tanishq": "https://www.tanishq.co.in/search?q={q}",
    "caratlane": "https://www.caratlane.com/search?q={q}",
    "melorra": "https://www.melorra.com/search?q={q}",
    "meesho": "https://www.meesho.com/search?q={q}",
}


# ---------------------------------------------------------------------------
# HOME PLANNER
# ---------------------------------------------------------------------------
def get_home_recommendations(budget_input: HomeBudgetInput) -> dict:
    """Generate home interior recommendations within budget, in INR, for the
    Indian market."""
    try:
        model = _get_model()

        prompt = f"""
I need interior design product recommendations for a home in India with a total budget of ₹{budget_input.total_budget:.2f}.

Requirements:
- {budget_input.num_lights} lights/lighting fixtures
- {budget_input.num_fans} ceiling fans
- {budget_input.num_furniture} furniture pieces
- {budget_input.num_dining_tables} dining tables

Additional rooms to consider:
{"- Living room" if budget_input.has_living_room else ""}
{"- Kitchen" if budget_input.has_kitchen else ""}
{"- Bedroom" if budget_input.has_bedroom else ""}

Additional requirements: {budget_input.additional_requirements or "None"}

Please provide a detailed budget breakdown with product recommendations **available in India**.
Use **Indian brands and pricing**. Include **search terms** suitable for Indian shopping platforms.

Respond with ONLY valid JSON (no markdown fences, no commentary) in this structure:
{{
  "total_budget": {budget_input.total_budget:.2f},
  "budget_breakdown": [
    {{
      "category": "lighting",
      "allocation": 0.0,
      "items": [
        {{
          "name": "",
          "description": "",
          "estimated_price": 0.0,
          "quantity": 0,
          "search_terms": ""
        }}
      ]
    }}
  ],
  "remaining_budget": 0.0,
  "additional_suggestions": []
}}

Ensure total costs stay within budget. Include search terms for each item to find it on shopping
websites like Flipkart, Amazon India, IKEA, Myntra, Ajio.
"""
        response = model.generate_content(prompt)
        result = extract_json_from_response(response.text)

        for category in result.get("budget_breakdown", []):
            for item in category.get("items", []):
                search_terms = item.get("search_terms", "")
                if search_terms:
                    item["shopping_links"] = _shopping_links(search_terms, HOME_PLATFORMS)

        # Build a calculation table summarising spend per category
        result["calculation_table"] = []
        for category in result.get("budget_breakdown", []):
            items = category.get("items", [])
            total_cost = sum(i.get("estimated_price", 0) * max(i.get("quantity", 1), 1) for i in items)
            pct = (total_cost / result["total_budget"] * 100) if result.get("total_budget") else 0
            result["calculation_table"].append({
                "category": category.get("category", "Misc"),
                "items_count": len(items),
                "total_cost": total_cost,
                "percentage_of_budget": round(pct, 1),
            })

        return result

    except Exception as e:
        raise RuntimeError(f"Error generating home recommendations: {str(e)}")


# ---------------------------------------------------------------------------
# PARTY PLANNER
# ---------------------------------------------------------------------------
def get_party_recommendations(budget_input: PartyBudgetInput) -> dict:
    """Generate party/event planning recommendations within budget, in INR."""
    try:
        model = _get_model()

        prompt = f"""
I need party planning recommendations for India with a total budget of ₹{budget_input.total_budget:.2f}.

Party details:
- Type: {budget_input.party_type}
- Number of guests: {budget_input.num_guests}
- Venue type: {budget_input.venue_type or "Not specified"}
- Catering needed: {"Yes" if budget_input.needs_catering else "No"}
- Decoration needed: {"Yes" if budget_input.needs_decoration else "No"}
- Entertainment needed: {"Yes" if budget_input.needs_entertainment else "No"}

Additional requirements: {budget_input.additional_requirements or "None"}

Please provide a detailed budget breakdown with specific recommendations available in India using
INR prices. Use Indian brands, services, and typical cost expectations.

Respond with ONLY valid JSON (no markdown fences, no commentary) in this structure:
{{
  "total_budget": {budget_input.total_budget:.2f},
  "budget_breakdown": [
    {{
      "category": "venue",
      "allocation": 0.0,
      "items": [
        {{
          "name": "",
          "description": "",
          "estimated_price": 0.0,
          "quantity": 0,
          "search_terms": ""
        }}
      ]
    }}
  ],
  "venue_suggestions": [
    {{
      "name": "",
      "type": "",
      "capacity": 0,
      "estimated_cost": 0.0,
      "search_terms": ""
    }}
  ],
  "remaining_budget": 0.0,
  "additional_suggestions": []
}}

Ensure all costs are in INR and total does not exceed the given budget.
Provide search terms suitable for Indian websites such as BookMyShow, Swiggy, Zomato, Flipkart, etc.
"""
        response = model.generate_content(prompt)
        result = extract_json_from_response(response.text)

        # Calculation table + shopping links per category
        result["calculation_table_inr"] = []
        categories = {}
        for category in result.get("budget_breakdown", []):
            cat_name = category.get("category", "Misc")
            cat_allocation = category.get("allocation", 0)
            if cat_name not in categories:
                categories[cat_name] = {
                    "category": cat_name,
                    "items_count": 0,
                    "total_cost": 0,
                    "percentage_of_budget": 0,
                }

            relevant_platforms = PARTY_PLATFORM_MAP.get(cat_name.lower(), PARTY_DEFAULT_PLATFORMS)

            for item in category.get("items", []):
                categories[cat_name]["items_count"] += 1
                categories[cat_name]["total_cost"] += item.get("estimated_price", 0)

                search_terms = item.get("search_terms", "")
                if search_terms:
                    item["shopping_links"] = _shopping_links(search_terms, relevant_platforms)

            if result.get("total_budget", 0) > 0:
                categories[cat_name]["percentage_of_budget"] = round(
                    (categories[cat_name]["total_cost"] / result["total_budget"]) * 100, 1
                )

        for cat_data in categories.values():
            result["calculation_table_inr"].append(cat_data)

        # Search links for venue suggestions
        for venue in result.get("venue_suggestions", []):
            search_terms = venue.get("search_terms", "")
            if search_terms:
                venue["search_links"] = _shopping_links(search_terms, PARTY_PLATFORM_MAP["venue"])

        return result

    except Exception as e:
        raise RuntimeError(f"Error generating party recommendations: {str(e)}")


# ---------------------------------------------------------------------------
# JEWELRY PLANNER
# ---------------------------------------------------------------------------
def get_jewelry_recommendations(
    budget_input: JewelryBudgetInput, image_path: Optional[str] = None
) -> dict:
    """Generate jewelry recommendations based on budget/occasion, and
    optionally an uploaded outfit image for color/style matching."""
    try:
        model = _get_model()

        base_prompt = f"""
I need jewelry recommendations for India with a total budget of ₹{budget_input.total_budget:.2f}.

Occasion: {budget_input.occasion}
Preferences: {budget_input.preferences or "Not specified"}
Provide only India-relevant styles, availability, and price ranges in INR.
"""

        if image_path:
            img = Image.open(image_path)
            prompt = base_prompt + """
An image of the outfit is uploaded. Suggest jewelry that complements it, considering color, design,
and occasion appropriateness.

Respond with ONLY valid JSON (no markdown fences, no commentary) in this structure:
{
  "outfit_analysis": {
    "colors": [],
    "style": "",
    "formality": ""
  },
  "total_budget": 0.0,
  "jewelry_recommendations": [
    {
      "item_type": "",
      "description": "",
      "style": "",
      "estimated_price": 0.0,
      "search_terms": ""
    }
  ],
  "remaining_budget": 0.0,
  "styling_tips": []
}

Make sure prices are in INR and stay within budget.
Include Indian-friendly search terms for shopping.
"""
            response = model.generate_content([prompt, img])
        else:
            prompt = base_prompt + """
Respond with ONLY valid JSON (no markdown fences, no commentary) in this structure:
{
  "total_budget": 0.0,
  "jewelry_recommendations": [
    {
      "item_type": "",
      "description": "",
      "style": "",
      "estimated_price": 0.0,
      "search_terms": ""
    }
  ],
  "remaining_budget": 0.0,
  "styling_tips": []
}

Keep prices in INR and relevant to Indian brands.
"""
            response = model.generate_content(prompt)

        result = extract_json_from_response(response.text)

        for item in result.get("jewelry_recommendations", []):
            search_terms = item.get("search_terms", "")
            if search_terms:
                item["shopping_links"] = _shopping_links(search_terms, JEWELRY_PLATFORMS)

        return result

    except Exception as e:
        raise RuntimeError(f"Error generating jewelry recommendations: {str(e)}")
