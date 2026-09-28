# PocketSmart AI — Your Smart Budget & Recommendation Assistant

An AI-powered, cross-platform budget & recommendation system built with **FastAPI** and
**Google Gemini 1.5 Flash**. It generates personalized, budget-aware suggestions for three
planning scenarios:

1. **Home Interior Planner** — lighting, fans, furniture, dining tables, room-by-room
2. **Party Planner** — venue, catering, decoration, entertainment, sized to guest count
3. **Jewelry Planner** — text + optional outfit image, matched to occasion & style

Recommendations link out to shopping-search URLs on Amazon, Flipkart, IKEA, Myntra, Ajio,
Swiggy, Zomato, BigBasket, BookMyShow, OYO Rooms, MakeMyTrip, Tanishq, BlueStone, CaratLane,
and more.

---

## Project structure

```
pocketsmart/
├── app.py                 # FastAPI app: routes, auth, sessions, history
├── gemini_utils.py         # Prompt building, Gemini calls, JSON parsing, shopping links
├── models.py               # Pydantic request/response & data models
├── requirements.txt
├── .env.example             # Copy to .env and fill in your keys
├── static/
│   ├── styles.css
│   └── uploads/             # Uploaded outfit images land here
└── templates/
    ├── base.html            # Shared navbar / footer / auth-fetch JS
    ├── index.html            # Landing page
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── home_planner.html
    ├── party_planner.html
    ├── jewelry_planner.html
    └── history.html
```

## How it works

- **Backend (FastAPI)** — `app.py` defines all routes: page routes that render Jinja2
  templates, auth routes (`/register`, `/login`, `/token`, `/logout`), session-info routes,
  and the three planner POST endpoints (`/home-budget`, `/party-budget`, `/jewelry-budget`).
- **AI layer (`gemini_utils.py`)** — builds a structured prompt per planner, calls Gemini,
  asks it to answer in strict JSON, parses/repairs that JSON, then decorates every
  recommended item with real shopping-search links for relevant Indian platforms.
- **Frontend** — server-rendered Jinja2 pages + a small amount of vanilla JS per planner
  page that POSTs the form as JSON (or `multipart/form-data` for the jewelry planner,
  since it can include an image) and renders the AI's response as cards/tables.
- **Auth & sessions** — JWT-based (`python-jose`), passwords hashed with `passlib`/bcrypt.
  For simplicity this reference build keeps users, sessions, and recommendation history in
  **in-memory Python dictionaries** — perfect for local development and demos. Swap these
  for a real database before deploying to production.

## Setup

1. **Create a virtual environment and install dependencies**

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Get a Gemini API key**

   - Go to [Google AI Studio](https://aistudio.google.com/), sign in, accept the terms,
     and click **Get API key** → **Create API key**.

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env`:

   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   SECRET_KEY=some_long_random_string
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   ```

4. **Run the app**

   ```bash
   python app.py
   ```

   or, for auto-reload during development:

   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

5. Open **http://localhost:8000** in your browser, create an account, sign in, and try
   the three planners from the dashboard.

   Interactive API docs are available at **http://localhost:8000/docs**.

## Notes & next steps for a production build

- Replace the in-memory `users_db` / `active_sessions` / `user_recommendations`
  dictionaries in `app.py` with a real database (PostgreSQL, MongoDB, etc.).
- The shopping links are **search-result URLs**, not live product/price lookups (the
  document's "third-party APIs" are simulated this way rather than scraped, since most of
  those platforms don't offer public product-search APIs).
- Add rate limiting / usage caps around the Gemini calls if you expose this publicly.
- Tighten `allow_origins=["*"]` in the CORS middleware before deploying.
- Consider moving uploaded outfit images to object storage (S3, GCS) instead of local disk.

## Milestones covered (per the original project spec)

- ✅ Milestone 1 — Gemini AI setup & initialization (`gemini_utils.configure_gemini`)
- ✅ Milestone 2 — Core functionality development (`gemini_utils.py` prompt logic)
- ✅ Milestone 3 — Backend FastAPI integration (`app.py`, all routes)
- ✅ Milestone 4 — UI development (Jinja2 templates + `static/styles.css`)
- ✅ Milestone 5 — Testing & optimization hooks (JSON-repair fallback, error handling,
  session cleanup task, additional-suggestions fallback content from the model)
