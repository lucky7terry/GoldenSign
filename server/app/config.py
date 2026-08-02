import os


SESSION_STORE_BACKEND = os.getenv("SESSION_STORE_BACKEND", "memory").lower()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
PUBLIC_WS_BASE_URL = os.getenv("PUBLIC_WS_BASE_URL")