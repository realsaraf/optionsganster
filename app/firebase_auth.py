"""Firebase ID token verification helpers."""
from __future__ import annotations

import httpx
from google.auth.transport.requests import Request
from google.oauth2 import id_token as google_id_token

from app.config import settings


class FirebaseAuthError(Exception):
    """Raised when Firebase auth is not configured or token verification fails."""


def firebase_enabled() -> bool:
    return all(
        [
            settings.FIREBASE_PROJECT_ID,
            settings.FIREBASE_API_KEY,
            settings.FIREBASE_AUTH_DOMAIN,
            settings.FIREBASE_APP_ID,
        ]
    )


def get_firebase_web_config() -> dict:
    return {
        "apiKey": settings.FIREBASE_API_KEY,
        "authDomain": settings.FIREBASE_AUTH_DOMAIN,
        "projectId": settings.FIREBASE_PROJECT_ID,
        "appId": settings.FIREBASE_APP_ID,
        "messagingSenderId": settings.FIREBASE_MESSAGING_SENDER_ID,
        "storageBucket": settings.FIREBASE_STORAGE_BUCKET,
        "measurementId": settings.FIREBASE_MEASUREMENT_ID,
    }


def _verify_via_identity_toolkit(token: str) -> dict:
    """Validate a Firebase ID token against the Identity Toolkit lookup API."""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={settings.FIREBASE_API_KEY}"
    try:
        response = httpx.post(url, json={"idToken": token}, timeout=10.0)
        response.raise_for_status()
    except Exception as exc:
        raise FirebaseAuthError("Invalid Firebase token") from exc

    payload = response.json()
    users = payload.get("users") or []
    if not users:
        raise FirebaseAuthError("Invalid Firebase token")

    user = users[0]
    return {
        "iss": f"https://securetoken.google.com/{settings.FIREBASE_PROJECT_ID}",
        "aud": settings.FIREBASE_PROJECT_ID,
        "user_id": user.get("localId", ""),
        "sub": user.get("localId", ""),
        "email": user.get("email", ""),
        "email_verified": user.get("emailVerified", False),
        "name": user.get("displayName", ""),
        "picture": user.get("photoUrl", ""),
    }


def verify_firebase_token(token: str) -> dict:
    if not firebase_enabled():
        raise FirebaseAuthError("Firebase auth is not configured")
    try:
        claims = google_id_token.verify_firebase_token(
            token,
            Request(),
            settings.FIREBASE_PROJECT_ID,
        )
    except Exception as exc:  # token verification exceptions vary by provider internals
        claims = _verify_via_identity_toolkit(token)

    if not claims:
        raise FirebaseAuthError("Invalid Firebase token")

    issuer = claims.get("iss", "")
    expected = f"https://securetoken.google.com/{settings.FIREBASE_PROJECT_ID}"
    if issuer != expected:
        raise FirebaseAuthError("Unexpected token issuer")

    if not claims.get("email"):
        raise FirebaseAuthError("Google account email missing from token")

    if not claims.get("email_verified", False):
        raise FirebaseAuthError("Google account email is not verified")

    return claims
