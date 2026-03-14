"""Mongo-backed auth user and session store."""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from pymongo import ReturnDocument

from app.mongo import get_db


UTC = timezone.utc
SESSION_DAYS = 7
VALID_ROLES = {"admin", "general"}


class AuthStore:
    def _db(self):
        return get_db()

    @staticmethod
    def normalize_email(email: str) -> str:
        return (email or "").strip().lower()

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(UTC)

    async def ensure_seed_users(self, legacy_users: dict[str, dict]) -> None:
        db = self._db()
        for email, record in legacy_users.items():
            normalized = self.normalize_email(email)
            role = record.get("role", "general")
            if role not in VALID_ROLES:
                role = "general"
            await db.users.update_one(
                {"email": normalized},
                {
                    "$setOnInsert": {
                        "email": normalized,
                        "display_name": record.get("display_name") or normalized.split("@")[0],
                        "role": role,
                        "is_authorized": True,
                        "is_deleted": False,
                        "auth_provider": "google",
                        "legacy_password_hash": record.get("password_hash", ""),
                        "created_at": self._utcnow(),
                    },
                    "$set": {"updated_at": self._utcnow()},
                },
                upsert=True,
            )

    async def get_authorized_user(self, email: str) -> Optional[dict]:
        db = self._db()
        return await db.users.find_one(
            {"email": self.normalize_email(email), "is_authorized": True, "is_deleted": {"$ne": True}},
            {"_id": 0},
        )

    async def record_google_login(self, claims: dict, existing_user: dict) -> dict:
        db = self._db()
        email = self.normalize_email(claims.get("email", ""))
        now = self._utcnow()
        display_name = claims.get("name") or existing_user.get("display_name") or email.split("@")[0]
        update = {
            "display_name": display_name,
            "google_uid": claims.get("user_id") or claims.get("sub", ""),
            "photo_url": claims.get("picture", existing_user.get("photo_url", "")),
            "last_login_at": now,
            "auth_provider": "google",
            "updated_at": now,
        }
        await db.users.update_one({"email": email}, {"$set": update})
        user = dict(existing_user)
        user.update(update)
        user["email"] = email
        return user

    async def create_session(self, user: dict) -> str:
        db = self._db()
        token = secrets.token_urlsafe(32)
        now = self._utcnow()
        session = {
            "token": token,
            "email": self.normalize_email(user["email"]),
            "role": user.get("role", "general"),
            "display_name": user.get("display_name") or user["email"].split("@")[0],
            "created_at": now,
            "updated_at": now,
            "expires_at": now + timedelta(days=SESSION_DAYS),
        }
        await db.auth_sessions.insert_one(session)
        return token

    async def get_session(self, token: str) -> Optional[dict]:
        db = self._db()
        now = self._utcnow()
        session = await db.auth_sessions.find_one(
            {"token": token, "expires_at": {"$gt": now}},
            {"_id": 0},
        )
        if not session:
            return None

        user = await db.users.find_one(
            {"email": session["email"], "is_authorized": True, "is_deleted": {"$ne": True}},
            {"_id": 0},
        )
        if not user:
            await db.auth_sessions.delete_one({"token": token})
            return None

        merged = {
            "email": user["email"],
            "role": user.get("role", "general"),
            "display_name": user.get("display_name") or user["email"].split("@")[0],
        }
        await db.auth_sessions.update_one(
            {"token": token},
            {"$set": {"updated_at": now}},
        )
        return merged

    async def delete_session(self, token: str) -> None:
        await self._db().auth_sessions.delete_one({"token": token})

    async def delete_sessions_for_email(self, email: str) -> None:
        await self._db().auth_sessions.delete_many({"email": self.normalize_email(email)})

    async def list_users(self) -> list[dict]:
        cursor = self._db().users.find({"is_deleted": {"$ne": True}}, {"_id": 0}).sort("email", 1)
        return await cursor.to_list(length=500)

    async def create_user(self, email: str, display_name: str = "", role: str = "general") -> dict:
        db = self._db()
        normalized = self.normalize_email(email)
        if not normalized:
            raise ValueError("Email is required")
        existing = await db.users.find_one({"email": normalized}, {"_id": 0})
        if existing and existing.get("is_deleted"):
            role = role if role in VALID_ROLES else "general"
            now = self._utcnow()
            restored = await db.users.find_one_and_update(
                {"email": normalized},
                {
                    "$set": {
                        "display_name": display_name.strip() or normalized.split("@")[0],
                        "role": role,
                        "is_authorized": True,
                        "is_deleted": False,
                        "deleted_at": None,
                        "deleted_by": None,
                        "updated_at": now,
                    }
                },
                projection={"_id": 0},
                return_document=ReturnDocument.AFTER,
            )
            return restored
        if existing:
            raise ValueError("User already exists")
        role = role if role in VALID_ROLES else "general"
        doc = {
            "email": normalized,
            "display_name": display_name.strip() or normalized.split("@")[0],
            "role": role,
            "is_authorized": True,
            "is_deleted": False,
            "auth_provider": "google",
            "created_at": self._utcnow(),
            "updated_at": self._utcnow(),
        }
        await db.users.insert_one(doc)
        return {k: v for k, v in doc.items() if k != "_id"}

    async def update_user(
        self,
        email: str,
        *,
        display_name: Optional[str] = None,
        role: Optional[str] = None,
        is_authorized: Optional[bool] = None,
    ) -> Optional[dict]:
        db = self._db()
        normalized = self.normalize_email(email)
        update: dict = {"updated_at": self._utcnow()}
        if display_name is not None:
            update["display_name"] = display_name.strip() or normalized.split("@")[0]
        if role is not None:
            update["role"] = role if role in VALID_ROLES else "general"
        if is_authorized is not None:
            update["is_authorized"] = bool(is_authorized)
        result = await db.users.find_one_and_update(
            {"email": normalized, "is_deleted": {"$ne": True}},
            {"$set": update},
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        return result

    async def soft_delete_user(self, email: str, *, deleted_by: str) -> Optional[dict]:
        db = self._db()
        normalized = self.normalize_email(email)
        result = await db.users.find_one_and_update(
            {"email": normalized, "is_deleted": {"$ne": True}},
            {
                "$set": {
                    "is_deleted": True,
                    "is_authorized": False,
                    "deleted_at": self._utcnow(),
                    "deleted_by": self.normalize_email(deleted_by),
                    "updated_at": self._utcnow(),
                }
            },
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        if result:
            await self.delete_sessions_for_email(normalized)
        return result


auth_store = AuthStore()
