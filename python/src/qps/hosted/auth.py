"""Cognito PKCE browser sessions. Workers and controller use separate credentials."""

import base64
import hashlib
import hmac
import json
import secrets
import time
from urllib.parse import urlencode

import httpx
import jwt
from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import delete, insert, select

from qps.hosted.database import sessions


class Auth:
    def __init__(self, settings, repository):
        self.settings = settings
        self.repo = repository
        self.fernet = Fernet(settings.session_key.encode()) if settings.session_key else None
        self.jwks = (
            jwt.PyJWKClient(settings.cognito_issuer + "/.well-known/jwks.json")
            if settings.cognito_issuer
            else None
        )

    def claims(self, token):
        try:
            key = self.jwks.get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                issuer=self.settings.cognito_issuer,
                options={"verify_aud": False},
                leeway=10,
            )
            if (
                claims.get("token_use") != "access"
                or claims.get("client_id") != self.settings.cognito_client_id
            ):
                raise ValueError("Wrong token audience")
            if "owner" not in claims.get("cognito:groups", []):
                raise ValueError("Owner group required")
            return claims
        except (jwt.PyJWTError, ValueError) as exc:
            raise HTTPException(401, "Invalid owner session") from exc

    def owner(self, request: Request, required=True, check_csrf=True):
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
        if self.settings.auth_mode == "development":
            token = bearer or request.cookies.get("qps_dev", "")
            authenticated = bool(token) and hmac.compare_digest(token, self.settings.dev_token)
            if authenticated:
                if check_csrf and not bearer and request.method not in ("GET", "HEAD", "OPTIONS"):
                    if request.headers.get("x-csrf-token") != self.settings.dev_token:
                        raise HTTPException(403, "CSRF token required")
                return {"username": "owner", "csrf": self.settings.dev_token}
        elif bearer:
            return self.claims(bearer)
        else:
            session_id = request.cookies.get("qps_session", "")
            if session_id:
                with self.repo.connection() as connection:
                    row = (
                        connection.execute(
                            select(sessions).where(
                                sessions.c.id == hashlib.sha256(session_id.encode()).hexdigest()
                            )
                        )
                        .mappings()
                        .first()
                    )
                if row and row["expires"] > time.time():
                    if (
                        check_csrf
                        and request.method not in ("GET", "HEAD", "OPTIONS")
                        and not hmac.compare_digest(
                            request.headers.get("x-csrf-token", ""), row["csrf"]
                        )
                    ):
                        raise HTTPException(403, "CSRF token required")
                    try:
                        tokens = json.loads(self.fernet.decrypt(row["encrypted"].encode()))
                    except InvalidToken as exc:
                        raise HTTPException(401, "Invalid session") from exc
                    try:
                        claims = self.claims(tokens["access_token"])
                    except HTTPException:
                        if required:
                            raise
                        return None
                    return {**claims, "csrf": row["csrf"]}
        if required:
            raise HTTPException(401, "Owner login required")
        return None

    def login(self):
        if self.settings.auth_mode != "cognito":
            raise HTTPException(400, "Development mode uses a configured bearer token")
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        query = {
            "response_type": "code",
            "client_id": self.settings.cognito_client_id,
            "redirect_uri": self.settings.public_url + "/api/v1/auth/callback",
            "scope": "openid profile",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        response = RedirectResponse(
            self.settings.cognito_domain + "/oauth2/authorize?" + urlencode(query)
        )
        value = self.fernet.encrypt(
            json.dumps(
                {"state": state, "verifier": verifier, "expires": time.time() + 600}
            ).encode()
        ).decode()
        response.set_cookie(
            "qps_login",
            value,
            max_age=600,
            secure=True,
            httponly=True,
            samesite="lax",
            path="/api/v1/auth",
        )
        return response

    def callback(self, request: Request, code: str, state: str):
        if not self.fernet:
            raise HTTPException(400, "Cognito is not configured")
        try:
            pending = json.loads(self.fernet.decrypt(request.cookies.get("qps_login", "").encode()))
        except (InvalidToken, ValueError) as exc:
            raise HTTPException(400, "Login state missing") from exc
        if pending["expires"] < time.time() or not hmac.compare_digest(pending["state"], state):
            raise HTTPException(400, "Login state mismatch or expired")
        with httpx.Client(timeout=15, follow_redirects=False, trust_env=False) as client:
            result = client.post(
                self.settings.cognito_domain + "/oauth2/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": self.settings.cognito_client_id,
                    "redirect_uri": self.settings.public_url + "/api/v1/auth/callback",
                    "code": code,
                    "code_verifier": pending["verifier"],
                },
            )
            if result.status_code != 200:
                raise HTTPException(401, "Authorization code exchange failed")
            tokens = result.json()
        self.claims(tokens["access_token"])
        identity = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        with self.repo.transaction() as connection:
            connection.execute(
                insert(sessions).values(
                    id=hashlib.sha256(identity.encode()).hexdigest(),
                    encrypted=self.fernet.encrypt(json.dumps(tokens).encode()).decode(),
                    csrf=csrf,
                    expires=time.time() + min(tokens.get("expires_in", 3600), 3600),
                )
            )
        response = RedirectResponse(self.settings.public_url + "/")
        response.delete_cookie("qps_login", path="/api/v1/auth")
        response.set_cookie(
            "qps_session",
            identity,
            max_age=3600,
            secure=True,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return response

    def logout(self, request):
        identity = request.cookies.get("qps_session", "")
        with self.repo.transaction() as connection:
            connection.execute(
                delete(sessions).where(
                    sessions.c.id == hashlib.sha256(identity.encode()).hexdigest()
                )
            )
        response = RedirectResponse(self.settings.public_url + "/", status_code=303)
        response.delete_cookie("qps_session", path="/")
        return response
