"""
Dual authentication support for HANI.

This module provides OAuth authentication alongside password authentication.
When dual auth is enabled, users can log in with either:
1. Username/password (via Panel's basic auth)
2. OAuth provider (GitHub, Google, etc.)

The OAuth flow is handled by custom Tornado handlers that integrate with
Panel's session management.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import urllib.parse
from typing import Any

from tornado.web import RequestHandler
from tornado.httpclient import AsyncHTTPClient, HTTPRequest

from hani.common import (
    OAUTH_PROVIDERS as CONFIGURED_OAUTH_PROVIDERS,
    OAUTH_REDIRECT_URI,
    COOKIE_SECRET,
    APP_URLS,
)


# OAuth provider endpoint configurations
OAUTH_PROVIDER_ENDPOINTS = {
    "github": {
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "user_url": "https://api.github.com/user",
        "scopes": ["read:user", "user:email"],
    },
    "google": {
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "user_url": "https://www.googleapis.com/oauth2/v2/userinfo",
        "scopes": ["openid", "email", "profile"],
    },
}


def is_server_running(host: str = "localhost", port: int = 5008) -> bool:
    """Check if a server is running on the given host:port.

    Args:
        host: The hostname to check
        port: The port to check

    Returns:
        True if server is accepting connections, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            result = sock.connect_ex((host, port))
            return result == 0
    except (socket.error, OSError):
        return False


def get_enabled_oauth_providers() -> dict[str, dict]:
    """Get all enabled OAuth providers with their credentials.

    Returns:
        Dict of provider_name -> {key, secret} for enabled providers
    """
    return CONFIGURED_OAUTH_PROVIDERS


def is_dual_auth_enabled() -> bool:
    """Check if dual auth (password + OAuth) is enabled.

    Dual auth is enabled when:
    1. AUTH_MODE is 'dual' or 'auto'
    2. At least one OAuth provider is configured
    """
    from hani.common import AUTH_MODE

    providers = get_enabled_oauth_providers()

    if AUTH_MODE.lower() == "dual":
        return True
    if AUTH_MODE.lower() == "auto" and providers:
        return True
    return False


def get_oauth_config() -> dict[str, Any]:
    """Get OAuth configuration for templates."""
    providers = get_enabled_oauth_providers()
    enabled_providers = list(providers.keys())

    # Check if guest server is running
    guest_url = APP_URLS.get("playground", "http://localhost:5008")
    # Extract port from URL
    try:
        from urllib.parse import urlparse

        parsed = urlparse(guest_url)
        guest_port = parsed.port or 5008
        guest_host = parsed.hostname or "localhost"
        guest_available = is_server_running(guest_host, guest_port)
    except Exception:
        guest_available = False

    return {
        "oauth_enabled": is_dual_auth_enabled() and bool(providers),
        "oauth_providers": enabled_providers,
        # Legacy single provider support (first enabled provider)
        "oauth_provider": enabled_providers[0] if enabled_providers else "github",
        "registration_url": APP_URLS.get("registration", "http://localhost:5007"),
        "guest_url": guest_url,
        "guest_available": guest_available,
    }


class OAuthStartHandler(RequestHandler):
    """Handler to start OAuth flow for a specific provider."""

    def get(self, provider_name: str | None = None):
        """Redirect to OAuth provider's authorization page.

        Args:
            provider_name: The OAuth provider (github, google, etc.)
        """
        # Get provider from URL path or default to first enabled
        if not provider_name:
            providers = get_enabled_oauth_providers()
            if not providers:
                self.set_status(500)
                self.write("No OAuth providers configured")
                return
            provider_name = next(iter(providers))

        provider_name = provider_name.lower()

        # Check if provider is configured
        configured_providers = get_enabled_oauth_providers()
        if provider_name not in configured_providers:
            self.set_status(400)
            self.write(f"OAuth provider '{provider_name}' is not configured")
            return

        # Get provider endpoints
        provider_endpoints = OAUTH_PROVIDER_ENDPOINTS.get(provider_name)
        if not provider_endpoints:
            self.set_status(400)
            self.write(f"Unknown OAuth provider: {provider_name}")
            return

        # Get credentials
        credentials = configured_providers[provider_name]
        oauth_key = credentials["key"]
        oauth_secret = credentials["secret"]

        # Generate state token to prevent CSRF (include provider name)
        state_data = f"{provider_name}:{secrets.token_urlsafe(32)}"
        self.set_secure_cookie("oauth_state", state_data, httponly=True)

        # Build authorization URL
        params = {
            "client_id": oauth_key,
            "redirect_uri": f"{OAUTH_REDIRECT_URI}/oauth/callback",
            "scope": " ".join(provider_endpoints["scopes"]),
            "state": state_data,
            "response_type": "code",
        }

        auth_url = (
            f"{provider_endpoints['authorize_url']}?{urllib.parse.urlencode(params)}"
        )
        self.redirect(auth_url)


class OAuthCallbackHandler(RequestHandler):
    """Handler for OAuth callback."""

    async def get(self):
        """Handle OAuth callback and create session."""
        # Verify state to prevent CSRF
        state = self.get_argument("state", None)
        stored_state = self.get_secure_cookie("oauth_state")

        if not state or not stored_state:
            self.set_status(400)
            self.write("Invalid OAuth state")
            return

        if isinstance(stored_state, bytes):
            stored_state = stored_state.decode("utf-8")

        if state != stored_state:
            self.set_status(400)
            self.write("OAuth state mismatch")
            return

        # Extract provider name from state (format: provider_name:token)
        if ":" not in state:
            self.set_status(400)
            self.write("Invalid OAuth state format")
            return

        provider_name = state.split(":")[0]

        # Clear state cookie
        self.clear_cookie("oauth_state")

        # Check for errors
        error = self.get_argument("error", None)
        if error:
            error_desc = self.get_argument("error_description", "Unknown error")
            self.set_status(400)
            self.write(f"OAuth error: {error_desc}")
            return

        # Get authorization code
        code = self.get_argument("code", None)
        if not code:
            self.set_status(400)
            self.write("No authorization code received")
            return

        # Get provider configuration
        configured_providers = get_enabled_oauth_providers()
        if provider_name not in configured_providers:
            self.set_status(500)
            self.write(f"OAuth provider '{provider_name}' not configured")
            return

        provider_endpoints = OAUTH_PROVIDER_ENDPOINTS.get(provider_name)
        if not provider_endpoints:
            self.set_status(500)
            self.write(f"Unknown OAuth provider: {provider_name}")
            return

        credentials = configured_providers[provider_name]

        try:
            access_token = await self._exchange_code(
                code, provider_endpoints, credentials
            )
            user_info = await self._get_user_info(
                access_token, provider_endpoints, provider_name
            )
        except Exception as e:
            self.set_status(500)
            self.write(f"OAuth error: {str(e)}")
            return

        # Create or update user in our system
        username = self._get_username(user_info, provider_name)

        # Store OAuth user in session
        # Panel uses 'user' cookie for authentication
        from hani.auth import create_user, get_user

        # Check if user exists, create if not
        existing_user = get_user(username)
        if not existing_user:
            # Create user with random password (OAuth users don't need password)
            random_pass = secrets.token_urlsafe(32)
            create_user(
                username=username,
                password=random_pass,
            )

        # Set Panel authentication cookie
        # Panel uses a signed cookie with the username
        self.set_secure_cookie("user", username, httponly=True)

        # Redirect to main app
        self.redirect("/")

    async def _exchange_code(
        self, code: str, provider_endpoints: dict, credentials: dict
    ) -> str:
        """Exchange authorization code for access token."""
        http_client = AsyncHTTPClient()

        params = {
            "client_id": credentials["key"],
            "client_secret": credentials["secret"],
            "code": code,
            "redirect_uri": f"{OAUTH_REDIRECT_URI}/oauth/callback",
            "grant_type": "authorization_code",
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        request = HTTPRequest(
            provider_endpoints["token_url"],
            method="POST",
            body=urllib.parse.urlencode(params),
            headers=headers,
        )

        response = await http_client.fetch(request)
        data = json.loads(response.body)

        if "error" in data:
            raise ValueError(data.get("error_description", data["error"]))

        return data["access_token"]

    async def _get_user_info(
        self, access_token: str, provider_endpoints: dict, provider_name: str
    ) -> dict:
        """Get user info from OAuth provider."""
        http_client = AsyncHTTPClient()

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

        # GitHub requires User-Agent header
        if provider_name == "github":
            headers["User-Agent"] = "HANI-App"

        request = HTTPRequest(
            provider_endpoints["user_url"],
            method="GET",
            headers=headers,
        )

        response = await http_client.fetch(request)
        return json.loads(response.body)

    def _get_username(self, user_info: dict, provider_name: str) -> str:
        """Extract username from user info based on provider."""
        if provider_name == "github":
            return user_info.get("login", user_info.get("email", "oauth_user"))
        elif provider_name == "google":
            # Use email prefix as username
            email = user_info.get("email", "")
            if email:
                return email.split("@")[0]
            return user_info.get("name", "oauth_user").replace(" ", "_").lower()
        else:
            return user_info.get("email", user_info.get("name", "oauth_user"))


class RegisterHandler(RequestHandler):
    """Handler for user registration."""

    def get(self):
        """Show registration form."""
        try:
            errormessage = self.get_argument("error")
        except Exception:
            errormessage = ""

        try:
            success = self.get_argument("success")
        except Exception:
            success = ""

        from panel.io.resources import CDN_DIST
        from hani.common import CONSENT_FILE

        # Load consent text
        consent_text = ""
        if CONSENT_FILE.exists():
            consent_text = CONSENT_FILE.read_text()
        else:
            consent_text = (
                "## Consent Form\n\nPlease read the terms carefully before proceeding."
            )

        # Get OAuth config
        oauth_config = get_oauth_config()

        html = REGISTER_TEMPLATE.render(
            errormessage=errormessage,
            success=success,
            consent_text=consent_text,
            PANEL_CDN=CDN_DIST,
            **oauth_config,
        )
        self.write(html)

    def post(self):
        """Handle registration form submission."""
        username = self.get_argument("username", "").strip()
        password = self.get_argument("password", "")
        password_confirm = self.get_argument("password_confirm", "")
        email = self.get_argument("email", "").strip()

        # Validation
        if not username or not password:
            self.redirect(
                "/register?error="
                + urllib.parse.quote("Username and password are required")
            )
            return

        if username.lower() == "ai":
            self.redirect(
                "/register?error=" + urllib.parse.quote("Cannot use 'AI' as username")
            )
            return

        if password != password_confirm:
            self.redirect(
                "/register?error=" + urllib.parse.quote("Passwords do not match")
            )
            return

        if len(password) < 4:
            self.redirect(
                "/register?error="
                + urllib.parse.quote("Password must be at least 4 characters")
            )
            return

        # Create user
        from hani.auth import create_user, load_users

        users = load_users()
        if username in users:
            self.redirect(
                "/register?error=" + urllib.parse.quote("Username already exists")
            )
            return

        success = create_user(username=username, password=password)
        if not success:
            self.redirect(
                "/register?error=" + urllib.parse.quote("Failed to create user")
            )
            return

        # Redirect to main app - Panel will show login page
        # Use success parameter to show success message
        self.redirect(
            "/register?success="
            + urllib.parse.quote("Registration successful! You can now log in at /app")
        )


# Registration page template
REGISTER_TEMPLATE_STR = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta content="width=device-width, initial-scale=1.0" name="viewport">
    <title>HANI | Register</title>
    <link rel="icon" type="image/x-icon" href="{{ PANEL_CDN }}images/favicon.ico">
    <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    html {
      height: 100%;
    }
    body {
      font-family: 'Segoe UI', sans-serif;
      font-size: 1em;
      height: 100%;
      line-height: 1.6;
    }
    .wrap {
      align-items: center;
      background: #fafafa;
      display: flex;
      min-height: 100%;
      justify-content: center;
      padding: 2em;
    }
    .register-form {
      background: #ffffff;
      border: 1px solid #ddd;
      margin: 0 auto;
      padding: 2em;
      max-width: 500px;
      width: 100%;
    }
    .form-input {
      background: #fafafa;
      border: 1px solid #eeeeee;
      padding: 12px;
      width: 100%;
    }
    .form-group {
      margin-bottom: 1em;
    }
    .form-button {
      background: #107bba;
      border: 1px solid #ddd;
      color: #ffffff;
      padding: 10px;
      text-transform: uppercase;
      width: 100%;
      cursor: pointer;
    }
    .form-button:hover {
      background: #0072b5;
    }
    .form-header {
      text-align: center;
      margin-bottom: 1em;
    }
    .form-footer {
      text-align: center;
      margin-top: 1em;
    }
    .error-message {
      color: red;
      font-weight: bold;
      text-align: center;
      margin-bottom: 1em;
    }
    .success-message {
      color: green;
      font-weight: bold;
      text-align: center;
      margin-bottom: 1em;
    }
    .divider {
      display: flex;
      align-items: center;
      text-align: center;
      margin: 1.5em 0;
      color: #888;
    }
    .divider::before,
    .divider::after {
      content: '';
      flex: 1;
      border-bottom: 1px solid #ddd;
    }
    .divider::before {
      margin-right: 0.5em;
    }
    .divider::after {
      margin-left: 0.5em;
    }
    .oauth-button {
      background: #24292e;
      border: 1px solid #24292e;
      color: #ffffff;
      padding: 10px;
      width: 100%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      font-size: 14px;
      text-decoration: none;
      text-transform: none;
    }
    .oauth-button:hover {
      background: #2f363d;
    }
    .oauth-button.github {
      background: #24292e;
    }
    .oauth-button.google {
      background: #4285f4;
      border-color: #4285f4;
    }
    .oauth-button svg {
      width: 18px;
      height: 18px;
      fill: currentColor;
    }
    .oauth-section {
      display: none;
    }
    .oauth-section.enabled {
      display: block;
    }
    </style>
</head>
<body>
  <div class="wrap">
    <form class="register-form" action="/register" method="post">
      <div class="form-header">
        <h2>Register for HANI</h2>
        <p>Human-Agent Negotiation Interface</p>
      </div>
      
      {% if errormessage %}
      <div class="error-message">{{ errormessage }}</div>
      {% endif %}
      
      {% if success %}
      <div class="success-message">{{ success }}</div>
      {% endif %}

      <div class="form-group">
        <input name="username" type="text" class="form-input" autocapitalize="off" autocorrect="off" placeholder="Username" required>
      </div>
      
      <div class="form-group">
        <input name="email" type="email" class="form-input" placeholder="Email (optional)">
      </div>
      
      <div class="form-group">
        <input name="password" type="password" class="form-input" placeholder="Password" required>
      </div>
      
      <div class="form-group">
        <input name="password_confirm" type="password" class="form-input" placeholder="Confirm Password" required>
      </div>
      
      <div class="form-group">
        <button class="form-button" type="submit">Register</button>
      </div>

      <!-- OAuth Section -->
      <div class="oauth-section {{ 'enabled' if oauth_enabled else '' }}">
        <div class="divider">or</div>
        {% for provider in oauth_providers %}
        <div class="form-group">
          <a href="/oauth/{{ provider }}" class="oauth-button {{ provider }}">
            {% if provider == 'github' %}
            <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            Register with GitHub
            {% elif provider == 'google' %}
            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
            </svg>
            Register with Google
            {% else %}
            Register with {{ provider|capitalize }}
            {% endif %}
          </a>
        </div>
        {% endfor %}
      </div>

      <div class="form-footer">
        <p>Already have an account? <a href="/app">Login here</a></p>
        {% if guest_available %}
        <p><a href="{{ guest_url }}">Try without registration</a></p>
        {% endif %}
      </div>
    </form>
  </div>
</body>
</html>
"""

from jinja2 import Template

REGISTER_TEMPLATE = Template(REGISTER_TEMPLATE_STR)


def get_oauth_handlers() -> list[tuple]:
    """Get Tornado handlers for OAuth and registration routes.

    Returns:
        List of (route, handler) tuples for Tornado application
    """
    return [
        (r"/oauth/callback", OAuthCallbackHandler),
        (r"/oauth/(\w+)", OAuthStartHandler),  # /oauth/github, /oauth/google
        (r"/oauth", OAuthStartHandler),  # Fallback to first enabled provider
        (r"/register", RegisterHandler),
    ]


def configure_dual_auth_template():
    """Configure Panel to pass OAuth config to login template.

    This patches Panel's BasicLoginHandler to include OAuth configuration
    in the template context.
    """
    try:
        from panel.auth import BasicLoginHandler
        from panel.io.resources import CDN_DIST
    except ImportError:
        print("Warning: Could not import Panel BasicLoginHandler")
        return

    # Store original get method
    original_get = BasicLoginHandler.get

    def patched_get(self):
        """Add OAuth config to template context."""
        try:
            errormessage = self.get_argument("error")
        except Exception:
            errormessage = ""

        from panel.io.state import state

        next_url = self.get_argument("next", state.base_url)
        if next_url:
            if state.base_url and not next_url.startswith(state.base_url):
                next_url = next_url.replace("/", state.base_url, 1)
            self.set_cookie("next_url", next_url)

        # Get OAuth config
        oauth_config = get_oauth_config()

        # Render template with OAuth config included
        html = self._login_template.render(
            login_endpoint=self._login_endpoint,
            errormessage=errormessage,
            PANEL_CDN=CDN_DIST,
            **oauth_config,
        )
        self.write(html)

    BasicLoginHandler.get = patched_get
    print("✓ Dual auth template configuration applied")
