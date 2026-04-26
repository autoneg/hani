# Authentication

HANI supports multiple authentication modes to fit different deployment scenarios.

## Configuration

Authentication is configured in `~/negmas/hani/settings/env.json`:

```json
{
    "auth": {
        "mode": "password",
        "cookie_secret": "change-this-in-production",
        "enforce_consent": false
    }
}
```

## Authentication Modes

### Password Mode (Default)

Simple username/password authentication. Best for local development and small deployments.

```json
{
    "auth": {
        "mode": "password"
    }
}
```

**Features:**

- Users stored in `~/negmas/hani/settings/users.json`
- Default users: `admin`/`adminpass` and `user`/`userpass`
- New users can register via `/register` endpoint
- Passwords are hashed with bcrypt

### Dual Mode

Both password and OAuth authentication. Users can choose their preferred login method.

```json
{
    "auth": {
        "mode": "dual"
    },
    "oauth": {
        "redirect_uri": "http://localhost:5006",
        "providers": {
            "github": {
                "enabled": true,
                "key": "your-github-client-id",
                "secret": "your-github-client-secret"
            },
            "google": {
                "enabled": true,
                "key": "your-google-client-id",
                "secret": "your-google-client-secret"
            }
        }
    }
}
```

### OAuth Mode

OAuth-only authentication. Users must log in via an OAuth provider.

```json
{
    "auth": {
        "mode": "oauth"
    }
}
```

### Auto Mode

Automatically selects OAuth if configured, otherwise uses password:

```json
{
    "auth": {
        "mode": "auto"
    }
}
```

## Setting Up OAuth

### GitHub OAuth

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click **New OAuth App**
3. Fill in the application details:
   - **Application name:** HANI (or your preferred name)
   - **Homepage URL:** `http://localhost:5006` (or your domain)
   - **Authorization callback URL:** `http://localhost:5006/oauth/callback`
4. Click **Register application**
5. Copy the **Client ID** and generate a **Client Secret**
6. Add to your `env.json`:

```json
{
    "oauth": {
        "redirect_uri": "http://localhost:5006",
        "providers": {
            "github": {
                "enabled": true,
                "key": "your-client-id",
                "secret": "your-client-secret"
            }
        }
    }
}
```

### Google OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create a new project or select an existing one
3. Click **Create Credentials** → **OAuth client ID**
4. Select **Web application**
5. Add authorized redirect URI: `http://localhost:5006/oauth/callback`
6. Copy the **Client ID** and **Client Secret**
7. Add to your `env.json`:

```json
{
    "oauth": {
        "redirect_uri": "http://localhost:5006",
        "providers": {
            "google": {
                "enabled": true,
                "key": "your-client-id",
                "secret": "your-client-secret"
            }
        }
    }
}
```

## Admin Access

### Password Mode

The user `admin` has administrator privileges by default.

### OAuth Mode

In OAuth mode, admin access is granted to users whose email is in the `admin.emails` list:

```json
{
    "admin": {
        "emails": ["admin@example.com", "another-admin@example.com"]
    }
}
```

## User Registration

Users can register at the `/register` endpoint. Registration creates a new entry in `users.json`.

To disable registration, you can modify the authentication handler or remove the registration route.

## Consent Form

HANI can require users to consent before participating in negotiations:

```json
{
    "auth": {
        "enforce_consent": true
    }
}
```

When enabled, users must accept the consent form (located at `~/negmas/hani/settings/consent.md`) before they can start negotiations.

## Security Recommendations

!!! warning "Production Deployment"
    For production deployments:
    
    1. **Change the cookie secret** - Generate a secure random string
    2. **Change default passwords** - Never use `adminpass` or `userpass`
    3. **Use HTTPS** - Configure a reverse proxy with SSL
    4. **Restrict admin emails** - Only add trusted emails to the admin list

### Generating a Secure Cookie Secret

```python
import secrets
print(secrets.token_urlsafe(32))
```

Add the generated value to your `env.json`:

```json
{
    "auth": {
        "cookie_secret": "your-generated-secret-here"
    }
}
```

## Viewing Current Configuration

Run `hani auth` to see your current authentication configuration and setup instructions:

```bash
hani auth
```
