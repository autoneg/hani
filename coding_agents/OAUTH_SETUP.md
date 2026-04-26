# OAuth Setup Guide for HANI

HANI now supports **dual authentication**: password-based auth for local development and OAuth for production deployments.

## Quick Start

### Local Development (Password Auth)
No setup needed! Just run:
```bash
hani --dev
```
Login with credentials from `~/negmas/hani/settings/users_hashed.json`

### Production (OAuth)
1. Get OAuth credentials from GitHub (see below)
2. Set environment variables
3. Run HANI

---

## Setting Up GitHub OAuth

### Step 1: Create OAuth App on GitHub

1. Go to: https://github.com/settings/developers
2. Click **"New OAuth App"**
3. Fill in the details:
   - **Application name**: HANI Negotiation Platform
   - **Homepage URL**: `http://localhost:5006` (or your domain)
   - **Authorization callback URL**: `http://localhost:5006` (IMPORTANT!)
   - **Description**: (optional)
4. Click **"Register application"**
5. You'll see your **Client ID**
6. Click **"Generate a new client secret"** to get your **Client Secret**
7. **Save both values** - you'll need them!

### Step 2: Configure Environment Variables

Create a `.env` file or set environment variables:

```bash
# For local testing with OAuth
export HANI_OAUTH_PROVIDER=github
export HANI_OAUTH_KEY=your_github_client_id_here
export HANI_OAUTH_SECRET=your_github_client_secret_here
export HANI_OAUTH_REDIRECT_URI=http://localhost:5006
```

Or if using direnv (recommended):
```bash
# Copy the example file
cp .env.example .env

# Edit with your credentials
nano .env

# Update these lines:
HANI_OAUTH_KEY=your_github_client_id_here
HANI_OAUTH_SECRET=your_github_client_secret_here
```

### Step 3: Run HANI with OAuth

```bash
# Make sure environment variables are loaded
source .env  # or `direnv allow` if using direnv

# Start HANI
hani --dev
```

You should see:
```
🔐 Authentication mode: OAUTH
  Provider: github
  Redirect URI: http://localhost:5006
✓ OAuth authentication configured

🚀 Starting HANI server...
```

### Step 4: Test Login

1. Open http://localhost:5006
2. You'll be redirected to GitHub login
3. Authorize the app
4. You'll be redirected back and logged in!

---

## Authentication Modes

HANI automatically detects which auth mode to use:

| Mode | When Used | Setup Required |
|------|-----------|----------------|
| **Password** | No OAuth credentials set | None - default |
| **OAuth** | OAuth credentials detected | GitHub OAuth App + env vars |

### Force Specific Mode

You can override auto-detection:

```bash
# Force password auth (ignore OAuth credentials)
export HANI_AUTH_MODE=password
hani --dev

# Force OAuth (will error if credentials missing)
export HANI_AUTH_MODE=oauth
hani --dev

# Auto-detect (default)
export HANI_AUTH_MODE=auto
hani --dev
```

---

## Deployment Scenarios

### Scenario 1: Local Development
**Use Case**: Working on HANI locally, quick iteration

```bash
# No setup needed
hani --dev
```
- Uses password authentication
- Login: admin / Yarab@Satrak19

### Scenario 2: Local Testing with OAuth
**Use Case**: Testing OAuth flow before deployment

```bash
# Set OAuth credentials
export HANI_OAUTH_KEY=your_client_id
export HANI_OAUTH_SECRET=your_client_secret
export HANI_OAUTH_REDIRECT_URI=http://localhost:5006

# Run with OAuth
hani --dev
```
- Uses GitHub OAuth
- Any GitHub user can login

### Scenario 3: Production Server
**Use Case**: Deployed on a public server

```bash
# Set OAuth for production domain
export HANI_OAUTH_KEY=your_client_id
export HANI_OAUTH_SECRET=your_client_secret
export HANI_OAUTH_REDIRECT_URI=https://yourdomain.com

# Run in production mode
hani
```
- Uses GitHub OAuth
- Secure authentication via GitHub

---

## Troubleshooting

### "OAuth credentials not configured" Error
**Problem**: HANI detected OAuth mode but credentials are missing

**Solution**: Either:
1. Set OAuth credentials: `export HANI_OAUTH_KEY=...` and `HANI_OAUTH_SECRET=...`
2. Force password mode: `export HANI_AUTH_MODE=password`

### "Callback URL mismatch" Error
**Problem**: GitHub OAuth redirect URL doesn't match configured URL

**Solution**: 
1. Check your GitHub OAuth app settings
2. Make sure callback URL matches exactly: `http://localhost:5006`
3. Update environment variable: `export HANI_OAUTH_REDIRECT_URI=http://localhost:5006`

### OAuth works locally but not in production
**Problem**: Callback URL is different in production

**Solution**:
1. Create a SECOND OAuth app on GitHub for production
2. Set production callback URL: `https://yourdomain.com`
3. Use different credentials in production environment

### Want to switch back to password auth
**Problem**: OAuth is configured but you want password auth

**Solution**:
```bash
# Option 1: Unset OAuth credentials
unset HANI_OAUTH_KEY
unset HANI_OAUTH_SECRET

# Option 2: Force password mode
export HANI_AUTH_MODE=password

# Then run
hani --dev
```

---

## Security Notes

### Development
- Password auth is fine for local development
- OAuth adds extra security layer

### Production
- **Always use OAuth** in production
- Never commit OAuth secrets to git
- Use environment variables or secrets management
- Consider using HTTPS for production deployments

### Cookie Secret
Generate a strong secret for production:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
export HANI_COOKIE_SECRET=<generated-secret>
```

---

## Advanced: Other OAuth Providers

HANI supports multiple OAuth providers through Panel:

### Google OAuth
```bash
export HANI_OAUTH_PROVIDER=google
export HANI_OAUTH_KEY=your_google_client_id
export HANI_OAUTH_SECRET=your_google_client_secret
export HANI_OAUTH_REDIRECT_URI=http://localhost:5006/oauth-callback
```

Get credentials: https://console.cloud.google.com/apis/credentials

### Azure AD
```bash
export HANI_OAUTH_PROVIDER=azure
export HANI_OAUTH_KEY=your_azure_client_id
export HANI_OAUTH_SECRET=your_azure_client_secret
export HANI_OAUTH_REDIRECT_URI=http://localhost:5006/oauth-callback
```

Get credentials: https://portal.azure.com

---

## Files Modified

The dual-auth implementation touches these files:

- `src/hani/common.py` - OAuth configuration constants
- `src/hani/auth.py` - Auth mode detection logic
- `src/hani/runapp.py` - Dual-auth startup logic
- `.env.example` - Configuration template

No changes to existing password files needed!

---

## Summary

✅ **Local Development**: Use password auth (default, no setup)
✅ **Production**: Use OAuth (secure, professional)
✅ **Automatic Detection**: HANI picks the right mode
✅ **Easy Switching**: Just set/unset environment variables
✅ **Backward Compatible**: Existing password auth still works

Questions? Check the troubleshooting section or the `.env.example` file.
