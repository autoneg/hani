# HANI Deployment Guide: Local vs Production

## Overview

HANI consists of **three separate apps** that run simultaneously:
1. **Main App** (port 5006) - Authenticated negotiation interface
2. **Registration** (port 5007) - Public user registration
3. **Playground** (port 5008) - Public guest/demo mode

The system supports two deployment modes controlled by the `HANI_ENV` environment variable.

---

## 🏠 Local Development Mode

### Environment Setup
```bash
export HANI_ENV=local
```

### URLs (from `env.local.json`)
```json
{
  "registration": "http://localhost:5007",
  "app": "http://localhost:5006",
  "playground": "http://localhost:5008"
}
```

### How It Works

#### 1. Starting All Apps
```bash
cd /Users/yasser/code/projects/han
export HANI_ENV=local  # or set in .envrc.local
hani
```

This starts three processes:
- **Registration** → `http://localhost:5007`
- **Main App** → `http://localhost:5006`
- **Playground** → `http://localhost:5008`

#### 2. Registration Flow
```
User visits: http://localhost:5007
    ↓
Fills registration form
    ↓
Clicks "Register"
    ↓
System creates:
  - ~/negmas/hani/settings/users_info.json (full details)
  - ~/negmas/hani/settings/users.json (plain passwords)
  - ~/negmas/hani/settings/users_hashed.json (SHA-256 hashes)
    ↓
Success message shows:
  "You can start negotiating here" → http://localhost:5006
    ↓
User clicks link
    ↓
Redirected to: http://localhost:5006
    ↓
User logs in with credentials
```

#### 3. Direct Access
Users can directly visit:
- `http://localhost:5007` - Register new account
- `http://localhost:5006` - Login to main app
- `http://localhost:5008` - Try playground mode (no login)

#### 4. App Communication
- Registration app reads `env.local.json` to know main app URL
- After registration, redirect link points to `http://localhost:5006`
- All apps share same settings directory: `~/negmas/hani/settings/`

---

## 🌐 Production Mode

### Environment Setup
```bash
export HANI_ENV=production
```

### URLs (from `env.production.json`)
```json
{
  "registration": "https://anac.cs.brown.edu/hanreg",
  "app": "https://anac.cs.brown.edu/hanapp",
  "playground": "https://anac.cs.brown.edu/hanguest"
}
```

### How It Works

#### 1. Starting All Apps
```bash
# On production server
cd /path/to/hani
export HANI_ENV=production
hani
```

Apps run on localhost but are proxied through web server:
- **Registration** → localhost:5007 → nginx → `https://anac.cs.brown.edu/hanreg`
- **Main App** → localhost:5006 → nginx → `https://anac.cs.brown.edu/hanapp`
- **Playground** → localhost:5008 → nginx → `https://anac.cs.brown.edu/hanguest`

#### 2. Registration Flow
```
User visits: https://anac.cs.brown.edu/hanreg
    ↓
Fills registration form
    ↓
Clicks "Register"
    ↓
System creates user files (same as local)
    ↓
Success message shows:
  "You can start negotiating here" → https://anac.cs.brown.edu/hanapp
    ↓
User clicks link
    ↓
Redirected to: https://anac.cs.brown.edu/hanapp
    ↓
User logs in with credentials
```

#### 3. Direct Access
Users can directly visit:
- `https://anac.cs.brown.edu/hanreg` - Register new account
- `https://anac.cs.brown.edu/hanapp` - Login to main app
- `https://anac.cs.brown.edu/hanguest` - Try playground mode

#### 4. Nginx Configuration Example
```nginx
# Registration app
location /hanreg {
    proxy_pass http://localhost:5007;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}

# Main app
location /hanapp {
    proxy_pass http://localhost:5006;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}

# Playground app
location /hanguest {
    proxy_pass http://localhost:5008;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

---

## 📋 URL Matrix

### Local Development
| App | Port | Internal URL | Public URL | Auth | Redirects To |
|-----|------|--------------|------------|------|--------------|
| Registration | 5007 | http://localhost:5007 | http://localhost:5007 | ❌ No | http://localhost:5006 |
| Main App | 5006 | http://localhost:5006 | http://localhost:5006 | ✅ Yes | - |
| Playground | 5008 | http://localhost:5008 | http://localhost:5008 | ❌ No | - |

### Production
| App | Port | Internal URL | Public URL | Auth | Redirects To |
|-----|------|--------------|------------|------|--------------|
| Registration | 5007 | http://localhost:5007 | https://anac.cs.brown.edu/hanreg | ❌ No | https://anac.cs.brown.edu/hanapp |
| Main App | 5006 | http://localhost:5006 | https://anac.cs.brown.edu/hanapp | ✅ Yes | - |
| Playground | 5008 | http://localhost:5008 | https://anac.cs.brown.edu/hanguest | ❌ No | - |

---

## 🔐 Authentication Modes

### Main App Only (Port 5006)
The main app supports **dual authentication**:

#### Local Development (Default)
```bash
export HANI_ENV=local
export HANI_AUTH_MODE=password
hani
```
- Uses password-based authentication
- Login: admin / Yarab@Satrak19
- Or any user registered via registration app

#### Production with OAuth
```bash
export HANI_ENV=production
export HANI_AUTH_MODE=oauth
export HANI_OAUTH_KEY=your_github_client_id
export HANI_OAUTH_SECRET=your_github_client_secret
export HANI_OAUTH_REDIRECT_URI=https://anac.cs.brown.edu/hanapp
export HANI_OAUTH_ENCRYPTION_KEY=$(panel oauth-secret)
hani
```
- Uses GitHub OAuth authentication
- Users login via GitHub

### Registration & Playground (Ports 5007, 5008)
Both are **always public** (no authentication required) in both modes.

---

## 🔄 Redirect Logic

### Registration → Main App
The registration app dynamically determines where to redirect based on `HANI_ENV`:

```python
# In register.py
from hani.common import APP_URLS

# After successful registration
main_app_url = APP_URLS.get("app")  # Reads from env.{HANI_ENV}.json
reg_message.object = f"[You can start negotiating here]({main_app_url})"
```

**Local:** Redirects to `http://localhost:5006`
**Production:** Redirects to `https://anac.cs.brown.edu/hanapp`

---

## 📁 Configuration Files

### Local Development
```
~/negmas/hani/settings/
├── env.local.json          # Local URLs
├── env.production.json           # Production URLs
├── env.json                # Fallback (currently local URLs)
├── users.json              # Plain text passwords
├── users_hashed.json       # SHA-256 hashed passwords (used for auth)
├── users_info.json         # Full user details
├── scenario_order.txt      # Scenario order
└── consent.md              # Registration consent form
```

### Environment Variable (in .envrc.local)
```bash
# Set deployment mode
export HANI_ENV=local           # or 'production'

# Set auth mode for main app
export HANI_AUTH_MODE=password  # or 'oauth' or 'auto'

# OAuth credentials (for OAuth mode)
export HANI_OAUTH_KEY=...
export HANI_OAUTH_SECRET=...
export HANI_OAUTH_ENCRYPTION_KEY=...
```

---

## 🚀 Deployment Checklist

### Local Development
- [x] Set `HANI_ENV=local` in `.envrc.local`
- [x] Set `HANI_AUTH_MODE=password`
- [x] Ensure `env.local.json` exists with localhost URLs
- [x] Run `direnv allow`
- [x] Run `hani --dev`
- [x] Access apps at localhost:5006, 5007, 5008

### Production Deployment
- [ ] Set `HANI_ENV=production` on server
- [ ] Set `HANI_AUTH_MODE=oauth` or `password`
- [ ] Ensure `env.production.json` exists with production URLs
- [ ] Configure OAuth credentials if using OAuth
- [ ] Set up nginx reverse proxy
- [ ] Configure SSL certificates
- [ ] Run `hani` (without --dev)
- [ ] Verify apps accessible at anac.cs.brown.edu URLs

---

## 🧪 Testing

### Test Local Mode
```bash
# Terminal 1
export HANI_ENV=local
hani --dev

# Browser
# 1. Visit http://localhost:5007
# 2. Register new user "testuser"
# 3. Click "start negotiating" link
# 4. Should redirect to http://localhost:5006
# 5. Login with testuser credentials
```

### Test Production Mode (Locally)
```bash
# Terminal 1
export HANI_ENV=production
hani --dev

# Browser
# 1. Visit http://localhost:5007
# 2. Register new user "testuser2"
# 3. Click "start negotiating" link
# 4. Should show link to https://anac.cs.brown.edu/hanapp
#    (won't work locally, but confirms production URLs are used)
```

---

## 🐛 Troubleshooting

### Issue: Registration redirects to wrong URL
**Solution:** Check `HANI_ENV` is set correctly
```bash
echo $HANI_ENV
# Should be 'local' or 'production'
```

### Issue: Can't login after registration
**Problem:** Hashed passwords not created
**Solution:** Check `users_hashed.json` exists and has your username
```bash
cat ~/negmas/hani/settings/users_hashed.json
```

### Issue: Apps start but URLs are wrong
**Solution:** Verify correct env file is being loaded
```bash
python -c "from hani.common import APP_URLS, HANI_ENV; print(f'ENV: {HANI_ENV}'); print(f'URLs: {APP_URLS}')"
```

---

## 📝 Summary

**Key Points:**
1. ✅ Three apps run simultaneously on different ports
2. ✅ `HANI_ENV` controls which URL configuration is loaded
3. ✅ Registration app redirects to main app based on environment
4. ✅ Only main app requires authentication
5. ✅ All apps share the same user database files
6. ✅ Production uses nginx to proxy localhost ports to public URLs

**Environment Variables:**
- `HANI_ENV` → `local` or `production` (controls URLs)
- `HANI_AUTH_MODE` → `password` or `oauth` (controls main app auth)

**URL Configuration Files:**
- `env.local.json` → Local development URLs (localhost)
- `env.production.json` → Production URLs (anac.cs.brown.edu)
