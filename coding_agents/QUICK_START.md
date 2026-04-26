# HANI Quick Start Guide

## Three Apps, Two Modes

### The Apps
1. **Main App** (5006) - Authenticated negotiation interface
2. **Registration** (5007) - Public user registration  
3. **Playground** (5008) - Public guest mode

### The Modes
- **Local** (`HANI_ENV=local`) - Development on localhost
- **Production** (`HANI_ENV=production`) - Deployed on anac.cs.brown.edu

---

## Local Development (Default)

```bash
# 1. Set environment
export HANI_ENV=local
export HANI_AUTH_MODE=password

# 2. Start all apps
cd /Users/yasser/code/projects/han
direnv allow
hani --dev

# 3. Access apps
# - Registration: http://localhost:5007
# - Main App:     http://localhost:5006 (login: admin / Yarab@Satrak19)
# - Playground:   http://localhost:5008
```

### What Happens
- User registers at `localhost:5007`
- Registration creates hashed password
- Success message links to `localhost:5006`
- User logs in at main app

---

## Production Deployment

```bash
# 1. Set environment
export HANI_ENV=production
export HANI_AUTH_MODE=password  # or oauth

# 2. Start all apps (on server)
hani

# 3. Configure nginx to proxy:
# - localhost:5007 → https://anac.cs.brown.edu/hanreg
# - localhost:5006 → https://anac.cs.brown.edu/hanapp
# - localhost:5008 → https://anac.cs.brown.edu/hanguest
```

### What Happens
- User registers at `anac.cs.brown.edu/hanreg`
- Registration creates hashed password
- Success message links to `anac.cs.brown.edu/hanapp`
- User logs in at main app

---

## Configuration Files

```
~/negmas/hani/settings/
├── env.local.json       # localhost URLs (local mode)
├── env.production.json  # anac.cs.brown.edu URLs (production mode)
├── users_hashed.json    # Hashed passwords (for authentication)
├── users_info.json      # Full user details
└── users.json           # Plain passwords (reference)
```

---

## Environment Variables

### Required
```bash
export HANI_ENV=local          # or 'production'
```

### Optional (Main App Auth)
```bash
export HANI_AUTH_MODE=password # or 'oauth' or 'auto'
export HANI_OAUTH_KEY=...      # if using OAuth
export HANI_OAUTH_SECRET=...   # if using OAuth
export HANI_OAUTH_ENCRYPTION_KEY=...  # if using OAuth
```

---

## URL Redirects

### Local Mode
| Action | User At | Redirects To |
|--------|---------|--------------|
| Register | http://localhost:5007 | http://localhost:5006 |

### Production Mode
| Action | User At | Redirects To |
|--------|---------|--------------|
| Register | https://anac.cs.brown.edu/hanreg | https://anac.cs.brown.edu/hanapp |

---

## Testing

### Test Registration Flow (Local)
```bash
# 1. Start apps
HANI_ENV=local hani --dev

# 2. Browser: http://localhost:5007
# 3. Register user "testuser"
# 4. Click "start negotiating" link
# 5. Should go to: http://localhost:5006
# 6. Login with testuser credentials
```

### Verify Environment Detection
```bash
# Check what environment is active
python -c "from hani.common import HANI_ENV, APP_URLS; import json; print(f'ENV: {HANI_ENV}'); print(json.dumps(APP_URLS, indent=2))"
```

---

## Quick Troubleshooting

```bash
# Check environment
echo $HANI_ENV

# Check which URLs will be used
python -c "from hani.common import APP_URLS; print(APP_URLS)"

# Check if hashed passwords exist
cat ~/negmas/hani/settings/users_hashed.json

# Test password hash
python -c "from hani.auth import hash_password; print(hash_password('yourpassword'))"
```

---

## Full Documentation

See `DEPLOYMENT_GUIDE.md` for complete details on:
- Nginx configuration
- OAuth setup
- Detailed flow diagrams
- Production checklist
