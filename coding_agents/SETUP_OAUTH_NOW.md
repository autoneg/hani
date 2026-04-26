# Quick OAuth Setup - Next Steps

## ✅ Files Created

- `.envrc.local` - Ready for your OAuth credentials
- `.envrc` - Updated to load `.envrc.local`
- `.gitignore` - Updated to protect your secrets
- `.envrc.local.example` - Template for reference

## 🔐 Your .envrc.local is Protected

```bash
# Verify it's not tracked by git
git status | grep envrc.local
# Should be empty = good! ✓
```

## 📝 Next Steps

### 1. Add Your GitHub OAuth Credentials

Edit `.envrc.local` and replace the placeholders:

```bash
nano .envrc.local
```

Replace these lines:
```bash
export HANI_OAUTH_KEY=PUT_YOUR_GITHUB_CLIENT_ID_HERE
export HANI_OAUTH_SECRET=PUT_YOUR_GITHUB_CLIENT_SECRET_HERE
```

With your actual credentials from GitHub:
```bash
export HANI_OAUTH_KEY=Ov23liABCDEF1234567890
export HANI_OAUTH_SECRET=1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t
```

### 2. Allow direnv to Load It

```bash
direnv allow
```

You should see:
```
direnv: loading ~/code/projects/han/.envrc
direnv: loading ~/code/projects/han/.envrc.local
```

### 3. Verify It Worked

```bash
# Check environment variables are set
echo $HANI_OAUTH_KEY
# Should print your actual Client ID

# Check auth mode detection
python -c "from src.hani.auth import get_auth_mode; print(f'Auth mode: {get_auth_mode()}')"
# Should print: Auth mode: oauth
```

### 4. Run HANI with OAuth

```bash
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

Then open http://localhost:5006 and login via GitHub!

---

## 🔄 Switching Between Auth Modes

### Use Password Auth (Temporarily)
```bash
# Option 1: Unset OAuth variables for this session
unset HANI_OAUTH_KEY HANI_OAUTH_SECRET
hani --dev

# Option 2: Force password mode in .envrc.local
echo "export HANI_AUTH_MODE=password" >> .envrc.local
direnv allow
hani --dev
```

### Back to OAuth
```bash
# Remove the force override
nano .envrc.local  # Comment out or remove HANI_AUTH_MODE line
direnv allow
hani --dev
```

---

## 🆘 If You Don't Have OAuth Credentials Yet

Get them from GitHub:

1. Go to: https://github.com/settings/developers
2. Click "New OAuth App"
3. Fill in:
   - **Application name**: `HANI Local Dev`
   - **Homepage URL**: `http://localhost:5006`
   - **Authorization callback URL**: `http://localhost:5006`
4. Click "Register application"
5. Copy the **Client ID**
6. Click "Generate a new client secret"
7. Copy the **Client Secret** (you won't see it again!)
8. Add both to `.envrc.local`

---

## 📋 Quick Checklist

- [ ] `.envrc.local` exists with your OAuth credentials
- [ ] `direnv allow` has been run
- [ ] `echo $HANI_OAUTH_KEY` shows your Client ID
- [ ] `git status` does NOT show `.envrc.local`
- [ ] HANI starts with "Authentication mode: OAUTH"

---

## 🎯 Current Status

✅ `.envrc` - Updated to load `.envrc.local`
✅ `.envrc.local` - Created and waiting for your credentials
✅ `.gitignore` - Updated to protect secrets
✅ Git - `.envrc.local` is ignored (safe!)

**Next:** Edit `.envrc.local` with your GitHub OAuth Client ID and Secret, then run `direnv allow`
