# Secure Environment Variable Setup

## Problem
Your dotfiles (including `.envrc`) are tracked in git, so we can't put OAuth secrets there.

## Solution
Use **`.envrc.local`** which is gitignored and automatically loaded by direnv.

---

## Setup Instructions

### Step 1: Create Your Local Environment File

Copy the example file and add your GitHub OAuth credentials:

```bash
# Copy the template
cp .envrc.local.example .envrc.local

# Edit with your real credentials
nano .envrc.local
```

Replace these values:
- `your_github_client_id_here` → Your actual Client ID from GitHub
- `your_github_client_secret_here` → Your actual Client Secret from GitHub

### Step 2: Allow direnv to Load It

```bash
# direnv will ask you to approve the new .envrc
direnv allow
```

### Step 3: Verify It's Working

```bash
# Check that environment variables are set
echo $HANI_OAUTH_KEY
# Should print your Client ID (not "your_github_client_id_here")

# Verify auth mode detection
cd /Users/yasser/code/projects/han
python -c "from src.hani.auth import get_auth_mode; print(f'Auth mode: {get_auth_mode()}')"
# Should print: Auth mode: oauth
```

---

## How It Works

### File Structure
```
.envrc                  ← Tracked in git ✅ Safe to commit
.envrc.local           ← NOT tracked ❌ Contains secrets
.envrc.local.example   ← Tracked in git ✅ Template only
.gitignore             ← Updated to ignore .envrc.local
```

### Loading Order
1. direnv loads `.envrc` (from git)
2. `.envrc` sources `.envrc.local` (if it exists)
3. Environment variables from `.envrc.local` are now available
4. HANI detects OAuth credentials and uses OAuth mode

---

## What's Protected

✅ `.envrc.local` - Your actual OAuth secrets (gitignored)
✅ `.env` - Alternative format (gitignored)
✅ `.env.local` - Another alternative (gitignored)
✅ `*.secret` - Any file ending in .secret (gitignored)

These files will **NEVER** be committed to git.

---

## Usage Examples

### Use Password Auth (Default)
Don't create `.envrc.local`, or leave OAuth variables unset:
```bash
hani --dev
# Uses password authentication
# Login: admin / Yarab@Satrak19
```

### Use OAuth Auth
Create `.envrc.local` with OAuth credentials:
```bash
# .envrc.local content:
export HANI_OAUTH_KEY=your_client_id
export HANI_OAUTH_SECRET=your_client_secret

# Then run:
direnv allow
hani --dev
# Uses OAuth authentication via GitHub
```

### Force Password Auth (Even with OAuth Credentials)
Add to `.envrc.local`:
```bash
export HANI_AUTH_MODE=password
```

This is useful when you have OAuth configured but want to test password auth.

---

## Getting GitHub OAuth Credentials

If you haven't created your GitHub OAuth app yet:

1. Go to: https://github.com/settings/developers
2. Click **"New OAuth App"**
3. Fill in:
   - Application name: `HANI Local Dev`
   - Homepage URL: `http://localhost:5006`
   - Authorization callback URL: `http://localhost:5006`
4. Click **"Register application"**
5. Copy the **Client ID**
6. Click **"Generate a new client secret"**
7. Copy the **Client Secret** (save it now - you won't see it again!)
8. Add both to your `.envrc.local`

---

## Security Best Practices

### ✅ DO
- Keep `.envrc.local` on your local machine only
- Use different OAuth apps for dev/production
- Regenerate secrets if they're accidentally committed
- Add `.envrc.local` to `.gitignore` (already done ✓)
- Back up `.envrc.local` securely (password manager, encrypted drive)

### ❌ DON'T
- Commit `.envrc.local` to git
- Share OAuth secrets in chat/email
- Use the same OAuth credentials for multiple projects
- Put secrets in `.envrc` (it's tracked in git!)

---

## Troubleshooting

### "Environment variables not set"
```bash
# Check if .envrc.local exists
ls -la .envrc.local

# If not, create it from the example
cp .envrc.local.example .envrc.local
nano .envrc.local

# Reload direnv
direnv allow
```

### "direnv: error .envrc is blocked"
```bash
# Allow direnv to load .envrc
direnv allow
```

### "Still using password auth instead of OAuth"
```bash
# Check if OAuth variables are actually set
echo $HANI_OAUTH_KEY
echo $HANI_OAUTH_SECRET

# If empty, check .envrc.local content
cat .envrc.local

# Make sure it has:
export HANI_OAUTH_KEY=your_actual_key_here
export HANI_OAUTH_SECRET=your_actual_secret_here

# Reload direnv
direnv allow

# Try again
hani --dev
```

### "Want to temporarily disable OAuth"
```bash
# Option 1: Unset the variables in current shell
unset HANI_OAUTH_KEY HANI_OAUTH_SECRET

# Option 2: Force password mode
export HANI_AUTH_MODE=password

# Option 3: Rename .envrc.local temporarily
mv .envrc.local .envrc.local.bak
direnv allow
```

---

## Alternative: One-Time Environment Variables

If you don't want to create `.envrc.local`, you can set variables for a single session:

```bash
# Set OAuth credentials for this terminal session only
export HANI_OAUTH_KEY=your_client_id
export HANI_OAUTH_SECRET=your_client_secret

# Run HANI
hani --dev

# These variables disappear when you close the terminal
```

---

## Verification Checklist

After setup, verify everything is correct:

- [ ] `.envrc.local` exists and contains your OAuth credentials
- [ ] `.envrc.local` is listed in `.gitignore`
- [ ] `direnv allow` has been run
- [ ] `echo $HANI_OAUTH_KEY` prints your Client ID
- [ ] `git status` does NOT show `.envrc.local`
- [ ] HANI starts in OAuth mode when you run `hani --dev`

Run this quick check:
```bash
# Should NOT appear in git
git status | grep envrc.local
# Empty output = good!

# SHOULD appear in gitignore
grep envrc.local .gitignore
# Should print: .envrc.local

# Variables should be set
echo "OAuth Key: ${HANI_OAUTH_KEY:0:10}..."  # Shows first 10 chars
echo "OAuth Secret: ${HANI_OAUTH_SECRET:0:10}..."
```

---

## Summary

✅ **Secrets are safe** - `.envrc.local` is gitignored
✅ **Automatic loading** - direnv loads it automatically
✅ **Flexible** - Can use password or OAuth auth
✅ **Simple** - Just copy and edit one file
✅ **Secure** - No secrets in git, ever

Your OAuth credentials are now safely stored locally and will never be committed to git!

---

## Quick Reference

```bash
# Create local env file
cp .envrc.local.example .envrc.local
nano .envrc.local

# Allow direnv
direnv allow

# Verify
echo $HANI_OAUTH_KEY

# Run with OAuth
hani --dev

# Check what's NOT in git
git status
# .envrc.local should NOT appear
```

**You're all set!** 🔒
