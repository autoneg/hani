# Registration App Fixes

## Changes Made

### 1. **Enabled Password Hashing**
- **Before**: Passwords were stored in plain text (line 8-9 had hashing disabled)
- **After**: Now uses the same SHA-256 hashing as the main app via `hani.auth.hash_password()`

### 2. **Created Hashed Password File**
- **Before**: Only created `users.json` and `users_info.json`
- **After**: Now creates three files:
  - `users_info.json` - Full user info (name, email, signature, plain password)
  - `users.json` - Plain text passwords (for reference)
  - `users_hashed.json` - SHA-256 hashed passwords (used by main app authentication)

### 3. **Dynamic Redirect to Main App**
- **Before**: Hardcoded redirect to `https://anac.cs.brown.edu/hanapp`
- **After**: Reads main app URL from `env.json` and redirects to local app
  - Defaults to `http://localhost:5006` if not in env.json
  - Uses same URL structure as other apps (registration, playground)

### 4. **Loaded App URLs from env.json**
- Added code to read `env.json` at startup
- Makes the registration app aware of other app URLs

## How It Works Now

### Registration Flow:
1. User fills registration form at `http://localhost:5007`
2. User submits registration
3. System creates three files:
   ```
   ~/negmas/hani/settings/users_info.json     - Full details + plain password
   ~/negmas/hani/settings/users.json           - Plain passwords
   ~/negmas/hani/settings/users_hashed.json    - Hashed passwords (SHA-256)
   ```
4. Success message shows with link to main app: `http://localhost:5006`
5. User clicks link and is redirected to main app
6. User logs in with their credentials (main app uses hashed passwords)

### Password Consistency:
- Registration app hashes passwords using `hani.auth.hash_password()`
- Main app authenticates using the same hashing algorithm
- Both use SHA-256 with salt `"hani-secure-salt-2025"`

## Files Modified

- `src/hani/register.py`:
  - Lines 1-16: Added imports and app URL loading
  - Lines 20-36: Updated `save_users()` to create hashed file
  - Lines 91-102: Updated success message with dynamic URL

## Testing

To test the full flow:

```bash
# Start all three apps
cd /Users/yasser/code/projects/han
direnv allow
hani --dev
```

Then:
1. Open `http://localhost:5007` (registration)
2. Register a new user
3. Click the link in success message
4. Should redirect to `http://localhost:5006` (main app)
5. Login with your new credentials

## Three Apps Overview

| App | Port | URL | Auth Required | Purpose |
|-----|------|-----|---------------|---------|
| **Main App** | 5006 | `http://localhost:5006` | ✅ Yes (Dual: Password/OAuth) | Main negotiation interface |
| **Registration** | 5007 | `http://localhost:5007` | ❌ No (Public) | User registration & profile |
| **Playground** | 5008 | `http://localhost:5008` | ❌ No (Public) | Guest/demo mode |

## Security Notes

- Registration app stores plain passwords in `users_info.json` for profile updates
- Main app ONLY uses hashed passwords from `users_hashed.json`
- Registration app is intentionally public (no authentication required)
- Playground app is intentionally public (guest access)

## Future Improvements

Consider for production:
1. Don't store plain passwords even in `users_info.json`
2. Add email verification
3. Add password strength requirements
4. Add rate limiting for registration
5. Add CAPTCHA to prevent spam registrations
