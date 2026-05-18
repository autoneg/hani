# Password Hashing Implementation

## Overview

The HANI application now uses SHA-256 hashed passwords instead of storing plain text passwords. This improves security by ensuring that even if the users file is compromised, the actual passwords cannot be easily recovered.

## Implementation

### Files Modified
1. **`src/hani/auth.py`** - New module providing:
   - `hash_password()` - Hash passwords with SHA-256 and salt
   - `create_hashed_users_file()` - Convert plain text to hashed format
   - `patch_panel_auth()` - Monkey-patch Panel's authentication to validate hashed passwords
   - `verify_password()` - Verify plain password against hash

2. **`src/hani/runapp.py`** - Updated to:
   - Automatically convert plain text passwords to hashed on first run
   - Patch Panel's authentication before starting server
   - Use hashed password file

### How It Works

1. **Hashing Algorithm**: SHA-256 with a salt
   ```python
   hash = SHA256(salt + password)
   ```

2. **Salt**: `"hani-secure-salt-2025"` (defined in `auth.py`)

3. **Automatic Conversion**: On first run of `runapp.py`:
   - Backs up plain text file to `users_plain_backup.json`
   - Creates `users_hashed.json` with hashed passwords
   - Uses hashed file for all subsequent authentication

4. **Authentication Flow**:
   ```
   User login → Plain password → Hash with salt → Compare with stored hash
   ```

## Usage

### For Development

The hashing happens automatically when you start the app:
```bash
hani --dev
# Or
python src/hani/runapp.py --dev
```

First run output:
```
⚠️  Converting plain text passwords to hashed format...
✓ Backed up plain text passwords to .../users_plain_backup.json
Created hashed users file at .../users_hashed.json
Converted 44 users
✓ Created hashed password file: .../users_hashed.json
✓ Panel authentication patched to use hashed passwords
```

### Manual Password Hashing

To manually hash a users file:
```bash
python src/hani/auth.py <plain_file> [hashed_output_file]
```

Example:
```bash
python src/hani/auth.py ~/negmas/hani/settings/users.json ~/negmas/hani/settings/users_hashed.json
```

### Adding New Users

**Option 1**: Add to plain text backup, then rehash:
```bash
# Edit the backup file
nano ~/negmas/hani/settings/users_plain_backup.json

# Rehash it
python src/hani/auth.py ~/negmas/hani/settings/users_plain_backup.json ~/negmas/hani/settings/users_hashed.json
```

**Option 2**: Hash individual password in Python:
```python
from hani.auth import hash_password

password_hash = hash_password("new_password")
# Add to users_hashed.json: "username": "hash_value"
```

## Security Notes

1. **Salt is hardcoded**: The salt is in the source code. For production, consider:
   - Using environment variables for the salt
   - Using per-user salts
   - Using more secure algorithms like bcrypt or argon2

2. **Plain text backup**: The `users_plain_backup.json` file contains plain passwords
   - Keep it secure or delete it after hashing
   - Add to `.gitignore`

3. **Hash Algorithm**: SHA-256 is fast but not ideal for passwords
   - Consider upgrading to bcrypt, scrypt, or argon2 for production
   - These are designed to be slow, making brute-force attacks harder

## Files

- `~/negmas/hani/settings/users.json` - Original plain text (will be replaced with hashed)
- `~/negmas/hani/settings/users_hashed.json` - Hashed passwords (used for auth)
- `~/negmas/hani/settings/users_plain_backup.json` - Backup of plain text

## Example

Plain text users.json:
```json
{
  "yasser": "yasser",
  "admin": "Yarabsatrak19"
}
```

Hashed users_hashed.json:
```json
{
  "yasser": "a7f8d9e1b2c3...",
  "admin": "f4e5d6c7b8a9..."
}
```

Login still works the same way - users enter their plain password, and the system hashes it before comparison.
