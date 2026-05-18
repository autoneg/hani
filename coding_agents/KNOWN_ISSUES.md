# Known Issues and Fixes

## stringcase SyntaxWarning

**Issue:** On Python 3.13, you may see this warning at startup:
```
/path/to/site-packages/stringcase.py:247: SyntaxWarning: invalid escape sequence '\W'
```

**Cause:** The `stringcase` library (v1.2.0, last updated 2017) uses an invalid escape sequence.

**Fix:** Patch the library file:
```bash
sed -i.bak 's/return re.sub("\\W+", "", string)/return re.sub(r"\\W+", "", string)/' \
  /path/to/site-packages/stringcase.py
```

Or manually edit line 247 to use a raw string: `r"\W+"`

**Note:** This fix needs to be reapplied if you recreate your virtual environment.

**Alternative:** Consider replacing `stringcase` with a more modern alternative if possible, or vendor the fixed version.

---

## Login Issues - Cannot authenticate with users.json

**Issue:** Cannot login even though username/password exists in `src/hani/users.json`.

**Cause:** The application looks for users at `~/negmas/hani/settings/users.json` (defined in `common.py`), not in the source directory. If you pass `--basic-auth` on the command line, it overrides the default path.

**Fix:**
1. **Remove `--basic-auth` from your command** - The app will automatically use `~/negmas/hani/settings/users.json`
2. **Or ensure the path is correct** - If using `run.sh`, make sure it doesn't override with wrong path

**Correct startup:**
```bash
# Let the app use the default path
hani --dev

# Or explicitly use the correct path
hani --basic-auth ~/negmas/hani/settings/users.json --dev
```

**What was wrong in run.sh:**
```bash
# Bad - wrong path that doesn't exist or overrides correct config
hani --basic-auth src/hani/users.json ...

# Good - removed --basic-auth, uses default from common.py
hani --cookie-secret my_super_safe_cookie_secret --dev ...
```

**Files involved:**
- `src/hani/common.py` - Defines `LOGIN_FILE` path
- `src/hani/runapp.py` - Uses `LOGIN_FILE` when starting panel serve
- `~/negmas/hani/settings/users.json` - Actual users file location

---

## FrozenInstanceError when loading scenarios

**Issue:** `FrozenInstanceError` when trying to login and load scenarios:
```
attr.exceptions.FrozenInstanceError
  File "/path/to/negmas/src/negmas/inout.py", line 897, in from_yaml_files
    os.path = Path(domain)
```

**Cause:** The outcome space objects are frozen (immutable using `@frozen` from attrs). The code was trying to set the `path` attribute after deserialization, but frozen instances cannot be modified after creation.

**Root Cause:** Lines 897 and 907-909 in `negmas/src/negmas/inout.py` were attempting:
```python
os = deserialize(...)
os.path = Path(domain)  # FAILS: cannot modify frozen object
```

**Fix:** Pass the `path` during deserialization by adding it to the dict before deserializing:

**Changed lines 892-897:**
```python
# Before:
os = deserialize(
    adjust_type(load(domain)),
    base_module="negmas",
    python_class_identifier=python_class_identifier,
)
os.path = Path(domain)  # This failed!

# After:
domain_dict = adjust_type(load(domain))
domain_dict["path"] = Path(domain)  # Add path to dict
os = deserialize(
    domain_dict,
    base_module="negmas",
    python_class_identifier=python_class_identifier,
)  # Path is set during construction
```

**Changed lines 899-909:**
```python
# Before:
utils = [deserialize(adjust_type(load(fname), domain=os), ...) for fname in ufuns]
for u, path in zip(utils, ufuns):
    u.outcome_space = os  # These fail on frozen objects
    u.path = path

# After:
utils = [
    deserialize(
        adjust_type(load(fname), domain=os) | {"path": path},  # Merge path into dict
        ...,
    )
    for fname, path in zip(ufuns, ufuns)
]  # No post-processing needed
```

**Files modified:**
- `/Users/yasser/code/projects/negmas/src/negmas/inout.py` - Lines 892-906

**Status:** ✅ Fixed
