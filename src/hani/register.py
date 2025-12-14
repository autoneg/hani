import panel as pn
import json
from hani.common import CONSENT_FILE, USERS_FILE, LOGIN_FILE, APP_URLS, HANI_ENV
from hani.auth import hash_password as hash_password_secure

pn.extension()


def hash_password(password):
    """Hash password using the same method as main app authentication"""
    return hash_password_secure(password)


def load_users():
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    """Save users to both users_info.json and create hashed password files"""
    # Save full user info (includes plain text password for profile updates)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

    # Create plain text passwords file (for reference)
    plain_passwords = {k: v["password"] for k, v in users.items()}
    with open(LOGIN_FILE, "w") as f:
        json.dump(plain_passwords, f, indent=2)

    # Create hashed passwords file (used for authentication)
    hashed_file = LOGIN_FILE.parent / "users_hashed.json"
    hashed_passwords = {k: hash_password(v["password"]) for k, v in users.items()}
    with open(hashed_file, "w") as f:
        json.dump(hashed_passwords, f, indent=2)


# Registration form
reg_consent = pn.pane.Markdown(CONSENT_FILE.read_text())
reg_signature = pn.widgets.Checkbox(
    name="I hereby confirm that I have read relevant details of Human Agent Negotiation Competition and all my questions (if any) were answered by the researcher. I consent to participate in this study voluntarily. "
)
reg_pre = pn.widgets.Checkbox(
    name="I filled the pre-competition questionnaire at https://forms.gle/c4tNxYof1Gm7ezmC7"
)
reg_name = pn.widgets.TextInput(name="Full Name (Signature)")
reg_date = pn.widgets.TextInput(name="Date of consent (yyyy-mm-dd)")
reg_username = pn.widgets.TextInput(name="Username")
reg_password = pn.widgets.PasswordInput(name="Password")
reg_email = pn.widgets.TextInput(name="Email")
reg_email_confirm = pn.widgets.TextInput(name="Confirm Email")
register_btn = pn.widgets.Button(name=" Register", button_type="primary")
reg_message = pn.pane.Markdown("")


def register(event):
    users = load_users()
    username = reg_username.value.strip()
    name = reg_name.value.strip()
    sig_date = reg_date.value.strip()
    signature = reg_signature.value
    email = reg_email.value.strip()
    email_confirm = reg_email_confirm.value.strip()
    if not signature:
        reg_message.object = "You MUST accept the consent form by checking the checkbox above to register."
        return
    if not reg_pre.value:
        reg_message.object = "You MUST complete the [Pre-Competition-Questionnaire](https://forms.gle/c4tNxYof1Gm7ezmC7) and check the corresponding checkbox above to be registered."
        return
    if not name:
        reg_message.object = "You MUST enter your full name."
        return
    if not sig_date:
        reg_message.object = "You MUST enter the date of consent."
        return
    if username.lower().strip() == "ai":
        reg_message.object = (
            "Cannot use AI as your username. Please choose a different username."
        )
        return
    if not username or not reg_password.value:
        reg_message.object = "Username and password required."
        return
    if email != email_confirm:
        reg_message.object = "Emails do not match."
        return
    if name in [_.get("name", "") for _ in users.values()]:
        reg_message.object = "Your full name is already registered."
        return
    if username in users:
        reg_message.object = "Username already exists."
        return
    users[username] = {
        "password": reg_password.value,  # Store plain text in users_info.json
        "email": email,
        "name": name,
        "signature": signature,
        "date_of_signature": sig_date,
    }
    save_users(users)

    # Get main app URL from env.json
    main_app_url = APP_URLS.get("app", "http://localhost:5006")

    reg_message.object = f"Registration successful!🎉🎉\n\nPlease be sure to complete the [Pre-Competition-Questionnaire](https://forms.gle/c4tNxYof1Gm7ezmC7) **before** conducting any negotiations.\n\n[You can start negotiating here]({main_app_url})."


register_btn.on_click(register)

registration_panel = pn.Column(
    "## Register for HAN",
    reg_consent,
    reg_signature,
    reg_pre,
    reg_name,
    reg_date,
    reg_username,
    reg_password,
    reg_email,
    reg_email_confirm,
    register_btn,
    reg_message,
)

# Login form
login_username = pn.widgets.TextInput(name="Username")
login_password = pn.widgets.PasswordInput(name="Password")
login_btn = pn.widgets.Button(name="✏️ Edit Profile", button_type="primary")
login_message = pn.pane.Markdown("")
logout_btn = pn.widgets.Button(name="Logout", button_type="danger", visible=False)

# Profile
profile_name = pn.widgets.TextInput(name="Name (Signature)", disabled=True)
profile_date = pn.widgets.TextInput(name="Date of Consent", disabled=True)
profile_email = pn.widgets.TextInput(name="Email")
profile_save_btn = pn.widgets.Button(name="Save Profile", button_type="primary")
profile_message = pn.pane.Markdown("")
profile_panel = pn.Column(
    "## Profile",
    profile_name,
    profile_date,
    profile_email,
    profile_save_btn,
    profile_message,
)
profile_panel.visible = False

current_user = {"username": None}


def login(event):
    users = load_users()
    username = login_username.value.strip()
    password = login_password.value
    # users_info.json stores plain text passwords for profile updates
    # Compare plain text to plain text
    if username in users and users[username]["password"] == password:
        current_user["username"] = username
        login_message.object = f"Welcome, {username}!"
        login_panel.visible = False
        registration_panel.visible = False
        logout_btn.visible = True
        profile_panel.visible = True
        profile_email.value = users[username].get("email", "")
        profile_name.value = users[username].get("name", "")
        profile_date.value = users[username].get("date_of_signature", "")
        profile_message.object = ""
    else:
        login_message.object = "Invalid username or password."


def logout(event):
    current_user["username"] = None
    login_panel.visible = True
    registration_panel.visible = True
    logout_btn.visible = False
    profile_panel.visible = False
    login_message.object = ""
    login_username.value = ""
    login_password.value = ""


def save_profile(event):
    username = current_user["username"]
    if not username:
        profile_message.object = "Not logged in."
        return
    users = load_users()
    users[username]["email"] = profile_email.value.strip()
    save_users(users)
    profile_message.object = "Profile updated."


login_btn.on_click(login)
logout_btn.on_click(logout)
profile_save_btn.on_click(save_profile)

login_panel = pn.Column(
    "## Profile Update",
    "After registration, you can update your profile here.",
    login_username,
    login_password,
    login_btn,
    login_message,
)

main = pn.Column(
    registration_panel,
    login_panel,
    logout_btn,
    profile_panel,
)

main.servable(title="HAN Registration & Profile Management")
