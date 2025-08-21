import panel as pn
import json
from pathlib import Path

pn.extension()

USERS_FILE = Path(__file__).parent / "users_info.json"
LOGIN_FILE = Path(__file__).parent / "users.json"


def hash_password(password):
    return password
    # return hashlib.sha256(password.encode()).hexdigest()


def load_users():
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)
    d = {k: hash_password(v["password"]) for k, v in users.items()}
    with open(LOGIN_FILE, "w") as f:
        json.dump(d, f, indent=2)


# Registration form
reg_consent = pn.pane.Markdown(
    """
**Participation Consent Form**

**Project Name:** Human-Agent Negotiation Competition

**Name and contact information of primary investigator of the project:**

Yasser Mohammad

E-mail: <y.mohammad@nec.com>

Reyhan Aydoğan

E-mail:
[reyhan.aydogan@ozyegin.edu.tr](mailto:reyhan.aydogan@ozyegin.edu.tr)

**Sources of funding or institutional support received for this study:**

[NEC-AIST Collaborative AI Research Laboratory](https://www.airc.aist.go.jp/en/project/overview.html) sponsors the [Human-Agent
                                                                                                                    Negotiation League](https://anac.cs.brown.edu/han) as part of the [ANAC 2025 competition](https://anac.cs.brown.edu).

**Project Purpose:** This competition is a pilot study for the
Human-Agent Negotiation League, which is a part of Automated
Negotiating Agents Competition (ANAC). The ANAC 2025 is held in
conjunction with the International Joint Conference of Artificial
Intelligence (IJCAI 2025), in Montreal, Canada. The goal is to
investigate how to develop AI agents and interfaces that can improve the
outcomes of human-agent negotiations. In this competition, we will
compare the performance of human negotiators when negotiating with AI
agents. We would like to use the collected data in the design of new
virtual agents.

**Process:** The procedure is as follows:

(1) Register a user-name, email and password at the ANAC booth to be
    able to participate in the competition. At the same time, fill the
    [pre-study questionnaire (Q1)](https://forms.gle/c4tNxYof1Gm7ezmC7). The email address is then used to
    announce the winners.

(2) Participants can visit the ANAC booth during its opening hours and
    use the HANI interface to negotiate as many times as they like
    within 30 minutes in each session. All scores will be saved. The
    participant can always check their accumulated score. They can
    conduct as many sessions as they want subject to availability of the
    machines and requests of other participants for participation. After
    the first session, the participant fills a [post-session
    questionnaire (Q2)](https://forms.gle/qT8VtQYrV5RuodML6).

(3) After the completion of all sessions, the organizers will send a
    [post-study questionnaire (Q3)](https://forms.gle/HEmb2CpruFD8aUvt8) to all participants to fill.

(4) The score of each participant will be calculated as the truncated
    mean of their scores (i.e. using only top 10 scores for each
    participant). Because we use the top-10 scores only, we do not
    conduct separate familiarization sessions as the participant can
    familiarize herself/himself with the interface for as long as they
    want provided that they conduct 10 negotiations after that that will
    be used in their scoring.

(5) The winners (i.e. participants with highest scores) will be
    announced during IJCAI and will be published in the ANAC webpage at
    <https://anac.brown.edu.eg/han>

(6) All the negotiation data and the HANI interface source code will be
    made publicly available after IJCAI. Questionnaires will NOT be made
    publicly available.

**Privacy:** We take the privacy of participants seriously:

1.  The emails collected will only be used to send the post-study
    questionnaire and announced the winners. It will be deleted from the
    system one month later. No emails will ever be published.

2.  We will not collect identifying information from the participants
    other than the name (signed with in this form) and will not publish
    any such information.

3.  The users can register any user-name they like.

4.  All data will be anonymized before being made publicly available. Negotiation logs and results will e saved without any identification of the competitors. The anonymized data will be published to help the research community to develop better negotiation agents and interfaces. Usernames will also be anonymized so feel free to use any username.

5.  Demographical data such as age, gender and education level could be
    saved for further analysis without any identification of the
    participants.

**Voluntary participation:** Participation in this study is voluntary.
Participants can seize participation at any point of the study with no
negative consequences. If you have any questions, concerns, or
suggestions regarding the ethical aspects of the research described in
this form and/or the details of the research, please contact the Özyeğin
University Ethics Committee at (216) 564 91 76.


You **must** satisfy the following conditions to be considered for the official competition and the monetary prize:

- You must be at least 18 years old.
- You must agree to the consent form you are reading now by writing your full name in the "Name" field and the current date in the "Date of Consent" field. The concatenation of these two fields will be considered your dated signature.
- You must fill the [Pre-Competition-Questionnaire](https://forms.gle/c4tNxYof1Gm7ezmC7) **before** conducting any negotiations with our system.
- You must fill the [Post-Competition-Questionnaire](https://forms.gle/HEmb2CpruFD8aUvt8) **after** conducting all negotiations. We will email you the link to this questionnaire after the competition ends.
- There are three different negotiation scenario types in HAN 2025 (Grocery, Island and Simple Trade). You need to log at least two negotiations on each scenario.

You can check the methodology, data management conditions and information declaration [here](https://anac.cs.brown.edu/files/han/privacy_and_conditions.pdf).

Please visit us at the ANAC competition booth if you have any further questions or for support in filling this form (Booth 7) from 9am to 12pm and 2pm to 5pm on August 20th, and 201st 2025 or from 9am to 12pm on August 22nd, 2025.
""",
)
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
register_btn = pn.widgets.Button(name="Register", button_type="primary")
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
        "password": hash_password(reg_password.value),
        "email": email,
        "name": name,
        "signature": signature,
        "date_of_signature": sig_date,
    }
    save_users(users)
    reg_message.object = "Registration successful!\n\nPlease be sure to complete the [Pre-Competition-Questionnaire](https://forms.gle/c4tNxYof1Gm7ezmC7) **before** conducting any negotiations.\n\n [You can start negotiating here](https://anac.cs.brown.edu/hanapp)."


register_btn.on_click(register)

registration_panel = pn.Column(
    "## Register",
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
login_btn = pn.widgets.Button(name="Login", button_type="primary")
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
    if username in users and users[username]["password"] == hash_password(password):
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
    "## Login", login_username, login_password, login_btn, login_message
)

main = pn.Column(
    "# User Registration and Login App",
    registration_panel,
    login_panel,
    logout_btn,
    profile_panel,
)

main.servable()
