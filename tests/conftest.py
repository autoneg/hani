"""
Pytest configuration and shared fixtures for HANI testing.

This module provides fixtures for testing the three HANI apps:
- Main app (port 5006) - Authenticated negotiation interface
- Registration app (port 5007) - User registration and profile management
- Playground app (port 5008) - Public guest/demo mode
"""

import json
import os
import pytest
import shutil
import time
from pathlib import Path
from playwright.sync_api import Page, expect
import subprocess
import signal

# Test configuration
TEST_SETTINGS_DIR = Path.home() / "negmas" / "hani" / "test_settings"
HANI_PORT = 5006
REG_PORT = 5007
GUEST_PORT = 5008

# Test user credentials
TEST_USER = "test_user"
TEST_PASSWORD = "TestPass123!"
ADMIN_USER = "admin"
ADMIN_PASSWORD = "Yarab@Satrak19"


@pytest.fixture(scope="session")
def test_settings_dir():
    """
    Create a test settings directory to isolate test data from real user data.

    This fixture:
    - Creates a temporary settings directory
    - Copies necessary files from real settings
    - Creates test user files
    - Cleans up after all tests complete
    """
    # Create test settings directory
    TEST_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Get real settings directory
    real_settings = Path.home() / "negmas" / "hani" / "settings"

    # Copy necessary configuration files
    files_to_copy = [
        "consent.md",
        "scenario_order.txt",
        "env.local.json",
        "env.production.json",
    ]

    for file_name in files_to_copy:
        src = real_settings / file_name
        if src.exists():
            shutil.copy(src, TEST_SETTINGS_DIR / file_name)

    # Create test users_info.json (plain text passwords for registration app)
    test_users_info = {
        TEST_USER: {
            "password": TEST_PASSWORD,
            "email": "test@example.com",
            "name": "Test User",
        },
        ADMIN_USER: {
            "password": ADMIN_PASSWORD,
            "email": "admin@example.com",
            "name": "Admin User",
        },
    }

    with open(TEST_SETTINGS_DIR / "users_info.json", "w") as f:
        json.dump(test_users_info, f, indent=2)

    # Create test users.json (plain text passwords for Panel's basic auth)
    test_users = {
        TEST_USER: TEST_PASSWORD,
        ADMIN_USER: ADMIN_PASSWORD,
    }

    with open(TEST_SETTINGS_DIR / "users.json", "w") as f:
        json.dump(test_users, f, indent=2)

    # Copy scenarios if they exist
    scenarios_src = real_settings / "scenarios"
    scenarios_dst = TEST_SETTINGS_DIR / "scenarios"
    if scenarios_src.exists():
        shutil.copytree(scenarios_src, scenarios_dst, dirs_exist_ok=True)

    yield TEST_SETTINGS_DIR

    # Cleanup after all tests
    # Note: Commented out to allow inspection after test runs
    # shutil.rmtree(TEST_SETTINGS_DIR)


@pytest.fixture(scope="session")
def hani_servers(test_settings_dir):
    """
    Start all three HANI servers for testing.

    This fixture:
    - Sets environment variables to use test settings
    - Starts main app, registration app, and playground app
    - Waits for servers to be ready
    - Terminates servers after all tests complete

    Returns:
        dict: URLs for each app
    """
    # Set environment to use test settings
    env = os.environ.copy()
    env["HANI_SETTINGS_DIR"] = str(test_settings_dir)
    env["HANI_ENV"] = "local"

    # Start servers
    processes = []

    # Start main app
    main_proc = subprocess.Popen(
        [
            "panel",
            "serve",
            "src/hani/app.py",
            "--port",
            str(HANI_PORT),
            "--show",
            "False",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    processes.append(main_proc)

    # Start registration app
    reg_proc = subprocess.Popen(
        [
            "panel",
            "serve",
            "src/hani/register.py",
            "--port",
            str(REG_PORT),
            "--show",
            "False",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    processes.append(reg_proc)

    # Start playground app
    guest_proc = subprocess.Popen(
        [
            "panel",
            "serve",
            "src/hani/runguest.py",
            "--port",
            str(GUEST_PORT),
            "--show",
            "False",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    processes.append(guest_proc)

    # Wait for servers to start
    time.sleep(5)

    urls = {
        "main": f"http://localhost:{HANI_PORT}",
        "registration": f"http://localhost:{REG_PORT}",
        "playground": f"http://localhost:{GUEST_PORT}",
    }

    yield urls

    # Cleanup: terminate all servers
    for proc in processes:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def browser_context(playwright):
    """
    Create a fresh browser context for each test.

    This ensures test isolation by:
    - Creating a new browser context
    - Clearing cookies and localStorage
    - Providing a clean slate for each test
    """
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        ignore_https_errors=True,
    )

    yield context

    context.close()
    browser.close()


@pytest.fixture
def page(browser_context):
    """Create a new page for each test."""
    page = browser_context.new_page()
    yield page
    page.close()


@pytest.fixture
def authenticated_page(page, hani_servers):
    """
    Provide a page that's already logged in to the main app.

    This fixture:
    - Navigates to the main app
    - Logs in with test credentials
    - Waits for successful login
    - Returns the authenticated page

    Useful for tests that need to skip the login flow.
    """
    page.goto(hani_servers["main"])

    # Fill in login form
    page.fill('input[name="username"]', TEST_USER)
    page.fill('input[name="password"]', TEST_PASSWORD)

    # Click login button
    page.click('button:has-text("Sign in")')

    # Wait for successful login (URL change or welcome message)
    page.wait_for_url(f"{hani_servers['main']}/**", timeout=10000)

    yield page


# Helper functions for common test operations


def wait_for_element(page: Page, selector: str, timeout: int = 10000):
    """Wait for an element to appear on the page."""
    page.wait_for_selector(selector, timeout=timeout)


def fill_form(page: Page, form_data: dict):
    """Fill in a form with the given data."""
    for name, value in form_data.items():
        page.fill(f'input[name="{name}"]', value)


def click_button(page: Page, text: str):
    """Click a button with the given text."""
    page.click(f'button:has-text("{text}")')


def assert_text_visible(page: Page, text: str):
    """Assert that text is visible on the page."""
    expect(page.locator(f'text="{text}"')).to_be_visible()


def assert_url_contains(page: Page, text: str):
    """Assert that the current URL contains the given text."""
    expect(page).to_have_url(f"**{text}**")
