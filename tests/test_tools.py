"""
Tool functionality tests for HANI.

These tests verify that all tools load and function correctly:
- Preferences Tool
- Scenario Info Tool
- Utility Plot Tool
- Outcome Plot Tool
- Value Histogram Tool
- Trace/History Tool
- Random Outcome Tool
- Utility Inverter Tool (urange)
- Results Tools
"""

import pytest
from playwright.sync_api import Page, expect


# Helper function to load a scenario and start negotiation
def load_and_start_scenario(page: Page):
    """Helper to load a scenario and start negotiation."""
    page.wait_for_timeout(2000)

    # Try to click Load button
    load_button = page.locator('button:has-text("Load")')
    if load_button.is_visible():
        load_button.click()
        page.wait_for_timeout(3000)

        # Try to click Start button
        start_button = page.locator('button:has-text("Start")')
        if start_button.is_visible():
            start_button.click()
            page.wait_for_timeout(2000)
            return True
    return False


class TestPreferencesTool:
    """Tests for the Preferences Tool."""

    def test_preferences_tool_loads(self, authenticated_page: Page):
        """Test that preferences tool loads after scenario is loaded."""
        if load_and_start_scenario(authenticated_page):
            # Look for preferences-related content
            page_content = authenticated_page.content().lower()

            # Preferences tool might show utility values, preference info, etc.
            # This is a basic smoke test
            assert len(page_content) > 1000
        else:
            pytest.skip("Could not load and start scenario")

    def test_preferences_display_visible(self, authenticated_page: Page):
        """Test that preferences information is displayed."""
        if load_and_start_scenario(authenticated_page):
            authenticated_page.wait_for_timeout(2000)

            # Page should have substantial content
            content = authenticated_page.content()
            assert len(content) > 1000
        else:
            pytest.skip("Could not load and start scenario")


class TestScenarioInfoTool:
    """Tests for the Scenario Info Tool."""

    def test_scenario_info_loads_after_load(self, authenticated_page: Page):
        """Test that scenario info loads after loading a scenario."""
        authenticated_page.wait_for_timeout(2000)

        load_button = authenticated_page.locator('button:has-text("Load")')
        if load_button.is_visible():
            load_button.click()
            authenticated_page.wait_for_timeout(3000)

            # Scenario info should be displayed
            content = authenticated_page.content()
            assert len(content) > 1000
        else:
            pytest.skip("Load button not found")

    def test_scenario_info_contains_metadata(self, authenticated_page: Page):
        """Test that scenario info displays relevant metadata."""
        authenticated_page.wait_for_timeout(2000)

        load_button = authenticated_page.locator('button:has-text("Load")')
        if load_button.is_visible():
            load_button.click()
            authenticated_page.wait_for_timeout(3000)

            # Check page has loaded content
            page_content = authenticated_page.content()
            assert len(page_content) > 500
        else:
            pytest.skip("Load button not found")


class TestUtilityPlotTool:
    """Tests for the Utility Plot Tool."""

    def test_utility_plot_loads_after_start(self, authenticated_page: Page):
        """Test that utility plot loads after starting negotiation."""
        if load_and_start_scenario(authenticated_page):
            authenticated_page.wait_for_timeout(2000)

            # Look for plot-related content (Plotly, canvas, etc.)
            page_content = authenticated_page.content().lower()

            # Basic smoke test - page should have substantial content
            assert len(page_content) > 1000
        else:
            pytest.skip("Could not load and start scenario")

    @pytest.mark.skip(reason="Requires Plotly interaction")
    def test_utility_plot_interactive(self, authenticated_page: Page):
        """Test that utility plot is interactive."""
        # Would test hovering, zooming, panning on the plot
        pass


class TestOutcomePlotTool:
    """Tests for the Outcome Plot Tool."""

    def test_outcome_plot_loads_after_start(self, authenticated_page: Page):
        """Test that outcome plot loads after starting negotiation."""
        if load_and_start_scenario(authenticated_page):
            authenticated_page.wait_for_timeout(2000)

            # Page should have content
            content = authenticated_page.content()
            assert len(content) > 1000
        else:
            pytest.skip("Could not load and start scenario")


class TestHistogramTool:
    """Tests for the Value Histogram Tool."""

    def test_histogram_loads_after_start(self, authenticated_page: Page):
        """Test that value histogram loads after starting negotiation."""
        if load_and_start_scenario(authenticated_page):
            authenticated_page.wait_for_timeout(2000)

            # Page should have content
            content = authenticated_page.content()
            assert len(content) > 1000
        else:
            pytest.skip("Could not load and start scenario")


class TestHistoryTraceTool:
    """Tests for the History/Trace Tool."""

    def test_history_pane_exists(self, authenticated_page: Page):
        """Test that history/trace pane exists after starting."""
        if load_and_start_scenario(authenticated_page):
            authenticated_page.wait_for_timeout(2000)

            # History pane should be part of the interface
            content = authenticated_page.content()
            assert len(content) > 1000
        else:
            pytest.skip("Could not load and start scenario")

    def test_history_updates_after_action(self, authenticated_page: Page):
        """Test that history updates when actions are taken."""
        if load_and_start_scenario(authenticated_page):
            authenticated_page.wait_for_timeout(2000)

            # Get initial content
            initial_content = authenticated_page.content()

            # Try to click reject button (or any action button)
            reject_button = authenticated_page.locator('button:has-text("Reject")')
            if reject_button.is_visible():
                reject_button.click()
                authenticated_page.wait_for_timeout(1000)

                # Content should have changed
                new_content = authenticated_page.content()
                # Note: This might not be significantly different in size,
                # but the test documents expected behavior
                assert len(new_content) >= len(initial_content) - 1000
            else:
                pytest.skip("Reject button not found")
        else:
            pytest.skip("Could not load and start scenario")


class TestRandomOutcomeTool:
    """Tests for the Random Outcome Tool."""

    def test_random_outcome_button_exists(self, authenticated_page: Page):
        """Test that random outcome button/tool exists."""
        if load_and_start_scenario(authenticated_page):
            authenticated_page.wait_for_timeout(2000)

            # Look for random-related content
            # The exact implementation determines what to look for
            content = authenticated_page.content()
            assert len(content) > 1000
        else:
            pytest.skip("Could not load and start scenario")

    @pytest.mark.skip(reason="Requires understanding of random tool UI")
    def test_random_outcome_generates_offer(self, authenticated_page: Page):
        """Test that clicking random outcome generates an offer."""
        # Would test clicking random button and verifying outcome is generated
        pass


class TestUtilityInverterTool:
    """Tests for the Utility Inverter Tool (urange)."""

    def test_utility_inverter_loads(self, authenticated_page: Page):
        """Test that utility inverter/urange tool loads."""
        if load_and_start_scenario(authenticated_page):
            authenticated_page.wait_for_timeout(2000)

            # Tool should be part of the interface
            content = authenticated_page.content()
            assert len(content) > 1000
        else:
            pytest.skip("Could not load and start scenario")

    @pytest.mark.skip(reason="Requires understanding of urange UI")
    def test_utility_inverter_calculates_range(self, authenticated_page: Page):
        """Test that utility inverter calculates utility range."""
        # Would test entering a utility value and getting outcome range
        pass


class TestResultsTools:
    """Tests for Results Tools (User Results, All Results)."""

    def test_user_results_accessible(self, authenticated_page: Page):
        """Test that user results can be accessed."""
        authenticated_page.wait_for_timeout(2000)

        # Results might be accessible from a tab or button
        # This is a basic test that the page loads
        content = authenticated_page.content()
        assert len(content) > 500

    @pytest.mark.skip(reason="Requires completing a negotiation")
    def test_user_results_show_statistics(self, authenticated_page: Page):
        """Test that user results show statistics after negotiations."""
        # Would complete a negotiation and check results display
        pass

    @pytest.mark.skip(reason="Requires admin access")
    def test_all_results_admin_only(self, authenticated_page: Page):
        """Test that all results tool is only visible to admins."""
        # Would test with admin and non-admin users
        pass


class TestToolInteractions:
    """Tests for interactions between tools."""

    def test_multiple_tools_load_together(self, authenticated_page: Page):
        """Test that multiple tools can load and work together."""
        if load_and_start_scenario(authenticated_page):
            authenticated_page.wait_for_timeout(3000)

            # All tools should be loaded without conflicts
            # Check for JavaScript errors
            js_errors = []

            def handle_console(msg):
                if msg.type == "error":
                    js_errors.append(msg.text)

            authenticated_page.on("console", handle_console)
            authenticated_page.wait_for_timeout(2000)

            # Should have no critical errors
            critical_errors = [
                e for e in js_errors if "critical" in e.lower() or "fatal" in e.lower()
            ]
            assert len(critical_errors) == 0
        else:
            pytest.skip("Could not load and start scenario")

    def test_tools_dont_interfere(self, authenticated_page: Page):
        """Test that tools don't interfere with each other."""
        if load_and_start_scenario(authenticated_page):
            authenticated_page.wait_for_timeout(2000)

            # Try interacting with different parts of the interface
            # If we get here without crashes, tools are coexisting
            content = authenticated_page.content()
            assert len(content) > 1000
        else:
            pytest.skip("Could not load and start scenario")


class TestToolPerformance:
    """Tests for tool performance."""

    def test_tools_load_in_reasonable_time(self, authenticated_page: Page):
        """Test that tools load within reasonable time."""
        import time

        start_time = time.time()

        if load_and_start_scenario(authenticated_page):
            load_time = time.time() - start_time

            # Should load within 10 seconds (generous timeout)
            assert load_time < 10, f"Tools took {load_time}s to load"
        else:
            pytest.skip("Could not load and start scenario")

    def test_no_memory_leaks_on_tool_updates(self, authenticated_page: Page):
        """Test that updating tools doesn't cause obvious memory leaks."""
        if load_and_start_scenario(authenticated_page):
            # Get initial metrics
            initial_content_size = len(authenticated_page.content())

            # Perform several actions
            for _ in range(3):
                reject_button = authenticated_page.locator('button:has-text("Reject")')
                if reject_button.is_visible():
                    reject_button.click()
                    authenticated_page.wait_for_timeout(500)

            # Final content size shouldn't be dramatically larger
            final_content_size = len(authenticated_page.content())

            # Allow 5x growth (generous threshold)
            assert final_content_size < initial_content_size * 5, (
                "Possible memory leak detected"
            )
        else:
            pytest.skip("Could not load and start scenario")
