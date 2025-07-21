import json
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from loguru import logger

from universal_mcp.applications import app_from_slug
from universal_mcp.integrations import AgentRIntegration
from universal_mcp.tools import ToolManager


class RichCLI:
    def __init__(self):
        self.console = Console()

    def display_welcome(self, agent_name: str):
        """Display welcome message"""
        welcome_text = f"""
# Welcome to {agent_name}!

Available commands:
- Type your questions naturally
- `/help` - Show help
- `/tools` - List available tools
- `/exit` - Exit the application
        """
        self.console.print(Panel(Markdown(welcome_text), title="🤖 AI Agent CLI", border_style="blue"))

    def display_agent_response(self, response: str, agent_name: str):
        """Display agent response with formatting"""
        self.console.print(Panel(Markdown(response), title=f"🤖 {agent_name}", border_style="green", padding=(1, 2)))

    @contextmanager
    def display_agent_response_streaming(self, agent_name: str):
        """Context manager for streaming agent response updates."""

        with Live(refresh_per_second=10, console=self.console) as live:

            class StreamUpdater:
                content = []

                def update(self, chunk: str):
                    self.content.append(chunk)
                    panel = Panel(
                        Markdown("".join(self.content)),
                        title=f"🤖 {agent_name}",
                        border_style="green",
                        padding=(1, 2),
                    )
                    live.update(panel)

            yield StreamUpdater()

    def display_thinking(self, thought: str):
        """Display agent's thinking process"""
        if thought:
            self.console.print(Panel(thought, title="💭 Thinking", border_style="yellow", padding=(1, 2)))

    def display_tools(self, tools: list):
        """Display available tools in a table"""
        table = Table(title="🛠️ Available Tools")
        table.add_column("Tool Name", style="cyan")
        table.add_column("Description", style="white")

        for tool in tools:
            func_info = tool["function"]
            table.add_row(func_info["name"], func_info["description"])

        self.console.print(table)

    def display_tool_call(self, tool_call: dict):
        """Display tool call"""
        tool_call_str = json.dumps(tool_call, indent=2)
        self.console.print(Panel(tool_call_str, title="🛠️ Tool Call", border_style="green", padding=(1, 2)))

    def display_tool_result(self, tool_result: dict):
        """Display tool result"""
        tool_result_str = json.dumps(tool_result, indent=2)
        self.console.print(Panel(tool_result_str, title="🛠️ Tool Result", border_style="green", padding=(1, 2)))

    def display_error(self, error: str):
        """Display error message"""
        self.console.print(Panel(error, title="❌ Error", border_style="red"))

    def get_user_input(self) -> str:
        """Get user input with rich prompt"""
        return Prompt.ask("[bold blue]You[/bold blue]", console=self.console)

    def display_info(self, message: str):
        """Display info message"""
        self.console.print(f"[bold cyan]ℹ️ {message}[/bold cyan]")

    def clear_screen(self):
        """Clear the screen"""
        self.console.clear()

    def handle_interrupt(self, interrupt) -> str | bool:
        interrupt_type = interrupt.value["type"]
        if interrupt_type == "text":
            value = Prompt.ask(interrupt.value["question"])
            return value
        elif interrupt_type == "bool":
            value = Prompt.ask(interrupt.value["question"], choices=["y", "n"], default="y")
            return value
        elif interrupt_type == "choice":
            value = Prompt.ask(
                interrupt.value["question"], choices=interrupt.value["choices"], default=interrupt.value["choices"][0]
            )
            return value
        else:
            raise ValueError(f"Invalid interrupt type: {interrupt.value['type']}")


from langchain_openai import AzureChatOpenAI


def get_llm(model: str):
    return AzureChatOpenAI(
        model="gpt-4o",
        api_version="2024-12-01-preview",
        azure_deployment="gpt-4o",
    )



class PlatformManager(ABC):
    """Abstract base class for platform-specific functionality.

    This class abstracts away platform-specific operations like fetching apps,
    loading actions, and managing integrations. This allows the AutoAgent to
    work with different platforms without being tightly coupled to any specific one.
    """

    @abstractmethod
    async def get_available_apps(self) -> list[dict[str, Any]]:
        """Get list of available apps from the platform.

        Returns:
            List of app dictionaries with at least 'id', 'name', 'description', and 'available' fields
        """
        pass

    @abstractmethod
    async def get_app_details(self, app_id: str) -> dict[str, Any]:
        """Get detailed information about a specific app.

        Args:
            app_id: The ID of the app to get details for

        Returns:
            Dictionary containing app details
        """
        pass

    @abstractmethod
    async def load_actions_for_app(self, app_id: str, tool_manager: ToolManager) -> None:
        """Load actions for a specific app and register them as tools.

        Args:
            app_id: The ID of the app to load actions for
            tool_manager: The tool manager to register tools with
        """
        pass


class AgentRPlatformManager(PlatformManager):
    """Platform manager implementation for AgentR platform."""

    def __init__(self, api_key: str, base_url: str = "https://api.agentr.dev"):
        """Initialize the AgentR platform manager.

        Args:
            api_key: The API key for AgentR
            base_url: The base URL for AgentR API
        """
        from universal_mcp.utils.agentr import AgentrClient

        self.api_key = api_key
        self.base_url = base_url
        self.client = AgentrClient(api_key=api_key, base_url=base_url)
        logger.debug("AgentRPlatformManager initialized successfully")

    async def get_available_apps(self) -> list[dict[str, Any]]:
        """Get list of available apps from AgentR.

        Returns:
            List of app dictionaries with id, name, description, and available fields
        """
        try:
            all_apps = self.client.list_all_apps()
            available_apps = [
                {
                    "id": app["id"],
                    "name": app["name"],
                    "description": app.get("description", ""),
                    "available": app.get("available", False),
                }
                for app in all_apps
                if app.get("available", False)
            ]
            logger.info(f"Found {len(available_apps)} available apps from AgentR")
            return available_apps
        except Exception as e:
            logger.error(f"Error fetching apps from AgentR: {e}")
            return []

    async def get_app_details(self, app_id: str) -> dict[str, Any]:
        """Get detailed information about a specific app from AgentR.

        Args:
            app_id: The ID of the app to get details for

        Returns:
            Dictionary containing app details
        """
        try:
            app_info = self.client.fetch_app(app_id)
            return {
                "id": app_info.get("id"),
                "name": app_info.get("name"),
                "description": app_info.get("description"),
                "category": app_info.get("category"),
                "available": app_info.get("available", True),
            }
        except Exception as e:
            logger.error(f"Error getting details for app {app_id}: {e}")
            return {
                "id": app_id,
                "name": app_id,
                "description": "Error loading details",
                "category": "Unknown",
                "available": True,
            }

    async def load_actions_for_app(self, app_id: str, tool_manager: ToolManager) -> None:
        """Load actions for a specific app from AgentR and register them as tools.

        Args:
            app_id: The ID of the app to load actions for
            tool_manager: The tool manager to register tools with
        """
        logger.info(f"Loading all actions for app: {app_id}")

        try:
            # Get all actions for the app
            app_actions = self.client.list_actions(app_id)

            if not app_actions:
                logger.warning(f"No actions available for app: {app_id}")
                return

            logger.debug(f"Found {len(app_actions)} actions for {app_id}")

            # Register all actions as tools
            app = app_from_slug(app_id)
            integration = AgentRIntegration(name=app_id, api_key=self.api_key, base_url=self.base_url)
            app_instance = app(integration=integration)
            logger.debug(f"Registering all tools for app: {app_id}")
            tool_manager.register_tools_from_app(app_instance)

            logger.info(f"Successfully loaded all {len(app_actions)} actions for app: {app_id}")

        except Exception as e:
            logger.error(f"Failed to load actions for app {app_id}: {e}")
            raise
