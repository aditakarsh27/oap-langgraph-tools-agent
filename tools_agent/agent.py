import os
import datetime
from contextlib import asynccontextmanager
from typing import Any, Optional, List
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from loguru import logger
from tools_agent.utils.tools import create_rag_tool
from tools_agent.utils.token import fetch_tokens
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from langchain_core.tools import StructuredTool
from tools_agent.utils.tools import (
    wrap_mcp_authenticate_tool,
    create_langchain_mcp_tool,
)

from composio_openai import ComposioToolSet, App, Action


UNEDITABLE_SYSTEM_PROMPT = "\nIf the tool throws an error requiring authentication, provide the user with a Markdown link to the authentication page and prompt them to authenticate."

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that has access to a variety of tools."
)


class TaskAnalysis(BaseModel):
    """Combined analysis of task type and app requirements"""

    requires_app: bool
    reasoning: str
    app_sets: list[list[str]] = []  # Multiple sets of app choices
    choice: list[bool] = []  # Whether user choice is needed for each app set


class RagConfig(BaseModel):
    rag_url: Optional[str] = None
    """The URL of the rag server"""
    collections: Optional[List[str]] = None
    """The collections to use for rag"""


class MCPConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class ComposioConfig(BaseModel):
    api_key: Optional[str] = None
    """The API key for the composio server"""



class GraphConfigPydantic(BaseModel):
    model_name: Optional[str] = Field(
        default="openai:gpt-4o",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "openai:gpt-4o",
                "description": "The model to use in all generations",
                "options": [
                    {
                        "label": "Claude 3.7 Sonnet",
                        "value": "anthropic:claude-3-7-sonnet-latest",
                    },
                    {
                        "label": "Claude 3.5 Sonnet",
                        "value": "anthropic:claude-3-5-sonnet-latest",
                    },
                    {"label": "GPT 4o", "value": "openai:gpt-4o"},
                    {"label": "GPT 4o mini", "value": "openai:gpt-4o-mini"},
                    {"label": "GPT 4.1", "value": "openai:gpt-4.1"},
                ],
            }
        },
    )
    temperature: Optional[float] = Field(
        default=0.7,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.7,
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Controls randomness (0 = deterministic, 2 = creative)",
            }
        },
    )
    max_tokens: Optional[int] = Field(
        default=4000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 4000,
                "min": 1,
                "description": "The maximum number of tokens to generate",
            }
        },
    )
    system_prompt: Optional[str] = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        metadata={
            "x_oap_ui_config": {
                "type": "textarea",
                "placeholder": "Enter a system prompt...",
                "description": f"The system prompt to use in all generations. The following prompt will always be included at the end of the system prompt:\n---{UNEDITABLE_SYSTEM_PROMPT}\n---",
                "default": DEFAULT_SYSTEM_PROMPT,
            }
        },
    )
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                # Here is where you would set the default tools.
                # "default": {
                #     "tools": ["Math_Divide", "Math_Mod"]
                # }
            }
        },
    )
    rag: Optional[RagConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "rag",
                # Here is where you would set the default collection. Use collection IDs
                # "default": {
                #     "collections": [
                #         "fd4fac19-886c-4ac8-8a59-fff37d2b847f",
                #         "659abb76-fdeb-428a-ac8f-03b111183e25",
                #     ]
                # },
            }
        },
    )
    composio: Optional[ComposioConfig] = Field(
        default=None,
        optional=True,
    )


def get_api_key_for_model(model_name: str, config: RunnableConfig):
    model_name = model_name.lower()
    model_to_key = {
        "openai:": "OPENAI_API_KEY",
        "anthropic:": "ANTHROPIC_API_KEY", 
        "google": "GOOGLE_API_KEY"
    }
    key_name = next((key for prefix, key in model_to_key.items() 
                    if model_name.startswith(prefix)), None)
    if not key_name:
        return None
    api_keys = config.get("configurable", {}).get("apiKeys", {})
    if api_keys and api_keys.get(key_name) and len(api_keys[key_name]) > 0:
        return api_keys[key_name]
    # Fallback to environment variable
    return os.getenv(key_name)


class AutoAgent:
    def __init__(self, config: RunnableConfig):
        logger.info("Initializing AutoAgent")
        self.config = config
        self.cfg = GraphConfigPydantic(**config.get("configurable", {}))
        
        # Initialize LLM
        api_key = get_api_key_for_model(self.cfg.model_name, config) or "No token found"
        if self.cfg.model_name.startswith("openai:gpt-4.1"):
            self.llm = AzureChatOpenAI(
                azure_deployment="gpt-4.1",
                model="gpt-4.1",
                api_version="2024-12-01-preview",
                api_key=api_key,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
        else:
            # Use standard LangChain model initialization for other models
            from langchain.chat_models import init_chat_model
            self.llm = init_chat_model(
                self.cfg.model_name,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                api_key=api_key
            )

        # Agent and conversation persistence
        self._agent: Any | None = None
        self._loaded_tools: list[str] = []
        self._conversation_history: list[BaseMessage] = []
        self._current_tools_hash: str | None = None
        self._mcp_tools: list[StructuredTool] = []
        self._rag_tools: list[StructuredTool] = []

        self.task_analysis_prompt = """You are a task analysis expert. Given a task description and available tools, determine:

        1. Whether the task requires external tools or can be handled through general reasoning
        2. If it requires tools, which tools are most relevant
        3. If the task requires multiple different types of functionality, organize tools into logical sets

        Tasks that typically require tools:
        - Searching the web for information
        - Sending emails
        - Creating or editing documents
        - Managing calendars or schedules
        - Processing data or files
        - Interacting with social media
        - Making API calls to external services
        - Retrieving documents from knowledge bases

        Tasks that typically don't require tools:
        - General reasoning and analysis
        - Mathematical calculations
        - Text summarization or analysis
        - Providing explanations or educational content
        - Planning and organization
        - Creative writing or brainstorming
        - Logical problem solving

        For complex tasks that require multiple types of functionality, organize tools into logical sets.
        For example, if a task requires both email and search functionality, you might create:
        - app_sets: [["outlook", "google-mail"], ["serpapi", "tavily"]]
        - choice: [True, False] (user chooses email tool, all search tools are loaded)

        The choice field should be an array of booleans with the same length as app_sets.
        Set choice[i] to True if the user should choose from app_sets[i].
        Set choice[i] to False if all tools in app_sets[i] should be automatically loaded.

        Analyze the given task and determine if it requires external tools or can be completed through general reasoning.
        If it requires tools, select the most relevant tools from the available list.
        If the task requires multiple different types of functionality, organize tools into logical sets.
        """

        logger.debug("AutoAgent initialized successfully")

    async def _load_mcp_tools(self):
        """Load MCP tools if configured"""
        if not self.cfg.mcp_config or not self.cfg.mcp_config.url or not self.cfg.mcp_config.tools:
            return

        # Check if MCP tools are already loaded
        if self._mcp_tools:
            return

        logger.info("Loading MCP tools")
        
        # Get MCP tokens if authentication is required
        mcp_tokens = None
        if self.cfg.mcp_config.auth_required:
            mcp_tokens = await fetch_tokens(self.config)
        
        if not mcp_tokens and self.cfg.mcp_config.auth_required:
            logger.warning("MCP authentication required but no tokens available")
            return

        server_url = self.cfg.mcp_config.url.rstrip("/") + "/mcp"
        tool_names_to_find = set(self.cfg.mcp_config.tools)
        fetched_mcp_tools_list: list[StructuredTool] = []

        # Set headers for authentication
        headers = (
            mcp_tokens is not None
            and {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
            or None
        )

        try:
            async with streamablehttp_client(server_url, headers=headers) as streams:
                read_stream, write_stream, _ = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    page_cursor = None
                    while True:
                        tool_list_page = await session.list_tools(cursor=page_cursor)

                        if not tool_list_page or not tool_list_page.tools:
                            break

                        for mcp_tool in tool_list_page.tools:
                            if mcp_tool.name in tool_names_to_find:
                                langchain_tool = create_langchain_mcp_tool(
                                    mcp_tool, mcp_server_url=server_url, headers=headers
                                )
                                fetched_mcp_tools_list.append(
                                    wrap_mcp_authenticate_tool(langchain_tool)
                                )

                        page_cursor = tool_list_page.nextCursor

                        if not page_cursor:
                            break

                    self._mcp_tools = fetched_mcp_tools_list
                    logger.info(f"Successfully loaded {len(self._mcp_tools)} MCP tools")
        except Exception as e:
            logger.error(f"Failed to fetch MCP tools: {e}")

    async def _load_rag_tools(self):
        """Load RAG tools if configured"""
        if not self.cfg.rag or not self.cfg.rag.rag_url or not self.cfg.rag.collections:
            return

        # Check if RAG tools are already loaded
        if self._rag_tools:
            return

        logger.info("Loading RAG tools")
        
        supabase_token = self.config.get("configurable", {}).get("x-supabase-access-token")
        if not supabase_token:
            logger.warning("No Supabase token available for RAG tools")
            return

        rag_tools = []
        for collection in self.cfg.rag.collections:
            try:
                rag_tool = await create_rag_tool(
                    self.cfg.rag.rag_url, collection, supabase_token
                )
                rag_tools.append(rag_tool)
                logger.info(f"Loaded RAG tool for collection: {collection}")
            except Exception as e:
                logger.error(f"Failed to load RAG tool for collection {collection}: {e}")

        self._rag_tools = rag_tools
        logger.info(f"Successfully loaded {len(self._rag_tools)} RAG tools")

    def _get_tools_hash(self) -> str:
        """Generate a hash of the current tools to detect changes"""
        tools = self._get_current_tools()
        tools_info = [(tool.name, tool.description) for tool in tools]
        return str(hash(str(tools_info)))

    def _get_current_tools(self) -> list[StructuredTool]:
        """Get the current list of available tools"""
        tools = []
        
        # Add MCP tools
        tools.extend(self._mcp_tools)
        
        # Add RAG tools
        tools.extend(self._rag_tools)
        
        return tools

    async def get_agent(self, force_recreate: bool = False):
        """Get or create an agent with tools. Reuses existing agent if tools haven't changed."""
        # Load tools if not already loaded
        await self._load_mcp_tools()
        await self._load_rag_tools()
        
        current_tools_hash = self._get_tools_hash()

        # Check if we need to recreate the agent
        if force_recreate or self._agent is None or self._current_tools_hash != current_tools_hash:
            logger.info("Creating new agent with tools")
            tools = self._get_current_tools()
            logger.debug(f"Created agent with {len(tools)} tools")

            # Get current datetime and timezone information
            current_time = datetime.datetime.now()
            utc_time = datetime.datetime.now(datetime.UTC)
            timezone_info = f"Current local time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | UTC time: {utc_time.strftime('%Y-%m-%d %H:%M:%S')}"

            self._agent = create_react_agent(
                self.llm,
                tools=tools,
                prompt=f"{self.cfg.system_prompt + UNEDITABLE_SYSTEM_PROMPT}\n\nCurrent time information: {timezone_info}",
            )
            self._current_tools_hash = current_tools_hash
            logger.info("Agent created successfully")
        else:
            logger.debug("Reusing existing agent")

        return self._agent

    def add_to_conversation_history(self, message: BaseMessage):
        """Add a message to the conversation history"""
        self._conversation_history.append(message)
        logger.debug(f"Added message to history. Total messages: {len(self._conversation_history)}")

    def get_conversation_history(self) -> list[BaseMessage]:
        """Get the current conversation history"""
        return self._conversation_history.copy()

    def clear_conversation_history(self):
        """Clear the conversation history"""
        self._conversation_history.clear()
        logger.info("Conversation history cleared")

    def reset_agent(self):
        """Reset the agent and clear conversation history"""
        self._agent = None
        self._current_tools_hash = None
        self._loaded_tools.clear()
        self._mcp_tools.clear()
        self._rag_tools.clear()
        self.clear_conversation_history()
        logger.info("Agent reset successfully")

    def get_loaded_tools(self) -> list[str]:
        """Get the list of currently loaded tools"""
        return self._loaded_tools.copy()

    def get_conversation_stats(self) -> dict:
        """Get statistics about the current conversation"""
        human_messages = [msg for msg in self._conversation_history if isinstance(msg, HumanMessage)]
        ai_messages = [msg for msg in self._conversation_history if isinstance(msg, AIMessage)]

        return {
            "total_messages": len(self._conversation_history),
            "human_messages": len(human_messages),
            "ai_messages": len(ai_messages),
            "loaded_tools": len(self._loaded_tools),
            "loaded_tool_names": self._loaded_tools.copy(),
            "mcp_tools": len(self._mcp_tools),
            "rag_tools": len(self._rag_tools),
            "has_agent": self._agent is not None,
        }

    def is_conversation_empty(self) -> bool:
        """Check if the conversation history is empty"""
        return len(self._conversation_history) == 0

    async def analyze_task_and_select_tools(
        self, task: str, available_tools: list[dict], interactive: bool = False
    ) -> TaskAnalysis:
        """Combined task analysis and tool selection to reduce LLM calls"""
        logger.info(f"Analyzing task and selecting tools: {task}")

        # Get conversation context
        conversation_history = self.get_conversation_history()
        context_summary = ""

        if len(conversation_history) > 1:  # More than just the current task
            # Create a summary of previous conversation context
            previous_messages = conversation_history[:-1]  # Exclude current task
            context_messages = []

            for msg in previous_messages[-5:]:  # Last 5 messages for context
                if isinstance(msg, HumanMessage):
                    context_messages.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    context_messages.append(f"Assistant: {msg.content[:200]}...")  # Truncate long responses

            if context_messages:
                context_summary = "\n\nPrevious conversation context:\n" + "\n".join(context_messages)
                logger.debug(f"Adding conversation context: {len(context_messages)} previous messages")

        prompt = f"""
        {self.task_analysis_prompt}

        Task: {task}
        Available tools: {available_tools}{context_summary}

        Determine if this task requires external tools or can be completed through general reasoning and knowledge.
        If it requires tools, select the most relevant tools from the available list.
        If the task requires multiple different types of functionality, organize tools into logical sets using the app_sets field.

        Consider the conversation context when making your decision. For example:
        - If the user previously mentioned specific tools or apps, prefer those
        - If the conversation is about a specific topic, choose tools relevant to that topic
        - If the user is continuing a previous task, maintain consistency in tool selection
        """

        # Use structured output with Pydantic model
        structured_llm = self.llm.with_structured_output(TaskAnalysis)
        response = await structured_llm.ainvoke(prompt)
        logger.debug(f"Task analysis response: {response}")

        logger.info(f"Task requires tools: {response.requires_app}")
        logger.info(f"Reasoning: {response.reasoning}")
        if response.requires_app:
            logger.info(f"Tool sets: {response.app_sets}")
            logger.info(f"Choice flags: {response.choice}")

        return response

    async def execute_task_without_tools(self, task: str) -> str:
        """Execute a task that doesn't require external tools using general reasoning"""
        logger.info(f"Executing task without tools: {task}")

        # Create a simple agent without any tools for general reasoning
        agent = await self.get_agent()

        # Execute the task with conversation history
        logger.info(f"Invoking agent for task: {task}")
        messages = self.get_conversation_history()
        results = await agent.ainvoke({"messages": messages})
        ai_message = results["messages"][-1]

        # Add the AI response to conversation history
        self.add_to_conversation_history(ai_message)

        logger.info("Task completed without additional tools")
        return ai_message.content

    async def continue_conversation(self, task: str) -> str:
        """Continue the conversation with a new task, maintaining all previous context"""
        logger.info(f"Continuing conversation with task: {task}")
        return await self.run(task, reset_conversation=False)

    async def start_new_conversation(self, task: str) -> str:
        """Start a new conversation, clearing all previous context"""
        logger.info(f"Starting new conversation with task: {task}")
        return await self.run(task, reset_conversation=True)

    async def run(self, task: str, reset_conversation: bool = False) -> str:
        logger.info(f"Starting task execution: {task}")

        # Reset conversation if requested
        if reset_conversation:
            self.clear_conversation_history()
            logger.info("Conversation history reset")

        # Add the new task to conversation history
        human_message = HumanMessage(content=task)
        self.add_to_conversation_history(human_message)

        # Get available tools (MCP and RAG)
        available_tools = []
        
        # Add MCP tools if configured
        if self.cfg.mcp_config and self.cfg.mcp_config.tools:
            available_tools.extend([{"id": tool, "name": tool, "description": f"MCP tool: {tool}"} for tool in self.cfg.mcp_config.tools])
            
        # Add RAG tools if configured
        if self.cfg.rag and self.cfg.rag.collections:
            available_tools.extend([{"id": collection, "name": f"RAG Collection: {collection}", "description": f"RAG collection: {collection}"} for collection in self.cfg.rag.collections])


        if self.cfg.composio:
            composio_toolset = ComposioToolSet(api_key=self.cfg.composio.api_key)  # Replace with your API key
            tools = composio_toolset.get_tools(actions=[Action.GMAIL_SEND_EMAIL,
    Action.GMAIL_CREATE_EMAIL_DRAFT,
    Action.GMAIL_SEND_DRAFT,
    Action.GMAIL_REPLY_TO_THREAD,
    Action.GMAIL_FETCH_EMAILS,
    Action.GMAIL_GET_ATTACHMENT,
    Action.GMAIL_DELETE_MESSAGE,
    Action.GMAIL_MOVE_TO_TRASH,
    Action.GMAIL_GET_CONTACTS,
    Action.GOOGLECALENDAR_CREATE_EVENT,
    Action.GOOGLECALENDAR_GET_EVENTS,
    Action.GOOGLECALENDAR_GET_EVENT_DETAILS,
    Action.GOOGLECALENDAR_UPDATE_EVENT,
    Action.GOOGLECALENDAR_DELETE_EVENT,
    Action.GOOGLECALENDAR_GET_CALENDARS,
    Action.GOOGLECALENDAR_GET_FREE_BUSY,
    Action.GOOGLECALENDAR_GET_PRIMARY_CALENDAR,
    Action.GOOGLECALENDAR_CREATE_ALL_DAY_EVENT,
    Action.GOOGLECALENDAR_CREATE_RECURRING_EVENT,
    Action.GOOGLECALENDAR_LIST_CALENDAR_LIST,
]


    ])
            available_tools.extend([{"id": tool.name, "name": tool.name, "description": tool.description} for tool in tools])

            
        
        logger.info(f"Found {len(available_tools)} available tools")

        # Analyze task and select tools
        task_analysis = await self.analyze_task_and_select_tools(task, available_tools)

        if not task_analysis.requires_app:
            logger.info("Task does not require tools, using general reasoning")
            return await self.execute_task_without_tools(task)

        if not task_analysis.app_sets:
            logger.warning(f"No suitable tools found for task: {task}")
            logger.info("Falling back to general reasoning for this task")
            return await self.execute_task_without_tools(task)

        # Load tools based on analysis
        loaded_tools = []
        for tool_set in task_analysis.app_sets:
            for tool_name in tool_set:
                if tool_name not in self._loaded_tools:
                    self._loaded_tools.append(tool_name)
                    loaded_tools.append(tool_name)
                    logger.info(f"Loaded tool: {tool_name}")

        if not loaded_tools:
            logger.warning("No tools loaded, using general reasoning")
            return await self.execute_task_without_tools(task)

        logger.info(f"Successfully loaded {len(loaded_tools)} tools: {', '.join(loaded_tools)}")

        # Get or create agent with the loaded tools
        agent = await self.get_agent()

        # Execute the task with conversation history
        logger.info(f"Invoking agent for task: {task}")
        messages = self.get_conversation_history()
        results = await agent.ainvoke({"messages": messages})
        ai_message = results["messages"][-1]

        # Add the AI response to conversation history
        self.add_to_conversation_history(ai_message)

        logger.info("Task completed successfully")
        return ai_message.content


async def graph(config: RunnableConfig):
    """Create and return the AutoAgent instance"""
    auto_agent = AutoAgent(config)
    
    # Return the agent's run method as the main entry point
    return auto_agent.run