#!/usr/bin/env python3
"""
Example usage of the modified AutoAgent

This script demonstrates how to use the AutoAgent with conversation management,
task analysis, and dynamic tool loading.
"""

import asyncio
import os
from typing import Dict, Any

# Mock configuration for demonstration
mock_config = {
    "configurable": {
        "model_name": "openai:gpt-4o",
        "temperature": 0.7,
        "max_tokens": 4000,
        "system_prompt": "You are a helpful assistant that can use various tools to complete tasks.",
        # "mcp_config": {
        #     "url": "https://example-mcp-server.com",
        #     "tools": ["search_web", "send_email", "create_document"],
        #     "auth_required": False
        # },
        # "rag": {
        #     "rag_url": "https://example-rag-server.com",
        #     "collections": ["knowledge_base", "documents"]
        # },
        "composio": {
            "api_key": os.getenv("COMPOSIO_API_KEY", "demo-key"),
        },
        "apiKeys": {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "demo-key")
        }
    }
}


async def demonstrate_auto_agent():
    """Demonstrate the AutoAgent functionality"""
    
    # Import the AutoAgent class
    from tools_agent.agent import AutoAgent
    
    print("🚀 Initializing AutoAgent...")
    
    # Create AutoAgent instance
    agent = AutoAgent(mock_config)
    
    print(f"✅ AutoAgent initialized successfully")
    print(f"📊 Initial stats: {agent.get_conversation_stats()}")
    
    # Example 1: Start a new conversation
    print("\n" + "="*50)
    print("EXAMPLE 1: Starting a new conversation")
    print("="*50)
    
    task1 = "Find my calendar events in the coming week"
    print(f"🤖 Task: {task1}")
    
    result1 = await agent.start_new_conversation(task1)
    print(f"📝 Result: {result1[:200]}...")
    print(f"📊 Stats after task 1: {agent.get_conversation_stats()}")
    
    # Example 2: Continue the conversation
    print("\n" + "="*50)
    print("EXAMPLE 2: Continuing the conversation")
    print("="*50)
    
    task2 = "Now add an event for tomorrow at 10am"
    print(f"🤖 Task: {task2}")
    
    result2 = await agent.continue_conversation(task2)
    print(f"📝 Result: {result2[:200]}...")
    print(f"📊 Stats after task 2: {agent.get_conversation_stats()}")
    
    # Example 3: Task that doesn't require tools
    print("\n" + "="*50)
    print("EXAMPLE 3: Task without external tools")
    print("="*50)
    
    task3 = "Explain the concept of machine learning in simple terms"
    print(f"🤖 Task: {task3}")
    
    result3 = await agent.continue_conversation(task3)
    print(f"📝 Result: {result3[:200]}...")
    print(f"📊 Stats after task 3: {agent.get_conversation_stats()}")
    
    # Example 4: Reset and start fresh
    print("\n" + "="*50)
    print("EXAMPLE 4: Reset and start fresh")
    print("="*50)
    
    agent.reset_agent()
    print("🔄 Agent reset successfully")
    
    task4 = "What is the weather like today?"
    print(f"🤖 Task: {task4}")
    
    result4 = await agent.start_new_conversation(task4)
    print(f"📝 Result: {result4[:200]}...")
    print(f"📊 Final stats: {agent.get_conversation_stats()}")




if __name__ == "__main__":
    print("🎯 AutoAgent Demonstration")
    print("This script demonstrates the modified AutoAgent with conversation management and task analysis.")
    print("Note: This is a demonstration with mock configuration. Real usage requires proper API keys and server URLs.")
    
    # Run demonstrations
    asyncio.run(demonstrate_auto_agent())
