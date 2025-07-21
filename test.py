from composio_openai import ComposioToolSet, App, Action
from composio import Composio
import os
composio_toolset = ComposioToolSet(api_key=os.getenv("COMPOSIO_API_KEY"))  # Replace with your API key
tools = composio_toolset.get_tools(actions=[Action.GOOGLECALENDAR_EVENTS_LIST])
print(tools)

composio = Composio()
tools = composio.tools.get(

    user_id,

    toolkits=["GITHUB"],

    limit=5,  # Returns the top 5 important tools from the toolkit

)
print(tools)