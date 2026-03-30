from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm 
from google.adk.tools.agent_tool import AgentTool
from google.genai import types



from google.adk.tools import google_search, LongRunningFunctionTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams, StreamableHTTPConnectionParams

from fedotmas import MAS, HttpMCPServer
from coscientist.instructions import hypotheses_instruction, research_instruction, fedot_instruction, orchestrator_instruction

from typing import Dict, Any
import uuid
import os
import asyncio
import json



MODEL = os.environ.get('MODEL',"openrouter/qwen/qwen3-235b-a22b-2507") # You can also try: gpt-4.1-mini, gpt-4o etc.
TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY')
MCP_URLS = json.loads(os.getenv("MCP_URLS"))





hypotheses_agent = LlmAgent(
    name="HypothesesAgent",
    model=LiteLlm(model=MODEL),
    instruction=hypotheses_instruction,
    description="Agent to generate scientific hypotheses and ideas for given task",
    output_key="hypotheses"
)

research_agent = LlmAgent(
    name="ResearchAgent",
    model=LiteLlm(model=MODEL),
    instruction=research_instruction,
    description="Agent to answer questions and knowledge mining using Literature and Web Search.",
    output_key="search_results",
    tools=[
            McpToolset(
                connection_params=StreamableHTTPConnectionParams(
                    url=f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
                ),
            ),
            ]
)


async def fedot_tool(task_description: str) -> Dict[str, Any]:
    """
    Tool for generating and executing multi-agent pipelines via FEDOT.MAS. Use it for experiments completion and calculations
    
    Args:
        task_description: Clear description of the task, including goals,
                          inputs, constraints, and expected outputs.
    
    Returns:
        Result of the executed MAS pipeline.
    """

    mas = MAS(mcp_servers={
            "automl_server": HttpMCPServer(
                url=MCP_URLS['auto_ml'],
                description="Remote server for automl training and predicting",
            ),
        })


    result = await mas.run(task_description)

    return {
        "status": "success",
        "result": result,
        # "coordinator": config.coordinator.name,
        # "num_workers": len(config.workers),
    }

fedot_agent = LlmAgent(
    name="ExperimentAgent",
    model=LiteLlm(model=MODEL),
    instruction=fedot_instruction,
    description="Agent to complete experiments and run calculations. Use it for any computation and idea validation. Includes automl",
    output_key="fedot_results",
    tools=[fedot_tool]
)


orchestrator_agent = LlmAgent(
    name="OrchestratorAgent",
    model=LiteLlm(model=MODEL),
    instruction=orchestrator_instruction,
    description="Main Orchestrator Agent",
    tools=[AgentTool(agent=hypotheses_agent), AgentTool(agent=research_agent), AgentTool(agent=fedot_agent)],
)



