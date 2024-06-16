# -*- coding: utf-8 -*-
# !pip install langchain langchain-cli "langserve[all]" google.generativeai langchain-google-genai pandas numexpr  langchain-openai

import json
import os

# basic langchain imports
import sys
from typing import Dict, List, Tuple, Union

import pandas as pd
from dotenv import load_dotenv

# agents
from langchain.agents import (
    AgentExecutor,
)
from langchain.agents.format_scratchpad import (
    format_log_to_str,
    format_to_openai_function_messages,
)
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.globals import set_debug, set_verbose

# Import things for functions
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from enum import Enum


class LlmModel(Enum):
    GOOGLE_GEMINI = 0
    GOOGLE_VERTEX = 1
    OPENAI = 2

LLM_MODEL = LlmModel.GOOGLE_VERTEX


load_dotenv()

url = "https://raw.githubusercontent.com/GoogleCloudPlatform/region-carbon-info/main/data/yearly/2022.csv"
df_cfe = pd.read_csv(url)
# print('cfe:')
df_cfe.rename(
    columns={"Google Cloud Region": "region", "Google CFE": "cfe"}, inplace=True
)


url = (
    "https://raw.githubusercontent.com/rcleveng/notebooks/main/gcp_latency_20240613.csv"
)
df_latency = pd.read_csv(url)


prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder("chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


class GcpRegionSchema(BaseModel):
    region: str = Field(description="name of the Google Cloud Region", default="")


class NoParameterInput(BaseModel):
    pass


class CfeAllTool(BaseTool):
    name = "cfe_all"
    description = "Fetch the carbon free energy (CFE) percent for every Google Cloud regions. No parameter needed from input"
    args_schema = NoParameterInput

    def is_single_input(self) -> bool:
        return True

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        """Ensure no args passed since this tool takes none"""
        print("_to_args_and_kwargs")
        return (), {}

    def _run(self):
        print("executing cfe-all-tool")
        # print(df_cfe.head(5))
        df_cfe.rename(
            columns={"Google Cloud Region": "region", "Google CFE": "cfe"}, inplace=True
        )
        return df_cfe[["region", "cfe"]].to_string(
            columns=["region", "cfe"],
            index=False,
            index_names=False,
            header=False,
        )

    def _arun(self):
        raise NotImplementedError("does not support async")


@tool("latency_tool", args_schema=GcpRegionSchema, return_direct=False)
def latency(region) -> str:
    """fetch the network latency from a given google cloud region to all other regions"""
    print(f"executing latency_tool for {region}")
    ms_formatter = lambda x: "%4.2f ms" % x

    return df_latency.loc[df_latency["sending_region"] == region].to_string(
        columns=["receiving_region", "milliseconds"],
        formatters={"milliseconds": ms_formatter},
        index=False,
        index_names=False,
        header=False,
    )


cfe_all = CfeAllTool()
tools = [cfe_all, latency]

# initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True
)

match LLM_MODEL:
    case LlmModel.GOOGLE_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Get the key from g://aistudio.google.com
        google_api_key = os.getenv("GOOGLE_API_KEY")
        # gemini-1.5-pro - "Exception: Multiple function calls are not currently supported"
        # gemini-1.5-flash - Wrong Answer: The region with the lowest latency from us-west1 with a CFE over 80% is **europe-north1**" or function calls error"
        # gemini-1.0-pro - Wrong Answer: The region with the lowest latency from us-west1 with a CFE over 80% is asia-northeast1 with a latency of 87.56 ms.
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=google_api_key,
            temperature=0,
        )
        # Work around gemini 1.0 lack of system message
        llm.convert_system_message_to_human = True
    case LlmModel.GOOGLE_VERTEX:
        import vertexai
        from langchain_google_vertexai import ChatVertexAI

        vertexai.init(project="robsite-assistant-prod", location="us-central1")

        llm  = ChatVertexAI(
            model="gemini-1.5-flash",             
            temperature=0,
        )
        #llm.convert_system_message_to_human = True
    case LlmModel.OPENAI:
        from langchain_openai import ChatOpenAI
        # Get the key from https://platform.openai.com/api-keys
        openai_api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=10,
            api_key=openai_api_key,
        )


# Add a debugging wrapper
# class that wraps another class and logs all function calls being executed
class Wrapper:
    def __init__(self, wrapped_class):
        self.wrapped_class = wrapped_class

    def __getattr__(self, attr):
        original_func = getattr(self.wrapped_class, attr)

        def wrapper(*args, **kwargs):
            print(f"Calling function: {attr}")
            print(f"Arguments: {args}, {kwargs}")
            result = original_func(*args, **kwargs)
            print(f"Response: {result}")
            return result

        return wrapper


# This seems to fail on vertext with:
# AttributeError: 'NoneType' object has no attribute 'stream_generate_content'
#
# llm.client = Wrapper(llm.client)

llm_with_tools = llm.bind_tools(tools=tools)

# set_verbose(True)
set_debug(True)


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


agent = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"])
    )
    | prompt
    | llm_with_tools
    | ToolsAgentOutputParser()
)


class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    # memory=conversational_memory,
    max_iterations=50,
    early_stopping_method="generate",
).with_types(input_type=AgentInput)

chat_history = conversational_memory.buffer_as_messages


def execute_query(query: str) -> Tuple[str, str]:
    """
    Executes a query, returns the result in the 1st str or the command name in the second
    """
    match query:
        case "quit":
            return None, query
    out = agent_executor.invoke(
        {
            "input": query,
            "chat_history": chat_history,
        }
    )
    return out["output"], None


print(
    execute_query(
        "What is the region with the lowest latency from us-west1 with a CFE over 80%?"
    )
)
sys.exit(0)

while True:
    query = input("Input: ")
    result, cmd = execute_query(query)
    if cmd:
        match cmd:
            case "quit":
                sys.exit(0)
    print(f"{result}\n\n\n")
