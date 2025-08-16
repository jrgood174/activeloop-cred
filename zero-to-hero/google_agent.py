from langchain.llms import OpenAI

from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent

from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv
import os

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="google-search",
        func=search.run,
        description="useful for when you need to answer a question about current events."
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=6
)

print(agent.run("What's the latest news on the mars rover?"))
