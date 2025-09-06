import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
from langchain_community.tools import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_tavily import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = 'ReAct Agent'

search_tool = TavilySearchResults()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'http://api.weatherstack.com/current?access_key=0e07dd417a4b185a6d7cf6c4cc1539b5&query={city}'

  response = requests.get(url)

  return response.json()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke
response = agent_executor.invoke({"input": "What is the release date of Dhadak 2?"})
print(response)

print(response['output'])