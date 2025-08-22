#easy mode
#from biomni.llm import get_llm
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from typing import TypedDict
from biomni.agent import A1
from biomni.llm import get_llm

agent = A1(path='./data',llm = 'claude-4-sonnet-2025',source='Custom')
llm1 = get_llm(source='Custom', model='claude-4-sonnet-2025', temperature=0.7)

class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_step: str | None

state = AgentState(
    messages=[HumanMessage(content="Describe to me the central dogma of molecular biology")],
    next_step=None
)

easy_messages = [SystemMessage('You are a helpful assistant.')] + state["messages"]
res = llm1.invoke(easy_messages )


messages = [SystemMessage(content=agent.system_prompt)] + state["messages"]

res = llm1.invoke(messages )

# Easy medium mode


llm2 = get_llm(source='Anthropic', model='claude-4-sonnet-2025', temperature=0.7)

llm2.invoke('What is the capital of France?')

# Medium mode

from biomni.agent import A1


agent = A1(path='./data',llm = 'claude-4-sonnet-2025',source='Custom')

agent.go("What is the capital of France?")

#hard mode



from biomni.agent import A1


agent = A1(path='./data',llm = 'claude-4-sonnet-2025',source='Custom')

agent.go("What is the capital of France?")
