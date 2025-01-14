from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Initialize LangChain with your local endpoint
llm = ChatOpenAI(
    openai_api_base="",  # Base URL of your endpoint
    openai_api_key="",  # Leave blank; no auth required
)

# Send a test query
response = llm([HumanMessage(content="Hello, how are you?")])
print(response.content)  # Should print the model's generated output
