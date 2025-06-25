from langchain_core.tools import Tool
from langchain import LLMMathChain

llm_math_chain = LLMMathChain.from_llm(llm=llm,verbose=True)
math_tool = Tool.from_function(
    func=llm_math_chain.invoke,
    name="Calculator",
    description="This tool is only for math questions, A tool for performing mathematical calculations using an LLM. It can solve equations")

class OutputFormat(BaseModel):
    answer: str = Field(..., description="Accurate answer to the question")
    reason: str = Field(..., description="Brief explanation to the answer")

# 2) Define input TypedDict
class InputFormat(TypedDict):
    question: str

math_assistant_prompt = """
<role>
You are advanced AI mathematics assistant. Your task is to provide accurate answer to only mathematical problems.
</role>

<context>
ID: {id}
{question}

use math_tool as calculator to solve math problems.

</context>

<result>
Return JSON with:
- answer: best accurate answer to the given mathematical question
- reason: A concise explanation justifying the answer (32–256 tokens)
</result>

<constraint>
- Every answer must be based on the context retrieved from the math_tool
- Use the math_tool to retrieve context from the question
- Reason must be >32 and <256 tokens
</constraint>
"""

llm = ChatOpenAI(model="typhoon-v2.1-12b-instruct"
                 ,api_key="sk-UhhyzDwMuJfFXDuYzoGOngETFNnBVQdoL4tgrluZnTYIXtUB"
                 ,base_url="https://api.opentyphoon.ai/v1"
                 , temperature=0.0)
# 6) Build the classification chain
math_agent = create_react_agent(
    model=llm,
    tools=[
        math_tool
    ],
    prompt=math_assistant_prompt,
    name="math_assistant")

# 7) Invoke the agent with a sample input
# response = math_agent.invoke(
#     {
#         "messages": [
#             (
#                 "human",
#                 f"""ID: 1223"""
#                 f"""Query: what is 2+2?"""

#             )
#         ]
#     }
# )