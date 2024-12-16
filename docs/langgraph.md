Most important high level API - https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/#usage

"Create React Agent" -> this API actually create a graph underneath and first call LLM and then check if requires calling a tool then it will call a tool and then result will be added 
to list and again it will call LLM and this iteration will be executed until there is no tool call.

