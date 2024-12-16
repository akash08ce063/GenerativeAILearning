Generative Agents - https://github.com/NirDiamant/GenAI_Agents

**Practicle Notes** -

- when feeding to model set of Messages like SystemMessage, Human MEssage, Tool MEssage, AIMessage, make sure that last message isn't AI Message, otherwise it does not work fine 
- Any agent work other than tool calling, make it like RAG, that fetch or present good source of information to LLM model and then ask questions to drive toward goal
- It is not necessary to stack up all the messages like HumanMessage, ToolMessage, AIMessage etc. before passing to LLM, you can pass the last message only with some systemMessage as an instruction to achieve final goal.
