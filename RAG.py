from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from VectorDB import get_context, get_query_engine

SAVED_MODEL_PATH = "./models"
LLM_PATH = str(Path(SAVED_MODEL_PATH, "LLM.h5"))
LLM_TOKENIZER_PATH = str(Path(SAVED_MODEL_PATH, "LLM_TOKENIZER.h5"))


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_TOKENIZER_PATH)
    model = AutoModelForCausalLM.from_pretrained(LLM_PATH)
    return tokenizer, model


if __name__ == "__main__":
    query_engine = get_query_engine()
    tokenizer, model = load_llm()
    # prompt (no context)
    intstructions_string = f"""SawserQGPT, functioning as a virtual Circassian history and culture expert, communicates in clear, accessible language, uses facts and reliable numbers upon request. \
    It reacts to feedback aptly and ends responses with its signature '–SawserQGPT'. \
    SawserQGPT will tailor its responses to match the user's input, providing concise acknowledgments to brief expressions of gratitude or feedback, \
    thus keeping the interaction natural and engaging.
    
    Please respond to the following user's input.
    """
    prompt_template = lambda user_input: f'''[INST] {intstructions_string} \n{user_input} \n[/INST]'''

    user_input = "Who are the Circassians?"

    prompt = prompt_template(user_input)
    print(prompt)

    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

    print(tokenizer.batch_decode(outputs)[0])

    ########################################################################################

    # prompt (with context)
    prompt_template_w_context = lambda context, comment: f"""[INST]SawserQGPT, functioning as a virtual Circassian history and culture expert, communicates in clear, accessible language, uses facts and reliable numbers upon request. \
    It reacts to feedback aptly and ends responses with its signature '–SawserQGPT'. \
    SawserQGPT will tailor its responses to match the user's input, providing concise acknowledgments to brief expressions of gratitude or feedback, \
    thus keeping the interaction natural and engaging.
    
    {context}
    Please respond to the following user's input. Use the context above if it is helpful.
    
    {user_input}
    [/INST]
    """

    context = get_context(query=user_input, query_engine=query_engine)
    prompt = prompt_template_w_context(context, user_input)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

    print(tokenizer.batch_decode(outputs)[0])
