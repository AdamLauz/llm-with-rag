
# load fine-tuned model from hub
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer




# import LLM

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

config = PeftConfig.from_pretrained("shawhin/shawgpt-ft")
model = PeftModel.from_pretrained(model, "shawhin/shawgpt-ft")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)



# Use LLM
# prompt (no context)
intstructions_string = f"""ShawGPT, functioning as a virtual data science consultant on YouTube, communicates in clear, accessible language, escalating to technical depth upon request. \
It reacts to feedback aptly and ends responses with its signature '–ShawGPT'. \
ShawGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, \
thus keeping the interaction natural and engaging.

Please respond to the following comment.
"""
prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''

comment = "What is fat-tailedness?"

prompt = prompt_template(comment)
print(prompt)



# prompt (with context)
prompt_template_w_context = lambda context, comment: f"""[INST]ShawGPT, functioning as a virtual data science consultant on YouTube, communicates in clear, accessible language, escalating to technical depth upon request. \
It reacts to feedback aptly and ends responses with its signature '–ShawGPT'. \
ShawGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, \
thus keeping the interaction natural and engaging.

{context}
Please respond to the following comment. Use the context above if it is helpful.

{comment}
[/INST]
"""

prompt = prompt_template_w_context(context, comment)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

print(tokenizer.batch_decode(outputs)[0])