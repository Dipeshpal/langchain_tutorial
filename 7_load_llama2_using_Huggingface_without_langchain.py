from transformers import pipeline
from transformers import AutoTokenizer

model = "meta-llama/Llama-2-7b-hf"
local_files_only = False

# Model: https://huggingface.co/meta-llama/Llama-2-7b-hf
pipe = pipeline("text-generation", model=model, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)

query = """
        You are a helpful assistant answers prompt to user input
        User Prompt: How are you?
        Based on above data, generate an answer for User Prompt in following format-
        AI_ANSWER: 
        """

sequences = pipe(
    query,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=148,
)
ans = ''
for seq in sequences:
    ans += seq["generated_text"]
print(ans)


