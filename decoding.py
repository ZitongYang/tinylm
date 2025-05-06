from vllm import LLM, SamplingParams


def get_model():
    model = LLM(model="Qwen/Qwen3-4B-Base",
                tokenizer="Qwen/Qwen3-4B-Base",
                device="cuda",
                tensor_parallel_size=1)
    return model

def decoding(model, prompt):
    # Set up sampling parameters
    sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=100,
                    skip_special_tokens=False)
    outputs = model.generate(prompt, sampling_params)
    return outputs[0].outputs[0].text

if __name__ == "__main__":
    model = get_model()
    import pdb; pdb.set_trace()
    prompt = "The captial of France is"
    print(f"Prompt: {prompt}\n",
          "="*100,
          f"Output: {decoding(model, prompt)}")





