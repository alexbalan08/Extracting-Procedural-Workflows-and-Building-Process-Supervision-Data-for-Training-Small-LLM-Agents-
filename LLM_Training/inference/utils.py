


def generate(model, tokenizer, messages, max_new_tokens, temperature):
    
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized["attention_mask"].to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else None,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
