"""Independent current questions in equal-length tensor rows; no padding/RAG.

No curriculum, expected answers or scoring enters this generation function.
Each row is a separate sequence; rows never share attention or chat history.
"""


def generate_equal_length(model, tokenizer, questions, max_new_tokens=20):
    import torch
    from model_support import format_chat
    if not questions or any(not isinstance(q, str) for q in questions):
        raise ValueError('Nonempty independent question strings required')
    texts = [format_chat(tokenizer, q) for q in questions]
    inputs = tokenizer(texts, return_tensors='pt').to(model.device)
    if inputs['input_ids'].ndim != 2 or not bool((inputs['attention_mask'] == 1).all().cpu()):
        raise ValueError('Equal-length no-padding independent prompts required')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
    prefix = inputs['input_ids'].shape[1]
    return [tokenizer.decode(row[prefix:], skip_special_tokens=True).strip().lower() for row in outputs]
