SYSTEM_TOKEN = "<SYSTEM>"
USER_TOKEN = "<USER>"
ASSISTANT_TOKEN = "<ASSISTANT>"
ROLE_TOKENS = [SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN]


def build_example_ids(tokenizer, system_msg, user_msg, assistant_msg):
    """One full training example: <SYSTEM> system_msg <USER> user_msg
    <ASSISTANT> assistant_msg, as token ids. Role-token ids are inserted
    directly rather than encoded from text -- CharTokenizer encodes
    character-by-character, so a literal "<SYSTEM>" string in raw text
    would be split into individual (mostly out-of-vocabulary) characters,
    not recognized as one atomic token."""
    ids = [tokenizer.char_to_id[SYSTEM_TOKEN]]
    ids += tokenizer.encode(system_msg)
    ids += [tokenizer.char_to_id[USER_TOKEN]]
    ids += tokenizer.encode(user_msg)
    ids += [tokenizer.char_to_id[ASSISTANT_TOKEN]]
    ids += tokenizer.encode(assistant_msg)
    return ids


def build_prompt_ids(tokenizer, system_msg, user_msg):
    """A prompt for INFERENCE: everything up through <ASSISTANT>, with
    nothing after it -- generation fills in what comes next."""
    ids = [tokenizer.char_to_id[SYSTEM_TOKEN]]
    ids += tokenizer.encode(system_msg)
    ids += [tokenizer.char_to_id[USER_TOKEN]]
    ids += tokenizer.encode(user_msg)
    ids += [tokenizer.char_to_id[ASSISTANT_TOKEN]]
    return ids
