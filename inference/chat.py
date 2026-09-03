"""Chat-mode inference on the <SYSTEM>/<USER>/<ASSISTANT> format. CLI usage
from the project root:

    .venv\\Scripts\\python.exe -m inference.chat --message "what is a checkpoint?"
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.generate import generate_ids  # noqa: E402
from tokenizer.char_tokenizer import CharTokenizer  # noqa: E402
from training.chat_format import ASSISTANT_TOKEN, SYSTEM_TOKEN, USER_TOKEN, build_prompt_ids  # noqa: E402
from training.checkpoint import load_for_inference  # noqa: E402

DEFAULT_SYSTEM_MESSAGE = "you are a small assistant that explains this project."
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "phase27_chat_tuned.pt"
DEFAULT_TOKENIZER = ROOT / "tokenizer" / "vocab_chat.json"


def chat(model, tokenizer, user_message, system_message=DEFAULT_SYSTEM_MESSAGE, max_new_tokens=100, temperature=0.8, greedy=False):
    """Formats one chat turn, generates a reply, and returns just the
    assistant's own text -- stopping generation the instant the model
    produces ANY role token again (a new <SYSTEM> or <USER> turn, or even
    another <ASSISTANT> -- a malformed back-to-back turn), rather than
    letting it run past its own turn. All three role tokens are stop
    tokens, not just SYSTEM/USER: a hallucinated second <ASSISTANT> is
    just as much "past the end of this reply" as a new user turn would be,
    and letting it through would leak the literal token into the display."""
    prompt_ids = build_prompt_ids(tokenizer, system_message, user_message)
    stop_ids = {tokenizer.char_to_id[SYSTEM_TOKEN], tokenizer.char_to_id[USER_TOKEN], tokenizer.char_to_id[ASSISTANT_TOKEN]}

    full_ids = generate_ids(
        model, prompt_ids, max_new_tokens, temperature=temperature, greedy=greedy, stop_token_ids=stop_ids,
    )
    generated_ids = full_ids[len(prompt_ids):]
    if generated_ids and generated_ids[-1] in stop_ids:
        generated_ids = generated_ids[:-1]  # drop the halting role token itself, don't display it
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", required=True)
    parser.add_argument("--system", default=DEFAULT_SYSTEM_MESSAGE)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--tokenizer-path", default=str(DEFAULT_TOKENIZER))
    args = parser.parse_args()

    tokenizer = CharTokenizer.load(args.tokenizer_path)
    model, _ = load_for_inference(args.checkpoint)

    reply = chat(
        model, tokenizer, args.message, system_message=args.system,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, greedy=args.greedy,
    )
    print(reply)


if __name__ == "__main__":
    main()
