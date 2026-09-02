import json
import logging

from inference.generate import stream_ids

logger = logging.getLogger("api.sse")


def format_event(payload):
    """SSE wire format: a 'data: ' line with a JSON payload, terminated by
    a blank line. The blank line is what tells the reader one event has
    ended -- SSE has no length prefix, just this delimiter."""
    return f"data: {json.dumps(payload)}\n\n"


async def sse_token_stream(model, tokenizer, generate_request, http_request):
    """Async generator yielding SSE-formatted events for each generated
    token, checking for client disconnection between tokens so an
    abandoned connection stops consuming CPU rather than generating (and
    discarding) a full response nobody will receive."""
    prompt_ids = tokenizer.encode(generate_request.prompt)
    token_gen = stream_ids(
        model,
        prompt_ids,
        generate_request.max_new_tokens,
        temperature=generate_request.temperature,
        top_k=generate_request.top_k,
        top_p=generate_request.top_p,
        greedy=generate_request.greedy,
    )
    try:
        for next_id in token_gen:
            if await http_request.is_disconnected():
                logger.info("client disconnected mid-stream, stopping generation")
                token_gen.close()
                return
            chunk = tokenizer.decode([next_id])
            yield format_event({"chunk": chunk})
        yield format_event({"done": True})
    except Exception:
        logger.exception("streaming generation failed")
        yield format_event({"error": "Generation failed. See server logs for details."})
