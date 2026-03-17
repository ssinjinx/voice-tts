#!/usr/bin/env python3
"""
article_to_harvey.py
LLM + Qwen3-TTS pipeline that reads a news article and outputs
audio in the style of Paul Harvey.

Usage:
    python article_to_harvey.py article.txt output.wav
    echo "Article text..." | python article_to_harvey.py - output.wav
    python article_to_harvey.py article.txt output.wav --clone-audio harvey_sample.wav
"""

import argparse
import re
import sys
import anthropic
from qwen_tts import Qwen3TTSModel

# ---------------------------------------------------------------------------
# Paul Harvey LLM prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are Paul Harvey, the legendary American radio broadcaster known for your
distinctive storytelling style on "The Rest of the Story" and "News and Comment."

Rewrite the provided article in your voice and style. Rules:

STRUCTURE:
- Open with a dramatic hook that pulls the listener in immediately
- Build suspense by withholding the key reveal until near the end
- Use "Page 2..." as a transition into the second half
- End with your signature: "And now you know... the rest of the story."
- Sign off with: "Good day!"

PACING AND PAUSES:
- Insert [pause] where you would take a dramatic breath or let a point land
- Use ellipses (...) for trailing, contemplative thoughts
- Short sentences for impact. Very short.
- Then a longer sentence to build and carry the listener along with you.

VOICE:
- Warm, personal, conversational — as if speaking to one friend, not a crowd
- Occasional wry humor; never sarcastic
- Patriotic undertone when relevant, never preachy
- Names and places spoken with familiarity, as if you know them personally
- Avoid jargon; prefer plain, vivid language

FORBIDDEN:
- Do not summarize — dramatize
- Do not use bullet points or lists
- Do not break character or mention AI
- Keep the factual content accurate; only style changes
"""

VOICE_DESCRIPTION = (
    "A warm, authoritative American male broadcaster voice. "
    "Deliberate, measured pacing with dramatic pauses between phrases. "
    "Slightly gravelly baritone texture. Mid-century radio announcer cadence. "
    "Confident, intimate storytelling delivery — as if speaking directly to one listener."
)

# ---------------------------------------------------------------------------

def read_article(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path) as f:
        return f.read()


def format_as_harvey(article: str, client: anthropic.Anthropic) -> str:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": article}],
    )
    return response.content[0].text


def strip_pause_markers(script: str) -> str:
    """Replace [pause] with a comma+space so TTS infers a natural pause."""
    return re.sub(r"\[pause\]", ", ", script)


def synthesize_voice_design(script: str, model: Qwen3TTSModel, output_path: str):
    tts_text = strip_pause_markers(script)
    audio = model.generate_voice_design(
        text=tts_text,
        voice_description=VOICE_DESCRIPTION,
    )
    model.save_audio(audio, output_path)


def synthesize_voice_clone(script: str, model: Qwen3TTSModel, output_path: str, clone_audio: str):
    tts_text = strip_pause_markers(script)
    audio = model.generate_voice_clone(
        text=tts_text,
        reference_audio=clone_audio,
    )
    model.save_audio(audio, output_path)


def main():
    parser = argparse.ArgumentParser(description="Paul Harvey TTS pipeline")
    parser.add_argument("article", help="Path to article text file, or - for stdin")
    parser.add_argument("output", help="Output audio file path (.wav)")
    parser.add_argument(
        "--clone-audio",
        metavar="PATH",
        help="Path to a 3-second Paul Harvey audio clip for voice cloning (optional; uses voice design if omitted)",
    )
    parser.add_argument(
        "--model-size",
        choices=["1.7B", "0.6B"],
        default="1.7B",
        help="Qwen3-TTS model size (default: 1.7B)",
    )
    parser.add_argument("--print-script", action="store_true", help="Print the LLM-formatted script before synthesizing")
    args = parser.parse_args()

    # --- Step 1: read article ---
    print("Reading article...")
    article = read_article(args.article)

    # --- Step 2: LLM formatting ---
    print("Formatting as Paul Harvey via Claude...")
    client = anthropic.Anthropic()
    script = format_as_harvey(article, client)

    if args.print_script:
        print("\n--- Paul Harvey Script ---")
        print(script)
        print("--------------------------\n")

    # --- Step 3: TTS synthesis ---
    if args.clone_audio:
        model_name = f"Qwen3-TTS-12Hz-{args.model_size}-Base"
        print(f"Loading {model_name} for voice clone...")
        model = Qwen3TTSModel(model_name)
        print("Synthesizing (voice clone)...")
        synthesize_voice_clone(script, model, args.output, args.clone_audio)
    else:
        model_name = f"Qwen3-TTS-12Hz-{args.model_size}-VoiceDesign"
        print(f"Loading {model_name} for voice design...")
        model = Qwen3TTSModel(model_name)
        print("Synthesizing (voice design)...")
        synthesize_voice_design(script, model, args.output)

    print(f"Done. Audio saved to: {args.output}")


if __name__ == "__main__":
    main()
