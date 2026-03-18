#!/usr/bin/env python3
"""
article_to_harvey.py
Ollama + Qwen3-TTS pipeline that reads a news article and outputs
audio in the style of Paul Harvey.

Usage:
    python article_to_harvey.py article.txt output.wav
    echo "Article text..." | python article_to_harvey.py - output.wav
    python article_to_harvey.py article.txt output.wav --clone-audio harvey_sample.wav
    python article_to_harvey.py article.txt output.wav --model minimax-m2.5:cloud
"""

import argparse
import re
import sys
import urllib.request
import urllib.error
import json
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
- Output ONLY the spoken script, no stage directions or labels
"""

VOICE_DESCRIPTION = (
    "A warm, authoritative American male broadcaster voice. "
    "Deliberate, measured pacing with dramatic pauses between phrases. "
    "Slightly gravelly baritone texture. Mid-century radio announcer cadence. "
    "Confident, intimate storytelling delivery — as if speaking directly to one listener."
)

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "minimax-m2.5:cloud"

# ---------------------------------------------------------------------------

def read_article(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path) as f:
        return f.read()


def format_as_harvey(article: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": article},
        ],
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def strip_pause_markers(script: str) -> str:
    """Replace [pause] with a comma so TTS infers a natural pause."""
    return re.sub(r"\[pause\]", ", ", script)


def synthesize_voice_design(script: str, tts_model: Qwen3TTSModel, output_path: str):
    tts_text = strip_pause_markers(script)
    audio = tts_model.generate_voice_design(
        text=tts_text,
        voice_description=VOICE_DESCRIPTION,
    )
    tts_model.save_audio(audio, output_path)


def synthesize_voice_clone(script: str, tts_model: Qwen3TTSModel, output_path: str, clone_audio: str):
    tts_text = strip_pause_markers(script)
    audio = tts_model.generate_voice_clone(
        text=tts_text,
        reference_audio=clone_audio,
    )
    tts_model.save_audio(audio, output_path)


def main():
    parser = argparse.ArgumentParser(description="Paul Harvey TTS pipeline")
    parser.add_argument("article", help="Path to article text file, or - for stdin")
    parser.add_argument("output", help="Output audio file path (.wav)")
    parser.add_argument(
        "--clone-audio",
        metavar="PATH",
        help="Path to a Paul Harvey audio clip for voice cloning (uses voice design if omitted)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use for text formatting (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--tts-size",
        choices=["1.7B", "0.6B"],
        default="1.7B",
        help="Qwen3-TTS model size (default: 1.7B)",
    )
    parser.add_argument("--print-script", action="store_true", help="Print the formatted script before synthesizing")
    args = parser.parse_args()

    # Step 1: read article
    print("Reading article...")
    article = read_article(args.article)

    # Step 2: LLM formatting via Ollama
    print(f"Formatting as Paul Harvey via Ollama ({args.model})...")
    script = format_as_harvey(article, args.model)

    if args.print_script:
        print("\n--- Paul Harvey Script ---")
        print(script)
        print("--------------------------\n")

    # Step 3: TTS synthesis
    if args.clone_audio:
        tts_model_name = f"Qwen3-TTS-12Hz-{args.tts_size}-Base"
        print(f"Loading {tts_model_name} for voice clone...")
        tts = Qwen3TTSModel(tts_model_name)
        print("Synthesizing (voice clone)...")
        synthesize_voice_clone(script, tts, args.output, args.clone_audio)
    else:
        tts_model_name = f"Qwen3-TTS-12Hz-{args.tts_size}-VoiceDesign"
        print(f"Loading {tts_model_name} for voice design...")
        tts = Qwen3TTSModel(tts_model_name)
        print("Synthesizing (voice design)...")
        synthesize_voice_design(script, tts, args.output)

    print(f"Done. Audio saved to: {args.output}")


if __name__ == "__main__":
    main()
