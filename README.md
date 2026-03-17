# Voice TTS Module

> Paul Harvey-style text-to-speech pipeline for long-form news articles

## Goal

Take long news articles, process them through an LLM to match Paul Harvey's storytelling style (dramatic pauses, deliberate pacing, narrative structure), then synthesize the audio using Qwen3-TTS to sound like Paul Harvey.

## Approach

### Step 1 — LLM Text Formatting
Feed raw article text to an LLM with a system prompt that:
- Rewrites content using Paul Harvey's structure ("Page 2..." transitions, dramatic reveals)
- Inserts pause markers `[pause]` at dramatic moments
- Controls pacing via punctuation and sentence breaks
- Applies his signature "And now you know... the rest of the story." closing

### Step 2 — TTS Synthesis (Qwen3-TTS)
Two strategies, starting with VoiceDesign:

**Option A — Voice Design (no audio sample needed)**  
Use `Qwen3-TTS-12Hz-1.7B-VoiceDesign` with a natural language description:
```
"A warm, authoritative American male broadcaster voice with deliberate pacing, 
dramatic pauses between phrases, slightly gravelly texture, mid-century radio 
announcer cadence, confident and storytelling-focused delivery."
```

**Option B — Voice Clone (with Paul Harvey audio sample)**  
Use `Qwen3-TTS-12Hz-1.7B-Base` with a 3-second audio reference clip.  
Better fidelity if a clean sample can be sourced.

### Pipeline Flow
```
Article text
    → LLM (Claude/GPT) with Paul Harvey style prompt
    → Formatted script with pause markers
    → Qwen3-TTS (VoiceDesign or Clone)
    → Audio output (.wav / .mp3)
```

## Tech Stack

| Component | Tool |
|-----------|------|
| TTS Model | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (1.7B VoiceDesign or Base) |
| LLM | Claude via API |
| Python Runtime | Python 3.12 + conda env |
| Package | `qwen-tts` (PyPI) |

## Models

| Model | Size | Use Case |
|-------|------|----------|
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1.7B | Describe Paul Harvey's voice in natural language |
| `Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | Clone from 3-sec audio reference |
| `Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | Lighter clone option for less VRAM |

Streaming supported — 97ms first-packet latency for real-time playback.

## Notes on StabilityMatrix

StabilityMatrix is a package manager for image-gen AI (ComfyUI, A1111, etc.). It does **not** natively support Qwen3-TTS. Run the TTS pipeline directly via Python/conda outside of StabilityMatrix.

## Next Steps

- [ ] Set up conda env with `qwen-tts`
- [ ] Write Paul Harvey LLM prompt template
- [ ] Test VoiceDesign with voice description
- [ ] Source a clean Paul Harvey audio clip for voice clone comparison
- [ ] Build pipeline script: `article_to_harvey.py`
- [ ] Evaluate output quality and iterate on voice description/prompt
