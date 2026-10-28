# Kennel — multi‑model crowd answer engine

## What it does
Kennel is a Python library that answers a question by querying multiple open models independently (through an OpenAI‑compatible endpoint), then reconciles agreement and dissent, grounds claims in supplied source passages, and learns from user feedback on the helpfulness of model responses.

## What is new in this project
This project was built from an empty folder during the Nebius × NVIDIA hackathon window.  
First commit (UTC): 2026-10-28T19:06:03Z.

## Clean-room statement
The code is written entirely from scratch; it contains no internal names, paths, keys or imports from any pre‑existing 3Dogs or other proprietary codebases. All file references are relative; no absolute paths appear in the source.

## Hackathon track
Nebius x NVIDIA Global AI Hackathon, Track 2: Best Apps and Agents. Runs on Nebius Token Factory with NVIDIA Nemotron 3 Super (nvidia/nemotron-3-super-120b-a12b) plus two independent open models from other makers: Qwen3 30B (Alibaba) and Gemma 3 27B (Google).

**Live demo:** https://hx6dz3wrnqydy4splilhkpfl5m0ygggd.lambda-url.us-east-2.on.aws/

Votes are kept in the function's temporary storage (`/tmp`), so they reset when the function restarts. Permanent vote storage, and using vote weights to rank answers, are next.

## Builder credits used
Builder credits used: Nebius Token Factory

## Run it
Deploy `src/kennel/handler.py` as an AWS Lambda Function URL. Set environment variables `KENNEL_API_KEY` (your Nebius Token Factory key), `KENNEL_MODELS` (comma‑separated model list; defaults include Nemotron and a non‑NVIDIA open model), and optionally `KENNEL_BASE_URL`.  
POST JSON `{"question": "...", "passages": [...]}` to get a crowd answer; POST `/vote` to record helpfulness votes; GET `/` for a simple HTML interface.

## License notice
Licensed by NVIDIA Corporation under the NVIDIA Open Model License.
