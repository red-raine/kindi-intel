# Troubleshooting

**`OutOfMemoryError` at start.** Something else holds GPU memory. `nvidia-smi` lists GPU processes (a desktop
session uses a few MB, which is fine; another CUDA or Vulkan program is not). Free the card, or lower `ctx`
(each 1,000 tokens of context is ~12 MB).

**`no model pack at ...`.** Run `kindi convert <model.gguf>` first, or point `pack` at the right directory.

**`needs the PTQ1_0 GGUF`.** Kindi reads the PTQ1_0 files of Ternary-Bonsai-2-2B, -4B, -9B and -27B. The Q2_0 / PQ2_0
files hold the same models in another format, and Kindi does not support them yet.

**`kindi-cpu: the pack at ... does not match the model layout`.** The CPU engine reads its tensors by name, and those
names come from the GGUF the pack was converted from, so a pack for another model - or one written by an older
`kindi convert` - is refused at load instead of failing later. The message lists what is missing, from which index,
against how many tensors the layout reads, and how many the pack holds that the layout never reads. Convert the
matching GGUF again: `kindi convert MODEL.gguf DIR`. A pack missing whole files (no `index.json`,
`bcodes_index.json`, `bcodes.bin`) is not a CPU pack at all: a Gemma 4 or Nemotron pack holds `model_path` and is
served by the GPU backend.

**CUDA errors such as `no kernel image is available` or `CUDA driver version is insufficient`.** Use the NVIDIA 470
driver and the CUDA 10.2 stack in the Docker image.

**The first start is slow.** Kindi compiles its CUDA kernels with NVRTC on first use (about a minute) and caches them
under `state_dir/kernels`.

**A client times out or retries on the first request.** Kindi reads the first long prompt at about 25-33 tokens/s. Raise the
client's timeout and limit retries (see [AGENTS.md](AGENTS.md)); watch progress with `curl -s
localhost:8093/health`.

**Answers are slow on every turn of an agent.** Check that the agent sends the same tools and system prompt each
time. `/health` and the server log show `prompt: N reused, M read` per request.

**`401 invalid API key`.** The server has `api_key` set; send `Authorization: Bearer <key>`.

**`the prompt has N tokens; the context is M`.** The conversation outgrew `ctx`. Let the client summarise older turns,
or raise `ctx` if memory allows.

**`kindi verify` fails.** The pack is from a different model file, or the GPU stack differs from the tested one.
Re-run `kindi convert --force` from the official file and check the driver.
