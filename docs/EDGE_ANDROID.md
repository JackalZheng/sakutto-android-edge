# Sakutto on Microsoft Edge for Android

This is the Edge for Android port of Sakutto (upstream:
<https://github.com/jammaru/sakutto>, MIT). Same on-device PLaMo 3
Translate engine, same local-only privacy promise — adapted for a
phone browser that has no keyboard, no Chrome Translator API, and a
much smaller extension surface than desktop Chrome.

## The one thing to understand first

Edge for Android does **not** let you sideload an unpacked extension,
and it does **not** open the full Chrome Web Store. Its stable
channel only installs extensions from Microsoft's own curated
in-app list (☰ → Extensions). The wider desktop catalog is only
reachable in Edge Canary with the **Android Extensions Search**
flag (`edge://flags`) turned on. So the release path is:

1. Build the Edge package in this repo (`pnpm build:edge`).
2. Submit it to **Microsoft Edge Add-ons** (the desktop store).
3. Test it on a phone in **Edge Canary** with the flag on.
4. Ask Microsoft / wait for it to appear in the stable Android list.

There is no supported way to "just install the zip on my phone"
on stable Edge. If that is a blocker, Firefox for Android is the
only mainstream phone browser that sideloads add-ons directly —
but that is a different port (upstream already ships a desktop
Firefox build; it is not marked Android-compatible).

## What changed in this branch (`edge-android`)

| Area                | Desktop Chrome                                                           | Edge for Android (this branch)                                                                                                  |
| ------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Target              | `wxt build`                                                              | `wxt build -b edge` / `pnpm build:edge`                                                                                         |
| Built-in translator | Chrome Translator API for 11 extra languages                             | **Not available on Edge** — PLaMo only, like Firefox (`hasBuiltInTranslatorUi()` is false on Edge)                              |
| Keyboard shortcuts  | Alt+A / Alt+B / Alt+S / Alt+T                                            | Same `suggested_key` as Chrome since 1.3.0 (desktop Edge binds them; Android ignores `commands`). Popup / in-page UI on a phone |
| Right-click menu    | `contextMenus`                                                           | Declared, but guarded: `installPageActions()` skips it silently when the API is absent on Android                               |
| Model               | 0.8B (573 MB) or 2B (1.8 GB)                                             | **Every model since 1.3.0.** Over 1 GB (Qwen3 1.7B, Hy-MT2 1.8B, PLaMo 3 2B) they run on the CPU (`phoneKeepsOffGpu`)           |
| Touch UI            | Desktop-sized controls                                                   | `base.css`: 44px minimum targets on `(pointer: coarse)`, 14px body text                                                         |
| Permissions         | offscreen, storage, unlimitedStorage, activeTab, scripting, contextMenus | Same set — Microsoft lists `offscreen` as supported on Android                                                                  |
| Package check       | `verify-package.ts` chrome/firefox                                       | Now accepts `--browser edge` (same permission set as Chrome)                                                                    |

New code: `src/shared/mobile.ts` (`isAndroidEdgeRuntime()` via the
`EdgA/` user-agent token, `MOBILE_MODEL_ID`, mobile limits),
`src/shared/manifest.ts` (`EDGE_PERMISSIONS`, `edgeCommands()`),
`src/shared/browser.ts` (`'edge'` build target, `isEdgeBuild()`).

## Build

```sh
pnpm install
pnpm build:wllama   # once, with Docker; see docs/BUILDING.md — tens of minutes
pnpm verify:wllama
pnpm build:edge     # production build in .output/edge-mv3
pnpm zip:edge       # Edge Add-ons zip in .output
```

For a smoke test without the real wasm (translation is faked, the
UI and manifest are real):

```sh
pnpm build:edge-fake   # SAKUTTO_FAKE_ENGINE=1 wxt build -b edge
```

Then check the built manifest:

```sh
python3 -c "import json; print(json.load(open('.output/edge-mv3/manifest.json')))"
# permissions: offscreen, storage, unlimitedStorage, activeTab, scripting, contextMenus
# commands: four actions, suggested_key Alt+A / Alt+B / Alt+S / Alt+T
# no minimum_chrome_version, no browser_specific_settings
```

## Test on a phone (Edge Canary)

1. On the phone, install **Microsoft Edge Canary** from Google Play.
2. Open `edge://flags`, enable **Android Extensions Search**, restart.
3. Open the Edge Add-ons / Extensions page and install the Sakutto
   listing (once it is published), or the desktop catalog entry.
4. Open a non-English page, tap the extensions button, open Sakutto.
5. First run: accept the PLaMo licence, then download the model —
   **use Wi-Fi, it is 573 MB**, and keep Edge in the foreground
   while it downloads. Expect the first load to be slow.
6. Translate a page, then try: selection translate (long-press
   select text, then use the popup), bilingual toggle, and the typed
   text page. Keyboard shortcuts and the right-click menu will not
   be there — that is expected on Android.

## v1.3.0 (store release): six models and six Adreno profiles

The store release is numbered 1.3.0 and includes the 1.3.1 development
notes below.

- **Models** (Settings → Model, smallest file first; on Android all six, the ones over 1 GB on the CPU):

  | Model                            | Size    | License                 | GPU on Adreno                  |
  | -------------------------------- | ------- | ----------------------- | ------------------------------ |
  | Qwen3 0.6B                       | 397 MB  | Apache-2.0              | experimental (Automatic → CPU) |
  | PLaMo 3 Translate 0.8B (default) | 573 MB  | PLaMo Community License | yes                            |
  | Gemma 3 1B IT                    | 806 MB  | Gemma Terms of Use      | experimental (Automatic → CPU) |
  | Qwen3 1.7B                       | 1.11 GB | Apache-2.0              | experimental (Automatic → CPU) |
  | Hy-MT2 1.8B                      | 1.13 GB | Apache-2.0              | yes                            |
  | PLaMo 3 Translate 2B             | 1.81 GB | PLaMo Community License | yes                            |

  The chosen model is stored in the settings (`modelId`) and the engine host
  starts on it. Prompts: `src/engine/prompt/model-prompt.ts`. Qwen3, Gemma 3
  and Hy-MT2 get one block per request with the inline `<i1>` tags removed;
  only PLaMo gets newline-packed requests and the tags.

- **Memory on a phone** (llama.cpp buffers, Q4_K_M, the phone's 128 MiB
  `maxStorageBufferBindingSize` sends the big embedding/output matrices to
  the CPU; on Adreno both sides are the same RAM):

  | Model        | Weights CPU + GPU | KV (2 slots / 2048) | Compute | Total     | With 1 slot / 1024 / batch 128 |
  | ------------ | ----------------- | ------------------- | ------- | --------- | ------------------------------ |
  | PLaMo 3 0.8B | 218 + 326 MiB     | 16 MiB              | 133 MiB | ~0.7 GiB  | (not applied, < 1 GB)          |
  | Hy-MT2 1.8B  | 194 + 882 MiB     | 128 MiB             | 148 MiB | ~1.32 GiB | ~1.19 GiB                      |
  | PLaMo 3 2B   | 363 + 1358 MiB    | 96 MiB              | ~150    | ~1.92 GiB | ~1.84 GiB                      |

  Hy-MT2 and PLaMo 2B closed Edge for Android while loading on WebGPU on a
  POCO F5 Pro, and both ran fine on the CPU there (the CPU path allocates no
  GPU buffers on top of the WASM heap). On a phone, a model over 1 GB runs on
  the CPU under Automatic and an explicit WebGPU choice alike
  (`phoneKeepsOffGpu` in `backend.ts`; the CPU tiers stay as chosen), and
  loads with `LARGE_MODEL_PHONE_LIMITS` (1 slot, n_ctx 1024, n_batch 128;
  `fitSelectionToModel`, every attempt). n_ctx 1024 is kept on the CPU too:
  it is what every WebGPU slot gets, the planner splits longer blocks, and
  2048 would only add KV (Hy-MT2 +64 MiB) next to a 1.1-1.8 GB file.

  The load-crash guard (`src/shared/model-load-guard.ts`) marks each load
  attempt with its backend. A new engine page that finds a mark left behind:
  - WebGPU: keeps the model and runs it on the CPU (`gpuCrashed`) until
    WebGPU is chosen again in the engine settings;
  - CPU: does not load that model again, resets the settings to PLaMo 3
    0.8B.
    The model list says which one happened.

- **Title speed, CPU vs WebGPU**: see "Titles vs paragraph" in the debug
  harness (GPU performance section). It runs 12 titles packed, 12 titles one
  request each (headings), and one paragraph through the production engine
  on cpu, cpu-2-slot, gpu-r6-r4 (1 slot), gpu-r6-r4-2-slot and
  gpu-r6-r4-4-slot, one config per page load, and reports wall time, first
  finished block and each request's ttft / prompt / decode timings.

- **One slot everywhere on a phone (1.3.0)**. The POCO F5 Pro harness:
  multi-slot was slower on the CPU and on WebGPU. Gemma 3 1B, titles one
  request each: cpu 15.1 s, cpu-2-slot 33.2 s, gpu 1 slot 14.7 s, 2 slots
  31.8 s, 4 slots 28.0 s; paragraph: cpu 12.8 s, gpu 1 slot 11.6 s, 2 slots
  13.1 s. Adreno runs the slots of a wave one after the other, and with more
  than one slot `cache_prompt` is off (`runtime.ts`), so every title
  prefilled the whole template again (prompt tokens per title: Gemma 11 ->
  36, PLaMo 15 -> 27). Now:

  |                                       | before                                         | after                                                                       |
  | ------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------- |
  | Adreno WebGPU (every profile)         | 2 slots, n_ctx 2048 (Low memory: 1 slot, 1024) | 1 slot, n_ctx 2048 (Low memory: 1 slot, 1024)                               |
  | Explicit WebGPU on a non-Adreno phone | 4 slots, n_ctx 4096                            | 1 slot, n_ctx 2048                                                          |
  | CPU                                   | 1 slot, n_ctx 2048                             | unchanged                                                                   |
  | Models over 1 GB on a phone           | CPU, 1 slot, n_ctx 1024, batch 128             | unchanged                                                                   |
  | Desktop non-Adreno WebGPU             | 4 slots (16 on measured GPUs)                  | unchanged: an RTX 4090 ran a long article in 8.1 s on 1 slot and 3.7 s on 4 |
  | Page requests in flight               | GPU tier 32, CPU tiers 2                       | 3 for a one-slot engine (GPU or CPU), 32 for a multi-slot one               |
  | Look-ahead                            | GPU 2 screens, CPU c8 1, c4/c1 0.5             | 2 screens on every backend (fast pace +1)                                   |

  When the in-flight limit is reached and a block scrolls into view, the
  page hands one prefetched request back if at least two have not streamed
  yet: it aborts the lowest-priority one (at least one band, 12, behind the
  visible block) and puts it back in its queue. With one slot the prompt cache is on, and
  a second wllama slot id is never needed.

- **Adreno GPU profiles** (`src/shared/adreno-profiles.ts`): Optimized
  (`q4v2r4+mvglu+fna`, 1 slot, n_ctx 2048), q4_K only (`q4v2r4`), Round 6
  2 rows (`q4v2r2+mvglu+fna`), Low memory (Optimized with n_ctx 1024),
  Compatible (round 5, no tokens), Round 4 (`bmv0+rwg256`). The subgroup
  profiles (`adreno-upstream`, `adreno-subgroups`, `adreno-sg-tiled`) garble
  and are harness-only; `sub64` lost the device; hybrid was removed.
- A broken-output verdict on a GPU-experimental model moves that load to the
  CPU without storing the per-version flag.

## v1.3.1 (development build): Automatic on Adreno = WebGPU (round 6)

- **Automatic on Adreno = WebGPU** with the Adreno profile (Edge for Android
  and Windows on Snapdragon). POCO F5 Pro (Adreno 730, PLaMo 3 0.8B, one
  paragraph, warm): round 6 2.25 s / decode 22.2 t/s, CPU (4 threads)
  2.91 s / 17.3 t/s, round 5 2.69 s / 15.9 t/s. Other Android GPUs keep
  Automatic on the CPU.
- **Adreno GPU profile:**
  - Optimized (default): round 6 = the ggml-webgpu Adreno defaults plus the
    tokens `q4v2r4+mvglu+fna` (q4_K mat-vec v2, 16 threads x 4 rows per
    group; fused gate+up mat-vec + SWIGLU; fused RMS_NORM+MUL+ADD chain).
  - Compatible (round 5): the ggml-webgpu Adreno defaults only (the 1.3.0
    Optimized profile).
  - Hybrid was removed (slower in every phone run, one 41 s stall); a stored
    `hybrid` is rewritten as `optimized`.
- **Broken-output flag** (`webgpuOutputBroken`) now stores the extension
  version and only counts for that version, so a flag from 1.3.0 (stored as
  `true`) does not keep the new kernels off. While it is set, Automatic and
  GPU (WebGPU) both run on the CPU; choosing GPU (WebGPU) clears it.
- **Debug harness:** the perf sweep skips the hybrid and per-op runs (the
  per-op run reloaded the page on the phone); per-op timing is an opt-in
  checkbox for a single breakdown.

## v1.3.0: compute backend and the Adreno GPU profile

Settings → **Compute backend**: Automatic (recommended) / GPU (WebGPU) /
CPU. The diagnostics list shows the detected GPU and the active backend.

- **Automatic on Adreno = CPU.** Measured on a POCO F5 Pro (PLaMo 3 0.8B,
  one paragraph): CPU 2.52 s, WebGPU round-4 kernels 3.77 s, hybrid
  (half the layers on the GPU) 3.28 s. Re-evaluate when the round-5
  kernels have phone numbers.
- **GPU (WebGPU) on Adreno** runs the Adreno defaults compiled into
  ggml-webgpu (no subgroups, mat-vec WG 64, SOFT_MAX without cache,
  row-reduction WG 128, batched mat-vec for 5–64 token prompts, 16 kernels
  per submit) with 1 slot and n_ctx 2048 (2 slots before 1.3.0). Flash attention is
  unavailable without subgroups, so it stays off.
- **Adreno GPU profile** (shown on Adreno only):
  - Optimized (default): the above.
  - Round-4 compatible: the same without batched mat-vec (`bmv0`).
  - Hybrid: offloads the output layer plus the last half of the blocks
    (0.8B: 9 of 16+1, 2B: 13 of 24+1); the rest runs on the CPU.
- **Fallback.** WebGPU init failure or device loss steps down to the CPU;
  garbled output (the Android output guard) sets a stored flag so later
  sessions stay on the CPU until GPU (WebGPU) is chosen again.
- **Release vs debug build.** `pnpm build:wllama` + `pnpm build:edge` /
  `pnpm zip:edge` leave out the GPU harness page and `test-backend-ops`;
  `pnpm build:wllama:debug` + `pnpm build:debug` include them.

## Known risks (verify on a real device, do not assume)

- **WebGPU on Android is device-dependent.** Chrome Android 121+
  supports WebGPU, but Adreno / Mali drivers vary; many phones will
  fall back to the WASM CPU path. On CPU the 0.8B model is much
  slower than on a desktop — measure time-to-first-token on the
  target phone before promising anything.
- **Offscreen document lifetime.** Android kills background work
  aggressively. A long page translation may be interrupted when
  Edge is backgrounded; re-opening Edge and re-running is the
  current workaround, there is no background keep-alive yet.
- **COEP/COOP are kept** (`require-corp` / `same-origin`) because
  the wasm runtime needs cross-origin isolation. If the model
  download from Hugging Face or the offscreen document fails on
  Edge Android, this header pair is the first thing to re-test.
- **Storage.** 573 MB model + cache needs `unlimitedStorage`,
  which is requested, but Android storage pressure can still evict
  it. Re-download is the fallback.
- **Edge Add-ons review** may ask for the same source / licence
  package as AMO / CWS (see `docs/AMO_UPLOAD.md`,
  `docs/CWS_UPLOAD.md` for the pattern). The wllama build is
  pinned and patched — keep `wllama-build.json` in the submission.

## Not in scope

- No native Android app, no Play Store listing — this stays a
  browser extension.
- No Chrome Web Store release for Android (Chrome for Android
  does not run extensions at all).
- No Chrome-only languages on Edge. Models over 1 GB run on the CPU on a phone.
