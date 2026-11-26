# Changelog

Release sections are added by `pnpm release`. Each section lists commit subjects since the previous `vX.Y.Z` tag.

## [1.3.0] - 2026-11-24

### Notes

- 1.3.0 is the first store release after 1.2.1. It folds in the unreleased 1.3.0 / 1.3.1 development builds (the version number stays 1.3.0).

### Features

- Settings → Model: all six translation models can be downloaded, switched and deleted, listed by file size: Qwen3 0.6B (397 MB, Apache-2.0), **PLaMo 3 Translate 0.8B** (573 MB, PLaMo Community License, default), Gemma 3 1B IT (806 MB, Gemma Terms of Use), Qwen3 1.7B (1.11 GB, Apache-2.0), Hy-MT2 1.8B (1.13 GB, Apache-2.0) and PLaMo 3 Translate 2B (1.81 GB). Each entry shows its size, license link, a short description, and whether it is downloaded or in use. The chosen model is remembered across restarts. Edge for Android offers every model; the ones over 1 GB run on the CPU there (see Bug Fixes).
- Every model gets a production translation prompt: PLaMo's PFML3; Qwen3 (ChatML, thinking off) and Gemma 3 turns with any supported source/target language, Japanese polite/plain style and glossary terms (page context is left out for these small chat models: it made them translate it); Hy-MT2's model-card templates (Chinese or English instruction, terminology and context forms). Each format has its own stop tokens, and stray `<think>` blocks or control tokens are stripped from the output.
- Qwen3 0.6B / 1.7B and Gemma 3 1B are labelled **GPU experimental**: on an Adreno GPU, Automatic runs them on the CPU (they drift from the CPU result there); choosing GPU (WebGPU) still runs them on the GPU for testing. A broken-output verdict for one of them does not keep PLaMo or Hy-MT2 off the GPU.
- Settings → Adreno GPU profile now offers six profiles built from the GPU harness setups that translated correctly on the test phone, each with its recommendation under the selector (zh-TW, en, zh-CN, ja, ko; English elsewhere): Optimized (recommended, r6-r4), q4_K kernel only (q4v2-r4), Round 6 with 2 rows (r6), Low memory (r6-r4 with 1 slot / n_ctx 1024), Compatible (round 5) and Round 4 fallback. The subgroup profiles that garble on Adreno are not offered. The selector appears when WebGPU is chosen, or under Automatic on an Adreno GPU.

### Improvements

- WebGPU on Adreno is now faster than the CPU, and Automatic uses it. The new default Adreno profile (Optimized) adds round-6 kernels: a q4_K mat-vec with 16-byte loads and 4 rows per thread group, a fused gate+up mat-vec + SwiGLU, and a fused RMS_NORM+MUL+ADD(+RMS_NORM+MUL) chain (decode kernels per token 355 → 277). On a POCO F5 Pro (Adreno 730, PLaMo 3 0.8B, one paragraph, warm) a translation took 2.25 s (decode 22.2 tokens/s) vs 2.91 s on the CPU (17.3 tokens/s) and 2.69 s with the 1.3.0 GPU kernels (15.9 tokens/s). (#18)
- On Adreno (Edge for Android and Windows on Snapdragon), Automatic now picks WebGPU with the Adreno profile. Other Android GPUs still use the CPU under Automatic; other desktop GPUs are unchanged.
- Adreno GPU profile: Optimized (recommended) = round 6; **Compatible (round 5)** = the 1.3.0 Optimized settings, as a fallback. The Hybrid option (half the layers on the GPU) is removed: it was slower than full GPU in every phone run and once stalled for 41 s. A saved Hybrid choice becomes Optimized; a saved Round-4 compatible choice now means round 5.
- The "WebGPU produced garbled text" flag is now kept per extension version. A flag stored by 1.3.0 or earlier no longer keeps Automatic off the GPU after the update; if the new kernels garble text too, the output guard moves to the CPU again and remembers it for this version (choose GPU (WebGPU) to retry).
- Debug build only: the GPU perf sweep no longer runs per-op timing (it crashed the page on the phone) or the hybrid configs; per-op timing is an opt-in checkbox for a single breakdown. The translation-run harness gains `gpu-release` (the 1.3.1 Adreno default tokens).

### Bug Fixes

- The popup's **Translate this page** did nothing on a page where the content script was already running (Muse, desktop Chromium; `page:state` stayed inactive with no error). Page commands from the popup and the context menu went through the keyboard-shortcut gate, which treated the empty Edge binding as a key cleared by the user. Commands now carry their source, and only a browser keyboard command is gated.
- The Edge package declares the Alt+A / Alt+B / Alt+S / Alt+T suggested keys again, the same as Chrome (desktop Edge and Chromium bind them at install; Edge for Android ignores `commands`).
- Qwen3 / Gemma 3 / Hy-MT2 translated some blocks two or three times (reported with Qwen3 0.6B on a phone). Short blocks were packed one per line; Qwen3 0.6B lost the line count (and answered in Japanese), so each line was translated again on its own, and a block with a link was then translated a third time because the chat models drop the inline `<i1>` tags. Only PLaMo gets packed requests and the tags now; the chat models get one plain block per request, and a plain answer for a tagged block is painted without another pass.
- Edge for Android: Hy-MT2 1.8B and PLaMo 3 Translate 2B closed the engine while loading on WebGPU on a POCO F5 Pro; both run fine on the CPU there. Measured llama.cpp buffers (with the phone's 128 MiB storage-binding limit): PLaMo 3 0.8B ~0.7 GiB, Hy-MT2 ~1.32 GiB, PLaMo 3 2B ~1.92 GiB, and on Adreno the GPU buffers come out of the same RAM as the page. On a phone, models over 1 GB (Qwen3 1.7B, Hy-MT2 1.8B, PLaMo 3 2B) now run on the CPU under Automatic and an explicit WebGPU choice alike (the settings say so), and load with one slot, n_ctx 1024 and a 128-token batch on every attempt. PLaMo 3 2B can be downloaded and used on a phone again. A load that closes the engine page is detected by the next one: a crash on WebGPU keeps the model and runs it on the CPU from then on (choosing WebGPU again in the settings retries the GPU); only a crash on the CPU switches back to PLaMo 3 0.8B. The model list explains which one happened.
- Debug harness: a **Titles vs paragraph (CPU vs GPU)** comparison runs the production pipeline on 12 short titles (packed, and one request each as headings) and on one paragraph, on cpu, cpu-2-slot, gpu-r6-r4 (1 slot), gpu-r6-r4-2-slot and gpu-r6-r4-4-slot, and reports the wall time and every request's ttft. The harness WebGPU configs on Adreno now use the production slot count (they ran 4 before).
- One slot on a phone. The POCO F5 Pro harness showed multi-slot slower on the CPU and on WebGPU (Gemma 3 1B, titles one request each: 1 slot 14.7 s, 2 slots 31.8 s on the GPU; CPU 15.1 s vs 33.2 s): Adreno ran the slots one after the other, and with several slots every title prefilled the whole prompt again (no prefix cache). Every Adreno profile now runs 1 slot with n_ctx 2048 (Low memory: n_ctx 1024), and an explicit WebGPU choice on another phone GPU runs 1 slot too. Desktop GPUs keep 4 (or 16) slots.
- Page scheduling no longer depends on the backend: the page looks 2 screens ahead on every backend (the CPU tiers looked 1 or 0.5), and keeps 3 requests in flight for a one-slot engine (GPU 32 / CPU 2 before; a multi-slot desktop GPU keeps 32). A block that scrolls into view takes over a prefetched request that has not started streaming.

### Features (1.3.0 development build)

- Settings: the performance mode is now **Compute backend**, with Automatic (recommended), GPU (WebGPU) and CPU. The diagnostics list shows the detected GPU (vendor, architecture and adapter name) next to the active backend.
- Settings: on Qualcomm Adreno GPUs (Edge for Android, Windows on Snapdragon) a new **Adreno GPU profile** selector picks how WebGPU runs: Optimized (recommended; batched mat-vec for prompts up to 64 tokens), Round-4 compatible (the phone-tested settings without batched mat-vec), or Hybrid (about half the layers on the GPU, the rest on the CPU). Each option has a short description. The profile applies only when the compute backend is GPU (WebGPU). (#16)

### Improvements (1.3.0 development build)

- WebGPU on Adreno now runs a tuned kernel set by default: no subgroups, 64-wide mat-vec, a SOFT_MAX shader without the workgroup cache, a 128-wide row reduction, batched mat-vec for 5–64 token prompts, and 16 kernels per submit. An explicit GPU choice on Adreno uses 1 slot and n_ctx 2048 (2 slots in the development builds); flash attention stays off there. Other GPUs keep the upstream defaults. (#14, #16)
- On Adreno, Automatic picks the CPU for now. On a POCO F5 Pro (Adreno 7xx, PLaMo 3 0.8B) a paragraph took about 2.52 s on the CPU, 3.77 s on WebGPU with the round-4 kernels and 3.28 s in the hybrid setup. GPU (WebGPU) stays selectable.
- If WebGPU fails to start, loses the device, or produces garbled text, Sakutto moves to the CPU and remembers it until you choose GPU (WebGPU) again in the settings.
- The release package no longer ships the GPU test harness page or the `test-backend-ops` build of the runtime (about 1.2 MB smaller per wasm). `pnpm build:wllama:debug` and `pnpm build:debug` still build them for diagnostics.

### Bug Fixes

- Chrome and Edge: the engine (offscreen document) now reads the stored compute backend, Adreno profile and broken-GPU flag through the background. Before, the offscreen document had no storage access, so a saved CPU/GPU choice only took effect after it was changed again in the settings.
- WebGPU on Adreno no longer produces wrong text from the Q6_K mat-vec and SOFT_MAX kernels. (#16)

### Known issues (1.3.0 development build)

- Qwen3 and Gemma 3 (GPU harness models only, not shipped in the translator) still drift from the CPU output on WebGPU for some prompts (Qwen3 on short prompts, Gemma on long ones). They are labelled “GPU experimental” in the debug harness. PLaMo 3 matches the CPU.
- GPU speed on Adreno has only been measured with the round-4 kernels; the round-5 batched mat-vec has been validated for correctness on the build machine, not timed on a phone yet.

## [1.2.1] - 2026-11-16

### Bug Fixes

- On Edge for Android, page translation no longer stalls at 0/N: the first-use quality benchmark and the engine warm-up no longer run ahead of a page translation on touch devices, and the progress counter shows partial progress while the first block streams.
- New installs now default the translation target to the browser language instead of Japanese, so a Traditional Chinese browser translates into Traditional Chinese.

### Features

- Mobile gestures for Edge for Android: a floating action button (tap to translate or restore, long-press for the selection or the writing translator), a bilingual toggle on the progress bar, and 44px touch targets for the selection and writing icons. The settings page shows a Gestures section instead of keyboard shortcuts on Android.

## [1.2.0] - 2026-11-10

### Features

- Translate the text you are typing, on the device. Focus a message box and press Alt+T (Option+T on a Mac), click the あA button next to the field, or use the right-click menu. A live preview bar appears above the field; keep typing and it updates. Press the same shortcut again, or Ctrl+Enter (⌘+Enter on a Mac), to replace the draft; Restore original puts the previous text back. You can drag the bar, park it below the field, or move it to the bottom-right of the window. The preview has a language picker, and settings can skip the preview so one shortcut press translates and replaces. Japanese drafts go to English (or to your saved page-translation target if that is not Japanese); other drafts use that target until you pick another language. Password fields, card numbers, and sites set to never translate are left alone. You can hide the あA button. (#42)

### Improvements

- The repository now has a Firefox build target for desktop Firefox 153 and later. It is not listed on Firefox Add-ons in this GitHub release. (#50, #54, #55)

### Bug Fixes

- Page translation no longer stops partway after a download or a cancelled "leave this site?" prompt. When the tab is hidden, work already in progress is kept instead of being thrown away and redone. (#31)
- On AMD and Intel GPUs, WebGPU failures now fall back to CPU more reliably, and a correctness check disables a bad GPU setup instead of producing garbage text. This does not mean AMD Radeon WebGPU now works. (#49)
- A shortcut you clear on Chrome's shortcut page now stays off. Fresh installs still get Alt+A, Alt+B, Alt+S, and Alt+T. If you had already left a shortcut unassigned in Chrome before this update, the in-page default may still work until Chrome assigns a key or you turn that command off in Sakutto. (#52)

### Contributors

- Thanks to @sabopura for fixing translation that stopped partway, and for stopping wasted recomputation when the tab is hidden. (#31)

## [1.1.0] - 2026-11-09

### Features

- Faster GPU page translation — on a long, text-heavy web page translated with the GPU (WebGPU), the whole page finished about 3.7 times faster (14.6 seconds → 4.0 seconds), and the first translated paragraph appeared about 1.6 times sooner (0.5 seconds → 0.3 seconds). These numbers are from the project's long-article benchmark. They do not describe CPU translation. Times vary by page and by computer. (#27)
- The extension's screens are now available in 14 languages: English, Japanese, Korean, Simplified Chinese, Traditional Chinese, Spanish, Brazilian Portuguese, French, German, Italian, Russian, Vietnamese, Thai, and Indonesian. Anyone can contribute translations. (#30)
- Keyboard shortcuts can now be changed from Sakutto's own screens. (#32)

### Contributors

- Thanks to @sabopura for making GPU page translation faster. (#27)

## [1.0.2] - 2026-11-08

### Features

- Faster GPU page translation — the first translated paragraph appears sooner, and long pages finish faster. On the project's long-article WebGPU benchmark, the first paragraph appeared in about 0.5 seconds instead of 1.1, and the full page finished in about 14.6 seconds instead of 21.6. CPU translation is unchanged. (#25)

### Improvements

- The GitHub repository now has a Sponsor button and a Support section. Sakutto stays free and open source, with no ads or paid features. (#24)

## [1.0.1] - 2026-11-08

### Features

- Larger glossaries — each stored list (global or per site) now keeps up to 5,000 terms, up from 1,000. A translate run can carry the merged lists up to 10,000. Only terms that appear in the current text go into the model prompt. A user asked for a much larger glossary, and that seemed reasonable. (#23)
- Longer typed-text translation — paste up to 100,000 characters at once, up from 20,000. Long text is still split for the model. Surrounding page context stays at 20,000 characters. (#23)
- Faster GPU page translation — two paragraphs decode at once, each with its own cache, so long pages no longer slow down after paired paragraphs. The translated text is unchanged. CPU translation is untouched. (#22)

### Improvements

- Keyboard hints show Mac keys next to the Windows ones — Option+A, Option+B, and Option+S beside Alt+A/B/S, and ⌘+Enter beside Ctrl+Enter.
- Chrome Web Store listing matches the packaged locales — the name is "Sakutto - Local AI translation" (Japanese: "Sakutto - ローカルAI翻訳"). The short summary no longer repeats "Built with PLaMo"; that credit stays in the detailed description. (#17, #18)
- Clearer privacy policy for readers, in English and Japanese — translation runs on your device, there is no account, advertising, or analytics, and the only file the extension downloads is the model. (#21)

### Bug Fixes

- Text typed into a search box, message box, or form is no longer treated as page copy, including inputs wrapped in custom elements. Closed shadow roots stay out of reach. (#19, #20)
- A run of dots or asterisks in ordinary page text is no longer treated as a hidden password — only a real password field is masked that way. (#20)
- The on-page translation overlay is isolated from the host page, so a site script cannot read it, rewrite it, or turn on "always translate this site" for you.
- Model download remains the only network use — no page text, no credentials — and extension pages do not load remote scripts, styles, or fonts. The model file is kept only when its hash matches the pin. These surfaces are checked in tests. (#20)

## [1.0.0] - 2026-11-06

- No commits since the previous release.

## [0.1.1] - 2026-11-06

- Initial public source release
- Rename the public product to Sakutto.
- Format the store zip example after the package rename.
- Redraw the Sakutto mark as a folded strip.
- Ignore local video renders.
- Restore onboarding after the pinned GGUF commits moved.
- Add Chrome Web Store listing copy and review notes (#3)
- Show a selection icon that translates the highlight immediately (#5)
- Add a manual Chrome Web Store upload workflow. (#4)
- Keep long pages moving while PLaMo translates
- Keep translated text readable and translate whole pages
- Translate pages with PLaMo only, with no 簡易訳 badge or Chrome draft (#8)
- Add a durable page-translation speed bench with headline numbers (#9)
- Fix ranking-table layout collapse and pack short blocks into one request (#10)
- Make whole-page PLaMo translation continuous: one prompt per paragraph, headline first, held tok/s, WebGPU fallback detection (#11)
- Describe PLaMo license terms in onboarding and record consent honestly (#12)
- Serve bundled licences as UTF-8, cover special page prose, add typed translation, clarify licence onboarding (#13)
- Hide start controls until PLaMo can answer, and compare typed text side by side (#14)
- Drop the typed page's outer card and separate the panes with rules
- Keep the local launch-video renders out of git and the linter
