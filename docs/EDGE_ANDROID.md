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

| Area                | Desktop Chrome                                                           | Edge for Android (this branch)                                                                                               |
| ------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| Target              | `wxt build`                                                              | `wxt build -b edge` / `pnpm build:edge`                                                                                      |
| Built-in translator | Chrome Translator API for 11 extra languages                             | **Not available on Edge** — PLaMo only, like Firefox (`hasBuiltInTranslatorUi()` is false on Edge)                           |
| Keyboard shortcuts  | Alt+A / Alt+B / Alt+S / Alt+T                                            | **No `suggested_key` in the manifest** — a phone has no Alt key. Actions live in the popup and the in-page UI                |
| Right-click menu    | `contextMenus`                                                           | Declared, but guarded: `installPageActions()` skips it silently when the API is absent on Android                            |
| Model               | 0.8B (573 MB) or 2B (1.8 GB)                                             | **0.8B only.** `ModelManager` hides 2B on Android Edge, and the background refuses 2B downloads / `useModel` calls there too |
| Touch UI            | Desktop-sized controls                                                   | `base.css`: 44px minimum targets on `(pointer: coarse)`, 14px body text                                                      |
| Permissions         | offscreen, storage, unlimitedStorage, activeTab, scripting, contextMenus | Same set — Microsoft lists `offscreen` as supported on Android                                                               |
| Package check       | `verify-package.ts` chrome/firefox                                       | Now accepts `--browser edge` (same permission set as Chrome)                                                                 |

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
# commands: four actions, no suggested_key
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
- No 2B model on mobile, no Chrome-only languages on Edge.
