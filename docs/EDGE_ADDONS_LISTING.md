# Sakutto for Android Edge（非官方移植）— Microsoft Edge Add-ons 送審資料

Package: Edge zip from `pnpm zip:edge` (CI artifact `edge-mv3-zip`). Images: `docs/edge-addons/`。Name / short description 在 Partner Center 是唯讀，來自 zip 的 `_locales/*/messages.json`。

## 1. Extension name（來自 manifest，唯讀）

| locale                                             | `extensionName`（manifest `name`，上限 75）   |
| -------------------------------------------------- | --------------------------------------------- |
| zh_TW                                              | Sakutto for Android Edge（非官方移植）        |
| zh_CN                                              | Sakutto for Android Edge（非官方移植）        |
| en（default_locale）                               | Sakutto for Android Edge (unofficial port)    |
| ja                                                 | Sakutto for Android Edge（非公式移植版）      |
| ko / de / es / fr / id / it / pt_BR / ru / th / vi | Sakutto for Android Edge + 各語「非官方移植」 |

`short_name` = `Sakutto`（7 字，上限 12）。

## 2. Short description（manifest `description`，唯讀，上限 132）

- zh_TW：本地 LLM 可翻譯您裝置上的頁面。免費，無需帳號。
- en：A local LLM translates pages on your device. Free, no account.
- ja：ローカルLLMが、この端末の中でページを翻訳します。アカウント不要の無料拡張機能です。

## 3. Properties

- **Category：** Productivity（生產力）
- **Website：** https://github.com/JackalZheng/sakutto-android-edge
- **Support contact：** https://github.com/JackalZheng/sakutto-android-edge/issues
- **Mature content：** 不勾
- **Privacy policy URL：** https://jackalzheng.github.io/sakutto-android-edge/privacy/
  - 備援（PR 合併前／Pages 未開時）：https://github.com/JackalZheng/sakutto-android-edge/blob/main/site/privacy/index.html

## 4. Privacy 頁

**Single purpose：**
Translate web pages on the user's device with the PLaMo 3 Translate model running locally inside the extension (no translation server).

**Permission justification：**

| permission                                            | justification                                                                                                                                                                   |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| activeTab                                             | Translate the current tab only when the user taps Translate in the popup; no access to other tabs.                                                                              |
| scripting                                             | Inject the translation content script into the tab the user asked to translate (or a site the user enabled for auto-translate) to read the text and write the translation back. |
| storage                                               | Save settings, glossary, per-site rules and the license-consent record on the device.                                                                                           |
| unlimitedStorage                                      | Store the ~573 MB PLaMo 0.8B model file and the local translation cache on the device so translation works offline.                                                             |
| offscreen                                             | Run the WebGPU/WASM inference engine (wllama) in an offscreen document, because MV3 service workers cannot host it.                                                             |
| contextMenus                                          | "Translate selection / page" menu items on desktop Edge. On Edge for Android the API is absent and the code skips it.                                                           |
| optional host permissions `https://*/*`, `http://*/*` | Requested at runtime only when the user turns on auto-translate for a specific site; never granted at install.                                                                  |

**Remote code：** No, I am not using remote code.（wasm/JS 全部打包在 zip 內；從 Hugging Face 下載的只有模型權重資料檔 .gguf，不是程式碼。）

**Data usage：** 全部不勾（不收集任何使用者資料）。勾選三項 certification（不販售、不用於無關用途、不用於信用評估）。

## 5. Store listing — Description（每個 locale 必填，250–10,000 字元）

### zh_TW（主要）

Sakutto for Android Edge 是開源擴充功能 Sakutto（作者 jammaru，MIT 授權）移植到 Android 版 Microsoft Edge 的非官方版本。本版本與原作者 jammaru 無關，也不是 Preferred Networks, Inc. 的產品。Built with PLaMo.

它讓 PLaMo 3 Translate 翻譯模型直接在你的手機上執行，把網頁翻譯成你的語言，網頁內容不會傳送到任何翻譯伺服器。

主要功能
• 一鍵翻譯整個頁面：在擴充功能選單點 Sakutto，再點「翻譯」。
• 雙語對照：原文與譯文並排顯示，方便學語言或核對。
• 翻譯選取的文字，以及在專用頁面輸入或貼上文字翻譯。
• 詞彙表與各網站規則：指定專有名詞的譯法，或讓特定網站自動翻譯。
• 支援 PLaMo 3 Translate 的 53 種語言互譯，介面有 14 種語言（含繁體中文）。
• 針對觸控螢幕放大按鈕與文字。

隱私
• 翻譯在裝置上完成；沒有帳號、廣告、分析或遙測。
• 唯一的網路連線是第一次使用時從 Hugging Face 下載模型。下載後可離線翻譯。
• 安裝時不要求所有網站的權限；只有你為某網站開啟自動翻譯時才會詢問。

使用須知
• 第一次使用需同意 PLaMo Community License，並下載約 573 MB 的 PLaMo 3 Translate 0.8B 模型（建議使用 Wi-Fi）。Android 版只提供 0.8B 模型。
• 需要支援 WebGPU 或 WebAssembly 的裝置；較舊的手機翻譯速度可能較慢。
• Edge 沒有 Chrome 內建的翻譯 API，因此只使用 PLaMo 翻譯。
• 免費、開源（MIT）。原始碼與問題回報：https://github.com/JackalZheng/sakutto-android-edge
• 上游專案：https://github.com/jammaru/sakutto（請勿把本移植版的問題回報給原作者）

### zh_CN

Sakutto for Android Edge 是开源扩展 Sakutto（作者 jammaru，MIT 许可）移植到 Android 版 Microsoft Edge 的非官方版本。本版本与原作者 jammaru 无关，也不是 Preferred Networks, Inc. 的产品。Built with PLaMo.

它让 PLaMo 3 Translate 翻译模型直接在你的手机上运行，把网页翻译成你的语言，网页内容不会发送到任何翻译服务器。

主要功能
• 一键翻译整个页面：在扩展菜单点 Sakutto，再点“翻译”。
• 双语对照：原文与译文并排显示。
• 翻译选中的文字，以及在专用页面输入或粘贴文字翻译。
• 词汇表与各网站规则。
• 支持 PLaMo 3 Translate 的 53 种语言互译，界面有 14 种语言。
• 针对触摸屏放大按钮与文字。

隐私
• 翻译在设备上完成；没有账号、广告、分析或遥测。
• 唯一的网络连接是首次使用时从 Hugging Face 下载模型，之后可离线翻译。
• 安装时不要求所有网站的权限。

使用须知
• 首次使用需同意 PLaMo Community License，并下载约 573 MB 的 PLaMo 3 Translate 0.8B 模型。Android 版只提供 0.8B 模型。
• Edge 没有 Chrome 内置的翻译 API，因此只使用 PLaMo。
• 免费、开源（MIT）：https://github.com/JackalZheng/sakutto-android-edge
• 上游项目：https://github.com/jammaru/sakutto

### en（也可貼到 ko / de / es / fr / id / it / pt_BR / ru / th / vi，見下方說明）

Sakutto for Android Edge is an unofficial port of the open-source extension Sakutto (by jammaru, MIT License) to Microsoft Edge for Android. It is not affiliated with or endorsed by jammaru, and it is not a product of Preferred Networks, Inc. Built with PLaMo.

It runs the PLaMo 3 Translate model directly on your phone and translates web pages into your language. Page text is never sent to a translation server.

Features
• Translate the whole page: open Sakutto from the Extensions menu and tap Translate.
• Bilingual view: show the original and the translation side by side.
• Translate a selection, or type/paste text on a dedicated translation page.
• Glossary and per-site rules, including optional auto-translate for chosen sites.
• 53 languages in either direction (PLaMo 3 Translate); interface in 14 languages.
• Larger touch targets and text for touch screens.

Privacy
• Translation runs on your device. No account, ads, analytics or telemetry.
• The only network request is the one-time model download from Hugging Face; after that, translation works offline.
• No access to all sites at install; site access is requested only when you enable auto-translate for that site.

Good to know
• On first use you accept the PLaMo Community License and download the PLaMo 3 Translate 0.8B model (about 573 MB; Wi-Fi recommended). Only the 0.8B model is offered on Android.
• Needs a device with WebGPU or WebAssembly support; older phones may translate slowly.
• Edge has no Chrome built-in Translator API, so this port uses PLaMo only.
• Free and open source (MIT). Source and issues: https://github.com/JackalZheng/sakutto-android-edge
• Upstream project: https://github.com/jammaru/sakutto (please report problems with this port here, not upstream).

### ja

Sakutto for Android Edge は、オープンソースの拡張機能 Sakutto（作者 jammaru、MIT License）を Android 版 Microsoft Edge に移植した非公式版です。原作者 jammaru とは関係がなく、Preferred Networks, Inc. の製品でもありません。Built with PLaMo.

翻訳モデル PLaMo 3 Translate をスマートフォンの中で動かし、ウェブページを翻訳します。ページの本文を翻訳サーバーへ送ることはありません。

主な機能
• ページ全体を翻訳：拡張機能メニューから Sakutto を開き「翻訳」をタップ。
• 対訳表示：原文と訳文を並べて表示。
• 選択範囲の翻訳、専用ページでの入力・貼り付けテキストの翻訳。
• 用語集とサイトごとの規則（指定サイトの自動翻訳も可能）。
• PLaMo 3 Translate の 53 言語を双方向に翻訳。画面は 14 言語に対応。
• タッチ操作向けに大きめのボタンと文字。

プライバシー
• 翻訳は端末の中で行います。アカウント、広告、分析、テレメトリはありません。
• 通信は初回の Hugging Face からのモデル取得だけで、その後はオフラインで翻訳できます。
• インストール時にすべてのサイトへのアクセスは求めません。

ご注意
• 初回に PLaMo Community License への同意と、PLaMo 3 Translate 0.8B モデル（約 573 MB）のダウンロードが必要です。Android では 0.8B モデルのみ提供します。
• Edge には Chrome 内蔵の Translator API がないため、PLaMo だけを使います。
• 無料・オープンソース（MIT）：https://github.com/JackalZheng/sakutto-android-edge
• 上流プロジェクト：https://github.com/jammaru/sakutto（この移植版の問題は原作者に報告しないでください）

### 其他 10 個 locale（ko, de, es, fr, id, it, pt_BR, ru, th, vi）

zip 內有 14 個 `_locales`，Partner Center 會為每個語言建立一列，每列都必填 Description + Logo。做法二選一：

1. 每列貼上英文 description，並用「Duplicate this logo for all languages」複製 logo（最快）。
2. 在 Store listings 頁按該語言的 Remove（若 Partner Center 允許移除套件內的語言）。

## 6. Search terms（最多 7 個、合計 21 個字、每個 ≤ 30 字元）

- zh_TW：`翻譯`、`網頁翻譯`、`離線翻譯`、`AI 翻譯`、`PLaMo`、`日文翻譯`、`Android`
- en：`translate`、`page translator`、`offline translation`、`local AI`、`PLaMo`、`Japanese`、`privacy`
- ja：`翻訳`、`ページ翻訳`、`オフライン翻訳`、`ローカルAI`、`PLaMo`、`日本語`、`Android`

## 7. 圖片

| 資產                           | 規格                            | 檔案                                 |
| ------------------------------ | ------------------------------- | ------------------------------------ |
| Extension logo（每語言必填）   | 1:1，建議 300×300，最小 128×128 | `store-logo-300x300.png`             |
| Small promotional tile（選填） | 440×280                         | `small-promo-tile-440x280.png`       |
| Large promotional tile（選填） | 1400×560 PNG                    | `large-promo-tile-1400x560.png`      |
| Screenshots（選填，最多 6 張） | 640×480 或 1280×800             | **未製作**，需實機截圖後補成上述尺寸 |
| YouTube video（選填）          | URL                             | 無                                   |

## 8. Notes for certification（貼到 Submit 頁）

```
Sakutto for Android Edge is an unofficial port of the MIT-licensed extension Sakutto
(https://github.com/jammaru/sakutto) to Microsoft Edge, targeting Edge for Android.
Source: https://github.com/JackalZheng/sakutto-android-edge (build: pnpm build:wllama && pnpm zip:edge).

No account or login is needed.

How to test (desktop Edge works too):
1. Install, then open the Sakutto popup from the toolbar / Extensions menu.
2. First run opens an onboarding page: read and accept the PLaMo Community License.
3. Tap/click the model download. The extension downloads the PLaMo 3 Translate 0.8B
   model (GGUF, about 573 MB) from the official Hugging Face repository
   (huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF). The model is NOT bundled in the
   package; the download is data (weights), not code. Allow a few minutes.
4. Open any English or Japanese news page, open the popup, press "Translate".
   Text is replaced with the translation; "Restore original" puts the source text back.
5. "Translate typed text" in the popup opens a page where pasted text is translated.

Notes:
- On Edge for Android only the 0.8B model is offered; the 2B model is hidden and refused.
  On desktop Edge the 2B model (about 1.8 GB) may also appear.
- Edge has no Chrome Translator API, so all translation is done locally by PLaMo
  (wllama WebAssembly/WebGPU, bundled in the zip). No page text leaves the device.
- Keyboard shortcuts have no default keys (phones have no Alt key); actions are in the popup.
- contextMenus is used on desktop only; on Android the code skips it when the API is absent.
- Site access (https://*/*, http://*/*) is optional and only requested when the user
  enables auto-translate for a specific site.
- Translation speed depends on WebGPU support; on CPU-only devices it is slower but works.
```

## 9. Android 上架（不是 Partner Center 的開關）

發佈到 Edge Add-ons 不會自動出現在 Edge for Android 的擴充清單。上架通過後還要：

1. 在 Edge Android（Canary）實機驗證核心功能。
2. 準備給手機使用者的價值說明與核心功能測試案例。
3. 填 Microsoft 的申請表單：https://forms.office.com/r/eYS5Rnf0d4
   來源：https://github.com/microsoft/MicrosoftEdge-Extensions/issues/529 、https://github.com/microsoft/MicrosoftEdge-Extensions/discussions/264
