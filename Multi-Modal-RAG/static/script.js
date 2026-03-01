/* ═══════════════════════════════════════════════════════════════════════════
   NeuralRAG — Frontend Controller  (fully audited & fixed)
   All HTML IDs wired, stopTypingCycle defined, latency/chunks/status live.
═══════════════════════════════════════════════════════════════════════════ */

const MAX_CHARS = 2000;
const SESSION_ID = `session-${Date.now()}`;

// ── State ─────────────────────────────────────────────────────────────────
let pendingFiles  = [];
let attachedImage = null;
let isGenerating  = false;
let currentMode   = "hybrid";

// ── DOM refs ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// Core
const sidebar        = $("sidebar");
const chatScroll     = $("chat-scroll");
const messages       = $("messages");
const question       = $("question");
const sendBtn        = $("send-btn");
const typingEl       = $("typing");
const typingStatus   = $("typing-status");   // FIX: was never grabbed
const charCount      = $("char-count");
const toast          = $("toast");

// Upload
const dropZone       = $("drop-zone");
const fileInput      = $("fileInput");
const browseLink     = $("browse-link");
const fileQueue      = $("file-queue");
const queueList      = $("queue-list");
const uploadBtn      = $("upload-btn");
const uploadBtnLabel = $("upload-btn-label"); // FIX: was never grabbed
const cancelQueueBtn = $("cancel-queue-btn"); // FIX: was never grabbed
const uploadProgress = $("upload-progress");
const progressFill   = $("progress-fill");
const progressLabel  = $("progress-label");
const progressPct    = $("progress-pct");     // FIX: was never grabbed
const perFileStatus  = $("per-file-status");  // FIX: was never grabbed

// Image
const imageInput   = $("imageInput");
const attachImgBtn = $("attach-img-btn");
const imgStrip     = $("img-preview-strip");
const imgThumb     = $("img-thumb");
const removeImgBtn = $("remove-img-btn");
const inputWrapper = $("input-wrapper");      // FIX: was never grabbed

// Namespace
const nsBadge    = $("ns-badge");             // FIX: was never grabbed
const nsLabel    = $("ns-label");
const nsDropdown = $("ns-dropdown");          // FIX: was never grabbed
const nsDropList = $("ns-dropdown-list");     // FIX: was never grabbed
const nsRefresh  = $("ns-refresh-btn");       // FIX: was never grabbed

// Status/perf chips
const statusDot    = $("status-dot");         // FIX: was never grabbed
const statusText   = $("status-text");        // FIX: was never grabbed
const statLatency  = $("stat-latency");       // FIX: was never grabbed
const latencyChip  = $("latency-chip");       // FIX: was never grabbed
const statChunks   = $("stat-chunks");        // FIX: was never grabbed
const chunksChip   = $("chunks-chip");        // FIX: was never grabbed
const perfBadge    = $("perf-badge");         // FIX: was never grabbed
const perfText     = $("perf-text");          // FIX: was never grabbed

// Retriever label
const retrieverLabel = $("retriever-label");  // FIX: was never grabbed
const modeDesc       = $("mode-desc");        // FIX: was never grabbed

// History
const historyList = $("history-list");        // FIX: was never grabbed

// ══════════════════════════════ UTILITIES ════════════════════════════════

function showToast(msg, type = "info", duration = 3200) {
    toast.textContent = msg;
    toast.className = `toast show ${type}`;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { toast.className = "toast"; }, duration);
}

function scrollBottom(smooth = true) {
    chatScroll.scrollTo({ top: chatScroll.scrollHeight, behavior: smooth ? "smooth" : "instant" });
}

function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
}

function fileIcon(filename) {
    const ext = filename.split(".").pop().toLowerCase();
    if (ext === "pdf") return "📄";
    if (ext === "md")  return "📝";
    return "📃";
}

function formatBytes(n) {
    if (n < 1024)    return n + " B";
    if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
    return (n / 1048576).toFixed(1) + " MB";
}

// ── Status helpers ─────────────────────────────────────────────────────────
function setStatusBusy(busy) {
    if (statusDot)  statusDot.className  = "status-dot-live" + (busy ? " busy" : "");
    if (statusText) statusText.textContent = busy ? "Processing…" : "Hybrid RAG Active";
}

function updateSendBtn() {
    sendBtn.disabled = !question.value.trim() && !attachedImage;
}

// ══════════════════════════════ THEME ════════════════════════════════════

const savedTheme = localStorage.getItem("theme") || "dark";
document.documentElement.dataset.theme = savedTheme;

$("theme-toggle")?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("theme", next);
});

// ══════════════════════════════ SIDEBAR ══════════════════════════════════

$("sidebar-toggle")?.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
});

// ══════════════════════════════ RETRIEVAL MODE ════════════════════════════

const modeDescs = {
    hybrid: "Dense + BM25 fused with Reciprocal Rank Fusion",
    dense:  "Vector similarity only (semantic search)",
    sparse: "BM25 keyword matching only",
};
const retrieverLabels = {
    hybrid: "Dense + BM25 RRF",
    dense:  "Dense Only",
    sparse: "BM25 Only",
};

document.querySelectorAll(".mode-pill").forEach(pill => {
    pill.addEventListener("click", () => {
        document.querySelectorAll(".mode-pill").forEach(p => p.classList.remove("active"));
        pill.classList.add("active");
        currentMode = pill.dataset.mode;
        // FIX: update mode-desc and retriever-label
        if (modeDesc)       modeDesc.textContent       = modeDescs[currentMode] || "";
        if (retrieverLabel) retrieverLabel.textContent  = retrieverLabels[currentMode] || currentMode;
        showToast(`Retrieval mode: ${currentMode}`, "success");
    });
});

// ══════════════════════════════ NEW CHAT ═════════════════════════════════

$("new-chat-btn")?.addEventListener("click", () => {
    if (!confirm("Start a new session?")) return;
    location.reload();
});

// ══════════════════════════════ CLEAR BUTTONS ════════════════════════════

$("clear-btn")?.addEventListener("click", async () => {
    if (!confirm("Clear conversation history?")) return;
    try {
        await fetch("/clear-cache", { method: "POST" });
        location.reload();
    } catch { showToast("Failed to clear cache", "error"); }
});

$("clear-cache-btn")?.addEventListener("click", async () => {
    try {
        await fetch("/clear-cache", { method: "POST" });
        showToast("Cache cleared", "success");
    } catch { showToast("Error clearing cache", "error"); }
});

// ══════════════════════════════ NAMESPACE SWITCHER ════════════════════════

let nsOpen = false;

// FIX: namespace dropdown was fully missing from JS
nsBadge?.addEventListener("click", toggleNsDropdown);

nsRefresh?.addEventListener("click", async () => {
    nsRefresh.classList.add("spin");
    await loadNamespaces();
    setTimeout(() => nsRefresh.classList.remove("spin"), 500);
});

document.addEventListener("click", e => {
    if (nsOpen && nsBadge && !nsBadge.contains(e.target) && nsDropdown && !nsDropdown.contains(e.target)) {
        closeNsDropdown();
    }
});

function toggleNsDropdown() {
    if (nsOpen) { closeNsDropdown(); return; }
    nsOpen = true;
    if (nsDropdown) nsDropdown.style.display = "block";
    loadNamespaces();
}

function closeNsDropdown() {
    nsOpen = false;
    if (nsDropdown) nsDropdown.style.display = "none";
}

async function loadNamespaces() {
    if (nsDropList) nsDropList.innerHTML = '<div class="ns-loading">Loading…</div>';
    try {
        const r = await fetch("/namespaces");
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        const nsList = d.namespaces || [];
        if (!nsDropList) return;
        if (!nsList.length) {
            nsDropList.innerHTML = '<div class="ns-loading">No namespaces yet</div>';
            return;
        }
        nsDropList.innerHTML = "";
        const currentNs = nsLabel?.textContent || "";
        nsList.forEach(n => {
            const el = document.createElement("div");
            el.className = "ns-item" + (n.name === currentNs ? " active" : "");
            el.innerHTML = `
                <span class="ns-item-name">${n.name}</span>
                <span class="ns-item-count">${n.vector_count} vecs · ${n.bm25_docs} bm25</span>
            `;
            el.addEventListener("click", () => switchNamespace(n.name));
            nsDropList.appendChild(el);
        });
    } catch (e) {
        if (nsDropList) nsDropList.innerHTML = `<div class="ns-loading" style="color:var(--red)">Error: ${e.message}</div>`;
    }
}

async function switchNamespace(ns) {
    try {
        await fetch("/switch-namespace", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ namespace: ns }),
        });
        if (nsLabel) nsLabel.textContent = ns;
        showToast(`Namespace: ${ns}`, "success");
        closeNsDropdown();
    } catch (e) {
        showToast("Switch failed: " + e.message, "error");
    }
}

// ══════════════════════════════ FILE UPLOAD ══════════════════════════════

browseLink?.addEventListener("click", e => { e.stopPropagation(); fileInput.click(); });
dropZone?.addEventListener("click", () => fileInput.click());
$("attach-doc-btn")?.addEventListener("click", () => fileInput.click());

// FIX: cancel button was never wired
cancelQueueBtn?.addEventListener("click", () => {
    pendingFiles = [];
    renderFileQueue();
});

// Drag & drop
dropZone?.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("over"); });
dropZone?.addEventListener("dragleave", () => dropZone.classList.remove("over"));
dropZone?.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("over");
    handleFileSelection([...e.dataTransfer.files]);
});

fileInput?.addEventListener("change", () => {
    handleFileSelection([...fileInput.files]);
    fileInput.value = "";
});

function handleFileSelection(files) {
    const allowed = [".pdf", ".txt", ".md"];
    const valid   = files.filter(f => allowed.some(ext => f.name.toLowerCase().endsWith(ext)));
    const invalid = files.filter(f => !allowed.some(ext => f.name.toLowerCase().endsWith(ext)));
    if (invalid.length) showToast(`Skipped ${invalid.length} unsupported file(s)`, "error");
    if (!valid.length) return;
    pendingFiles.push(...valid);
    renderFileQueue();
}

function renderFileQueue() {
    if (!pendingFiles.length) {
        fileQueue.style.display = "none";
        dropZone.style.display  = "";
        return;
    }
    dropZone.style.display  = "none";
    fileQueue.style.display = "flex";

    queueList.innerHTML = "";
    pendingFiles.forEach((f, i) => {
        const item = document.createElement("div");
        item.className = "queue-item";
        item.innerHTML = `
            <span class="file-icon">${fileIcon(f.name)}</span>
            <span class="file-name">${f.name}</span>
            <span class="file-size">${formatBytes(f.size)}</span>
            <span class="remove-file" data-index="${i}">✕</span>
        `;
        queueList.appendChild(item);
    });

    // FIX: update upload-btn-label instead of clobbering the whole button innerHTML
    if (uploadBtnLabel) {
        uploadBtnLabel.textContent = `Upload ${pendingFiles.length} File${pendingFiles.length > 1 ? "s" : ""}`;
    }

    queueList.querySelectorAll(".remove-file").forEach(btn => {
        btn.addEventListener("click", e => {
            pendingFiles.splice(parseInt(e.target.dataset.index), 1);
            renderFileQueue();
        });
    });
}

uploadBtn?.addEventListener("click", uploadFiles);

async function uploadFiles() {
    if (!pendingFiles.length) return;

    const multi = pendingFiles.length > 1;
    fileQueue.style.display    = "none";
    uploadProgress.style.display = "flex";
    progressFill.style.width   = "0%";
    if (progressPct) progressPct.textContent = "0%";
    if (progressLabel) progressLabel.textContent = multi
        ? `Uploading ${pendingFiles.length} files…`
        : `Uploading ${pendingFiles[0].name}…`;

    // FIX: render per-file status rows
    if (perFileStatus) {
        perFileStatus.innerHTML = "";
        pendingFiles.forEach((f, i) => {
            const row = document.createElement("div");
            row.className = "pf-item processing";
            row.id = `pf-row-${i}`;
            row.innerHTML = `
                <span class="pf-icon">⏳</span>
                <span class="pf-name">${f.name}</span>
                <span class="pf-chunks"></span>
            `;
            perFileStatus.appendChild(row);
        });
    }

    setStatusBusy(true);

    let fakePct = 0;
    const ticker = setInterval(() => {
        fakePct = Math.min(fakePct + 1.5, 88);
        progressFill.style.width = fakePct + "%";
        // FIX: update progress % label
        if (progressPct) progressPct.textContent = Math.round(fakePct) + "%";
    }, 100);

    try {
        let result;
        if (multi) {
            const form = new FormData();
            pendingFiles.forEach(f => form.append("files", f));
            const resp = await fetch("/upload/multiple", { method: "POST", body: form });
            result = await resp.json();
        } else {
            const form = new FormData();
            form.append("file", pendingFiles[0]);
            const resp = await fetch("/upload", { method: "POST", body: form });
            result = await resp.json();
        }

        clearInterval(ticker);
        progressFill.style.width = "100%";
        if (progressPct) progressPct.textContent = "100%";

        if (result.error) {
            showToast(`Upload error: ${result.error}`, "error");
            markAllPfRows("err", "❌");
        } else {
            // FIX: update namespace badge
            if (result.namespace && nsLabel) nsLabel.textContent = result.namespace;

            // FIX: update chunks chip
            let totalChunks = 0;
            if (multi && result.results) {
                Object.entries(result.results).forEach(([fn, info], i) => {
                    const row = $(`pf-row-${i}`);
                    if (row) {
                        if (info.status === "ok") {
                            row.className = "pf-item done";
                            row.querySelector(".pf-icon").textContent   = "✅";
                            row.querySelector(".pf-chunks").textContent = info.chunks + " chunks";
                            totalChunks += info.chunks || 0;
                        } else {
                            row.className = "pf-item err";
                            row.querySelector(".pf-icon").textContent   = "❌";
                            row.querySelector(".pf-chunks").textContent = (info.error || "").slice(0, 30);
                        }
                    }
                });
            } else {
                markAllPfRows("done", "✅", (result.chunks || 0) + " chunks");
                totalChunks = result.chunks || 0;
            }

            // FIX: show chunk count in sidebar chip
            if (statChunks) statChunks.textContent = totalChunks;
            if (chunksChip) chunksChip.style.opacity = "1";

            const ok  = multi ? Object.values(result.results || {}).filter(r => r.status === "ok") : [result];
            const err = multi ? Object.values(result.results || {}).filter(r => r.status === "error") : [];
            const msg = multi
                ? `✅ ${ok.length} file(s) indexed → ns: ${result.namespace}${err.length ? `, ${err.length} failed` : ""} (${result.time})`
                : `✅ ${result.filename} — ${result.chunks} chunks in ${result.time}`;

            if (progressLabel) progressLabel.textContent = msg.replace("✅ ", "");
            showToast(msg, "success", 5000);
            appendSystemMessage(msg);
        }

    } catch (err) {
        clearInterval(ticker);
        markAllPfRows("err", "❌");
        showToast("Upload failed: " + err.message, "error");
    } finally {
        setStatusBusy(false);
        setTimeout(() => {
            uploadProgress.style.display = "none";
            dropZone.style.display = "";
            pendingFiles = [];
        }, 3000);
    }
}

function markAllPfRows(cls, icon, info = "") {
    document.querySelectorAll(".pf-item").forEach(row => {
        row.className = "pf-item " + cls;
        const ic = row.querySelector(".pf-icon"); if (ic) ic.textContent = icon;
        const ch = row.querySelector(".pf-chunks"); if (ch) ch.textContent = info;
    });
}

// ══════════════════════════════ IMAGE ATTACH ══════════════════════════════

attachImgBtn?.addEventListener("click", () => imageInput.click());

imageInput?.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;
    imageInput.value = "";

    const reader = new FileReader();
    reader.onload = e => {
        attachedImage = { dataUrl: e.target.result, base64: e.target.result.split(",")[1] };
        imgThumb.src           = e.target.result;
        imgStrip.style.display = "flex";             // FIX: was "inline-flex", CSS expects flex
        if (inputWrapper) inputWrapper.classList.add("has-image"); // FIX: was never toggled
        updateSendBtn();
        showToast("Image attached — will send with your next message", "success");
    };
    reader.readAsDataURL(file);
});

removeImgBtn?.addEventListener("click", () => {
    attachedImage = null;
    imgStrip.style.display = "none";
    imgThumb.src = "";
    if (inputWrapper) inputWrapper.classList.remove("has-image"); // FIX: was never toggled
    updateSendBtn();
});

// ══════════════════════════════ TYPING CYCLE ══════════════════════════════

// FIX: stopTypingCycle was called 4× but never defined — caused a ReferenceError crash
const typingMessages = [
    "Generating answer…",
];
let _typingTimer = null;
let _typingIdx   = 0;

function startTypingCycle() {
    _typingIdx = 0;
    if (typingStatus) typingStatus.textContent = typingMessages[0];
    _typingTimer = setInterval(() => {
        _typingIdx = (_typingIdx + 1) % typingMessages.length;
        if (typingStatus) typingStatus.textContent = typingMessages[_typingIdx];
    }, 1800);
}

function stopTypingCycle() {
    clearInterval(_typingTimer);
    _typingTimer = null;
}

// ══════════════════════════════ CHAT ══════════════════════════════════════

// FIX: send button was always disabled; needs to enable when text is typed
question?.addEventListener("input", () => {
    autoResize(question);
    const len = question.value.length;
    charCount.textContent = `${len} / ${MAX_CHARS}`;
    charCount.style.color = len > MAX_CHARS * 0.9 ? "#F87171" : "";
    updateSendBtn(); // FIX: re-evaluate disabled state on every keystroke
});

question?.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn?.addEventListener("click", sendMessage);

// FIX: suggestion cards used wrong span selector — was picking up the icon span
document.querySelectorAll(".suggestion-card").forEach(card => {
    card.addEventListener("click", () => {
        question.value = card.dataset.q || "";
        autoResize(question);
        updateSendBtn();
        question.focus();
    });
});

async function sendMessage() {
    const text = question.value.trim();
    if ((!text && !attachedImage) || isGenerating) return;
    if (text.length > MAX_CHARS) { showToast("Message too long", "error"); return; }

    isGenerating     = true;
    sendBtn.disabled = true;
    setStatusBusy(true);

    const welcome = $("welcome");
    if (welcome) welcome.style.display = "none";

    appendUserMessage(text, attachedImage?.dataUrl);
    question.value = "";
    autoResize(question);
    charCount.textContent = `0 / ${MAX_CHARS}`;

    const imgB64  = attachedImage?.base64 || null;
    attachedImage = null;
    imgStrip.style.display = "none";
    imgThumb.src = "";
    if (inputWrapper) inputWrapper.classList.remove("has-image");
    updateSendBtn();

    typingEl.style.display = "flex";
    startTypingCycle(); // FIX: now defined above
    scrollBottom();

    const t0 = performance.now();

    try {
        const resp = await fetch("/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question:     text,
                session_id:   SESSION_ID,
                image_base64: imgB64,
                mode:         currentMode,
            }),
        });

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

        // Keep typing visible until first chunk arrives (no blank bubble flash)
        let aiBubble  = null;
        let contentEl = null;
        const reader  = resp.body.getReader();
        const dec     = new TextDecoder();
        let rawText   = "";
        let buf       = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buf += dec.decode(value, { stream: true });
            const lines = buf.split("\n");
            buf = lines.pop();

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                try {
                    const data = JSON.parse(line.slice(6));
                    if (data.chunk) {
                        // First chunk: hide typing indicator, create bubble
                        if (!aiBubble) {
                            typingEl.style.display = "none";
                            stopTypingCycle();
                            aiBubble  = appendAIMessage("");
                            contentEl = aiBubble.querySelector(".msg-content");
                        }
                        rawText += data.chunk;
                        contentEl.innerHTML = marked.parse(rawText);
                        contentEl.querySelectorAll("pre code").forEach(el => hljs.highlightElement(el));
                        scrollBottom(false);
                    }
                    if (data.done) {
                        // FIX: update namespace label from stream done event
                        if (data.namespace && nsLabel) nsLabel.textContent = data.namespace;

                        // FIX: show latency in sidebar chip and topbar perf badge
                        const ms = Math.round(performance.now() - t0);
                        if (statLatency) statLatency.textContent = ms;
                        if (latencyChip) latencyChip.style.opacity = "1";
                        if (perfText)   perfText.textContent = ms + "ms";
                        if (perfBadge)  perfBadge.style.display = "block";
                    }
                } catch { /* ignore partial JSON lines */ }
            }
        }

        // Safety: if stream ended with zero chunks (error path)
        if (!aiBubble) {
            typingEl.style.display = "none";
            stopTypingCycle();
            aiBubble = appendAIMessage("*(No response received — please try again)*");
        }

        // FIX: update message count chip
        const msgCount = $("stat-msgs");
        if (msgCount) msgCount.textContent = parseInt(msgCount.textContent || "0") + 2;

        // Wire copy button on fresh bubble
        aiBubble.querySelector(".copy-btn")?.addEventListener("click", () => {
            navigator.clipboard.writeText(rawText).then(() => showToast("Copied!", "success"));
        });

        // FIX: add question to sidebar history
        if (text) addToHistory(text);

    } catch (err) {
        typingEl.style.display = "none";
        stopTypingCycle(); // FIX: was crashing here before (undefined)
        appendAIMessage(`⚠️ Error: ${err.message}`);
        showToast("Request failed", "error");
    } finally {
        isGenerating     = false;
        sendBtn.disabled = false;
        setStatusBusy(false);
        updateSendBtn();
        scrollBottom();
    }
}

// ── Message builders ──────────────────────────────────────────────────────

function appendUserMessage(text, imageUrl = null) {
    const div = document.createElement("div");
    div.className = "msg user-msg";
    div.style.animation = "fade-up .35s cubic-bezier(.16,1,.3,1) both";

    let extra = "";
    if (imageUrl) {
        extra = `<img src="${imageUrl}" style="max-height:120px;border-radius:8px;margin-bottom:6px;display:block;">`;
    }

    div.innerHTML = `
        <div class="msg-content">${extra}${escapeHtml(text)}</div>
        <div class="avatar user-av">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                <circle cx="12" cy="7" r="4"/>
            </svg>
        </div>
    `;
    messages.appendChild(div);
    return div;
}

function appendAIMessage(html) {
    const div = document.createElement("div");
    div.className = "msg ai-msg";
    div.style.animation = "fade-up .35s cubic-bezier(.16,1,.3,1) both";
    div.innerHTML = `
        <div class="avatar ai-av">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="url(#av-dyn)" stroke-width="2">
                <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/>
                <defs><linearGradient id="av-dyn" x1="2" y1="2" x2="22" y2="22">
                    <stop stop-color="#7DF9C4"/><stop offset="1" stop-color="#6366F1"/>
                </linearGradient></defs>
            </svg>
        </div>
        <div class="msg-body">
            <div class="msg-content markdown-content">${html ? marked.parse(html) : ""}</div>
            <div class="msg-actions">
                <button class="msg-act copy-btn" title="Copy">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2"/>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                    </svg>
                </button>
                <button class="msg-act like-btn" title="Helpful">👍</button>
                <button class="msg-act dislike-btn" title="Not helpful">👎</button>
            </div>
        </div>
    `;
    messages.appendChild(div);

    // Copy handler on bubble itself
    div.querySelector(".copy-btn")?.addEventListener("click", () => {
        const t = div.querySelector(".msg-content")?.innerText || "";
        navigator.clipboard.writeText(t).then(() => showToast("Copied!", "success"));
    });

    return div;
}

function appendSystemMessage(text) {
    const div = document.createElement("div");
    div.style.cssText = "text-align:center;font-size:12px;color:var(--text-muted);padding:8px 0;animation:fade-up .3s ease both";
    div.textContent = text;
    messages.appendChild(div);
    scrollBottom();
}

function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ── FIX: sidebar history – was never populated dynamically ────────────────
function addToHistory(q) {
    if (!historyList) return;
    // Remove "no history" placeholder if present
    const empty = historyList.querySelector(".empty-hint");
    if (empty) empty.remove();

    const item = document.createElement("div");
    item.className = "history-item";
    item.dataset.q = q;
    item.innerHTML = `
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
        </svg>
        <span>${q.slice(0, 36)}${q.length > 36 ? "…" : ""}</span>
    `;
    // Click → re-fill the input
    item.addEventListener("click", () => {
        question.value = item.dataset.q;
        autoResize(question);
        updateSendBtn();
        question.focus();
    });

    historyList.insertBefore(item, historyList.firstChild);
    // Keep max 8 entries
    while (historyList.querySelectorAll(".history-item").length > 8) {
        historyList.lastElementChild?.remove();
    }
}

// ── Wire existing server-rendered copy buttons ────────────────────────────
document.querySelectorAll(".copy-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        const text = btn.closest(".msg-body")?.querySelector(".msg-content")?.innerText || "";
        navigator.clipboard.writeText(text).then(() => showToast("Copied!", "success"));
    });
});

// ── Highlight existing server-rendered code blocks ────────────────────────
document.querySelectorAll("pre code").forEach(el => hljs.highlightElement(el));

// ── Init ──────────────────────────────────────────────────────────────────
updateSendBtn();       // set correct initial disabled state
scrollBottom(false);   // scroll to bottom on page load