/* ═══════════════════════════════════════════════════════════════════════
   NeuralRAG  v2  —  Frontend Controller
   Full chat history: load sessions, switch sessions, delete, rename.
═══════════════════════════════════════════════════════════════════════ */

const MAX_CHARS   = 2000;
let   SESSION_ID  = `session-${Date.now()}-${Math.random().toString(36).slice(2,8)}`;

// ── State ─────────────────────────────────────────────────────────────
let pendingFiles    = [];
let attachedImage   = null;
let isGenerating    = false;
let currentMode     = "hybrid";
let ctxTargetId     = null;   // session id for right-click context menu
let activeSessions  = {};     // id → { title, preview, updated_at }

// ── DOM ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const sidebar       = $("sidebar");
const chatArea      = $("chat-scroll");
const messages      = $("messages");
const welcomeEl     = $("welcome");
const question      = $("question");
const sendBtn       = $("send-btn");
const typingEl      = $("typing");
const charCount     = $("char-count");
const toast         = $("toast");
const sessionList   = $("session-list");
const sessionsLoading = $("sessions-loading");

// Upload
const uploadPanel   = $("upload-panel");
const dropZone      = $("drop-zone");
const fileInput     = $("fileInput");
const browseLink    = $("browse-link");
const fileQueue     = $("file-queue");
const queueList     = $("queue-list");
const uploadBtn     = $("upload-btn");
const uploadBtnLabel= $("upload-btn-label");
const cancelQueueBtn= $("cancel-queue-btn");
const uploadProgress= $("upload-progress");
const progressFill  = $("progress-fill");
const progressLabel = $("progress-label");
const progressPct   = $("progress-pct");
const perFileStatus = $("per-file-status");

// Image
const imageInput    = $("imageInput");
const attachImgBtn  = $("attach-img-btn");
const imgStrip      = $("img-preview-strip");
const imgThumb      = $("img-thumb");
const removeImgBtn  = $("remove-img-btn");

// Namespace
const nsBadge       = $("ns-badge");
const nsLabel       = $("ns-label");
const nsDropdown    = $("ns-dropdown");
const nsDropList    = $("ns-dropdown-list");

// Topbar / perf
const perfBadge     = $("perf-badge");
const perfText      = $("perf-text");
const retrieverLabel= $("retriever-label");

// Context menu + modal
const ctxMenu       = $("ctx-menu");
const ctxRename     = $("ctx-rename");
const ctxDelete     = $("ctx-delete");
const renameModal   = $("rename-modal");
const renameInput   = $("rename-input");
const renameConfirm = $("rename-confirm");
const renameCancel  = $("rename-cancel");

// ══════════════════════════════ UTILS ════════════════════════════════

function showToast(msg, type = "info", dur = 3200) {
    toast.textContent = msg;
    toast.className = `toast show ${type}`;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { toast.className = "toast"; }, dur);
}

function scrollBottom(smooth = true) {
    chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: smooth ? "smooth" : "instant" });
}

function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 150) + "px";
}

function escapeHtml(s) {
    return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function updateSendBtn() {
    sendBtn.disabled = !question.value.trim() && !attachedImage;
}

function fileIcon(name) {
    const ext = name.split(".").pop().toLowerCase();
    return ext === "pdf" ? "📄" : ext === "md" ? "📝" : "📃";
}
function formatBytes(n) {
    if (n < 1024) return n + " B";
    if (n < 1048576) return (n/1024).toFixed(1) + " KB";
    return (n/1048576).toFixed(1) + " MB";
}
function timeAgo(ts) {
    if (!ts) return "";
    const s = Math.floor(Date.now()/1000 - ts);
    if (s < 60)   return "just now";
    if (s < 3600) return Math.floor(s/60) + "m ago";
    if (s < 86400)return Math.floor(s/3600) + "h ago";
    if (s < 604800)return Math.floor(s/86400) + "d ago";
    return new Date(ts*1000).toLocaleDateString();
}
function groupDate(ts) {
    if (!ts) return "Older";
    const d = new Date(ts*1000);
    const now = new Date();
    const diff = (now - d) / 86400000;
    if (diff < 1) return "Today";
    if (diff < 2) return "Yesterday";
    if (diff < 7) return "This Week";
    if (diff < 30)return "This Month";
    return "Older";
}

// ══════════════════════════════ THEME ════════════════════════════════

const savedTheme = localStorage.getItem("theme") || "dark";
document.documentElement.dataset.theme = savedTheme;
$("theme-toggle")?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("theme", next);
});

// ══════════════════════════════ SIDEBAR ══════════════════════════════

$("sidebar-toggle")?.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
});

// ══════════════════════════════ RETRIEVAL MODE ════════════════════════

const retrieverLabels = { hybrid:"Dense + BM25 · RRF", dense:"Dense Only", sparse:"BM25 Only" };
document.querySelectorAll(".mode-pill-mini").forEach(pill => {
    pill.addEventListener("click", () => {
        document.querySelectorAll(".mode-pill-mini").forEach(p => p.classList.remove("active"));
        pill.classList.add("active");
        currentMode = pill.dataset.mode;
        if (retrieverLabel) retrieverLabel.textContent = retrieverLabels[currentMode] || currentMode;
        showToast(`Mode: ${currentMode}`, "info");
    });
});

// ══════════════════════════════ NAMESPACE ════════════════════════════

let nsOpen = false;
nsBadge?.addEventListener("click", e => { e.stopPropagation(); toggleNs(); });
document.addEventListener("click", e => {
    if (nsOpen && !nsBadge?.contains(e.target) && !nsDropdown?.contains(e.target)) closeNs();
});

function toggleNs() { nsOpen ? closeNs() : openNs(); }
function openNs() {
    nsOpen = true;
    if (nsDropdown) nsDropdown.style.display = "block";
    loadNamespaces();
}
function closeNs() {
    nsOpen = false;
    if (nsDropdown) nsDropdown.style.display = "none";
}
async function loadNamespaces() {
    if (nsDropList) nsDropList.innerHTML = '<div class="ns-loading">Loading…</div>';
    try {
        const r = await fetch("/namespaces");
        const d = await r.json();
        if (!nsDropList) return;
        const list = d.namespaces || [];
        if (!list.length) { nsDropList.innerHTML = '<div class="ns-loading">No namespaces</div>'; return; }
        nsDropList.innerHTML = "";
        list.forEach(n => {
            const el = document.createElement("div");
            el.className = "ns-item" + (n.name === (nsLabel?.textContent || "") ? " active" : "");
            el.innerHTML = `<span class="ns-item-name">${n.name}</span><span class="ns-item-count">${n.vector_count}v</span>`;
            el.addEventListener("click", () => switchNamespace(n.name));
            nsDropList.appendChild(el);
        });
    } catch { if (nsDropList) nsDropList.innerHTML = '<div class="ns-loading">Error loading</div>'; }
}
async function switchNamespace(ns) {
    await fetch("/switch-namespace", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({namespace: ns}),
    });
    if (nsLabel) nsLabel.textContent = ns;
    showToast(`Namespace: ${ns}`, "success");
    closeNs();
}

// ══════════════════════════════ CLEAR / CACHE ═════════════════════════

$("clear-btn")?.addEventListener("click", async () => {
    try { await fetch("/clear-cache", {method:"POST"}); showToast("Cache cleared","success"); }
    catch { showToast("Error","error"); }
});
$("clear-cache-btn")?.addEventListener("click", async () => {
    try { await fetch("/clear-cache", {method:"POST"}); location.reload(); }
    catch { showToast("Error","error"); }
});

// ══════════════════════════════ SESSIONS ═════════════════════════════

async function loadSessions() {
    if (sessionsLoading) sessionsLoading.style.display = "flex";
    try {
        const r = await fetch("/sessions");
        const d = await r.json();
        renderSessions(d.sessions || []);
    } catch (e) {
        if (sessionList) sessionList.innerHTML = '<div class="sessions-empty">Could not load sessions.</div>';
    } finally {
        if (sessionsLoading) sessionsLoading.style.display = "none";
    }
}

function renderSessions(sessions) {
    if (!sessionList) return;
    if (!sessions.length) {
        sessionList.innerHTML = '<div class="sessions-empty">No conversations yet.<br>Start chatting to see your history here.</div>';
        return;
    }

    // Group by date
    const groups = {};
    sessions.forEach(s => {
        const g = groupDate(s.updated_at);
        if (!groups[g]) groups[g] = [];
        groups[g].push(s);
        activeSessions[s.session_id] = s;
    });

    const order = ["Today","Yesterday","This Week","This Month","Older"];
    sessionList.innerHTML = "";

    order.forEach(gName => {
        if (!groups[gName]) return;
        const header = document.createElement("div");
        header.className = "session-date-group";
        header.textContent = gName;
        sessionList.appendChild(header);

        groups[gName].forEach(s => {
            sessionList.appendChild(buildSessionItem(s));
        });
    });
}

function buildSessionItem(s) {
    const item = document.createElement("div");
    item.className = "session-item" + (s.session_id === SESSION_ID ? " active" : "");
    item.dataset.id = s.session_id;

    item.innerHTML = `
        <div class="session-icon">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
        </div>
        <div class="session-meta">
            <div class="session-title">${escapeHtml(s.title || "New Chat")}</div>
            <div class="session-preview">${escapeHtml(s.preview || "")}</div>
        </div>
        <div class="session-actions">
            <button class="session-action-btn rename-session-btn" title="Rename">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
            </button>
            <button class="session-action-btn danger delete-session-btn" title="Delete">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            </button>
        </div>
    `;

    // Click → switch session
    item.addEventListener("click", e => {
        if (e.target.closest(".session-actions")) return;
        switchSession(s.session_id);
    });

    // Right-click → context menu
    item.addEventListener("contextmenu", e => {
        e.preventDefault();
        showCtxMenu(e.clientX, e.clientY, s.session_id);
    });

    // Button click handlers
    item.querySelector(".rename-session-btn")?.addEventListener("click", e => {
        e.stopPropagation();
        openRenameModal(s.session_id, s.title || "");
    });
    item.querySelector(".delete-session-btn")?.addEventListener("click", e => {
        e.stopPropagation();
        confirmDeleteSession(s.session_id);
    });

    return item;
}

async function switchSession(sessionId) {
    if (sessionId === SESSION_ID) return;
    SESSION_ID = sessionId;

    // Mark active
    document.querySelectorAll(".session-item").forEach(el => {
        el.classList.toggle("active", el.dataset.id === sessionId);
    });

    // Clear current chat
    messages.innerHTML = "";
    if (welcomeEl) welcomeEl.style.display = "none";

    // Load history from server
    try {
        const r = await fetch(`/sessions/${sessionId}`);
        const d = await r.json();
        const history = d.messages || [];

        if (history.length === 0) {
            if (welcomeEl) welcomeEl.style.display = "flex";
        } else {
            history.forEach(m => {
                if (m.role === "user") {
                    appendUserMessage(m.content);
                } else {
                    const bubble = appendAIMessage(m.content || "");
                    // Wire copy btn
                    bubble.querySelector(".copy-btn")?.addEventListener("click", () => {
                        navigator.clipboard.writeText(m.content || "");
                        showToast("Copied!", "success");
                    });
                }
            });
            scrollBottom(false);
        }
    } catch (e) {
        showToast("Failed to load session", "error");
    }
}

async function confirmDeleteSession(sessionId) {
    if (!confirm("Delete this conversation? This cannot be undone.")) return;
    await deleteSessionById(sessionId);
}

async function deleteSessionById(sessionId) {
    try {
        await fetch(`/sessions/${sessionId}`, {method: "DELETE"});
        showToast("Conversation deleted", "success");

        // Remove from DOM
        const item = sessionList.querySelector(`[data-id="${sessionId}"]`);
        item?.remove();
        delete activeSessions[sessionId];

        // If deleted the active session → start fresh
        if (sessionId === SESSION_ID) {
            SESSION_ID = `session-${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
            messages.innerHTML = "";
            if (welcomeEl) welcomeEl.style.display = "";
        }

        // Remove empty date groups
        document.querySelectorAll(".session-date-group").forEach(group => {
            if (!group.nextElementSibling || group.nextElementSibling.classList.contains("session-date-group")) {
                group.remove();
            }
        });
    } catch {
        showToast("Failed to delete", "error");
    }
}

function openRenameModal(sessionId, currentTitle) {
    ctxTargetId = sessionId;
    renameInput.value = currentTitle;
    if (renameModal) renameModal.style.display = "grid";
    setTimeout(() => { renameInput.focus(); renameInput.select(); }, 50);
}

renameConfirm?.addEventListener("click", async () => {
    const newTitle = renameInput.value.trim();
    if (!newTitle || !ctxTargetId) return;
    try {
        await fetch(`/sessions/${ctxTargetId}/rename`, {
            method: "POST",
            headers: {"Content-Type":"application/json"},
            body: JSON.stringify({title: newTitle}),
        });
        // Update DOM
        const item = sessionList.querySelector(`[data-id="${ctxTargetId}"]`);
        if (item) item.querySelector(".session-title").textContent = newTitle;
        if (activeSessions[ctxTargetId]) activeSessions[ctxTargetId].title = newTitle;
        showToast("Renamed", "success");
    } catch { showToast("Rename failed", "error"); }
    if (renameModal) renameModal.style.display = "none";
    ctxTargetId = null;
});

renameCancel?.addEventListener("click", () => {
    if (renameModal) renameModal.style.display = "none";
    ctxTargetId = null;
});

// Close modal on overlay click
renameModal?.addEventListener("click", e => {
    if (e.target === renameModal) {
        renameModal.style.display = "none";
        ctxTargetId = null;
    }
});

// ── Context menu ──────────────────────────────────────────────────────
function showCtxMenu(x, y, sessionId) {
    ctxTargetId = sessionId;
    ctxMenu.style.display = "block";
    ctxMenu.style.left = Math.min(x, window.innerWidth - 160) + "px";
    ctxMenu.style.top  = Math.min(y, window.innerHeight - 80) + "px";
}
function hideCtxMenu() { ctxMenu.style.display = "none"; }
document.addEventListener("click", hideCtxMenu);
document.addEventListener("keydown", e => { if (e.key === "Escape") { hideCtxMenu(); if (renameModal) renameModal.style.display = "none"; } });

ctxRename?.addEventListener("click", () => {
    if (!ctxTargetId) return;
    const s = activeSessions[ctxTargetId];
    openRenameModal(ctxTargetId, s?.title || "");
    hideCtxMenu();
});
ctxDelete?.addEventListener("click", () => {
    if (!ctxTargetId) return;
    confirmDeleteSession(ctxTargetId);
    hideCtxMenu();
});

// ── Refresh sessions ──────────────────────────────────────────────────
const refreshBtn = $("refresh-sessions");
refreshBtn?.addEventListener("click", async () => {
    refreshBtn.classList.add("spinning");
    await loadSessions();
    setTimeout(() => refreshBtn.classList.remove("spinning"), 400);
});

// ── New chat ──────────────────────────────────────────────────────────
$("new-chat-btn")?.addEventListener("click", async () => {
    // 1. Generate a new session ID and pre-create it on the server
    SESSION_ID = `session-${Date.now()}-${Math.random().toString(36).slice(2,8)}`;

    try {
        await fetch("/sessions/new", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: SESSION_ID }),
        });
    } catch { /* non-fatal — session will be created on first message anyway */ }

    // 2. Reset the chat UI
    messages.innerHTML = "";
    if (welcomeEl) welcomeEl.style.display = "";
    question.value = "";
    autoResize(question);
    updateSendBtn();

    // 3. Deselect all session items in sidebar
    document.querySelectorAll(".session-item").forEach(el => el.classList.remove("active"));
    question.focus();
});

// ══════════════════════════════ UPLOAD ═══════════════════════════════

$("upload-trigger-btn")?.addEventListener("click", () => {
    if (uploadPanel) uploadPanel.style.display = uploadPanel.style.display === "none" ? "block" : "none";
});
$("close-upload-panel")?.addEventListener("click", () => {
    if (uploadPanel) uploadPanel.style.display = "none";
});

browseLink?.addEventListener("click", e => { e.stopPropagation(); fileInput.click(); });
dropZone?.addEventListener("click", () => fileInput.click());

cancelQueueBtn?.addEventListener("click", () => { pendingFiles = []; renderFileQueue(); });

dropZone?.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("over"); });
dropZone?.addEventListener("dragleave", () => dropZone.classList.remove("over"));
dropZone?.addEventListener("drop", e => {
    e.preventDefault(); dropZone.classList.remove("over");
    handleFileSelection([...e.dataTransfer.files]);
});
fileInput?.addEventListener("change", () => {
    handleFileSelection([...fileInput.files]); fileInput.value = "";
});

function handleFileSelection(files) {
    const allowed = [".pdf",".txt",".md"];
    const valid   = files.filter(f => allowed.some(ext => f.name.toLowerCase().endsWith(ext)));
    const invalid = files.filter(f => !allowed.some(ext => f.name.toLowerCase().endsWith(ext)));
    if (invalid.length) showToast(`Skipped ${invalid.length} unsupported file(s)`, "error");
    if (!valid.length) return;
    pendingFiles.push(...valid);
    renderFileQueue();
}

function renderFileQueue() {
    if (!pendingFiles.length) {
        fileQueue.style.display = "none"; dropZone.style.display = "";
        return;
    }
    dropZone.style.display = "none"; fileQueue.style.display = "flex";
    queueList.innerHTML = "";
    pendingFiles.forEach((f, i) => {
        const item = document.createElement("div");
        item.className = "queue-item";
        item.innerHTML = `
            <span>${fileIcon(f.name)}</span>
            <span class="file-name">${f.name}</span>
            <span class="file-size">${formatBytes(f.size)}</span>
            <span class="remove-file" data-i="${i}">✕</span>
        `;
        queueList.appendChild(item);
    });
    if (uploadBtnLabel) uploadBtnLabel.textContent = `Upload ${pendingFiles.length} File${pendingFiles.length>1?"s":""}`;
    queueList.querySelectorAll(".remove-file").forEach(btn => {
        btn.addEventListener("click", e => { pendingFiles.splice(+e.target.dataset.i,1); renderFileQueue(); });
    });
}

uploadBtn?.addEventListener("click", uploadFiles);
async function uploadFiles() {
    if (!pendingFiles.length) return;
    const multi = pendingFiles.length > 1;
    fileQueue.style.display = "none";
    uploadProgress.style.display = "flex";
    progressFill.style.width = "0%";
    if (progressPct) progressPct.textContent = "0%";
    if (progressLabel) progressLabel.textContent = multi ? `Uploading ${pendingFiles.length} files…` : `Uploading ${pendingFiles[0].name}…`;
    if (perFileStatus) {
        perFileStatus.innerHTML = "";
        pendingFiles.forEach((f,i) => {
            const row = document.createElement("div");
            row.className = "pf-item processing"; row.id = `pf-row-${i}`;
            row.innerHTML = `<span class="pf-icon">⏳</span><span class="pf-name">${f.name}</span><span class="pf-chunks"></span>`;
            perFileStatus.appendChild(row);
        });
    }

    let pct = 0;
    const ticker = setInterval(() => {
        pct = Math.min(pct+1.5, 88);
        progressFill.style.width = pct+"%";
        if (progressPct) progressPct.textContent = Math.round(pct)+"%";
    }, 100);

    try {
        let result;
        if (multi) {
            const form = new FormData();
            pendingFiles.forEach(f => form.append("files", f));
            result = await (await fetch("/upload/multiple",{method:"POST",body:form})).json();
        } else {
            const form = new FormData();
            form.append("file", pendingFiles[0]);
            result = await (await fetch("/upload",{method:"POST",body:form})).json();
        }
        clearInterval(ticker);
        progressFill.style.width = "100%";
        if (progressPct) progressPct.textContent = "100%";

        if (result.error) {
            showToast(`Upload error: ${result.error}`, "error");
        } else {
            if (result.namespace && nsLabel) nsLabel.textContent = result.namespace;
            if (multi && result.results) {
                Object.entries(result.results).forEach(([fn,info],i) => {
                    const row = $(`pf-row-${i}`);
                    if (!row) return;
                    if (info.status === "ok") {
                        row.className="pf-item done";
                        row.querySelector(".pf-icon").textContent="✅";
                        row.querySelector(".pf-chunks").textContent=info.chunks+" chunks";
                    } else {
                        row.className="pf-item err";
                        row.querySelector(".pf-icon").textContent="❌";
                    }
                });
                const ok = Object.values(result.results).filter(r=>r.status==="ok");
                showToast(`✅ ${ok.length} file(s) indexed → ${result.namespace} (${result.time})`, "success", 5000);
            } else {
                document.querySelectorAll(".pf-item").forEach(row => {
                    row.className="pf-item done";
                    row.querySelector(".pf-icon").textContent="✅";
                    row.querySelector(".pf-chunks").textContent=(result.chunks||0)+" chunks";
                });
                showToast(`✅ ${result.filename} — ${result.chunks} chunks (${result.time})`, "success", 5000);
            }
            if (progressLabel) progressLabel.textContent = "Upload complete";
        }
    } catch (err) {
        clearInterval(ticker);
        showToast("Upload failed: "+err.message, "error");
    } finally {
        setTimeout(() => {
            uploadProgress.style.display = "none";
            dropZone.style.display = "";
            pendingFiles = [];
        }, 3000);
    }
}

// ══════════════════════════════ IMAGE ATTACH ══════════════════════════

attachImgBtn?.addEventListener("click", () => imageInput.click());
imageInput?.addEventListener("change", () => {
    const file = imageInput.files[0]; if (!file) return; imageInput.value = "";
    const r = new FileReader();
    r.onload = e => {
        attachedImage = {dataUrl: e.target.result, base64: e.target.result.split(",")[1]};
        imgThumb.src = e.target.result;
        imgStrip.style.display = "flex";
        updateSendBtn();
        showToast("Image attached", "success");
    };
    r.readAsDataURL(file);
});
removeImgBtn?.addEventListener("click", () => {
    attachedImage = null; imgStrip.style.display = "none"; imgThumb.src = "";
    updateSendBtn();
});

// ══════════════════════════════ SIDEBAR LIVE UPDATE ══════════════════
/**
 * Immediately insert or update a session item in the sidebar without
 * doing a full reload. Called as soon as we know the title/preview
 * (i.e. right when the stream's `done` event fires).
 */
function upsertSessionInSidebar(sessionId, title, preview) {
    activeSessions[sessionId] = { session_id: sessionId, title, preview, updated_at: Date.now()/1000 };

    // Remove old entry if present
    const existing = sessionList?.querySelector(`[data-id="${sessionId}"]`);
    existing?.closest(".session-date-group + .session-item") // nothing — just remove the item
    existing?.remove();

    // Also remove a stale "Today" group header if it became orphaned
    // (handled by full reload below — don't over-engineer here)

    if (!sessionList) return;

    // Build the new item
    const item = buildSessionItem(activeSessions[sessionId]);
    item.classList.add("active");

    // Ensure a "Today" group header exists at the top
    let todayGroup = sessionList.querySelector(".session-date-group");
    if (!todayGroup || todayGroup.textContent !== "Today") {
        const header = document.createElement("div");
        header.className = "session-date-group";
        header.textContent = "Today";
        sessionList.insertBefore(header, sessionList.firstChild);
        todayGroup = header;
    }

    // Insert right after the "Today" header
    todayGroup.insertAdjacentElement("afterend", item);

    // Deselect everything else
    sessionList.querySelectorAll(".session-item").forEach(el => {
        el.classList.toggle("active", el.dataset.id === sessionId);
    });

    // Remove the empty-state placeholder if present
    sessionList.querySelector(".sessions-empty")?.remove();
}



const typingMessages = ["Retrieving context…","Thinking…","Generating answer…"];
let _typingTimer = null, _typingIdx = 0;

function startTypingCycle() {
    _typingIdx = 0;
    const statusEl = $("typing-status");
    if (statusEl) statusEl.textContent = typingMessages[0];
    _typingTimer = setInterval(() => {
        _typingIdx = (_typingIdx+1) % typingMessages.length;
        const s = $("typing-status"); if (s) s.textContent = typingMessages[_typingIdx];
    }, 1800);
}
function stopTypingCycle() { clearInterval(_typingTimer); _typingTimer = null; }

// ══════════════════════════════ CHAT ══════════════════════════════════

question?.addEventListener("input", () => {
    autoResize(question);
    const len = question.value.length;
    charCount.textContent = `${len} / ${MAX_CHARS}`;
    charCount.style.color = len > MAX_CHARS*.9 ? "var(--red)" : "";
    updateSendBtn();
});
question?.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
sendBtn?.addEventListener("click", sendMessage);

document.querySelectorAll(".suggestion-card").forEach(card => {
    card.addEventListener("click", () => {
        question.value = card.dataset.q || "";
        autoResize(question); updateSendBtn(); question.focus();
    });
});

async function sendMessage() {
    const text = question.value.trim();
    if ((!text && !attachedImage) || isGenerating) return;
    if (text.length > MAX_CHARS) { showToast("Message too long","error"); return; }

    isGenerating = true; sendBtn.disabled = true;
    if (welcomeEl) welcomeEl.style.display = "none";

    // Capture the question text BEFORE clearing the input (used for session title)
    const questionText = text;

    appendUserMessage(text, attachedImage?.dataUrl);
    question.value = ""; autoResize(question); charCount.textContent = `0 / ${MAX_CHARS}`;

    const imgB64 = attachedImage?.base64 || null;
    attachedImage = null; imgStrip.style.display = "none"; imgThumb.src = "";
    updateSendBtn();

    typingEl.style.display = "flex";
    startTypingCycle();
    scrollBottom();

    const t0 = performance.now();

    try {
        const resp = await fetch("/chat/stream", {
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body: JSON.stringify({question:text, session_id:SESSION_ID, image_base64:imgB64, mode:currentMode}),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

        let aiBubble=null, contentEl=null, rawText="", buf="";
        const reader = resp.body.getReader();
        const dec = new TextDecoder();

        while (true) {
            const {done,value} = await reader.read();
            if (done) break;
            buf += dec.decode(value, {stream:true});
            const lines = buf.split("\n"); buf = lines.pop();
            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                try {
                    const data = JSON.parse(line.slice(6));
                    if (data.chunk) {
                        if (!aiBubble) {
                            typingEl.style.display = "none"; stopTypingCycle();
                            aiBubble = appendAIMessage("");
                            contentEl = aiBubble.querySelector(".msg-content");
                        }
                        rawText += data.chunk;
                        contentEl.innerHTML = marked.parse(rawText);
                        contentEl.querySelectorAll("pre code").forEach(el => hljs.highlightElement(el));
                        scrollBottom(false);
                    }
                    if (data.done) {
                        if (data.namespace && nsLabel) nsLabel.textContent = data.namespace;
                        const ms = Math.round(performance.now()-t0);
                        if (perfText) perfText.textContent = ms+"ms";
                        if (perfBadge) perfBadge.style.opacity = "1";

                        // ── Immediately update sidebar with the new/updated session ──
                        // Title = first 45 chars of the user question
                        const sessionTitle = questionText.length > 45
                            ? questionText.slice(0, 45) + "…"
                            : questionText;
                        // Preview = first 80 chars of AI response
                        const sessionPreview = rawText.slice(0, 80);
                        upsertSessionInSidebar(SESSION_ID, sessionTitle, sessionPreview);

                        // Background reload to sync any ordering/grouping changes
                        setTimeout(loadSessions, 1200);
                    }
                } catch { /* skip */ }
            }
        }

        if (!aiBubble) {
            typingEl.style.display = "none"; stopTypingCycle();
            aiBubble = appendAIMessage("*(No response received — please try again)*");
            // Still upsert the session so it appears in the list
            upsertSessionInSidebar(SESSION_ID, questionText.slice(0,45), "");
            setTimeout(loadSessions, 1200);
        }

        aiBubble.querySelector(".copy-btn")?.addEventListener("click", () => {
            navigator.clipboard.writeText(rawText).then(() => showToast("Copied!","success"));
        });

    } catch (err) {
        typingEl.style.display = "none"; stopTypingCycle();
        appendAIMessage(`⚠️ Error: ${err.message}`);
        showToast("Request failed","error");
    } finally {
        isGenerating = false; sendBtn.disabled = false; updateSendBtn(); scrollBottom();
    }
}

// ── Message builders ──────────────────────────────────────────────────

function appendUserMessage(text, imageUrl = null) {
    const div = document.createElement("div");
    div.className = "msg user-msg";
    let extra = imageUrl ? `<img src="${imageUrl}" style="max-height:110px;border-radius:8px;margin-bottom:5px;display:block;">` : "";
    div.innerHTML = `
        <div class="msg-content">${extra}${escapeHtml(text)}</div>
        <div class="avatar user-avatar">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
        </div>`;
    messages.appendChild(div);
    return div;
}

function appendAIMessage(content) {
    const div = document.createElement("div");
    div.className = "msg ai-msg";
    const html = content ? marked.parse(content) : "";
    div.innerHTML = `
        <div class="avatar ai-avatar">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="url(#avgi)" stroke-width="2">
                <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/>
                <defs><linearGradient id="avgi" x1="2" y1="2" x2="22" y2="22">
                    <stop stop-color="#60A5FA"/><stop offset="1" stop-color="#A78BFA"/>
                </linearGradient></defs>
            </svg>
        </div>
        <div class="msg-body">
            <div class="msg-content markdown-content">${html}</div>
            <div class="msg-actions">
                <button class="msg-act copy-btn" title="Copy">
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                    Copy
                </button>
                <button class="msg-act like-btn" title="Helpful">👍</button>
                <button class="msg-act dislike-btn" title="Not helpful">👎</button>
            </div>
        </div>`;
    if (content) {
        div.querySelectorAll("pre code").forEach(el => hljs.highlightElement(el));
    }
    messages.appendChild(div);
    div.querySelector(".copy-btn")?.addEventListener("click", () => {
        const t = div.querySelector(".msg-content")?.innerText || "";
        navigator.clipboard.writeText(t).then(() => showToast("Copied!","success"));
    });
    return div;
}

// ══════════════════════════════ INIT ══════════════════════════════════

updateSendBtn();
scrollBottom(false);

// Load sessions on start
loadSessions();

// Wire existing server-rendered code blocks
document.querySelectorAll("pre code").forEach(el => hljs.highlightElement(el));