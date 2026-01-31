// ============================================================================
// MODERN RAG ASSISTANT - INTERACTIVE SCRIPT
// ============================================================================

// --- GLOBAL STATE ---
const state = {
    isTyping: false,
    currentTheme: localStorage.getItem('theme') || 'light',
    messageCount: 0
};

// --- DOM ELEMENTS ---
const elements = {
    chatWindow: document.getElementById('chat-window'),
    messagesContainer: document.querySelector('.messages-container'),
    typingIndicator: document.getElementById('typing-indicator'),
    chatForm: document.getElementById('chat-form'),
    questionInput: document.getElementById('question'),
    sendBtn: document.getElementById('send-btn'),
    charCount: document.getElementById('char-count'),
    sidebar: document.getElementById('sidebar'),
    sidebarToggle: document.getElementById('sidebar-toggle'),
    themeToggle: document.getElementById('theme-toggle'),
    fileInput: document.getElementById('fileInput'),
    attachBtn: document.getElementById('attachBtn'),
    toast: document.getElementById('toast')
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function scrollToBottom(smooth = true) {
    elements.chatWindow.scrollTo({
        top: elements.chatWindow.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto'
    });
}

function showTypingIndicator() {
    if (elements.typingIndicator) {
        elements.typingIndicator.style.display = 'flex';
        scrollToBottom();
    }
}

function hideTypingIndicator() {
    if (elements.typingIndicator) {
        elements.typingIndicator.style.display = 'none';
    }
}

function disableInput(disabled = true) {
    elements.questionInput.disabled = disabled;
    elements.sendBtn.disabled = disabled;
    state.isTyping = disabled;
}

function showToast(message, duration = 3000) {
    elements.toast.textContent = message;
    elements.toast.classList.add('show');
    
    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, duration);
}

// ============================================================================
// AUTO-RESIZE TEXTAREA
// ============================================================================
function autoResizeTextarea() {
    elements.questionInput.style.height = 'auto';
    elements.questionInput.style.height = elements.questionInput.scrollHeight + 'px';
}

elements.questionInput.addEventListener('input', () => {
    autoResizeTextarea();
    updateCharCount();
});

// ============================================================================
// CHARACTER COUNTER
// ============================================================================
function updateCharCount() {
    const length = elements.questionInput.value.length;
    const max = 500;
    elements.charCount.textContent = `${length} / ${max}`;
    
    if (length > max * 0.9) {
        elements.charCount.style.color = 'var(--error)';
    } else if (length > max * 0.75) {
        elements.charCount.style.color = 'var(--warning)';
    } else {
        elements.charCount.style.color = 'var(--text-tertiary)';
    }
}

// ============================================================================
// MESSAGE CREATION
// ============================================================================
function createMessageHTML(content, type = 'assistant') {
    const isUser = type === 'user';
    
    return `
        <div class="message-group ${isUser ? 'user-message' : 'assistant-message'}">
            ${!isUser ? `
                <div class="message-avatar">
                    <div class="avatar assistant-avatar">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                            <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                        </svg>
                    </div>
                </div>
            ` : ''}
            <div class="message-bubble">
                <div class="message-text ${!isUser ? 'markdown-content' : ''}">
                    ${content}
                </div>
                ${!isUser ? `
                    <div class="message-actions">
                        <button class="action-btn copy-btn" title="Copy">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                            </svg>
                        </button>
                        <button class="action-btn like-btn" title="Good">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>
                            </svg>
                        </button>
                        <button class="action-btn dislike-btn" title="Bad">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/>
                            </svg>
                        </button>
                    </div>
                ` : ''}
            </div>
            ${isUser ? `
                <div class="message-avatar">
                    <div class="avatar user-avatar">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                            <circle cx="12" cy="7" r="4"/>
                        </svg>
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

function appendMessage(content, type = 'assistant') {
    const messageHTML = createMessageHTML(content, type);
    
    if (!elements.messagesContainer) {
        // Create messages container if it doesn't exist
        const container = document.createElement('div');
        container.className = 'messages-container';
        elements.chatWindow.appendChild(container);
        elements.messagesContainer = container;
        
        // Remove welcome screen
        const welcomeScreen = elements.chatWindow.querySelector('.welcome-screen');
        if (welcomeScreen) {
            welcomeScreen.remove();
        }
    }
    
    elements.messagesContainer.insertAdjacentHTML('beforeend', messageHTML);
    
    // Add event listeners to new message actions
    const lastMessage = elements.messagesContainer.lastElementChild;
    attachMessageActions(lastMessage);
    
    // Render markdown for assistant messages
    if (type === 'assistant') {
        renderMarkdown(lastMessage.querySelector('.markdown-content'));
    }
    
    scrollToBottom();
    state.messageCount++;
}

// ============================================================================
// MESSAGE ACTIONS
// ============================================================================
function attachMessageActions(messageElement) {
    const copyBtn = messageElement.querySelector('.copy-btn');
    const likeBtn = messageElement.querySelector('.like-btn');
    const dislikeBtn = messageElement.querySelector('.dislike-btn');
    
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const text = messageElement.querySelector('.message-text').textContent;
            copyToClipboard(text);
        });
    }
    
    if (likeBtn) {
        likeBtn.addEventListener('click', (e) => handleFeedback(e, 'like'));
    }
    
    if (dislikeBtn) {
        dislikeBtn.addEventListener('click', (e) => handleFeedback(e, 'dislike'));
    }
}

async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('✅ Copied to clipboard!');
    } catch (err) {
        showToast('❌ Failed to copy');
    }
}

function handleFeedback(event, type) {
    const btn = event.currentTarget;
    const actions = btn.parentElement;
    const allBtns = actions.querySelectorAll('.action-btn');
    
    // Toggle active state
    if (btn.classList.contains('active')) {
        btn.classList.remove('active');
        showToast('Feedback removed');
    } else {
        allBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        showToast(`${type === 'like' ? '👍' : '👎'} Feedback recorded`);
    }
}

// ============================================================================
// MARKDOWN RENDERING
// ============================================================================
function renderMarkdown(element) {
    if (!element) return;
    
    const rawText = element.textContent;
    
    // Configure marked options
    marked.setOptions({
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
        gfm: true
    });
    
    element.innerHTML = marked.parse(rawText);
}

// ============================================================================
// STREAMING MESSAGE
// ============================================================================
async function streamMessage(text) {
    const messageHTML = createMessageHTML('', 'assistant');
    elements.messagesContainer.insertAdjacentHTML('beforeend', messageHTML);
    
    const messageElement = elements.messagesContainer.lastElementChild;
    const textElement = messageElement.querySelector('.message-text');
    
    // Stream character by character
    let index = 0;
    const speed = 15; // ms per character
    
    return new Promise((resolve) => {
        const interval = setInterval(() => {
            if (index < text.length) {
                textElement.textContent = text.slice(0, index + 1);
                index++;
                
                // Scroll every 50 characters
                if (index % 50 === 0) {
                    scrollToBottom();
                }
            } else {
                clearInterval(interval);
                
                // Render markdown
                renderMarkdown(textElement);
                attachMessageActions(messageElement);
                scrollToBottom();
                
                resolve();
            }
        }, speed);
    });
}

// ============================================================================
// FORM SUBMISSION
// ============================================================================
elements.chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const question = elements.questionInput.value.trim();
    if (!question || state.isTyping) return;
    
    // Add user message
    appendMessage(question, 'user');
    
    // Clear input
    elements.questionInput.value = '';
    elements.questionInput.style.height = 'auto';
    updateCharCount();
    
    // Show typing
    disableInput(true);
    showTypingIndicator();
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });
        
        const data = await response.json();
        
        hideTypingIndicator();
        
        if (data.error) {
            appendMessage(`⚠️ ${data.error}`, 'assistant');
            showToast('❌ ' + data.error);
            return;
        }
        
        // Extract plain text from HTML
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = data.answer;
        const plainText = tempDiv.textContent || tempDiv.innerText;
        
        // Stream the message
        await streamMessage(plainText);
        
        // Show latency if available
        if (data.latency) {
            console.log(`Response time: ${data.latency}ms`);
        }
        
    } catch (error) {
        hideTypingIndicator();
        appendMessage(`❌ Error: ${error.message}`, 'assistant');
        showToast('❌ Request failed');
    } finally {
        disableInput(false);
        elements.questionInput.focus();
    }
});

// ============================================================================
// FILE UPLOAD
// ============================================================================
elements.attachBtn.addEventListener('click', () => {
    elements.fileInput.click();
});

elements.fileInput.addEventListener('change', async () => {
    const file = elements.fileInput.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    showToast('📤 Uploading document...');
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            showToast('❌ ' + data.error);
            return;
        }
        
        showToast(`✅ Uploaded ${data.filename} • ${data.chunks} chunks`);
        
        // Add system message
        appendMessage(`📄 Document uploaded: **${data.filename}**\n${data.chunks} text chunks indexed and ready for questions.`, 'assistant');
        
    } catch (err) {
        showToast('❌ Upload failed');
    } finally {
        elements.fileInput.value = '';
    }
});

// ============================================================================
// KEYBOARD SHORTCUTS
// ============================================================================
elements.questionInput.addEventListener('keydown', (e) => {
    // Enter to send (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        elements.chatForm.dispatchEvent(new Event('submit'));
    }
    
    // Shift+Enter for new line (default behavior)
});

// ============================================================================
// SIDEBAR TOGGLE
// ============================================================================
elements.sidebarToggle.addEventListener('click', () => {
    elements.sidebar.classList.toggle('collapsed');
});

// Close sidebar on mobile when clicking outside
document.addEventListener('click', (e) => {
    if (window.innerWidth <= 768) {
        if (!elements.sidebar.contains(e.target) && !elements.sidebarToggle.contains(e.target)) {
            elements.sidebar.classList.add('collapsed');
        }
    }
});

// ============================================================================
// THEME TOGGLE
// ============================================================================
elements.themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    
    const isDark = document.body.classList.contains('dark-mode');
    state.currentTheme = isDark ? 'dark' : 'light';
    localStorage.setItem('theme', state.currentTheme);
});

// Load saved theme
if (state.currentTheme === 'dark') {
    document.body.classList.add('dark-mode');
}

// ============================================================================
// WELCOME SUGGESTIONS
// ============================================================================
document.querySelectorAll('.suggestion-card').forEach(card => {
    card.addEventListener('click', () => {
        const question = card.dataset.question;
        elements.questionInput.value = question;
        elements.questionInput.focus();
        autoResizeTextarea();
        updateCharCount();
    });
});

// ============================================================================
// NEW CHAT
// ============================================================================
document.getElementById('new-chat-btn')?.addEventListener('click', () => {
    if (confirm('Start a new chat? Current conversation will be cleared.')) {
        location.reload();
    }
});

// ============================================================================
// CLEAR HISTORY
// ============================================================================
document.getElementById('clear-history-btn')?.addEventListener('click', async () => {
    if (confirm('Clear all chat history? This cannot be undone.')) {
        try {
            const response = await fetch('/clear-cache', {
                method: 'DELETE'
            });
            
            if (response.ok) {
                showToast('✅ History cleared');
                setTimeout(() => location.reload(), 1000);
            } else {
                showToast('❌ Failed to clear history');
            }
        } catch (err) {
            showToast('❌ Request failed');
        }
    }
});

// ============================================================================
// EXPORT CHAT
// ============================================================================
document.getElementById('export-btn')?.addEventListener('click', () => {
    const messages = Array.from(document.querySelectorAll('.message-group'));
    
    const text = messages.map(msg => {
        const isUser = msg.classList.contains('user-message');
        const content = msg.querySelector('.message-text').textContent;
        return `${isUser ? 'You' : 'Assistant'}: ${content}`;
    }).join('\n\n' + '='.repeat(50) + '\n\n');
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rag-chat-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    
    showToast('✅ Chat exported!');
});

// ============================================================================
// SETTINGS
// ============================================================================
document.getElementById('settings-btn')?.addEventListener('click', async () => {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        
        showToast(`
📊 System Stats
⏱️ Startup: ${data.startup_time}
💾 Cache: ${data.cache?.size || 0} queries
💬 Messages: ${state.messageCount}
        `.trim(), 5000);
    } catch (err) {
        showToast('❌ Failed to load stats');
    }
});

// ============================================================================
// INITIALIZATION
// ============================================================================
window.addEventListener('load', () => {
    // Render existing markdown
    document.querySelectorAll('.markdown-content').forEach(el => {
        renderMarkdown(el);
    });
    
    // Attach actions to existing messages
    document.querySelectorAll('.message-group').forEach(msg => {
        attachMessageActions(msg);
    });
    
    // Focus input
    elements.questionInput.focus();
    
    // Scroll to bottom
    scrollToBottom(false);
    
    console.log('🚀 RAG Assistant initialized');
});

// Auto-scroll on resize
window.addEventListener('resize', () => {
    scrollToBottom(false);
});