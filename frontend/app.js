/* ============================================
   Multimodal RAG Platform — Frontend Application
   Wired to FastAPI backend at localhost:8000
   ============================================ */

const API_BASE = 'http://localhost:8000';

// ============================================
// DOM Elements
// ============================================
const elements = {
    menuBtn: document.getElementById('menuBtn'),
    notebookTitle: document.getElementById('notebookTitle'),
    sourcesPanel: document.getElementById('sourcesPanel'),
    chatPanel: document.getElementById('chatPanel'),
    studioPanel: document.getElementById('studioPanel'),
    toggleSources: document.getElementById('toggleSources'),
    toggleStudio: document.getElementById('toggleStudio'),
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    sourcesList: document.getElementById('sourcesList'),
    addSourceBtn: document.getElementById('addSourceBtn'),
    chatMessages: document.getElementById('chatMessages'),
    chatInput: document.getElementById('chatInput'),
    sendBtn: document.getElementById('sendBtn'),
    welcomeMessage: document.getElementById('welcomeMessage'),
    citationCards: document.getElementById('citationCards'),
    sessionSelect: document.getElementById('sessionSelect'),
    newSessionBtn: document.getElementById('newSessionBtn'),
    healthIndicator: document.getElementById('healthIndicator'),
    generateAudio: document.getElementById('generateAudio'),
    addNote: document.getElementById('addNote'),
    notesGrid: document.getElementById('notesGrid'),
    modalOverlay: document.getElementById('modalOverlay'),
    modalClose: document.getElementById('modalClose'),
    toastContainer: document.getElementById('toastContainer'),
};

// ============================================
// Application State
// ============================================
const state = {
    currentSessionId: null,
    sessions: [],
    sources: [],
    notes: [],
    activeModalities: ['text', 'image', 'audio'],
    isStreaming: false,
    pollIntervals: {},  // job_id -> interval
};

// ============================================
// Initialization
// ============================================
async function init() {
    bindEventListeners();
    setupTextareaAutoResize();
    await checkHealth();
    await loadSessions();

    // Restore session from localStorage
    const saved = localStorage.getItem('currentSessionId');
    if (saved && state.sessions.find(s => s.session_id === saved)) {
        state.currentSessionId = saved;
        elements.sessionSelect.value = saved;
        await loadSessionSources();
    }
}

// ============================================
// Health Check
// ============================================
async function checkHealth() {
    const dot = elements.healthIndicator?.querySelector('.health-dot');
    const text = elements.healthIndicator?.querySelector('.health-text');
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        if (dot) dot.classList.add('connected');
        if (text) text.textContent = 'Backend online';
        elements.healthIndicator?.setAttribute('title',
            `Models: E5=${data.models_loaded?.e5_small_v2 ? '✅' : '❌'} ` +
            `Florence=${data.models_loaded?.florence_2 ? '✅' : '❌'} ` +
            `Whisper=${data.models_loaded?.whisper_tiny ? '✅' : '❌'}`
        );
    } catch {
        if (dot) dot.classList.remove('connected');
        if (text) text.textContent = 'Backend offline';
    }
}

// ============================================
// Sessions API
// ============================================
async function loadSessions() {
    try {
        const res = await fetch(`${API_BASE}/api/sessions`);
        const data = await res.json();
        state.sessions = data.sessions || [];
        renderSessionSelect();
    } catch (e) {
        console.error('Failed to load sessions:', e);
    }
}

function renderSessionSelect() {
    if (!elements.sessionSelect) return;
    elements.sessionSelect.innerHTML = '<option value="">— Select session —</option>';
    state.sessions.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s.session_id;
        opt.textContent = s.session_id.replace('session_', '').replace(/_/g, ' ') +
            ` (${s.source_count} sources)`;
        elements.sessionSelect.appendChild(opt);
    });
    if (state.currentSessionId) {
        elements.sessionSelect.value = state.currentSessionId;
    }
}

async function createSession() {
    try {
        const res = await fetch(`${API_BASE}/api/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: null }),
        });
        const session = await res.json();
        state.currentSessionId = session.session_id;
        localStorage.setItem('currentSessionId', session.session_id);
        await loadSessions();
        elements.sessionSelect.value = session.session_id;
        state.sources = [];
        renderSources();
        showToast(`Session created: ${session.session_id}`, 'success');
    } catch (e) {
        showToast('Failed to create session', 'error');
    }
}

async function switchSession(sessionId) {
    if (!sessionId) {
        state.currentSessionId = null;
        state.sources = [];
        renderSources();
        return;
    }
    state.currentSessionId = sessionId;
    localStorage.setItem('currentSessionId', sessionId);
    await loadSessionSources();
}

async function loadSessionSources() {
    if (!state.currentSessionId) return;
    try {
        const res = await fetch(`${API_BASE}/api/sessions/${state.currentSessionId}`);
        const data = await res.json();
        state.sources = (data.sources || []).map((s, i) => ({
            id: i + 1,
            name: s.file_name,
            modality: s.modality,
            embedded: s.embedded,
            job_status: s.job_status || 'done',
        }));
        renderSources();
    } catch (e) {
        console.error('Failed to load session:', e);
    }
}

// ============================================
// Event Listeners
// ============================================
function bindEventListeners() {
    elements.toggleSources?.addEventListener('click', () => togglePanel('sources'));
    elements.toggleStudio?.addEventListener('click', () => togglePanel('studio'));
    elements.menuBtn?.addEventListener('click', () => togglePanel('sources'));

    elements.uploadZone?.addEventListener('click', () => elements.fileInput?.click());
    elements.uploadZone?.addEventListener('dragover', handleDragOver);
    elements.uploadZone?.addEventListener('dragleave', handleDragLeave);
    elements.uploadZone?.addEventListener('drop', handleDrop);
    elements.fileInput?.addEventListener('change', handleFileSelect);

    elements.addSourceBtn?.addEventListener('click', openModal);
    elements.modalClose?.addEventListener('click', closeModal);
    elements.modalOverlay?.addEventListener('click', (e) => {
        if (e.target === elements.modalOverlay) closeModal();
    });

    document.querySelectorAll('.source-option').forEach(opt => {
        opt.addEventListener('click', () => handleSourceOption(opt.dataset.type));
    });

    elements.chatInput?.addEventListener('input', handleInputChange);
    elements.chatInput?.addEventListener('keydown', handleKeyDown);
    elements.sendBtn?.addEventListener('click', sendMessage);

    document.querySelectorAll('.suggested-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            elements.chatInput.value = btn.textContent;
            handleInputChange();
            sendMessage();
        });
    });

    // Session controls
    elements.sessionSelect?.addEventListener('change', (e) => switchSession(e.target.value));
    elements.newSessionBtn?.addEventListener('click', createSession);

    // Studio
    document.querySelectorAll('.generate-card').forEach(card => {
        card.addEventListener('click', () => handleGenerate(card.dataset.type));
    });
    elements.generateAudio?.addEventListener('click', handleGenerateAudio);
    elements.addNote?.addEventListener('click', addNewNote);

    document.addEventListener('keydown', handleGlobalKeydown);
}

// ============================================
// Panel Management
// ============================================
function togglePanel(panel) {
    if (panel === 'sources') {
        elements.sourcesPanel?.classList.toggle('collapsed');
        elements.sourcesPanel?.classList.toggle('open');
    } else if (panel === 'studio') {
        elements.studioPanel?.classList.toggle('collapsed');
    }
}

// ============================================
// File Upload → Backend (with upload progress via XHR)
// ============================================
async function uploadFileToBackend(file) {
    if (!state.currentSessionId) {
        await createSession();
    }

    // Immediately add source to UI in 'uploading' state
    const sourceId = Date.now();
    const guessedModality = guessModality(file.name);
    const newSource = {
        id: sourceId,
        name: file.name,
        modality: guessedModality,
        embedded: false,
        job_status: 'uploading',
        uploadPercent: 0,
        job_id: null,
        previewUrl: null,   // will be set for images
    };

    // For images: read as data URL for instant preview
    if (guessedModality === 'image') {
        newSource.previewUrl = await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = () => resolve(null);
            reader.readAsDataURL(file);
        });
    }

    state.sources.push(newSource);
    renderSources();

    const formData = new FormData();
    formData.append('session_id', state.currentSessionId);
    formData.append('file', file);

    try {
        // Use XHR for upload progress
        const data = await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', `${API_BASE}/api/upload`);

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const pct = Math.round((e.loaded / e.total) * 100);
                    newSource.uploadPercent = pct;
                    newSource.progress = `Uploading… ${pct}%`;
                    renderSources();
                }
            };

            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    try {
                        const err = JSON.parse(xhr.responseText);
                        reject(new Error(err.detail || `Upload failed (${xhr.status})`));
                    } catch {
                        reject(new Error(`Upload failed (${xhr.status})`));
                    }
                }
            };

            xhr.onerror = () => reject(new Error('Network error during upload'));
            xhr.send(formData);
        });

        // Upload succeeded — switch to processing state
        showToast(`Uploaded "${data.file_name}" — processing…`, 'success');
        newSource.name = data.file_name;
        newSource.modality = data.modality;
        newSource.job_status = data.status || 'queued';
        newSource.job_id = data.job_id;
        newSource.uploadPercent = 100;
        newSource.progress = 'Processing…';
        renderSources();

        // Start polling for backend processing status
        pollJobStatus(data.job_id, sourceId);

    } catch (e) {
        newSource.job_status = 'error';
        newSource.progress = e.message;
        renderSources();
        showToast(`Upload error: ${e.message}`, 'error');
    }
}

function guessModality(fileName) {
    const ext = fileName.split('.').pop()?.toLowerCase() || '';
    if (['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'].includes(ext)) return 'image';
    if (['mp3', 'wav', 'm4a', 'ogg', 'flac'].includes(ext)) return 'audio';
    return 'text';
}

function pollJobStatus(jobId, sourceId) {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/upload/status/${jobId}`);
            const data = await res.json();

            const source = state.sources.find(s => s.id === sourceId);
            if (source) {
                source.job_status = data.status;
                source.progress = data.progress;
                if (data.status === 'done') {
                    source.embedded = true;
                    clearInterval(interval);
                    showToast(`"${source.name}" processed ✅`, 'success');
                } else if (data.status === 'error') {
                    clearInterval(interval);
                    showToast(`"${source.name}" failed: ${data.error}`, 'error');
                }
                renderSources();
            }
        } catch {
            // Silently retry
        }
    }, 2000);
    state.pollIntervals[jobId] = interval;
}

// ============================================
// Source Rendering (with processing states)
// ============================================
const MODALITY_ICONS = {
    text: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
             <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
             <polyline points="14 2 14 8 20 8"></polyline>
             <line x1="16" y1="13" x2="8" y2="13"></line>
             <line x1="16" y1="17" x2="8" y2="17"></line>
           </svg>`,
    image: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <circle cx="8.5" cy="8.5" r="1.5"></circle>
              <polyline points="21 15 16 10 5 21"></polyline>
            </svg>`,
    audio: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 18V5l12-2v13"></path>
              <circle cx="6" cy="18" r="3"></circle>
              <circle cx="18" cy="16" r="3"></circle>
            </svg>`,
};

function getModalityLabel(mod) {
    return { text: '📄', image: '🖼️', audio: '🎵' }[mod] || '📄';
}

function getModalityName(mod) {
    return { text: 'doc', image: 'image', audio: 'audio' }[mod] || 'file';
}

// Update the natural status bar above chat
function updateSourceStatusBar() {
    const bar = document.getElementById('sourceStatusBar');
    if (!bar) return;
    const ready = state.sources.filter(s => s.job_status === 'done');
    if (ready.length === 0) { bar.style.display = 'none'; return; }

    const counts = {};
    ready.forEach(s => { counts[s.modality] = (counts[s.modality] || 0) + 1; });
    const parts = Object.entries(counts).map(([mod, n]) =>
        `${getModalityLabel(mod)} ${n} ${getModalityName(mod)}${n > 1 ? 's' : ''}`
    );
    bar.style.display = 'flex';
    bar.textContent = parts.join('  ·  ');
}
function renderSources() {
    if (!elements.sourcesList) return;

    if (state.sources.length === 0) {
        elements.sourcesList.innerHTML = '';
        updateSourceStatusBar();
        return;
    }

    elements.sourcesList.innerHTML = state.sources.map(source => {
        const isUploading = source.job_status === 'uploading';
        const isProcessing = source.job_status === 'queued' || source.job_status === 'processing';
        const isError = source.job_status === 'error';
        const statusClass = (isUploading || isProcessing) ? 'processing' : isError ? 'error' : 'done';
        const pct = source.uploadPercent || 0;
        const hasPreview = source.modality === 'image' && source.previewUrl;

        return `
        <div class="source-item ${statusClass}" data-id="${source.id}" id="src-${source.id}">
            ${hasPreview ? `
                <div class="source-image-preview">
                    <img src="${source.previewUrl}" alt="${source.name}" class="source-thumb" />
                    ${(isUploading || isProcessing) ? `<div class="source-thumb-overlay"><span class="spinner"></span></div>` : ''}
                    ${isError ? `<div class="source-thumb-overlay error-overlay">⚠️</div>` : ''}
                </div>
            ` : `
                <div class="source-icon ${source.modality}">
                    ${MODALITY_ICONS[source.modality] || MODALITY_ICONS.text}
                </div>
            `}
            <div class="source-info">
                <div class="source-name">${source.name}</div>
                ${isUploading ? `
                    <div class="upload-progress-bar">
                        <div class="upload-progress-fill" style="width:${pct}%"></div>
                    </div>
                    <div class="source-meta">
                        <span class="source-progress">
                            <span class="spinner"></span>
                            Uploading… ${pct}%
                        </span>
                    </div>
                ` : `
                    <div class="source-meta">
                        ${isProcessing ? `
                            <span class="source-progress">
                                <span class="spinner"></span>
                                ${source.progress || 'Processing…'}
                            </span>
                        ` : isError ? `
                            <span class="source-error">⚠️ Failed</span>
                        ` : `
                            <span class="source-done">✅ Ready</span>
                        `}
                    </div>
                `}
            </div>
        </div>
        `;
    }).join('');

    updateSourceStatusBar();
}

// ============================================
// Drag & Drop
// ============================================
function handleDragOver(e) {
    e.preventDefault();
    elements.uploadZone?.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadZone?.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadZone?.classList.remove('dragover');
    Array.from(e.dataTransfer.files).forEach(f => uploadFileToBackend(f));
}

function handleFileSelect(e) {
    Array.from(e.target.files).forEach(f => uploadFileToBackend(f));
    e.target.value = '';
}

// ============================================
// Modal
// ============================================
function openModal() {
    elements.modalOverlay?.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    elements.modalOverlay?.classList.remove('active');
    document.body.style.overflow = '';
}

function handleSourceOption(type) {
    closeModal();
    if (type === 'upload') {
        elements.fileInput?.click();
    } else {
        showToast(`${type} import coming soon`, 'success');
    }
}

// ============================================
// Chat — SSE Streaming to Backend
// ============================================
function setupTextareaAutoResize() {
    const ta = elements.chatInput;
    if (!ta) return;
    ta.addEventListener('input', () => {
        ta.style.height = 'auto';
        ta.style.height = Math.min(ta.scrollHeight, 150) + 'px';
    });
}

function handleInputChange() {
    const has = elements.chatInput?.value.trim().length > 0;
    if (elements.sendBtn) elements.sendBtn.disabled = !has || state.isStreaming;
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

async function sendMessage() {
    const content = elements.chatInput?.value.trim();
    if (!content || state.isStreaming) return;

    if (!state.currentSessionId) {
        showToast('Please select or create a session first', 'error');
        return;
    }

    // Hide welcome
    if (elements.welcomeMessage) {
        elements.welcomeMessage.style.display = 'none';
    }

    // Add user message
    addUserMessage(content);

    // Clear input
    elements.chatInput.value = '';
    elements.chatInput.style.height = 'auto';
    handleInputChange();

    // Stream AI response
    await sendMessageToAPI(content);
}

function addUserMessage(content) {
    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML = `
        <div class="message-avatar">U</div>
        <div class="message-content">${escapeHtml(content)}</div>
    `;
    elements.chatMessages?.appendChild(div);
    scrollToBottom();
}

function createEmptyAIMessage() {
    const div = document.createElement('div');
    div.className = 'message ai';
    div.innerHTML = `
        <div class="message-avatar" style="background: var(--accent-gradient);">✨</div>
        <div class="message-content ai-streaming">
            <div class="ai-text"></div>
        </div>
    `;
    elements.chatMessages?.appendChild(div);
    scrollToBottom();
    return div.querySelector('.ai-text');
}

// ============================================
// Core: sendMessageToAPI — SSE streaming
// ============================================
async function sendMessageToAPI(query) {
    state.isStreaming = true;
    if (elements.sendBtn) elements.sendBtn.disabled = true;

    // Show typing indicator
    showTypingIndicator();

    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: state.currentSessionId,
                query,
                top_k: 5,
                modalities: state.activeModalities,
            }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(err.detail || 'Chat request failed');
        }

        hideTypingIndicator();

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        const aiTextEl = createEmptyAIMessage();
        let fullText = '';
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep incomplete line in buffer

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const event = JSON.parse(line.slice(6));

                    if (event.type === 'sources') {
                        renderCitationCards(event.chunks);
                    }
                    if (event.type === 'token') {
                        fullText += event.content;
                        appendStreamToken(aiTextEl, fullText);
                    }
                    if (event.type === 'error') {
                        fullText += `\n\n⚠️ Error: ${event.message}`;
                        appendStreamToken(aiTextEl, fullText);
                    }
                    if (event.type === 'done') {
                        // Final render with full markdown
                        renderFinalMarkdown(aiTextEl, fullText);
                    }
                } catch (parseErr) {
                    // Skip malformed SSE lines
                }
            }
        }

        // Ensure final render even if 'done' event was missed
        if (fullText) {
            renderFinalMarkdown(aiTextEl, fullText);
        }

    } catch (err) {
        hideTypingIndicator();
        showToast(`Chat error: ${err.message}`, 'error');
        // Show error message in chat
        const errDiv = createEmptyAIMessage();
        errDiv.innerHTML = `<span style="color:#ef4444;">⚠️ ${escapeHtml(err.message)}</span>`;
    } finally {
        state.isStreaming = false;
        handleInputChange();
    }
}

function appendStreamToken(el, fullText) {
    // Render incrementally — use marked for partial markdown
    if (typeof marked !== 'undefined') {
        el.innerHTML = marked.parse(fullText);
    } else {
        el.textContent = fullText;
    }
    scrollToBottom();
}

function renderFinalMarkdown(el, fullText) {
    if (typeof marked !== 'undefined') {
        el.innerHTML = marked.parse(fullText);
    } else {
        el.textContent = fullText;
    }
    el.closest('.ai-streaming')?.classList.remove('ai-streaming');
    scrollToBottom();
}

// ============================================
// Citation Cards — rendered from SSE sources event
// ============================================

// Global registry: citation index → source file name (for click-to-highlight)
let _citationRegistry = [];

function scrollToSource(idx) {
    const entry = _citationRegistry[idx];
    if (!entry) return;
    // Find by file name in state.sources
    const source = state.sources.find(s => s.name === entry.source);
    const domId = source ? `src-${source.id}` : null;
    const el = domId ? document.getElementById(domId) : null;
    if (el) {
        // Open sources panel if collapsed
        const panel = document.getElementById('sourcesPanel');
        panel?.classList.remove('collapsed');
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        el.classList.add('citation-highlight');
        setTimeout(() => el.classList.remove('citation-highlight'), 1800);
    }
    // Also scroll the citation card into view
    const card = document.querySelector(`.citation-card[data-index="${idx}"]`);
    card?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderCitationCards(chunks) {
    if (!elements.citationCards || !chunks || chunks.length === 0) return;

    // Build registry for click handlers
    _citationRegistry = chunks.map(c => ({ source: c.source || 'unknown', modality: c.modality || 'text' }));

    elements.citationCards.style.display = 'block';
    elements.citationCards.innerHTML = `
        <div class="citations-header">
            <span class="citations-label">Sources Used</span>
            <button class="citations-toggle" onclick="this.parentElement.parentElement.classList.toggle('collapsed')">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
            </button>
        </div>
        <div class="citations-grid">
            ${chunks.map((chunk, i) => {
        const mod = chunk.modality || 'text';
        const icon = getModalityLabel(mod);
        const score = chunk.score || 0;
        const scorePercent = Math.round(score * 100);
        const excerpt = (chunk.text || '').slice(0, 150);
        const source = chunk.source || 'unknown';
        const tsHint = (mod === 'audio' && chunk.timestamp)
            ? ` · ${chunk.timestamp[0]?.toFixed(1)}s–${chunk.timestamp[1]?.toFixed(1)}s`
            : '';

        return `
                <div class="citation-card" data-index="${i}" onclick="scrollToSource(${i})" title="Click to highlight source">
                    <div class="citation-card-header">
                        <span class="citation-index-badge">${i + 1}</span>
                        <span class="citation-source" title="${escapeHtml(source)}">${escapeHtml(source)}${escapeHtml(tsHint)}</span>
                        <span class="citation-modality-icon">${icon}</span>
                    </div>
                    <div class="citation-score-bar">
                        <div class="citation-score-fill" style="width:${scorePercent}%"></div>
                        <span class="citation-score-label">${scorePercent}% match</span>
                    </div>
                    <div class="citation-excerpt">${escapeHtml(excerpt)}${excerpt.length >= 150 ? '…' : ''}</div>
                </div>
                `;
    }).join('')}
        </div>
    `;

    // Inject inline numbered superscripts — no emoji labels, just clean [1] [2] links
    const citationsInChat = document.createElement('div');
    citationsInChat.className = 'message-citations';
    citationsInChat.innerHTML = chunks.map((c, i) =>
        `<button class="inline-citation" onclick="scrollToSource(${i})" title="${escapeHtml(c.source || '')}">[${i + 1}] ${escapeHtml(c.source || 'unknown')}</button>`
    ).join('');

    // Append citations after the latest AI message
    const lastAi = elements.chatMessages?.querySelector('.message.ai:last-child .message-content');
    if (lastAi && !lastAi.querySelector('.message-citations')) {
        lastAi.appendChild(citationsInChat);
    }

    scrollToBottom();
}

// ============================================
// Typing Indicator
// ============================================
function showTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'typing-indicator';
    div.id = 'typingIndicator';
    div.innerHTML = `
        <div class="message-avatar" style="background: var(--accent-gradient);">✨</div>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    elements.chatMessages?.appendChild(div);
    scrollToBottom();
}

function hideTypingIndicator() {
    document.getElementById('typingIndicator')?.remove();
}

function scrollToBottom() {
    if (elements.chatMessages) {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }
}

// ============================================
// Studio (kept simple)
// ============================================
function handleGenerate(type) {
    const labels = { 'study-guide': 'Study Guide', briefing: 'Briefing Doc', timeline: 'Timeline', faq: 'FAQ' };
    showToast(`Generating ${labels[type]}...`, 'success');
    setTimeout(() => addNoteWithContent(labels[type], `Generated ${labels[type]} from your sources.`), 2000);
}

function handleGenerateAudio() {
    showToast('Generating Audio Overview...', 'success');
    const btn = elements.generateAudio;
    if (btn) {
        btn.textContent = 'Generating...';
        btn.disabled = true;
        setTimeout(() => { btn.textContent = 'Generate'; btn.disabled = false; }, 3000);
    }
}

function addNewNote() {
    state.notes.push({ id: Date.now(), title: 'New Note', content: 'Click to edit...', date: new Date().toLocaleDateString() });
    renderNotes();
}

function addNoteWithContent(title, content) {
    state.notes.push({ id: Date.now(), title, content, date: new Date().toLocaleDateString() });
    renderNotes();
    showToast(`${title} created!`, 'success');
}

function renderNotes() {
    if (!elements.notesGrid) return;
    const html = state.notes.map(n => `
        <div class="note-card" data-id="${n.id}">
            <div class="note-title">${n.title}</div>
            <div class="note-preview">${n.content}</div>
        </div>
    `).join('');
    elements.notesGrid.innerHTML = `
        <div class="note-card add-note" id="addNote">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line>
            </svg><span>Add note</span>
        </div>
        ${html}
    `;
    document.getElementById('addNote')?.addEventListener('click', addNewNote);
}

// ============================================
// Toast Notifications
// ============================================
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icon = type === 'success'
        ? `<svg class="toast-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>`
        : `<svg class="toast-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
    toast.innerHTML = `${icon}<span class="toast-message">${message}</span>`;
    elements.toastContainer?.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'toastSlide 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ============================================
// Utilities
// ============================================
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function handleGlobalKeydown(e) {
    if (e.key === 'Escape') closeModal();
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') { e.preventDefault(); elements.chatInput?.focus(); }
    if ((e.ctrlKey || e.metaKey) && e.key === 'b') { e.preventDefault(); togglePanel('sources'); }
}

// ============================================
// Initialize
// ============================================
document.addEventListener('DOMContentLoaded', init);
