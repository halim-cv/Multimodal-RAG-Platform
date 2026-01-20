/* ============================================
   NotebookLM UI Clone - Application Logic
   ============================================ */

// DOM Elements
const elements = {
    // Header
    menuBtn: document.getElementById('menuBtn'),
    notebookTitle: document.getElementById('notebookTitle'),
    
    // Panels
    sourcesPanel: document.getElementById('sourcesPanel'),
    chatPanel: document.getElementById('chatPanel'),
    studioPanel: document.getElementById('studioPanel'),
    toggleSources: document.getElementById('toggleSources'),
    toggleStudio: document.getElementById('toggleStudio'),
    
    // Sources
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    sourcesList: document.getElementById('sourcesList'),
    addSourceBtn: document.getElementById('addSourceBtn'),
    
    // Chat
    chatMessages: document.getElementById('chatMessages'),
    chatInput: document.getElementById('chatInput'),
    sendBtn: document.getElementById('sendBtn'),
    
    // Studio
    generateAudio: document.getElementById('generateAudio'),
    addNote: document.getElementById('addNote'),
    notesGrid: document.getElementById('notesGrid'),
    
    // Modal
    modalOverlay: document.getElementById('modalOverlay'),
    sourceModal: document.getElementById('sourceModal'),
    modalClose: document.getElementById('modalClose'),
    
    // Toast
    toastContainer: document.getElementById('toastContainer')
};

// State
const state = {
    sources: [],
    messages: [],
    notes: [],
    activeSource: null
};

// Sample sources for demo
const sampleSources = [
    { id: 1, name: 'Research Paper.pdf', type: 'pdf', date: 'Jan 15, 2026' },
    { id: 2, name: 'Meeting Notes.txt', type: 'txt', date: 'Jan 14, 2026' },
    { id: 3, name: 'Project Plan.docx', type: 'doc', date: 'Jan 12, 2026' }
];

// ============================================
// Initialization
// ============================================
function init() {
    loadSampleData();
    bindEventListeners();
    setupTextareaAutoResize();
}

function loadSampleData() {
    state.sources = [...sampleSources];
    renderSources();
}

// ============================================
// Event Listeners
// ============================================
function bindEventListeners() {
    // Panel toggles
    elements.toggleSources?.addEventListener('click', () => togglePanel('sources'));
    elements.toggleStudio?.addEventListener('click', () => togglePanel('studio'));
    elements.menuBtn?.addEventListener('click', () => togglePanel('sources'));
    
    // Upload zone
    elements.uploadZone?.addEventListener('click', () => elements.fileInput?.click());
    elements.uploadZone?.addEventListener('dragover', handleDragOver);
    elements.uploadZone?.addEventListener('dragleave', handleDragLeave);
    elements.uploadZone?.addEventListener('drop', handleDrop);
    elements.fileInput?.addEventListener('change', handleFileSelect);
    
    // Add source button
    elements.addSourceBtn?.addEventListener('click', openModal);
    
    // Modal
    elements.modalClose?.addEventListener('click', closeModal);
    elements.modalOverlay?.addEventListener('click', (e) => {
        if (e.target === elements.modalOverlay) closeModal();
    });
    
    // Source options in modal
    document.querySelectorAll('.source-option').forEach(option => {
        option.addEventListener('click', () => handleSourceOption(option.dataset.type));
    });
    
    // Chat
    elements.chatInput?.addEventListener('input', handleInputChange);
    elements.chatInput?.addEventListener('keydown', handleKeyDown);
    elements.sendBtn?.addEventListener('click', sendMessage);
    
    // Suggested questions
    document.querySelectorAll('.suggested-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            elements.chatInput.value = btn.textContent;
            handleInputChange();
            sendMessage();
        });
    });
    
    // Studio generate buttons
    document.querySelectorAll('.generate-card').forEach(card => {
        card.addEventListener('click', () => handleGenerate(card.dataset.type));
    });
    
    // Audio generate
    elements.generateAudio?.addEventListener('click', handleGenerateAudio);
    
    // Add note
    elements.addNote?.addEventListener('click', addNewNote);
    
    // Keyboard shortcuts
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
// Source Management
// ============================================
function renderSources() {
    if (!elements.sourcesList) return;
    
    elements.sourcesList.innerHTML = state.sources.map(source => `
        <div class="source-item ${state.activeSource === source.id ? 'active' : ''}" data-id="${source.id}">
            <div class="source-icon">
                ${getFileIcon(source.type)}
            </div>
            <div class="source-info">
                <div class="source-name">${source.name}</div>
                <div class="source-meta">
                    <span>${source.type.toUpperCase()}</span>
                    <span>•</span>
                    <span>${source.date}</span>
                </div>
            </div>
            <button class="source-remove" data-id="${source.id}" title="Remove source">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </button>
        </div>
    `).join('');
    
    // Bind click events to source items
    document.querySelectorAll('.source-item').forEach(item => {
        item.addEventListener('click', (e) => {
            if (!e.target.closest('.source-remove')) {
                selectSource(parseInt(item.dataset.id));
            }
        });
    });
    
    // Bind click events to remove buttons
    document.querySelectorAll('.source-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            removeSource(parseInt(btn.dataset.id));
        });
    });
}

function getFileIcon(type) {
    const icons = {
        pdf: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
              </svg>`,
        txt: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
              </svg>`,
        doc: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
              </svg>`,
        default: `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
                    <polyline points="13 2 13 9 20 9"></polyline>
                  </svg>`
    };
    return icons[type] || icons.default;
}

function selectSource(id) {
    state.activeSource = state.activeSource === id ? null : id;
    renderSources();
}

function removeSource(id) {
    state.sources = state.sources.filter(s => s.id !== id);
    if (state.activeSource === id) {
        state.activeSource = null;
    }
    renderSources();
    showToast('Source removed', 'success');
}

function addSource(file) {
    const newSource = {
        id: Date.now(),
        name: file.name,
        type: file.name.split('.').pop().toLowerCase(),
        date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    };
    state.sources.push(newSource);
    renderSources();
    showToast(`Added "${file.name}"`, 'success');
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
    
    const files = e.dataTransfer.files;
    if (files.length) {
        Array.from(files).forEach(file => addSource(file));
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length) {
        Array.from(files).forEach(file => addSource(file));
    }
    e.target.value = '';
}

// ============================================
// Modal Management
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
    
    switch (type) {
        case 'upload':
            elements.fileInput?.click();
            break;
        case 'paste':
            showToast('Paste text feature coming soon', 'success');
            break;
        case 'website':
            showToast('Website import coming soon', 'success');
            break;
        case 'youtube':
            showToast('YouTube import coming soon', 'success');
            break;
    }
}

// ============================================
// Chat Functionality
// ============================================
function setupTextareaAutoResize() {
    const textarea = elements.chatInput;
    if (!textarea) return;
    
    textarea.addEventListener('input', () => {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    });
}

function handleInputChange() {
    const hasContent = elements.chatInput?.value.trim().length > 0;
    if (elements.sendBtn) {
        elements.sendBtn.disabled = !hasContent;
    }
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function sendMessage() {
    const content = elements.chatInput?.value.trim();
    if (!content) return;
    
    // Hide welcome message
    const welcomeMessage = elements.chatMessages?.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }
    
    // Add user message
    addMessage('user', content);
    
    // Clear input
    elements.chatInput.value = '';
    elements.chatInput.style.height = 'auto';
    handleInputChange();
    
    // Show typing indicator
    showTypingIndicator();
    
    // Simulate AI response
    setTimeout(() => {
        hideTypingIndicator();
        generateAIResponse(content);
    }, 1500 + Math.random() * 1000);
}

function addMessage(type, content, citations = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const avatar = type === 'ai' ? '✨' : 'U';
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            ${content}
            ${citations.length > 0 ? `
                <div class="message-citations">
                    ${citations.map(c => `
                        <span class="citation">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            </svg>
                            ${c}
                        </span>
                    `).join('')}
                </div>
            ` : ''}
        </div>
    `;
    
    elements.chatMessages?.appendChild(messageDiv);
    scrollToBottom();
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="message-avatar" style="background: var(--accent-gradient);">✨</div>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    elements.chatMessages?.appendChild(typingDiv);
    scrollToBottom();
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    typingIndicator?.remove();
}

function generateAIResponse(userMessage) {
    // Simulated AI responses based on user input
    const responses = [
        {
            content: `Based on your sources, I can help you understand that topic better. Here's what I found:

The key points relate to the concepts you've mentioned. Your documents contain relevant information that supports this analysis.

Would you like me to elaborate on any specific aspect?`,
            citations: state.sources.length > 0 ? [state.sources[0].name] : []
        },
        {
            content: `That's an interesting question! After analyzing your uploaded sources, I've identified several relevant themes:

1. **Main Concept**: The primary idea discussed across your documents
2. **Supporting Details**: Evidence and examples that reinforce the main points
3. **Connections**: How different sources relate to each other

Let me know if you'd like me to dive deeper into any of these areas.`,
            citations: state.sources.slice(0, 2).map(s => s.name)
        },
        {
            content: `I've reviewed your sources to answer your question. Here's a comprehensive summary:

The information suggests that the topic you're asking about is well-documented in your materials. The sources provide a strong foundation for understanding the subject matter.

Is there anything specific you'd like me to clarify?`,
            citations: state.sources.slice(0, 1).map(s => s.name)
        }
    ];
    
    const response = responses[Math.floor(Math.random() * responses.length)];
    addMessage('ai', response.content, response.citations);
}

function scrollToBottom() {
    if (elements.chatMessages) {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }
}

// ============================================
// Studio Functionality
// ============================================
function handleGenerate(type) {
    const typeLabels = {
        'study-guide': 'Study Guide',
        'briefing': 'Briefing Doc',
        'timeline': 'Timeline',
        'faq': 'FAQ'
    };
    
    showToast(`Generating ${typeLabels[type]}...`, 'success');
    
    // Simulate generation delay
    setTimeout(() => {
        addNoteWithContent(typeLabels[type], `Generated ${typeLabels[type]} from your sources. Click to view details.`);
    }, 2000);
}

function handleGenerateAudio() {
    showToast('Generating Audio Overview...', 'success');
    
    // Show generation progress
    const btn = elements.generateAudio;
    if (btn) {
        btn.textContent = 'Generating...';
        btn.disabled = true;
        
        setTimeout(() => {
            btn.textContent = 'Generate';
            btn.disabled = false;
            showToast('Audio Overview ready! (Demo)', 'success');
        }, 3000);
    }
}

function addNewNote() {
    const noteId = Date.now();
    const note = {
        id: noteId,
        title: 'New Note',
        content: 'Click to edit...',
        date: new Date().toLocaleDateString()
    };
    state.notes.push(note);
    renderNotes();
}

function addNoteWithContent(title, content) {
    const note = {
        id: Date.now(),
        title: title,
        content: content,
        date: new Date().toLocaleDateString()
    };
    state.notes.push(note);
    renderNotes();
    showToast(`${title} created!`, 'success');
}

function renderNotes() {
    if (!elements.notesGrid) return;
    
    const notesHTML = state.notes.map(note => `
        <div class="note-card" data-id="${note.id}">
            <div class="note-title">${note.title}</div>
            <div class="note-preview">${note.content}</div>
        </div>
    `).join('');
    
    elements.notesGrid.innerHTML = `
        <div class="note-card add-note" id="addNote">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="12" y1="5" x2="12" y2="19"></line>
                <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            <span>Add note</span>
        </div>
        ${notesHTML}
    `;
    
    // Rebind add note event
    document.getElementById('addNote')?.addEventListener('click', addNewNote);
}

// ============================================
// Toast Notifications
// ============================================
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'success' 
        ? `<svg class="toast-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
             <polyline points="20 6 9 17 4 12"></polyline>
           </svg>`
        : `<svg class="toast-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
             <circle cx="12" cy="12" r="10"></circle>
             <line x1="12" y1="8" x2="12" y2="12"></line>
             <line x1="12" y1="16" x2="12.01" y2="16"></line>
           </svg>`;
    
    toast.innerHTML = `
        ${icon}
        <span class="toast-message">${message}</span>
    `;
    
    elements.toastContainer?.appendChild(toast);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'toastSlide 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ============================================
// Keyboard Shortcuts
// ============================================
function handleGlobalKeydown(e) {
    // Escape to close modal
    if (e.key === 'Escape') {
        closeModal();
    }
    
    // Ctrl/Cmd + K to focus chat
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        elements.chatInput?.focus();
    }
    
    // Ctrl/Cmd + B to toggle sources panel
    if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
        e.preventDefault();
        togglePanel('sources');
    }
}

// ============================================
// Initialize App
// ============================================
document.addEventListener('DOMContentLoaded', init);
