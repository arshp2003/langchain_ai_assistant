const API_BASE = 'http://127.0.0.1:8000';
const REQUEST_TIMEOUT_MS = 90_000;

let currentThreadId = null;
let isWaitingForApproval = false;
let attachedDocuments = [];

function fetchWithTimeout(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(timer));
}

const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const hitlMode = document.getElementById('hitl-mode');
const userIdInput = document.getElementById('user-id');
const approvalPanel = document.getElementById('approval-panel');
const approvalContent = document.getElementById('approval-content');
const approvalStatus = document.getElementById('approval-status');
const feedbackInput = document.getElementById('feedback-input');
const approveBtn = document.getElementById('approve-btn');
const rejectBtn = document.getElementById('reject-btn');
const attachedFilesSection = document.getElementById('attached-files');
const attachedChips = document.getElementById('attached-chips');

function addMessage(content, type = 'assistant') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Parse markdown-style links [text](url) and bare URLs
    let htmlContent = content;
    
    // First, handle markdown links: [text](url)
    const markdownLinkRegex = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g;
    htmlContent = htmlContent.replace(markdownLinkRegex, (match, text, url) => {
        return `<a href="${url}" target="_blank" rel="noopener noreferrer">${text}</a>`;
    });
    
    // Then, handle bare URLs that aren't already inside links
    const bareUrlRegex = /(?<!["'>])(https?:\/\/[^\s<]+)/g;
    htmlContent = htmlContent.replace(bareUrlRegex, (url) => {
        // Clean up trailing punctuation that might not be part of the URL
        const cleanUrl = url.replace(/[.,;:!?)]+$/, '');
        const trailing = url.slice(cleanUrl.length);
        return `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer">${cleanUrl}</a>${trailing}`;
    });
    
    // Convert newlines to <br> for proper formatting
    htmlContent = htmlContent.replace(/\n/g, '<br>');
    
    contentDiv.innerHTML = htmlContent;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

function addMessageWithFiles(content, files, type = 'user') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Add file attachments at the top with distinct styling
    if (files && files.length > 0) {
        const filesDiv = document.createElement('div');
        filesDiv.className = 'message-files';
        files.forEach(filename => {
            const fileSpan = document.createElement('span');
            fileSpan.className = 'message-file-chip';
            fileSpan.innerHTML = `<span class="file-icon">📎</span><span class="file-name">${filename}</span>`;
            filesDiv.appendChild(fileSpan);
        });
        contentDiv.appendChild(filesDiv);
    }
    
    // Add message text below files
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    
    let htmlContent = content;
    
    // Parse markdown-style links [text](url) and bare URLs
    const markdownLinkRegex = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g;
    htmlContent = htmlContent.replace(markdownLinkRegex, (match, text, url) => {
        return `<a href="${url}" target="_blank" rel="noopener noreferrer">${text}</a>`;
    });
    
    const bareUrlRegex = /(?<!["'>])(https?:\/\/[^\s<]+)/g;
    htmlContent = htmlContent.replace(bareUrlRegex, (url) => {
        const cleanUrl = url.replace(/[.,;:!?)]+$/, '');
        const trailing = url.slice(cleanUrl.length);
        return `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer">${cleanUrl}</a>${trailing}`;
    });
    
    htmlContent = htmlContent.replace(/\n/g, '<br>');
    textDiv.innerHTML = htmlContent;
    
    contentDiv.appendChild(textDiv);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

function addLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant loading';
    messageDiv.id = 'loading-message';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<div class="loading"></div><span>Thinking...</span>';
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeLoadingMessage() {
    const loadingMsg = document.getElementById('loading-message');
    if (loadingMsg) {
        loadingMsg.remove();
    }
}

function showApprovalPanel(pendingApproval) {
    approvalPanel.classList.remove('hidden');
    approvalStatus.textContent = 'Awaiting your decision';
    feedbackInput.value = '';
    
    let content = '';
    
    if (pendingApproval.pending_node === 'tools') {
        content = '<strong>Action:</strong> Execute tool(s)<br><br>';
        content += '<strong>Tool calls:</strong><br>';
        pendingApproval.tool_calls.forEach(tc => {
            content += `  - ${tc.name}(${JSON.stringify(tc.args)})<br>`;
        });
    } else if (pendingApproval.pending_node === 'extract_memory') {
        content = '<strong>Action:</strong> Extract and save memories<br><br>';
        content += pendingApproval.message || 'The agent will analyze the conversation and save personal info/preferences to long-term memory.';
    } else {
        content = `<strong>Pending node:</strong> ${pendingApproval.pending_node}<br>`;
        content += `<strong>Action:</strong> ${pendingApproval.action}`;
    }
    
    approvalContent.innerHTML = content;
}

function hideApprovalPanel() {
    approvalPanel.classList.add('hidden');
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message || isWaitingForApproval) return;
    
    const userId = userIdInput.value.trim() || 'user-123';
    
    // Capture attached files before clearing
    const filesToSend = [...attachedDocuments];
    
    // Clear attached files immediately
    attachedDocuments = [];
    renderAttachedChips();
    
    // Add message with file attachments shown separately
    addMessageWithFiles(message, filesToSend, 'user');
    userInput.value = '';
    sendBtn.disabled = true;
    addLoadingMessage();
    
    try {
        // Always use HITL - it auto-detects based on tool priority
        await startHITLConversation(message, userId);
    } catch (error) {
        removeLoadingMessage();
        if (error.name === 'AbortError') {
            addMessage('Request timed out (90s). The model may be overloaded — please try again.', 'system');
        } else {
            addMessage(`Error: ${error.message}`, 'system');
        }
        console.error('Error:', error);
    } finally {
        sendBtn.disabled = false;
    }
}

async function startNormalConversation(message, userId) {
    const response = await fetchWithTimeout(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question: message,
            user_id: userId,
            attached_documents: attachedDocuments.length > 0 ? attachedDocuments : null,
            thread_id: currentThreadId
        })
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();
    removeLoadingMessage();
    addMessage(data.answer, 'assistant');
}

async function startHITLConversation(message, userId) {
    const response = await fetchWithTimeout(`${API_BASE}/chat/hitl`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question: message,
            user_id: userId,
            attached_documents: attachedDocuments.length > 0 ? attachedDocuments : null,
            thread_id: currentThreadId
        })
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();
    currentThreadId = data.thread_id;

    removeLoadingMessage();

    if (data.answer) {
        addMessage(data.answer, 'assistant');
    }

    if (data.pending_approval && data.pending_approval.status === 'awaiting_approval') {
        isWaitingForApproval = true;
        showApprovalPanel(data.pending_approval);
    }
}

async function handleApproval(approved) {
    if (!currentThreadId || !isWaitingForApproval) return;
    
    const feedback = feedbackInput.value.trim();
    
    approveBtn.disabled = true;
    rejectBtn.disabled = true;
    approvalStatus.textContent = 'Processing...';
    
    try {
        const response = await fetchWithTimeout(`${API_BASE}/approval/${currentThreadId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ approved, feedback })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        
        const data = await response.json();
        
        if (approved) {
            addMessage(`[Approved] ${feedback ? feedback : 'Action approved'}`, 'approval');
        } else {
            addMessage(`[Rejected] ${feedback || 'Action rejected'}`, 'rejection');
        }
        
        if (data.answer) {
            addMessage(data.answer, 'assistant');
        }
        
        if (data.still_pending) {
            const pendingResponse = await fetch(`${API_BASE}/approval/${currentThreadId}`);
            const pendingData = await pendingResponse.json();
            
            if (pendingData.status === 'awaiting_approval') {
                showApprovalPanel(pendingData);
            } else {
                hideApprovalPanel();
                isWaitingForApproval = false;
            }
        } else {
            hideApprovalPanel();
            isWaitingForApproval = false;
            // Keep currentThreadId: it is what makes the next question a turn in
            // this conversation instead of a brand-new thread with no history.
        }
        
    } catch (error) {
        addMessage(`Error: ${error.message}`, 'system');
        console.error('Approval error:', error);
        hideApprovalPanel();
        isWaitingForApproval = false;
    } finally {
        approveBtn.disabled = false;
        rejectBtn.disabled = false;
    }
}

sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

approveBtn.addEventListener('click', () => handleApproval(true));
rejectBtn.addEventListener('click', () => handleApproval(false));

userIdInput.addEventListener('change', () => {
    if (isWaitingForApproval) {
        addMessage('Note: User ID changed. Current HITL session will continue with the original user.', 'system');
    }
});

// -----------------------------
// Document Attachment
// -----------------------------

const fileInput = document.getElementById('file-input');
const attachBtn = document.getElementById('attach-btn');

attachBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    // Show file as attachment preview above input (like ChatGPT)
    attachDocument(file.name);

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetchWithTimeout(`${API_BASE}/documents/upload`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.status === 'uploaded') {
            // File uploaded successfully - it's now attached and ready to send with next message
            console.log(`File "${file.name}" uploaded and indexed (${data.chunks_created} chunks)`);
            if (data.message) addMessage(data.message, 'system');
        } else {
            addMessage(`Upload error: ${data.error}`, 'system');
            detachDocument(file.name);
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            addMessage('Upload timed out. Please try again.', 'system');
        } else {
            addMessage(`Upload failed: ${error.message}`, 'system');
        }
        detachDocument(file.name);
    }

    fileInput.value = '';
});

function attachDocument(filename) {
    if (!attachedDocuments.includes(filename)) {
        attachedDocuments.push(filename);
        renderAttachedChips();
    }
}

function detachDocument(filename) {
    attachedDocuments = attachedDocuments.filter(f => f !== filename);
    renderAttachedChips();
}

function renderAttachedChips() {
    if (attachedDocuments.length === 0) {
        attachedFilesSection.classList.add('hidden');
        attachedChips.innerHTML = '';
        return;
    }

    attachedFilesSection.classList.remove('hidden');
    attachedChips.innerHTML = attachedDocuments.map(filename => `
        <span class="attached-chip">
            <span class="chip-icon">📎</span>
            <span class="chip-name">${filename}</span>
            <span class="chip-remove" onclick="detachDocument('${filename}')">×</span>
        </span>
    `).join('');
}

console.log('Chat interface initialized');
console.log(`API Base: ${API_BASE}`);
