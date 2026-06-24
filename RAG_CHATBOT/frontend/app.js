document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const toggleUploadBtn = document.getElementById('toggle-upload-btn');
    const closeDrawerBtn = document.getElementById('close-drawer-btn');
    const uploadDrawer = document.getElementById('upload-drawer');
    const drawerOverlay = document.getElementById('drawer-overlay');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const statusMessage = document.getElementById('status-message');

    // UI State: Toggle Upload Drawer
    const openDrawer = () => {
        uploadDrawer.classList.add('open');
        drawerOverlay.classList.add('open');
    };

    const closeDrawer = () => {
        uploadDrawer.classList.remove('open');
        drawerOverlay.classList.remove('open');
    };

    toggleUploadBtn.addEventListener('click', openDrawer);
    closeDrawerBtn.addEventListener('click', closeDrawer);
    drawerOverlay.addEventListener('click', closeDrawer);

    // Chat: Submit Form
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = chatInput.value.trim();
        if (!question) return;

        // Clear input
        chatInput.value = '';

        // Add user message
        appendMessage(question, 'user');

        // Add loading typing indicator
        const typingEl = appendTypingIndicator();
        scrollToBottom();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });

            // Remove typing indicator
            typingEl.remove();

            if (!response.ok) {
                const err = await response.json();
                appendMessage(`Error: ${err.detail || 'Could not reach server'}`, 'bot error-message');
                return;
            }

            const data = await response.json();
            appendMessage(data.answer, 'bot', data.sources);
        } catch (error) {
            typingEl.remove();
            appendMessage(`Error: Failed to connect to server`, 'bot error-message');
        }

        scrollToBottom();
    });

    // Helper: Append Chat Message
    function appendMessage(text, sender, sources = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Simple paragraph text
        const textPara = document.createElement('p');
        textPara.textContent = text;
        contentDiv.appendChild(textPara);

        // Append sources if available
        if (sources && sources.length > 0) {
            const sourcesContainer = document.createElement('div');
            sourcesContainer.className = 'sources-container';

            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'sources-toggle';
            toggleBtn.innerHTML = `
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="transform: rotate(90deg); transition: transform 0.2s;"><polyline points="9 18 15 12 9 6"></polyline></svg>
                <span>Show Sources (${sources.length})</span>
            `;

            const listDiv = document.createElement('div');
            listDiv.className = 'sources-list';
            listDiv.style.display = 'none';

            sources.forEach(src => {
                const item = document.createElement('div');
                item.className = 'source-item';
                item.textContent = src.text;

                const meta = document.createElement('span');
                meta.className = 'source-meta';
                meta.textContent = `Page ${src.page_number} (Relevance Score: ${(1 - src.distance).toFixed(2)})`;
                
                item.appendChild(meta);
                listDiv.appendChild(item);
            });

            toggleBtn.addEventListener('click', () => {
                const isOpen = listDiv.style.display === 'flex';
                listDiv.style.display = isOpen ? 'none' : 'flex';
                const svg = toggleBtn.querySelector('svg');
                svg.style.transform = isOpen ? 'rotate(90deg)' : 'rotate(270deg)';
                toggleBtn.querySelector('span').textContent = isOpen ? `Show Sources (${sources.length})` : 'Hide Sources';
            });

            sourcesContainer.appendChild(toggleBtn);
            sourcesContainer.appendChild(listDiv);
            contentDiv.appendChild(sourcesContainer);
        }

        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    // Helper: Append Typing Indicator
    function appendTypingIndicator() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message typing-msg';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';

        contentDiv.appendChild(indicator);
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        return messageDiv;
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // File Dropzone Actions
    dropzone.addEventListener('click', () => fileInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // Upload & Index File via API
    async function handleFileUpload(file) {
        if (file.type !== 'application/pdf') {
            alert('Please select a valid PDF file.');
            return;
        }

        // Show status loading state
        uploadStatus.style.display = 'flex';
        const spinner = uploadStatus.querySelector('.status-spinner');
        spinner.style.display = 'block';
        statusMessage.textContent = `Uploading and indexing '${file.name}'...`;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Upload failed');
            }

            // Success state
            spinner.style.display = 'none';
            statusMessage.innerHTML = `<span style="color: var(--success-color)">✓</span> Indexed successfully! Added ${data.chunks_added} chunks.`;
            
            // Add a clean message to main chat
            appendMessage(`System: Loaded and indexed document '${file.name}' successfully (${data.chunks_added} chunks added).`, 'system');

            // Reset status after a delay
            setTimeout(() => {
                uploadStatus.style.display = 'none';
            }, 5000);

        } catch (error) {
            spinner.style.display = 'none';
            statusMessage.innerHTML = `<span style="color: var(--error-color)">✗</span> Error: ${error.message}`;
            setTimeout(() => {
                uploadStatus.style.display = 'none';
            }, 5000);
        }
    }
});
