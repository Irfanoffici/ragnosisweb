// Ragnosis AI - Enhanced Frontend JavaScript with Session Management
class RagnosisAI {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.initializeApp();
        this.conversationHistory = [];
        this.sessionId = this.generateSessionId();
        this.isProcessing = false;
    }

    initializeElements() {
        // Core elements
        this.chatMessages = document.getElementById('chat-messages');
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.typingIndicator = document.getElementById('typing-indicator');
        this.clearChatButton = document.getElementById('clear-chat');
        
        // API configuration
        this.API_BASE = window.location.origin;
        
        console.log('🤖 Ragnosis AI Frontend initialized');
    }

    bindEvents() {
        // Send button click
        this.sendButton.addEventListener('click', () => this.handleSendMessage());
        
        // Enter key press
        this.messageInput.addEventListener('keydown', (e) => this.handleKeydown(e));
        
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => this.handleInputResize());
        
        // Enable/disable send button
        this.messageInput.addEventListener('input', () => this.updateSendButtonState());
        
        // Clear chat button
        if (this.clearChatButton) {
            this.clearChatButton.addEventListener('click', () => this.clearChat());
        }
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    initializeApp() {
        this.updateSendButtonState();
        this.messageInput.focus();
        this.addWelcomeMessage();
        
        // Load any existing session from localStorage
        this.loadSession();
        
        console.log('🚀 Ragnosis AI Ready! Session:', this.sessionId);
    }

    loadSession() {
        const savedSession = localStorage.getItem('ragnosis_session');
        if (savedSession) {
            try {
                const sessionData = JSON.parse(savedSession);
                this.sessionId = sessionData.sessionId;
                this.conversationHistory = sessionData.conversationHistory || [];
                
                // Restore chat messages
                if (this.conversationHistory.length > 0) {
                    this.restoreChatHistory();
                }
                
                console.log('💾 Session restored:', this.sessionId);
            } catch (e) {
                console.error('❌ Error loading session:', e);
                this.clearSession();
            }
        }
    }

    saveSession() {
        const sessionData = {
            sessionId: this.sessionId,
            conversationHistory: this.conversationHistory,
            timestamp: Date.now()
        };
        localStorage.setItem('ragnosis_session', JSON.stringify(sessionData));
    }

    clearSession() {
        localStorage.removeItem('ragnosis_session');
        this.sessionId = this.generateSessionId();
        this.conversationHistory = [];
        console.log('🗑️ Session cleared, new session:', this.sessionId);
    }

    restoreChatHistory() {
        this.chatMessages.innerHTML = '';
        
        // Add welcome message
        this.addWelcomeMessage();
        
        // Restore conversation history
        this.conversationHistory.forEach(msg => {
            if (msg.role === 'user') {
                this.addMessage(msg.content, 'user');
            } else if (msg.role === 'assistant') {
                this.addBotMessage(msg.content, msg.sources || [], msg.follow_up_questions || []);
            }
        });
        
        this.scrollToBottom();
    }

    addWelcomeMessage() {
        const welcomeMessage = `
            <div class="message bot-message">
                <div class="avatar bot-avatar">🤖</div>
                <div class="message-content">
                    <strong>👋 Hello! I'm Ragnosis AI</strong><br><br>
                    Your friendly AI health companion! I'm here to help with medical questions, health information, and wellness guidance.<br><br>
                    💬 <strong>You can ask me about:</strong><br>
                    • Symptoms and conditions<br>
                    • Treatment options<br>
                    • Medical explanations<br>
                    • Health advice<br>
                    • Or just have a friendly chat!<br><br>
                    <div class="disclaimer-inline">
                    💡 Remember: I'm an AI assistant for informational purposes. Always consult healthcare professionals for medical advice, diagnoses, or treatment.
                    </div>
                </div>
            </div>
        `;
        this.chatMessages.innerHTML += welcomeMessage;
    }

    handleKeydown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.handleSendMessage();
        }
    }

    handleInputResize() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    updateSendButtonState() {
        this.sendButton.disabled = this.messageInput.value.trim() === '' || this.isProcessing;
        if (this.isProcessing) {
            this.sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        } else {
            this.sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
    }

    async handleSendMessage() {
        if (this.isProcessing) return;
        
        const message = this.messageInput.value.trim();
        
        if (!message) return;

        console.log('📤 Sending message:', message);
        console.log('💾 Session ID:', this.sessionId);
        
        try {
            this.isProcessing = true;
            this.updateSendButtonState();
            
            // Clear input and update UI
            this.messageInput.value = '';
            this.messageInput.style.height = 'auto';
            
            // Add user message to chat
            this.addMessage(message, 'user');
            
            // Show typing indicator
            this.showTypingIndicator();
            
            // Call API
            const response = await this.callRagnosisAPI(message);
            
            // Hide typing indicator and show response
            this.hideTypingIndicator();
            this.addBotMessage(response.answer, response.sources, response.follow_up_questions);
            
            // Update conversation history and session
            this.conversationHistory.push({
                role: 'user',
                content: message,
                timestamp: Date.now()
            });
            this.conversationHistory.push({
                role: 'assistant',
                content: response.answer,
                sources: response.sources,
                follow_up_questions: response.follow_up_questions,
                timestamp: Date.now()
            });
            
            // Update session ID if changed
            if (response.session_id && response.session_id !== this.sessionId) {
                this.sessionId = response.session_id;
            }
            
            // Save session
            this.saveSession();
            
        } catch (error) {
            console.error('❌ Error:', error);
            this.hideTypingIndicator();
            this.showError("I'm having trouble connecting right now. Please try again! 😅");
        } finally {
            this.isProcessing = false;
            this.updateSendButtonState();
            this.messageInput.focus();
        }
    }

    async callRagnosisAPI(message) {
        const payload = {
            question: message,
            session_id: this.sessionId,
            conversation_history: this.conversationHistory.slice(-10)
        };

        const response = await fetch(`${this.API_BASE}/ask`, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ragnosis-ai-token-2024'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            // Even if the request fails, we'll handle it gracefully in the UI
            throw new Error(`API request failed: ${response.status}`);
        }
        
        return await response.json();
    }

    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = sender === 'user' ? '👤' : '🤖';
        const avatarClass = sender === 'user' ? 'user-avatar' : 'bot-avatar';
        
        messageDiv.innerHTML = `
            <div class="avatar ${avatarClass}">${avatar}</div>
            <div class="message-content">${this.escapeHtml(content)}</div>
        `;
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addBotMessage(answer, sources = [], followUpQuestions = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        
        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            sourcesHtml = `
                <div class="sources">
                    <div class="sources-title">📚 Sources & References:</div>
                    ${sources.map(source => 
                        `<div class="source-item">${this.escapeHtml(source)}</div>`
                    ).join('')}
                </div>
            `;
        }
        
        let followUpHtml = '';
        if (followUpQuestions && followUpQuestions.length > 0) {
            followUpHtml = `
                <div class="follow-up-questions">
                    <div class="follow-up-title">💭 You might want to ask:</div>
                    ${followUpQuestions.map(question => 
                        `<div class="follow-up-item" onclick="ragnosisAI.useFollowUpQuestion('${this.escapeHtml(question)}')">
                            ${this.escapeHtml(question)}
                        </div>`
                    ).join('')}
                </div>
            `;
        }
        
        messageDiv.innerHTML = `
            <div class="avatar bot-avatar">🤖</div>
            <div class="message-content">
                ${this.formatMessage(answer)}
                ${sourcesHtml}
                ${followUpHtml}
            </div>
        `;
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    useFollowUpQuestion(question) {
        this.messageInput.value = question;
        this.messageInput.focus();
        this.updateSendButtonState();
        this.handleInputResize();
    }

    clearChat() {
        if (confirm('Are you sure you want to clear the chat history? This will start a new conversation.')) {
            this.chatMessages.innerHTML = '';
            this.clearSession();
            this.addWelcomeMessage();
            this.messageInput.focus();
        }
    }

    showError(errorMessage) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message bot-message';
        errorDiv.innerHTML = `
            <div class="avatar bot-avatar">🤖</div>
            <div class="message-content">
                <div class="error-message">
                    😅 <strong>Connection Issue</strong><br>
                    ${this.escapeHtml(errorMessage)}
                </div>
            </div>
        `;
        
        this.chatMessages.appendChild(errorDiv);
        this.scrollToBottom();
    }

    showTypingIndicator() {
        this.typingIndicator.style.display = 'block';
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }

    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }

    // Utility functions
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatMessage(text) {
        if (!text) return '';
        
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>')
            .replace(/💡 Remember: (.*?)$/, '<div class="disclaimer-inline">💡 Remember: $1</div>');
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.ragnosisAI = new RagnosisAI();
});
