/**
 * app.js
 * BarPal.ai — Legal Document Q&A Frontend
 * Handles: file upload, document library, chat Q&A, source citations
 */

const API = 'http://localhost:8000';

// ── DOM References ────────────────────────────────────────────────────────────
const sidebar        = document.getElementById('sidebar');
const sidebarToggle  = document.getElementById('sidebarToggle');
const topbarMenu     = document.getElementById('topbarMenu');
const uploadZone     = document.getElementById('uploadZone');
const fileInput      = document.getElementById('fileInput');
const browseBtn      = document.getElementById('browseBtn');
const uploadProgress = document.getElementById('uploadProgress');
const progressBar    = document.getElementById('progressBar');
const progressText   = document.getElementById('progressText');
const docList        = document.getElementById('docList');
const docCount       = document.getElementById('docCount');
const docFilter      = document.getElementById('docFilter');
const chatArea       = document.getElementById('chatArea');
const welcomeScreen  = document.getElementById('welcomeScreen');
const questionInput  = document.getElementById('questionInput');
const sendBtn        = document.getElementById('sendBtn');
const clearChatBtn   = document.getElementById('clearChatBtn');
const statusDot      = document.getElementById('statusDot');
const statusText     = document.getElementById('statusText');
const toast          = document.getElementById('toast');

// Evaluation
const tabChat        = document.getElementById('tabChat');
const tabEval        = document.getElementById('tabEval');
const panelChat      = document.getElementById('panelChat');
const panelEval      = document.getElementById('panelEval');
const chatInputWrap  = document.getElementById('chatInputWrap');
const evalQuestion   = document.getElementById('evalQuestion');
const evalBtn        = document.getElementById('evalBtn');
const evalResults    = document.getElementById('evalResults');
const evalLoading    = document.getElementById('evalLoading');
const metricCards    = document.getElementById('metricCards');
const overallPct     = document.getElementById('overallPct');
const ringProgress   = document.getElementById('ringProgress');
const evalAnswerBox  = document.getElementById('evalAnswerBox');
const evalSources    = document.getElementById('evalSources');

// ── State ─────────────────────────────────────────────────────────────────────
let documents     = [];
let chatHistory   = [];
let isLoading     = false;
let toastTimeout  = null;

// ── Init ──────────────────────────────────────────────────────────────────────
(async function init() {
  await checkHealth();
  await fetchDocuments();
  setupEventListeners();
})();

// ── Health Check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`);
    if (res.ok) {
      statusDot.className  = 'status-dot online';
      statusText.textContent = 'Backend connected';
    } else {
      throw new Error('Non-OK response');
    }
  } catch {
    statusDot.className    = 'status-dot offline';
    statusText.textContent = 'Backend offline';
    showToast('Backend not reachable. Start the server first.', 'error', 5000);
  }
}

// ── Event Listeners ───────────────────────────────────────────────────────────
function setupEventListeners() {
  // Sidebar toggle
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
  });
  topbarMenu.addEventListener('click', () => {
    sidebar.classList.toggle('mobile-open');
  });

  // Tab switching
  tabChat.addEventListener('click', () => switchTab('chat'));
  tabEval.addEventListener('click', () => switchTab('eval'));

  // Upload zone
  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });
  uploadZone.addEventListener('click', () => fileInput.click());
  uploadZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
  });

  // Drag and drop
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFileUpload(fileInput.files[0]);
  });

  // Chat
  sendBtn.addEventListener('click', handleSend);
  questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  // Auto-resize textarea
  questionInput.addEventListener('input', () => {
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 180) + 'px';
  });

  // Example cards
  document.querySelectorAll('.example-card').forEach(card => {
    card.addEventListener('click', () => {
      questionInput.value = card.dataset.q;
      questionInput.dispatchEvent(new Event('input'));
      questionInput.focus();
    });
  });

  // Clear chat
  clearChatBtn.addEventListener('click', () => {
    chatHistory = [];
    const messages = chatArea.querySelectorAll('.message');
    messages.forEach(m => m.remove());
    welcomeScreen.classList.remove('hidden');
    showToast('Chat cleared', 'info');
  });

  // Evaluation
  evalBtn.addEventListener('click', handleEvaluation);
  evalQuestion.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleEvaluation();
    }
  });
}

// ════════════════════════════════════════════════════════════════════════════
// TAB SWITCHING
// ════════════════════════════════════════════════════════════════════════════
function switchTab(tab) {
  const isChat = tab === 'chat';
  tabChat.classList.toggle('active', isChat);
  tabEval.classList.toggle('active', !isChat);
  tabChat.setAttribute('aria-selected', isChat);
  tabEval.setAttribute('aria-selected', !isChat);
  panelChat.classList.toggle('hidden', !isChat);
  panelEval.classList.toggle('hidden', isChat);
  chatInputWrap.style.display = isChat ? '' : 'none';
}

// ════════════════════════════════════════════════════════════════════════════
// RAGAS EVALUATION
// ════════════════════════════════════════════════════════════════════════════
async function handleEvaluation() {
  const question = evalQuestion.value.trim();
  if (!question) {
    showToast('Enter a test question first.', 'error');
    return;
  }

  // Show loading
  evalResults.classList.add('hidden');
  evalLoading.classList.remove('hidden');
  evalBtn.disabled = true;

  try {
    const res = await fetch(`${API}/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        doc_filter: docFilter.value || null,
      }),
    });

    const data = await res.json();
    evalLoading.classList.add('hidden');

    if (!res.ok) {
      showToast(data.detail || 'Evaluation failed.', 'error', 5000);
      return;
    }

    renderEvaluationResults(data);

  } catch (err) {
    evalLoading.classList.add('hidden');
    showToast('Connection error. Is the server running?', 'error');
  } finally {
    evalBtn.disabled = false;
  }
}

function renderEvaluationResults(data) {
  // ── Overall ring ───────────────────────────────────────────────────────
  const pct = Math.round(data.overall_score * 100);
  overallPct.textContent = pct + '%';

  // Animate the SVG circle: dashoffset from 264 (empty) to 264*(1-score) (filled)
  const circumference = 264;
  const offset = circumference * (1 - data.overall_score);
  // Set colour based on score
  ringProgress.style.stroke = scoreColor(data.overall_score);
  setTimeout(() => {
    ringProgress.style.strokeDashoffset = offset;
  }, 50);

  // ── Metric cards ───────────────────────────────────────────────────────
  metricCards.innerHTML = '';
  const metricOrder = ['faithfulness', 'answer_relevance', 'context_precision'];
  metricOrder.forEach(key => {
    const m = data.metrics[key];
    if (!m) return;
    const tier = scoreTier(m.score);
    const pctM = Math.round(m.score * 100);

    const card = document.createElement('div');
    card.className = 'metric-card';
    card.innerHTML = `
      <div class="metric-card-header">
        <span class="metric-name">${escapeHtml(m.label)}</span>
        <span class="metric-score-badge ${tier}">${pctM}%</span>
      </div>
      <div class="metric-bar-wrap">
        <div class="metric-bar ${tier}" id="bar-${key}" style="width:0%"></div>
      </div>
      <p class="metric-desc">${escapeHtml(m.description)}</p>
      ${m.reason ? `<p class="metric-reason">"${escapeHtml(m.reason)}"</p>` : ''}
    `;
    metricCards.appendChild(card);

    // Animate bar after paint
    setTimeout(() => {
      const bar = document.getElementById(`bar-${key}`);
      if (bar) bar.style.width = pctM + '%';
    }, 100);
  });

  // ── Answer box ─────────────────────────────────────────────────────────
  evalAnswerBox.textContent = data.answer;

  // ── Sources ────────────────────────────────────────────────────────────
  evalSources.innerHTML = '';
  if (data.sources && data.sources.length > 0) {
    data.sources.forEach(s => {
      const chip = document.createElement('span');
      chip.className = 'source-chip';
      chip.innerHTML = `
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
        </svg>
        ${escapeHtml(truncate(s.filename, 24))} · p.${s.page}
      `;
      evalSources.appendChild(chip);
    });
  } else {
    evalSources.textContent = 'No sources returned.';
  }

  // Show results
  evalResults.classList.remove('hidden');
}

function scoreTier(score) {
  if (score >= 0.7) return 'good';
  if (score >= 0.4) return 'medium';
  return 'poor';
}

function scoreColor(score) {
  if (score >= 0.7) return '#2f9e44';
  if (score >= 0.4) return '#e67700';
  return '#c92a2a';
}



// ════════════════════════════════════════════════════════════════════════════
// FILE UPLOAD
// ════════════════════════════════════════════════════════════════════════════
async function handleFileUpload(file) {
  const allowed = ['.pdf', '.docx', '.doc'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showToast(`Unsupported file type: ${ext}. Use PDF or DOCX.`, 'error');
    return;
  }

  // Show progress
  uploadProgress.classList.add('show');
  progressBar.style.width = '0%';
  progressText.textContent = `Uploading ${file.name}…`;
  animateProgress(0, 40, 600);

  const formData = new FormData();
  formData.append('file', file);

  try {
    animateProgress(40, 75, 800);
    const res = await fetch(`${API}/upload`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Upload failed');

    animateProgress(75, 100, 400);
    await delay(500);

    uploadProgress.classList.remove('show');
    progressBar.style.width = '0%';
    fileInput.value = '';

    showToast(`✓ ${data.filename} — ${data.chunks} chunks indexed from ${data.pages} pages`, 'success', 4000);
    await fetchDocuments();

  } catch (err) {
    uploadProgress.classList.remove('show');
    showToast(err.message || 'Upload failed. Please try again.', 'error');
  }
}

function animateProgress(from, to, duration) {
  const start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / duration, 1);
    progressBar.style.width = (from + (to - from) * easeOut(t)) + '%';
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function easeOut(t) { return 1 - Math.pow(1 - t, 2); }
function delay(ms)  { return new Promise(r => setTimeout(r, ms)); }

// ════════════════════════════════════════════════════════════════════════════
// DOCUMENT LIBRARY
// ════════════════════════════════════════════════════════════════════════════
async function fetchDocuments() {
  try {
    const res  = await fetch(`${API}/documents`);
    const data = await res.json();
    documents  = data.documents || [];
    renderDocumentLibrary();
  } catch {
    // Silently fail — server may not be up yet
  }
}

function renderDocumentLibrary() {
  docCount.textContent = documents.length;

  // Update filter dropdown
  const currentFilter = docFilter.value;
  docFilter.innerHTML = '<option value="">All documents</option>';
  documents.forEach(doc => {
    const opt = document.createElement('option');
    opt.value = doc.filename;
    opt.textContent = truncate(doc.filename, 30);
    if (doc.filename === currentFilter) opt.selected = true;
    docFilter.appendChild(opt);
  });

  // Render list
  if (documents.length === 0) {
    docList.innerHTML = `<li class="doc-empty">No documents yet.<br/>Upload your first legal document above.</li>`;
    return;
  }

  docList.innerHTML = '';
  documents.forEach(doc => {
    const li = document.createElement('li');
    li.className = 'doc-item';
    li.dataset.docId = doc.doc_id;
    li.innerHTML = `
      <div class="doc-icon">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
        </svg>
      </div>
      <div class="doc-info">
        <span class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
        <div class="doc-meta">
          <span>${doc.chunk_count} chunks</span>
        </div>
      </div>
      <button class="doc-delete" title="Remove document" data-doc-id="${doc.doc_id}" aria-label="Delete ${escapeHtml(doc.filename)}">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="3 6 5 6 21 6"/>
          <path d="M19 6l-1 14H6L5 6"/>
        </svg>
      </button>
    `;

    li.querySelector('.doc-delete').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteDocument(doc.doc_id, doc.filename);
    });

    docList.appendChild(li);
  });
}

async function deleteDocument(docId, filename) {
  if (!confirm(`Remove "${filename}" from the knowledge base?`)) return;
  try {
    const res  = await fetch(`${API}/documents/${docId}`, { method: 'DELETE' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Delete failed');
    showToast(`Removed ${filename}`, 'info');
    await fetchDocuments();
  } catch (err) {
    showToast(err.message || 'Could not delete document.', 'error');
  }
}

// ════════════════════════════════════════════════════════════════════════════
// CHAT
// ════════════════════════════════════════════════════════════════════════════
async function handleSend() {
  const question = questionInput.value.trim();
  if (!question || isLoading) return;

  isLoading = true;
  sendBtn.disabled = true;

  // Hide welcome, add user message
  welcomeScreen.classList.add('hidden');
  appendMessage('user', question);
  questionInput.value = '';
  questionInput.style.height = 'auto';

  // Typing indicator
  const typingId = appendTyping();

  try {
    const res = await fetch(`${API}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        doc_filter: docFilter.value || null,
      }),
    });

    const data = await res.json();
    removeTyping(typingId);

    if (!res.ok) {
      appendMessage('assistant', `**Error:** ${data.detail || 'Something went wrong.'}`, [], 0);
    } else {
      appendMessage('assistant', data.answer, data.sources, data.chunks_retrieved);
    }

    chatHistory.push({ role: 'user', content: question });
    chatHistory.push({ role: 'assistant', content: data.answer });

  } catch (err) {
    removeTyping(typingId);
    appendMessage('assistant', `**Connection error:** Could not reach the backend.\n\nMake sure the server is running on \`http://localhost:8000\``, [], 0);
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
    questionInput.focus();
  }
}

function appendMessage(role, text, sources = [], chunksRetrieved = 0) {
  const div = document.createElement('div');
  div.className = `message ${role}`;

  const avatarLabel = role === 'user' ? 'U' : '⚖';
  const avatar = `<div class="msg-avatar">${avatarLabel}</div>`;

  // Render markdown-like formatting
  const htmlContent = simpleMarkdown(text);

  // Build sources chips
  let sourcesHtml = '';
  if (sources && sources.length > 0) {
    const chips = sources.map(s => `
      <span class="source-chip" title="Relevance: ${Math.round(s.score * 100)}%">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
        </svg>
        ${escapeHtml(truncate(s.filename, 22))} · p.${s.page}
      </span>
    `).join('');

    const retrieval = chunksRetrieved > 0
      ? `<div class="msg-retrieval">
           <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
             <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
           </svg>
           ${chunksRetrieved} relevant passage${chunksRetrieved !== 1 ? 's' : ''} retrieved
         </div>`
      : '';

    sourcesHtml = `
      ${retrieval}
      <div class="msg-sources">${chips}</div>
    `;
  }

  div.innerHTML = `
    ${avatar}
    <div class="msg-body">
      <div class="msg-bubble">${htmlContent}</div>
      ${sourcesHtml}
    </div>
  `;

  chatArea.appendChild(div);
  scrollToBottom();
  return div;
}

function appendTyping() {
  const id = 'typing-' + Date.now();
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.id = id;
  div.innerHTML = `
    <div class="msg-avatar">⚖</div>
    <div class="msg-body">
      <div class="msg-bubble">
        <div class="typing-dots">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>
  `;
  chatArea.appendChild(div);
  scrollToBottom();
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function scrollToBottom() {
  chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
}

// ════════════════════════════════════════════════════════════════════════════
// SIMPLE MARKDOWN RENDERER
// ════════════════════════════════════════════════════════════════════════════
function simpleMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);

  // Headings
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Bold / italic
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g,     '<em>$1</em>');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Blockquote
  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

  // Unordered list
  html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');

  // Ordered list
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

  // Line breaks → paragraphs
  html = html
    .split(/\n\n+/)
    .map(p => p.trim())
    .filter(Boolean)
    .map(p => {
      if (/^<(h[1-3]|ul|ol|blockquote|li)/.test(p)) return p;
      return `<p>${p.replace(/\n/g, '<br/>')}</p>`;
    })
    .join('\n');

  return html;
}

// ════════════════════════════════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ════════════════════════════════════════════════════════════════════════════
function showToast(message, type = 'info', duration = 3500) {
  clearTimeout(toastTimeout);
  toast.textContent = message;
  toast.className = `toast ${type} show`;
  toastTimeout = setTimeout(() => {
    toast.className = `toast ${type}`;
  }, duration);
}

// ════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ════════════════════════════════════════════════════════════════════════════
function escapeHtml(str = '') {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function truncate(str, maxLen) {
  if (!str) return '';
  return str.length > maxLen ? str.slice(0, maxLen - 1) + '…' : str;
}
