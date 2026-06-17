// ============================================================
// CosyVoice TTS WebUI — JavaScript
// ============================================================

// ---- Config ----
const CFG = window.__COSYVOICE_CONFIG__ || {};

// ---- State ----
let speakers = [];
let statusData = {};
let isGenerating = false;
let seedLocked = false;
let historyEntries = [];
const MAX_HISTORY = 20;

// Audio trim state (speaker extraction)
let waveformState = null;
let previewAudio = null;

// Recording state (inside Extract tab)
let extractRecState = null; // { mediaRecorder, chunks, startTime, stream, analyser, dataArray, audioCtx, source, animId }

// Toast timer IDs
let toastTimers = [];

// ---- localStorage persistence ----
const ADV_PARAM_IDS = [
    'tts-seed','tts-temp','tts-topk','tts-topp','tts-winsize','tts-taur',
    'tts-fadein','tts-textnorm','tts-split','tts-fastsplit',
    'tts-mode','tts-format',
    'model-max-llm','model-kv-k','model-kv-v','model-buffer-policy',
    'model-threads','model-backend'
];

function saveAdvParam(id) {
    try {
        const el = document.getElementById(id);
        if (!el) return;
        const key = 'cosyvoice_' + id;
        if (el.type === 'checkbox') localStorage.setItem(key, el.checked ? '1' : '0');
        else if (el.tagName === 'SELECT') localStorage.setItem(key, el.value);
        else localStorage.setItem(key, el.value);
    } catch(e) { /* ignore */ }
}

function restoreAdvParams() {
    try {
        ADV_PARAM_IDS.forEach(id => {
            const key = 'cosyvoice_' + id;
            const saved = localStorage.getItem(key);
            if (saved === null) return;
            const el = document.getElementById(id);
            if (!el) return;
            if (el.type === 'checkbox') el.checked = saved === '1';
            else if (el.tagName === 'SELECT') {
                for (let i = 0; i < el.options.length; i++) {
                    if (el.options[i].value === saved) { el.selectedIndex = i; break; }
                }
            } else el.value = saved;
        });
    } catch(e) { /* ignore */ }
}

function clearAdvParams() {
    ADV_PARAM_IDS.forEach(id => localStorage.removeItem('cosyvoice_' + id));
}

// ---- SVG Icons ----
const ICONS = {
    mic: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>',
    speaker: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>',
    music: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>',
    cpu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    wrench: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>',
    user: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
    play: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>',
    stop: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>',
    download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    scissors: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><line x1="20" y1="4" x2="8.12" y2="15.88"/><line x1="14.47" y1="14.48" x2="20" y2="20"/><line x1="8.12" y1="8.12" x2="12" y2="12"/></svg>',
    refresh: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>',
    trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>',
    copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    x: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
    alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    settings: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
    clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    wave: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h.01M7 12h.01M11 12h.01M15 12h.01M19 12h.01"/><path d="M1 9a5 5 0 0 0 0 6"/><path d="M5 7a9 9 0 0 0 0 10"/><path d="M9 5a13 13 0 0 0 0 14"/><path d="M13 4a15 15 0 0 0 0 16"/></svg>',
    volume: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>',
    play_small: '<svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14"><polygon points="5 3 19 12 5 21 5 3"/></svg>',
};

// ---- Toast Notification System ----
function showToast(message, type, duration) {
    type = type || 'info';
    duration = duration || (type === 'error' ? 5000 : 3000);

    // Clean up expired toasts with same type to avoid clutter
    const container = document.getElementById('toast-container');
    if (!container) return;

    const icons = { success: ICONS.check, error: ICONS.alert, warning: ICONS.alert, info: ICONS.info };

    const toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.innerHTML = icons[type] || icons.info;
    const span = document.createElement('span');
    span.textContent = message;
    toast.appendChild(span);

    toast.addEventListener('click', () => {
        toast.classList.add('toast-out');
        setTimeout(() => { if (toast.parentNode) toast.parentNode.removeChild(toast); }, 300);
    });

    container.appendChild(toast);

    const timer = setTimeout(() => {
        toast.classList.add('toast-out');
        setTimeout(() => { if (toast.parentNode) toast.parentNode.removeChild(toast); }, 300);
    }, duration);
    toastTimers.push(timer);
}

// ---- DOM Shortcuts ----
const $ = id => document.getElementById(id);
const els = {};

// ---- Init DOM references after DOM ready ----
function initEls() {
    const ids = [
        // Status
        'status-dot', 'status-text', 'server-info-grid',

        // Model
        'model-path', 'model-backend', 'model-threads',
        'model-kv-k', 'model-kv-v', 'model-buffer-policy', 'model-max-llm',
        'btn-reset-model-config',
        'btn-load-model', 'btn-unload-model',
        'model-load-area', 'model-loaded-area', 'model-loaded-info',
        'model-error', 'model-success',

        // Frontend
        'cfg-tokenizer', 'cfg-campplus',
        'btn-load-frontend', 'btn-unload-frontend',
        'frontend-not-loaded', 'frontend-loaded', 'frontend-loaded-info',
        'frontend-error', 'frontend-success',

        // Speakers
        'speaker-list', 'speaker-count',
        'load-name', 'load-path', 'btn-load-gguf',
        'load-error', 'load-success',
        'extract-name', 'extract-text', 'extract-audio', 'btn-extract',
        'extract-error', 'extract-success', 'extract-disabled', 'extract-form',
        'trim-area', 'waveform-canvas',
        'wf-start-time', 'wf-end-time', 'wf-selected', 'wf-total',
        'wf-zoomin', 'wf-zoomout', 'wf-reset', 'wf-preview',

        // TTS
        'tts-voice', 'tts-mode', 'tts-format',
        'tts-text', 'tts-instructions',
        'tts-speed', 'speed-val',
        'tts-seed', 'tts-temp', 'tts-topk', 'tts-topp', 'tts-winsize', 'tts-taur',
        'tts-fadein', 'tts-textnorm', 'tts-split', 'tts-fastsplit',
        'seed-dice', 'seed-lock',
        'btn-tts', 'btn-reset-gen-config',
        'player-area', 'audio-player', 'download-link',
        'tts-error',

        // History
        'history-list', 'history-area', 'btn-clear-history',

        // Stats
        'stat-model', 'stat-speakers', 'stat-frontend', 'stat-sample-rate',

        // Info Card visibility
        'info-card',

        // Frontend card
        'frontend-card',

        // Extract tab recording
        'extract-record-btn', 'erm-label', 'erm-status', 'erm-timer',
        'erm-waveform', 'erm-canvas', 'erm-level',
    ];

    ids.forEach(id => { els[id] = $(id); });
}

// ---- Utilities ----
function showError(el, msg) {
    el.innerHTML = ICONS.alert + ' ' + escapeHtml(msg);
    el.classList.add('show');
}
function hideError(el) { el.classList.remove('show'); el.innerHTML = ''; }
function showSuccess(el, msg) {
    el.innerHTML = ICONS.check + ' ' + escapeHtml(msg);
    el.classList.add('show');
}
function hideSuccess(el) { el.classList.remove('show'); el.innerHTML = ''; }

async function apiFetch(url, opts) {
    const res = await fetch(url, opts);
    if (!res.ok) {
        let msg = 'HTTP ' + res.status;
        try { const j = await res.json(); msg = j.error || j.message || msg; } catch(e) { /* ignore */ }
        throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
    }
    return res;
}

function jsonBody(data) {
    return {
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    };
}

function escapeHtml(s) {
    if (!s) return '';
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(1);
    return m > 0 ? m + 'm ' + s + 's' : s + 's';
}

// ---- Tab Switching ----
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        if (tab.disabled) return;
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        $(tab.dataset.tab).classList.add('active');
    });
});

// ---- Speed Display ----
function initSpeedDisplay() {
    const slider = document.getElementById('tts-speed');
    const display = document.getElementById('speed-val');
    if (slider && display) {
        display.textContent = parseFloat(slider.value).toFixed(2);
        slider.addEventListener('input', () => {
            display.textContent = parseFloat(slider.value).toFixed(2);
        });
    }
}

// ---- Fetch Server Status ----
async function fetchStatus() {
    try {
        const res = await apiFetch('/status');
        statusData = await res.json();
        renderStatus();
        renderFrontendConfig();
        checkExtractAvailability();
        updateModelUI();
        updateFrontendUI();
    } catch(e) {
        if (els['status-text']) els['status-text'].textContent = 'Offline';
        if (els['status-dot']) { els['status-dot'].className = 'status-dot err'; }
        showToast('Connection failed: ' + e.message, 'error', 5000);
    }
}

function renderStatus() {
    const modelName = statusData.model_loaded ? statusData.model : null;

    // Status dot
    if (els['status-dot']) {
        els['status-dot'].className = 'status-dot ' + (statusData.model_loaded ? 'on' : 'off');
    }
    if (els['status-text']) {
        els['status-text'].textContent = statusData.model_loaded
            ? 'Model: ' + modelName
            : 'Ready (no model)';
    }

    // Info grid
    if (els['server-info-grid']) {
        const gridData = [
            ['Status', '<span class="status-badge on">' + ICONS.check + ' Running</span>'],
            ['Model', statusData.model_loaded ? ('<b>' + escapeHtml(statusData.model || '') + '</b>') : '<span class="status-badge off">Not loaded</span>'],
            ['Sample Rate', statusData.sample_rate ? statusData.sample_rate + ' Hz' : '—'],
            ['Frontend', statusData.frontend_available ? '<span class="status-badge on">' + ICONS.check + ' Configured</span>' : '<span class="status-badge off">Not configured</span>'],
            ['Speakers', (statusData.speakers && statusData.speakers.length) || '0'],
        ];
        els['server-info-grid'].innerHTML = gridData.map(([k, v]) =>
            '<span class="label">' + k + '</span><span class="value">' + v + '</span>'
        ).join('');
    }

    // Stats
    if (els['stat-model']) {
        els['stat-model'].querySelector('.stat-num').textContent = statusData.model_loaded ? '1' : '0';
        els['stat-model'].querySelector('.stat-sub').textContent = statusData.model_arch || '';
    }
    if (els['stat-speakers']) {
        els['stat-speakers'].querySelector('.stat-num').textContent = (statusData.speakers && statusData.speakers.length) || '0';
    }
    if (els['stat-frontend']) {
        const avail = statusData.frontend_available;
        els['stat-frontend'].querySelector('.stat-num').textContent = avail ? '✓' : '✗';
        els['stat-frontend'].querySelector('.stat-num').style.color = avail ? 'var(--success)' : 'var(--text-muted)';
    }
    if (els['stat-sample-rate'] && statusData.sample_rate) {
        els['stat-sample-rate'].querySelector('.stat-num').textContent = (statusData.sample_rate / 1000).toFixed(1) + 'k';
        els['stat-sample-rate'].querySelector('.stat-sub').textContent = statusData.sample_rate + ' Hz';
    }
}

function renderFrontendConfig() {
    if (els['cfg-tokenizer']) els['cfg-tokenizer'].value = CFG.speechTokenizer || '';
    if (els['cfg-campplus']) els['cfg-campplus'].value = CFG.campplus || '';
}

function updateModelUI() {
    const hasModel = statusData.model_loaded;
    if (els['model-load-area']) els['model-load-area'].style.display = hasModel ? 'none' : 'block';
    if (els['model-loaded-area']) els['model-loaded-area'].style.display = hasModel ? 'block' : 'none';
    if (hasModel && els['model-loaded-info']) {
        const arch = statusData.model_arch || statusData.model || 'Model';
        const kv_k = statusData.k_cache_type || '?';
        const kv_v = statusData.v_cache_type || '?';
        const buf = statusData.buffer_policy || '?';
        const mll = statusData.max_llm_len || '?';
        els['model-loaded-info'].innerHTML = '<b>' + escapeHtml(arch) + '</b><br>'
            + 'KV Cache: K=' + kv_k + ', V=' + kv_v + '<br>'
            + 'Buffer Policy: ' + buf + '<br>'
            + 'Max LLM Length: ' + mll + '<br>'
            + 'Sample Rate: ' + (statusData.sample_rate || '?') + ' Hz';
    }
}

function updateFrontendUI() {
    const avail = statusData.frontend_available;
    if (els['frontend-not-loaded']) els['frontend-not-loaded'].style.display = avail ? 'none' : 'block';
    if (els['frontend-loaded']) els['frontend-loaded'].style.display = avail ? 'block' : 'none';
}

function checkExtractAvailability() {
    const avail = statusData.frontend_available;
    if (els['extract-disabled'] && els['extract-form']) {
        if (!avail) {
            els['extract-disabled'].style.display = 'block';
            els['extract-form'].style.display = 'none';
        } else {
            els['extract-disabled'].style.display = 'none';
            els['extract-form'].style.display = 'block';
        }
    }
}

// ---- Fetch Speakers ----
async function fetchSpeakers() {
    try {
        const res = await apiFetch('/speaker');
        const data = await res.json();
        if (Array.isArray(data)) {
            speakers = data.length > 0 && typeof data[0] === 'object' ? data : data.map(n => ({ name: n }));
        } else {
            speakers = data.speakers || [];
            if (speakers.length > 0 && typeof speakers[0] === 'string')
                speakers = speakers.map(n => ({ name: n }));
        }
        renderSpeakers();
    } catch(e) {
        console.error('Failed to load speakers:', e);
    }
}

function renderSpeakers() {
    const names = speakers.map(s => (typeof s === 'string' ? s : s.name)).filter(Boolean);

    // Speaker count badge
    if (els['speaker-count']) {
        els['speaker-count'].textContent = names.length;
    }

    if (names.length === 0) {
        if (els['speaker-list']) {
            els['speaker-list'].innerHTML = '<div class="empty-state">'
                + ICONS.mic
                + '<br>No speakers registered<br>'
                + '<span style="font-size:12px;color:var(--text-muted)">Load a GGUF file or extract from audio to get started.</span>'
                + '</div>';
        }
    } else if (els['speaker-list']) {
        els['speaker-list'].innerHTML = names.map(n =>
            '<div class="speaker-item">'
                + '<div class="speaker-name">' + ICONS.user + '<span>' + escapeHtml(n) + '</span></div>'
                + '<div class="speaker-actions">'
                    + '<span class="action-link" data-name="' + escapeHtml(n) + '" onclick="saveSpeaker(\'' + escapeHtml(n) + '\')">' + ICONS.download + ' Save</span>'
                    + '<span class="action-link" style="color:var(--danger)" data-name="' + escapeHtml(n) + '" onclick="deleteSpeaker(\'' + escapeHtml(n) + '\')">' + ICONS.trash + '</span>'
                + '</div>'
            + '</div>'
        ).join('');
    }

    // Update voice selector
    if (els['tts-voice']) {
        els['tts-voice'].innerHTML = '';
        if (names.length === 0) {
            els['tts-voice'].innerHTML = '<option value="">— No speakers —</option>';
        } else {
            names.forEach(n => {
                const opt = document.createElement('option');
                opt.value = n;
                opt.textContent = n;
                els['tts-voice'].appendChild(opt);
            });
        }
    }
}

// ---- Delete Speaker ----
async function deleteSpeaker(name) {
    try {
        await apiFetch('/speaker/' + encodeURIComponent(name), { method: 'DELETE' });
        showToast('Speaker "' + name + '" deleted', 'success');
        await fetchStatus();
        await fetchSpeakers();
    } catch(e) {
        showToast('Failed to delete: ' + e.message, 'error');
    }
}

// ---- Save Speaker Prompt Speech ----
async function saveSpeaker(name) {
    const path = prompt('Save prompt speech for "' + name + '" to server path:', '');
    if (!path) return;
    try {
        await apiFetch('/speaker/save', {
            method: 'POST',
            ...jsonBody({ name: name, path: path }),
        });
        showToast('Speaker saved to: ' + path, 'success');
    } catch(e) {
        showToast('Save failed: ' + e.message, 'error');
    }
}

// ==========================================================
// Model Management
// ==========================================================

// Load Model
function initModelLoad() {
    const btn = els['btn-load-model'];
    if (!btn) return;
    btn.addEventListener('click', async () => {
        hideError(els['model-error']);
        hideSuccess(els['model-success']);
        const model = els['model-path'].value.trim();
        if (!model) { showError(els['model-error'], 'Please enter a model file path'); return; }
        if (!els['model-max-llm'].value) { showError(els['model-error'], 'Max LLM Length is required'); return; }

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span>Loading...';
        try {
            const body = {
                model: model,
                backend: els['model-backend'].value,
                n_threads: parseInt(els['model-threads'].value, 10) || 0,
            };
            const kt = els['model-kv-k'].value;
            const vt = els['model-kv-v'].value;
            body.llm_kv_cache_type = (kt === vt) ? kt : 'k=' + kt + ',v=' + vt;
            const bp = els['model-buffer-policy'].value;
            if (bp) body.inference_buffer_policy = bp;
            const ml = parseInt(els['model-max-llm'].value, 10);
            if (ml > 0) body.max_llm_len = ml;

            await apiFetch('/model/load', { method: 'POST', ...jsonBody(body) });
            showSuccess(els['model-success'], 'Model loaded successfully');
            showToast('Model loaded: ' + model, 'success');
            await fetchStatus();
            await fetchDefaults();
            await fetchSpeakers();
        } catch(e) {
            showError(els['model-error'], 'Failed to load model: ' + e.message);
            showToast('Failed to load model', 'error');
        }
        btn.disabled = false;
        btn.innerHTML = ICONS.cpu + ' Load Model';
    });
}

// Unload Model
function initModelUnload() {
    const btn = els['btn-unload-model'];
    if (!btn) return;
    btn.addEventListener('click', async () => {
        hideError(els['model-error']);
        hideSuccess(els['model-success']);
        if (!confirm('Unload the current model? All speakers will be removed.')) return;

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span>Unloading...';
        try {
            await apiFetch('/model/unload', { method: 'POST' });
            showSuccess(els['model-success'], 'Model unloaded');
            showToast('Model unloaded', 'info');
            await fetchStatus();
            await fetchSpeakers();
        } catch(e) {
            showError(els['model-error'], 'Failed to unload: ' + e.message);
        }
        btn.disabled = false;
        btn.innerHTML = ICONS.cpu + ' Unload Model';
    });
}

// ==========================================================
// Frontend Model Management
// ==========================================================

function initFrontendLoad() {
    const btn = els['btn-load-frontend'];
    if (!btn) return;
    btn.addEventListener('click', async () => {
        hideError(els['frontend-error']);
        hideSuccess(els['frontend-success']);
        const st = els['cfg-tokenizer'].value.trim();
        const cp = els['cfg-campplus'].value.trim();
        if (!st || !cp) { showError(els['frontend-error'], 'Both Speech Tokenizer and Campplus paths are required'); return; }

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span>Loading...';
        try {
            await apiFetch('/frontend/model/load', { method: 'POST', ...jsonBody({ speech_tokenizer: st, campplus: cp }) });
            showSuccess(els['frontend-success'], 'Frontend ONNX models loaded');
            showToast('Frontend models loaded', 'success');
            await fetchStatus();
        } catch(e) {
            showError(els['frontend-error'], 'Failed to load frontend: ' + e.message);
        }
        btn.disabled = false;
        btn.innerHTML = ICONS.wrench + ' Load Frontend Models';
    });
}

function initFrontendUnload() {
    const btn = els['btn-unload-frontend'];
    if (!btn) return;
    btn.addEventListener('click', async () => {
        hideError(els['frontend-error']);
        hideSuccess(els['frontend-success']);
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span>Unloading...';
        try {
            await apiFetch('/frontend/model/unload', { method: 'POST' });
            showSuccess(els['frontend-success'], 'Frontend models unloaded');
            showToast('Frontend models unloaded', 'info');
            await fetchStatus();
        } catch(e) {
            showError(els['frontend-error'], 'Failed to unload: ' + e.message);
        }
        btn.disabled = false;
        btn.innerHTML = ICONS.wrench + ' Unload Frontend Models';
    });
}

// ==========================================================
// Load GGUF Speaker
// ==========================================================

function initLoadGguf() {
    const btn = els['btn-load-gguf'];
    if (!btn) return;
    btn.addEventListener('click', async () => {
        hideError(els['load-error']);
        hideSuccess(els['load-success']);
        const name = els['load-name'].value.trim();
        const path = els['load-path'].value.trim();
        if (!name) { showError(els['load-error'], 'Please enter a speaker name'); return; }
        if (!path) { showError(els['load-error'], 'Please enter a GGUF file path'); return; }

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span>Loading...';
        try {
            await apiFetch('/speaker', { method: 'POST', ...jsonBody({ type: 'gguf', name: name, path: path }) });
            showSuccess(els['load-success'], 'Speaker "' + name + '" loaded');
            showToast('Speaker "' + name + '" loaded', 'success');
            els['load-name'].value = '';
            els['load-path'].value = '';
            await fetchStatus();
            await fetchSpeakers();
        } catch(e) {
            showError(els['load-error'], 'Load failed: ' + e.message);
        }
        btn.disabled = false;
        btn.innerHTML = ICONS.user + ' Load Speaker';
    });
}

// ==========================================================
// Waveform Editor (Speaker Extraction)
// ==========================================================

function initWaveform(canvas, audioBuffer) {
    const total = audioBuffer.duration;
    const wf = {
        canvas: canvas,
        audioBuffer: audioBuffer,
        startPct: 0,
        endPct: 100,
        zoomStart: 0,
        zoomEnd: 100,
        dragging: null,
        totalDuration: total,
    };
    canvas.width = canvas.clientWidth * 2;
    canvas.height = 100 * 2;
    canvas.style.width = canvas.clientWidth + 'px';
    canvas.style.height = '100px';
    drawWaveform(wf);
    return wf;
}

function drawWaveform(wf, canvasId) {
    const cvs = wf.canvas;
    const ctx = cvs.getContext('2d');
    const W = cvs.width;
    const H = cvs.height;
    const buffer = wf.audioBuffer;
    const isRetina = W !== cvs.clientWidth;

    ctx.clearRect(0, 0, W, H);

    const visStart = wf.zoomStart / 100;
    const visEnd = wf.zoomEnd / 100;
    const visLen = visEnd - visStart;
    if (visLen <= 0) return;

    const totalFrames = buffer.length;
    const startFrame = Math.round(totalFrames * visStart);
    const endFrame = Math.round(totalFrames * visEnd);
    const frameRange = endFrame - startFrame;

    const channelData = buffer.getChannelData(0);
    const midY = H / 2;
    const halfH = H * 0.42;

    // ---- Background ----
    ctx.fillStyle = 'rgba(10, 10, 30, 0.3)';
    ctx.fillRect(0, 0, W, H);

    // ---- Background grid ----
    ctx.strokeStyle = 'rgba(255,255,255,0.025)';
    ctx.lineWidth = 1;
    for (let g = 1; g < 10; g++) {
        const gx = (g / 10) * W;
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }
    for (let g = 1; g < 5; g++) {
        const gy = (g / 4) * H;
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }

    const selStartX = ((wf.startPct / 100) - visStart) / visLen * W;
    const selEndX = ((wf.endPct / 100) - visStart) / visLen * W;

    // ---- Darken area outside selection ----
    if (selStartX > 0) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.25)';
        ctx.fillRect(0, 0, selStartX, H);
    }
    if (selEndX < W) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.25)';
        ctx.fillRect(selEndX, 0, W - selEndX, H);
    }

    // ---- Selection gradient highlight ----
    const selGrad = ctx.createLinearGradient(selStartX, 0, selEndX, 0);
    selGrad.addColorStop(0, 'rgba(139, 124, 247, 0.08)');
    selGrad.addColorStop(0.3, 'rgba(139, 124, 247, 0.18)');
    selGrad.addColorStop(0.7, 'rgba(139, 124, 247, 0.18)');
    selGrad.addColorStop(1, 'rgba(139, 124, 247, 0.08)');
    ctx.fillStyle = selGrad;
    ctx.fillRect(selStartX, 0, selEndX - selStartX, H);

    // ---- Waveform bars (inside selection = brighter) ----
    for (let pass = 0; pass < 2; pass++) {
        ctx.beginPath();
        for (let i = 0; i < W; i += 1) {
            const pct = i / W;
            const idx = startFrame + Math.round(frameRange * pct);
            const idx2 = startFrame + Math.round(frameRange * ((i + 1) / W));
            if (idx >= buffer.length) break;

            let maxVal = 0;
            for (let j = idx; j <= idx2 && j < buffer.length; j++) {
                const v = Math.abs(channelData[j]);
                if (v > maxVal) maxVal = v;
            }

            const barH = Math.max(1, maxVal * halfH);
            const inside = i >= selStartX && i <= selEndX;
            const y1 = midY - barH;
            const y2 = midY + barH;

            if (pass === 0 && !inside) {
                if (i === 0) ctx.moveTo(i, y1);
                else ctx.lineTo(i, y1);
                ctx.moveTo(i, y2);
                ctx.lineTo(i, y2);
            } else if (pass === 1 && inside) {
                if (i === 0) ctx.moveTo(i, y1);
                else ctx.lineTo(i, y1);
                ctx.moveTo(i, y2);
                ctx.lineTo(i, y2);
            }
        }
        if (pass === 0) {
            ctx.strokeStyle = 'rgba(144, 144, 176, 0.35)';
            ctx.lineWidth = 1;
            ctx.stroke();
        } else {
            ctx.strokeStyle = '#8b7cf7';
            ctx.lineWidth = 1.5;
            ctx.shadowColor = 'rgba(139, 124, 247, 0.3)';
            ctx.shadowBlur = 4;
            ctx.stroke();
            ctx.shadowBlur = 0;
        }
    }

    // ---- Center line ----
    ctx.strokeStyle = 'rgba(139, 124, 247, 0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(W, midY);
    ctx.stroke();
    ctx.setLineDash([]);

    // ---- Boundary markers + handles ----
    const total = wf.totalDuration;
    const sTime = total * wf.startPct / 100;
    const eTime = total * wf.endPct / 100;
    const markerScale = isRetina ? 1 : 1;

    function drawMarker(x, color, label) {
        // Vertical line
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.shadowColor = color + '40';
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Top circle handle
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, 8 * markerScale, 5 * markerScale, 0, Math.PI * 2);
        ctx.fill();

        // Bottom circle handle
        ctx.beginPath();
        ctx.arc(x, H - 8 * markerScale, 5 * markerScale, 0, Math.PI * 2);
        ctx.fill();

        // Time label at top
        ctx.fillStyle = color;
        ctx.font = 'bold ' + (10 * markerScale) + 'px ' + getComputedStyle(document.body).fontFamily;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(label + 's', x, 18 * markerScale);
    }

    drawMarker(selStartX, '#ff6b6b', sTime.toFixed(2));
    drawMarker(selEndX, '#51cf66', eTime.toFixed(2));

    // ---- Selection time in center ----
    const selW = selEndX - selStartX;
    if (selW > 80) {
        ctx.fillStyle = 'rgba(139, 124, 247, 0.5)';
        ctx.font = '11px ' + getComputedStyle(document.body).fontFamily;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const selTime = total * (wf.endPct - wf.startPct) / 100;
        ctx.fillText('▎' + formatTime(selTime) + '▎', (selStartX + selEndX) / 2, midY);
    }

    // ---- Update time display ----
    const startEl = document.getElementById('wf-start-time');
    const endEl = document.getElementById('wf-end-time');
    const selEl = document.getElementById('wf-selected');
    const totEl = document.getElementById('wf-total');
    if (startEl) startEl.textContent = sTime.toFixed(2);
    if (endEl) endEl.textContent = eTime.toFixed(2);
    if (selEl) selEl.textContent = (eTime - sTime).toFixed(2);
    if (totEl) totEl.textContent = total.toFixed(2);
}

// ---- Audio File Selection ----
function initExtractAudio() {
    const input = els['extract-audio'];
    if (!input) return;

    input.addEventListener('change', async () => {
        const file = input.files[0];
        if (!file) {
            if (els['trim-area']) els['trim-area'].classList.remove('show');
            waveformState = null;
            return;
        }

        try {
            const arrayBuffer = await file.arrayBuffer();
            const audioCtx = new AudioContext();
            const decoded = await audioCtx.decodeAudioData(arrayBuffer);
            await audioCtx.close();

            if (els['trim-area']) els['trim-area'].classList.add('show');
            await new Promise(r => setTimeout(r, 50));

            const canvas = els['waveform-canvas'];
            if (!canvas) return;
            waveformState = initWaveform(canvas, decoded);
            setupWaveformEvents(waveformState, canvas, 'wf-');
        } catch(e) {
            console.warn('Audio decode failed (trimming disabled):', e);
            if (els['trim-area']) els['trim-area'].classList.remove('show');
        }
    });
}

function setupWaveformEvents(wf, canvas, prefix) {
    function getMousePct(e) {
        const rect = canvas.getBoundingClientRect();
        const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const ws = wf.zoomStart / 100;
        const we = wf.zoomEnd / 100;
        return (ws + x * (we - ws)) * 100;
    }

    function getThreshold() {
        return (wf.zoomEnd - wf.zoomStart) / 100 * 4;
    }

    canvas.onmousedown = (e) => {
        const pct = getMousePct(e);
        const threshold = getThreshold();
        const distStart = Math.abs(pct - wf.startPct);
        const distEnd = Math.abs(pct - wf.endPct);

        if (distEnd < threshold && distEnd <= distStart) {
            wf.dragging = 'end';
        } else if (distStart < threshold) {
            wf.dragging = 'start';
        } else {
            wf.dragging = (pct - wf.startPct < wf.endPct - pct) ? 'start' : 'end';
            if (wf.dragging === 'start') wf.startPct = pct;
            else wf.endPct = pct;
            drawWaveform(wf);
        }
    };

    canvas.onmousemove = (e) => {
        const pct = getMousePct(e);
        const threshold = getThreshold();
        const distStart = Math.abs(pct - wf.startPct);
        const distEnd = Math.abs(pct - wf.endPct);

        if (wf.dragging) {
            canvas.style.cursor = 'ew-resize';
            if (wf.dragging === 'start') {
                wf.startPct = Math.max(wf.zoomStart, Math.min(pct, wf.endPct - 0.5));
            } else {
                wf.endPct = Math.min(wf.zoomEnd, Math.max(pct, wf.startPct + 0.5));
            }
            drawWaveform(wf);
        } else {
            // Cursor feedback based on position
            if (distStart < threshold || distEnd < threshold) {
                canvas.style.cursor = 'ew-resize';
            } else {
                canvas.style.cursor = 'crosshair';
            }
        }
    };

    const onUp = () => { wf.dragging = null; };
    canvas.onmouseup = onUp;
    canvas.onmouseleave = (e) => { wf.dragging = null; canvas.style.cursor = 'crosshair'; };

    // Waveform buttons
    const zoomIn = document.getElementById(prefix + 'zoomin');
    const zoomOut = document.getElementById(prefix + 'zoomout');
    const reset = document.getElementById(prefix + 'reset');
    const preview = document.getElementById(prefix + 'preview');
    const downloadBtn = document.getElementById(prefix + 'download');

    if (zoomIn) zoomIn.addEventListener('click', () => {
        const range = wf.endPct - wf.startPct;
        if (range < 0.5) return;
        wf.zoomStart = wf.startPct;
        wf.zoomEnd = wf.endPct;
        drawWaveform(wf);
    });

    if (zoomOut) zoomOut.addEventListener('click', () => {
        const range = wf.zoomEnd - wf.zoomStart;
        const expand = range * 0.5;
        wf.zoomStart = Math.max(0, wf.zoomStart - expand);
        wf.zoomEnd = Math.min(100, wf.zoomEnd + expand);
        drawWaveform(wf);
    });

    if (reset) reset.addEventListener('click', () => {
        wf.zoomStart = 0;
        wf.zoomEnd = 100;
        drawWaveform(wf);
    });

    if (preview) preview.addEventListener('click', () => {
        if (!wf) return;
        if (previewAudio) { previewAudio.pause(); previewAudio = null; }

        const buffer = wf.audioBuffer;
        const totalFrames = buffer.length;
        const startFrame = Math.round(totalFrames * wf.startPct / 100);
        const endFrame = Math.round(totalFrames * wf.endPct / 100);
        const frames = endFrame - startFrame;
        if (frames <= 0) return;

        const channels = buffer.numberOfChannels;
        const out = new Float32Array(frames * channels);
        for (let ch = 0; ch < channels; ch++) {
            const src = buffer.getChannelData(ch);
            for (let i = 0; i < frames; i++) {
                out[i * channels + ch] = src[startFrame + i];
            }
        }

        const blob = encodeWav(out, buffer.sampleRate, channels);
        const url = URL.createObjectURL(blob);
        previewAudio = new Audio(url);
        previewAudio.play().catch(() => {});

        function animateCursor() {
            if (!previewAudio || previewAudio.paused || previewAudio.ended) {
                drawWaveform(wf);
                return;
            }
            drawWaveform(wf);
            requestAnimationFrame(animateCursor);
        }
        requestAnimationFrame(animateCursor);
    });

    // Download cropped audio
    if (downloadBtn) downloadBtn.addEventListener('click', () => {
        if (!wf) return;
        const buffer = wf.audioBuffer;
        const totalFrames = buffer.length;
        const startFrame = Math.round(totalFrames * wf.startPct / 100);
        const endFrame = Math.round(totalFrames * wf.endPct / 100);
        const frames = endFrame - startFrame;
        if (frames <= 0) { showToast('Selected range is empty', 'warning'); return; }

        const channels = buffer.numberOfChannels;
        const out = new Float32Array(frames * channels);
        for (let ch = 0; ch < channels; ch++) {
            const src = buffer.getChannelData(ch);
            for (let i = 0; i < frames; i++) {
                out[i * channels + ch] = src[startFrame + i];
            }
        }

        const blob = encodeWav(out, buffer.sampleRate, channels);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'cropped_audio.wav';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast('Cropped audio downloaded (' + formatFileSize(blob.size) + ')', 'success');
    });

    // Seek on canvas click during preview
    canvas.addEventListener('click', (e) => {
        if (!wf || !previewAudio || previewAudio.paused) return;
        const rect = canvas.getBoundingClientRect();
        const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const ws = wf.zoomStart / 100;
        const we = wf.zoomEnd / 100;
        const pct = (ws + x * (we - ws)) * 100;
        if (pct < wf.startPct || pct > wf.endPct) return;

        const rangeDur = previewAudio.duration || 0;
        const previewPct = (pct - wf.startPct) / (wf.endPct - wf.startPct);
        previewAudio.currentTime = Math.max(0, Math.min(rangeDur, previewPct * rangeDur));
    });
}

// ---- Crop Audio Buffer ----
function cropAudioBuffer(buffer, startPct, endPct) {
    const totalFrames = buffer.length;
    const channels = buffer.numberOfChannels;
    const startFrame = Math.round(totalFrames * startPct / 100);
    const endFrame = Math.round(totalFrames * endPct / 100);
    const outFrames = endFrame - startFrame;
    if (outFrames <= 0) return buffer;

    const out = new Float32Array(outFrames * channels);
    for (let ch = 0; ch < channels; ch++) {
        const src = buffer.getChannelData(ch);
        for (let i = 0; i < outFrames; i++) {
            out[i * channels + ch] = src[startFrame + i];
        }
    }
    return { data: out, channels: channels, sampleRate: buffer.sampleRate, frames: outFrames };
}

// ---- Encode WAV ----
function encodeWav(pcm, sampleRate, channels) {
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const dataBytes = pcm.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataBytes);
    const view = new DataView(buffer);

    function writeString(offset, str) {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    }

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataBytes, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, channels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * channels * bytesPerSample, true);
    view.setUint16(32, channels * bytesPerSample, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(36, 'data');
    view.setUint32(40, dataBytes, true);

    let offset = 44;
    for (let i = 0; i < pcm.length; i++) {
        let s = pcm[i];
        s = Math.max(-1, Math.min(1, s));
        const val = s < 0 ? s * 0x8000 : s * 0x7FFF;
        view.setInt16(offset, val, true);
        offset += 2;
    }
    return new Blob([buffer], { type: 'audio/wav' });
}

// ---- Extract Speaker ----
function initExtractSpeaker() {
    const btn = els['btn-extract'];
    if (!btn) return;
    btn.addEventListener('click', async () => {
        hideError(els['extract-error']);
        hideSuccess(els['extract-success']);
        const name = els['extract-name'].value.trim();
        const text = els['extract-text'].value.trim();
        const file = els['extract-audio'].files[0];
        if (!name) { showError(els['extract-error'], 'Please enter a speaker name'); return; }
        if (!file && !waveformState) { showError(els['extract-error'], 'Please select an audio file or record from microphone'); return; }

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span>Extracting...';
        try {
            let audioBlob;
            if (waveformState) {
                const cropped = cropAudioBuffer(waveformState.audioBuffer, waveformState.startPct, waveformState.endPct);
                audioBlob = encodeWav(cropped.data, cropped.sampleRate, cropped.channels);
            } else {
                audioBlob = file;
            }

            const fd = new FormData();
            fd.append('name', name);
            fd.append('text', text);
            fd.append('audio', audioBlob, 'reference.wav');
            await apiFetch('/speaker?type=extract', { method: 'POST', body: fd });
            showSuccess(els['extract-success'], 'Speaker "' + name + '" extracted successfully');
            showToast('Speaker "' + name + '" extracted', 'success');
            els['extract-name'].value = '';
            els['extract-text'].value = '';
            els['extract-audio'].value = '';
            waveformState = null;
            if (els['trim-area']) els['trim-area'].classList.remove('show');
            await fetchStatus();
            await fetchSpeakers();
        } catch(e) {
            showError(els['extract-error'], 'Extraction failed: ' + e.message);
        }
        btn.disabled = false;
        btn.innerHTML = ICONS.user + ' Extract Speaker';
    });
}

// ==========================================================
// TTS Generation
// ==========================================================

function initTts() {
    const btn = els['btn-tts'];
    if (!btn) return;

    btn.addEventListener('click', generateTts);

    // Mode helper text
    if (els['tts-mode']) {
        els['tts-mode'].addEventListener('change', () => {
            const mode = els['tts-mode'].value;
            if (els['tts-instructions']) {
                els['tts-instructions'].parentElement.style.display = 'block';
            }
        });
    }
}

async function generateTts() {
    hideError(els['tts-error']);
    const voice = els['tts-voice'].value;
    const text = els['tts-text'].value.trim();
    if (!voice) { showError(els['tts-error'], 'Please select a voice'); return; }
    if (!text) { showError(els['tts-error'], 'Please enter synthesis text'); return; }

    const body = {
        text: text,
        voice: voice,
        mode: els['tts-mode'].value,
        speed: parseFloat(els['tts-speed'].value),
        format: els['tts-format'].value,
        fade_in: els['tts-fadein'] ? els['tts-fadein'].checked : true,
        text_normalization: els['tts-textnorm'] ? els['tts-textnorm'].checked : true,
        split_text: els['tts-split'] ? els['tts-split'].checked : true,
        fast_split: els['tts-fastsplit'] ? els['tts-fastsplit'].checked : true,
    };

    const instr = els['tts-instructions'] ? els['tts-instructions'].value.trim() : '';
    if (instr) body.instructions = instr;

    const adv = [
        ['seed', els['tts-seed']],
        ['temperature', els['tts-temp']],
        ['top_k', els['tts-topk']],
        ['top_p', els['tts-topp']],
        ['win_size', els['tts-winsize']],
        ['tau_r', els['tts-taur']],
    ];
    for (const [k, el] of adv) {
        const v = el.value.trim();
        if (v === '') {
            showError(els['tts-error'], 'Please fill in all advanced parameters (empty: "' + k + '").');
            return;
        }
        body[k] = k === 'seed' || k === 'top_k' || k === 'win_size'
            ? parseInt(v, 10) : parseFloat(v);
    }

    isGenerating = true;
    const btn = els['btn-tts'];
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Generating...';

    try {
        const res = await apiFetch('/tts', { method: 'POST', ...jsonBody(body) });
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        if (els['audio-player']) els['audio-player'].src = url;
        if (els['player-area']) els['player-area'].style.display = 'block';

        const ext = body.format || 'wav';
        if (els['download-link']) {
            els['download-link'].href = url;
            els['download-link'].download = 'cosyvoice_' + voice + '.' + ext;
            els['download-link'].innerHTML = ICONS.download + ' Download ' + ext.toUpperCase() + ' (' + formatFileSize(blob.size) + ')';
        }

        // Auto-play
        if (els['audio-player']) els['audio-player'].play().catch(() => {});

        // Add to history
        addHistory({
            voice: voice,
            text: text,
            blob: blob,
            url: url,
            format: ext,
            mode: els['tts-mode'].value,
            timestamp: new Date(),
        });

        // Auto-roll seed
        if (!seedLocked && els['seed-dice']) els['seed-dice'].click();
    } catch(e) {
        showError(els['tts-error'], 'Synthesis failed: ' + e.message);
        showToast('TTS generation failed', 'error');
    }
    isGenerating = false;
    btn.disabled = false;
    btn.innerHTML = ICONS.speaker + ' Generate Speech';
}

// ==========================================================
// TTS History
// ==========================================================

function addHistory(entry) {
    historyEntries.unshift(entry);
    if (historyEntries.length > MAX_HISTORY) {
        const removed = historyEntries.pop();
        if (removed.url) URL.revokeObjectURL(removed.url);
    }
    renderHistory();
    if (els['history-area']) els['history-area'].style.display = 'block';
}

function renderHistory() {
    const list = els['history-list'];
    if (!list) return;

    if (historyEntries.length === 0) {
        list.innerHTML = '<div class="empty-state">' + ICONS.clock + '<br>No generation history yet</div>';
        return;
    }

    list.innerHTML = historyEntries.map((e, i) => {
        const time = e.timestamp.toLocaleTimeString();
        const preview = e.text.length > 40 ? e.text.substring(0, 40) + '…' : e.text;
        const filesize = formatFileSize(e.blob.size);
        return '<div class="history-item">'
            + '<span class="h-time">' + escapeHtml(time) + '</span>'
            + '<span class="h-voice">' + ICONS.user + ' ' + escapeHtml(e.voice) + '</span>'
            + '<span class="h-text" title="' + escapeHtml(e.text) + '">' + escapeHtml(preview) + '</span>'
            + '<span style="color:var(--text-muted);font-size:11px">' + e.format.toUpperCase() + ' ' + filesize + '</span>'
            + '<span class="h-actions">'
                + '<button class="btn btn-sm" onclick="playHistory(' + i + ')" title="Play">' + ICONS.play_small + '</button>'
                + '<button class="btn btn-sm" onclick="downloadHistory(' + i + ')" title="Download">' + ICONS.download + '</button>'
            + '</span>'
            + '</div>';
    }).join('');
}

function playHistory(index) {
    const entry = historyEntries[index];
    if (!entry || !entry.url) return;
    if (els['audio-player']) {
        els['audio-player'].src = entry.url;
        els['audio-player'].play().catch(() => {});
        if (els['player-area']) els['player-area'].style.display = 'block';
    }
}

function downloadHistory(index) {
    const entry = historyEntries[index];
    if (!entry || !entry.url) return;
    const a = document.createElement('a');
    a.href = entry.url;
    a.download = 'cosyvoice_' + entry.voice + '_history.' + entry.format;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function clearHistory() {
    historyEntries.forEach(e => { if (e.url) URL.revokeObjectURL(e.url); });
    historyEntries = [];
    renderHistory();
    if (els['history-area']) els['history-area'].style.display = 'none';
}

// ==========================================================
// Inline Recording (inside Extract tab)
// ==========================================================

function initExtractRecording() {
    const btn = els['extract-record-btn'];
    if (!btn) return;

    btn.addEventListener('click', () => {
        if (!extractRecState) {
            startExtractRecording();
        } else {
            stopExtractRecording();
        }
    });
}

async function startExtractRecording() {
    // Check support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === 'undefined') {
        showToast('Recording not supported in this browser', 'error');
        return;
    }

    // Update UI
    const btn = els['extract-record-btn'];
    const status = els['erm-status'];
    if (btn) btn.classList.add('recording');
    if (status) { status.textContent = 'Recording...'; status.className = 'erm-status recording'; }

    // Clear any previous recording
    if (els['erm-waveform']) els['erm-waveform'].style.display = 'none';

    // Clear file input if previously set
    if (els['extract-audio']) els['extract-audio'].value = '';

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 }
        });

        const audioCtx = new AudioContext();
        const source = audioCtx.createMediaStreamSource(stream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);
        const dataArray = new Uint8Array(analyser.frequencyBinCount);

        const mimeType = getBestMimeType();
        const mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
        const chunks = [];
        const startTime = Date.now();

        mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };

        mediaRecorder.onstop = () => {
            if (extractRecState && extractRecState.animId) {
                cancelAnimationFrame(extractRecState.animId);
            }
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(chunks, { type: mimeType || 'audio/webm' });

            // Update UI
            if (status) { status.textContent = 'Processing audio...'; status.className = 'erm-status'; }

            // Decode and feed into trim area
            decodeAndShowTrim(blob);
            extractRecState = null;
            if (audioCtx.state !== 'closed') audioCtx.close();
        };

        mediaRecorder.start(100);

        extractRecState = {
            mediaRecorder, chunks, startTime, stream, analyser, dataArray, audioCtx, source, animId: null
        };

        // Show live waveform
        if (els['erm-waveform']) els['erm-waveform'].style.display = 'block';
        drawExtractLiveWaveform();

    } catch (e) {
        if (btn) btn.classList.remove('recording');
        if (status) { status.textContent = 'Microphone unavailable'; status.className = 'erm-status'; }
        showToast('Recording failed: ' + (e.name === 'NotAllowedError' ? 'microphone access denied' : e.message), 'error');
    }
}

function stopExtractRecording() {
    if (!extractRecState) return;
    const rs = extractRecState;
    if (els['erm-status']) els['erm-status'].textContent = 'Processing...';

    try {
        if (rs.mediaRecorder && rs.mediaRecorder.state !== 'inactive') rs.mediaRecorder.stop();
    } catch(e) { /* ignore */ }
    if (rs.stream) rs.stream.getTracks().forEach(t => t.stop());
}

function getBestMimeType() {
    const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4', ''];
    for (const t of types) {
        if (!t || MediaRecorder.isTypeSupported(t)) return t || '';
    }
    return '';
}

function decodeAndShowTrim(blob) {
    const reader = new FileReader();
    reader.onload = async () => {
        try {
            const audioCtx = new AudioContext();
            const decoded = await audioCtx.decodeAudioData(reader.result);
            await audioCtx.close();

            // Set up trim area with recorded audio (same as file upload flow)
            if (els['trim-area']) els['trim-area'].classList.add('show');
            await new Promise(r => setTimeout(r, 50));

            const canvas = els['waveform-canvas'];
            if (!canvas) return;

            waveformState = initWaveform(canvas, decoded);
            setupWaveformEvents(waveformState, canvas, 'wf-');

            // Update status
            if (els['erm-status']) {
                els['erm-status'].textContent = 'Recorded ' + formatTime(decoded.duration) + ' — trim below or extract';
                els['erm-status'].className = 'erm-status';
            }
            const btn = els['extract-record-btn'];
            if (btn) btn.classList.remove('recording');

            // Reset timer
            if (els['erm-timer']) { els['erm-timer'].textContent = ''; els['erm-timer'].className = 'erm-timer'; }

            showToast('Recorded ' + formatTime(decoded.duration), 'success');

        } catch(e) {
            showToast('Failed to process recorded audio', 'error');
        }
    };
    reader.readAsArrayBuffer(blob);
}

function drawExtractLiveWaveform() {
    if (!extractRecState) return;

    const canvas = els['erm-canvas'];
    const levelBar = els['erm-level'];
    if (!canvas) return;

    canvas.width = canvas.clientWidth * 2;
    canvas.height = 60 * 2;
    canvas.style.width = canvas.clientWidth + 'px';
    canvas.style.height = '60px';

    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const { analyser, dataArray, startTime } = extractRecState;

    function animate() {
        if (!extractRecState) return;
        extractRecState.animId = requestAnimationFrame(animate);

        // Timer
        const elapsed = (Date.now() - startTime) / 1000;
        const mins = Math.floor(elapsed / 60);
        const secs = Math.floor(elapsed % 60);
        if (els['erm-timer']) {
            els['erm-timer'].textContent = String(mins).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
            els['erm-timer'].className = 'erm-timer recording';
        }

        // Frequency data
        analyser.getByteFrequencyData(dataArray);

        // Level
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
        const avg = sum / dataArray.length;
        const level = Math.min(100, (avg / 255) * 100);
        if (levelBar) levelBar.style.width = level + '%';

        // Draw bars
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = 'rgba(10, 10, 30, 0.4)';
        ctx.fillRect(0, 0, W, H);

        const barCount = 48;
        const barW = W / barCount;
        const midY = H / 2;

        for (let i = 0; i < barCount; i++) {
            const idx = Math.floor((i / barCount) * dataArray.length * 0.8) + 1;
            const val = dataArray[idx] || 0;
            const pct = val / 255;
            const barH = Math.max(1, pct * H * 0.8);

            let color;
            if (pct < 0.3) color = 'rgba(139, 124, 247, ' + (0.4 + pct * 1.5) + ')';
            else if (pct < 0.6) color = 'rgba(0, 206, 201, ' + (0.5 + (pct - 0.3) * 1.5) + ')';
            else color = 'rgba(232, 67, 147, ' + (0.5 + (pct - 0.6) * 1.5) + ')';

            ctx.fillStyle = color;
            ctx.fillRect(i * barW + 1, midY - barH / 2, barW - 2, barH);
        }
    }
    animate();
}

// ==========================================================
// Keyboard Shortcuts
// ==========================================================

document.addEventListener('keydown', (e) => {
    // Ctrl+Enter / Cmd+Enter → Generate TTS
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const btn = els['btn-tts'];
        if (btn && !btn.disabled) {
            e.preventDefault();
            generateTts();
        }
    }

    // Escape → Clear error messages
    if (e.key === 'Escape') {
        document.querySelectorAll('.msg-box.show').forEach(el => {
            el.classList.remove('show');
            el.innerHTML = '';
        });
    }
});

// ==========================================================
// Init Helpers / Seed / Defaults
// ==========================================================

function initSeedControls() {
    const dice = document.getElementById('seed-dice');
    const lock = document.getElementById('seed-lock');
    if (dice) {
        dice.addEventListener('click', () => {
            const input = document.getElementById('tts-seed');
            if (input) input.value = Math.floor(Math.random() * 2147483647) + 1;
        });
    }
    if (lock) {
        lock.addEventListener('click', () => {
            seedLocked = !seedLocked;
            lock.textContent = seedLocked ? '🔒' : '🔓';
            if (seedLocked) {
                const input = document.getElementById('tts-seed');
                if (input && !input.value && dice) dice.click();
            }
        });
    }
}

function initResetButtons() {
    const resetModel = document.getElementById('btn-reset-model-config');
    if (resetModel) {
        resetModel.addEventListener('click', () => {
            clearAdvParams();
            fetchDefaults().then(() => {
                restoreAdvParams();
                const input = document.getElementById('tts-seed');
                const dice = document.getElementById('seed-dice');
                if (input && !input.value && dice) dice.click();
            });
        });
    }

    const resetGen = document.getElementById('btn-reset-gen-config');
    if (resetGen) {
        resetGen.addEventListener('click', () => {
            ADV_PARAM_IDS.forEach(id => {
                if (id.startsWith('tts') || id === 'modelMaxLlm') localStorage.removeItem('cosyvoice_' + id);
            });
            fetchDefaults().then(() => updateFastSplitState());
        });
    }

    const clearHist = document.getElementById('btn-clear-history');
    if (clearHist) clearHist.addEventListener('click', clearHistory);

    const splitCheck = document.getElementById('tts-split');
    if (splitCheck) splitCheck.addEventListener('change', updateFastSplitState);
}

// ---- Fetch Defaults ----
async function fetchDefaults() {
    try {
        const res = await apiFetch('/model/defaults');
        const d = await res.json();

        const r = (v, n) => v !== undefined ? parseFloat(Number(v).toFixed(n)) : undefined;

        if (d.default_max_llm_len) els['model-max-llm'].value = d.default_max_llm_len;
        if (d.max_llm_len) els['model-max-llm'].value = d.max_llm_len;
        if (d.default_k_cache_type) {
            const kSel = els['model-kv-k'];
            for (let i = 0; i < kSel.options.length; i++) {
                if (kSel.options[i].value === d.default_k_cache_type) { kSel.selectedIndex = i; break; }
            }
        }
        if (d.default_v_cache_type) {
            const vSel = els['model-kv-v'];
            for (let i = 0; i < vSel.options.length; i++) {
                if (vSel.options[i].value === d.default_v_cache_type) { vSel.selectedIndex = i; break; }
            }
        }
        els['model-threads'].value = d.default_n_threads !== undefined ? d.default_n_threads : 0;

        if (d.temperature !== undefined) els['tts-temp'].value = r(d.temperature, 6);
        if (d.top_k !== undefined) els['tts-topk'].value = d.top_k;
        if (d.top_p !== undefined) els['tts-topp'].value = r(d.top_p, 6);
        if (d.win_size !== undefined) els['tts-winsize'].value = d.win_size;
        if (d.tau_r !== undefined) els['tts-taur'].value = r(d.tau_r, 6);

        if (d.text_normalization !== undefined) els['tts-textnorm'].checked = d.text_normalization;
        if (d.split_text !== undefined) els['tts-split'].checked = d.split_text;
        if (d.fast_split !== undefined) els['tts-fastsplit'].checked = d.fast_split;
        if (d.fade_in !== undefined) els['tts-fadein'].checked = d.fade_in;

        if (!els['tts-seed'].value && els['seed-dice']) {
            els['seed-dice'].click();
        }
    } catch(e) {
        console.warn('Failed to fetch defaults:', e);
    }
}

function updateFastSplitState() {
    if (els['tts-fastsplit'] && els['tts-split']) {
        els['tts-fastsplit'].disabled = !els['tts-split'].checked;
        if (!els['tts-split'].checked) els['tts-fastsplit'].checked = false;
    }
}

// ---- Fetch Backends ----
async function fetchBackends() {
    try {
        const res = await apiFetch('/backends');
        const list = await res.json();
        const sel = els['model-backend'];
        if (!sel) return;
        while (sel.options.length > 1) sel.remove(1);
        for (const b of list) {
            if (b.name === 'auto') continue;
            const opt = document.createElement('option');
            opt.value = b.name;
            opt.textContent = b.description
                ? b.name + ' — ' + b.description
                : b.name;
            sel.appendChild(opt);
        }
    } catch(e) {
        console.warn('Failed to fetch backends:', e);
    }
}

// ---- Fetch Formats ----
async function fetchFormats() {
    try {
        const res = await apiFetch('/formats');
        const list = await res.json();
        const sel = els['tts-format'];
        if (!sel) return;
        sel.innerHTML = '';
        list.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f;
            opt.textContent = f.toUpperCase();
            if (f === 'wav' || f === 'mp3') opt.selected = true;
            sel.appendChild(opt);
        });
    } catch(e) {
        console.warn('Failed to fetch formats:', e);
    }
}

// ==========================================================
// Init
// ==========================================================

async function init() {
    initEls();

    if (CFG.speechTokenizer && els['cfg-tokenizer']) els['cfg-tokenizer'].value = CFG.speechTokenizer;
    if (CFG.campplus && els['cfg-campplus']) els['cfg-campplus'].value = CFG.campplus;
    if (CFG.defaultMaxLlmLen && els['model-max-llm']) els['model-max-llm'].value = CFG.defaultMaxLlmLen;

    // Init all modules
    initModelLoad();
    initModelUnload();
    initFrontendLoad();
    initFrontendUnload();
    initLoadGguf();
    initExtractAudio();
    initExtractSpeaker();
    initExtractRecording();
    initTts();
    initSpeedDisplay();
    initSeedControls();
    initResetButtons();

    // Fetch data
    await fetchBackends();
    await fetchFormats();
    await fetchDefaults();
    restoreAdvParams();
    updateFastSplitState();
    await fetchStatus();
    await fetchSpeakers();

    // Track changes on advanced param elements
    ADV_PARAM_IDS.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        const evt = el.type === 'checkbox' ? 'change' : 'input';
        el.addEventListener(evt, () => saveAdvParam(id));
    });
}

// Wait for DOM
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => { try { init(); } catch(e) { console.error('Init error:', e); } });
} else {
    try { init(); } catch(e) { console.error('Init error:', e); }
}
