/**
 * TruthLens AI — Frontend Application Logic
 *
 * Handles:
 *   - API communication with /analyze endpoint
 *   - Dynamic result rendering
 *   - Score ring animations
 *   - Server health polling
 *   - Error handling
 */

// ── Configuration ──────────────────────────────────────────────────────
const API_BASE = window.location.origin;
const ANALYZE_ENDPOINT = `${API_BASE}/analyze`;
const HEALTH_ENDPOINT = `${API_BASE}/health`;

// ── DOM References ─────────────────────────────────────────────────────
const claimInput = document.getElementById('claimInput');
const charCount = document.getElementById('charCount');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const errorBanner = document.getElementById('errorBanner');
const errorText = document.getElementById('errorText');
const serverStatus = document.getElementById('serverStatus');

// ── State ──────────────────────────────────────────────────────────────
let isAnalyzing = false;
let currentResult = null;

// ── Character Counter ──────────────────────────────────────────────────
claimInput.addEventListener('input', () => {
    const len = claimInput.value.length;
    charCount.textContent = `${len.toLocaleString()} / 10,000`;

    if (len > 9500) {
        charCount.style.color = 'var(--accent-danger)';
    } else if (len > 8000) {
        charCount.style.color = 'var(--accent-warning)';
    } else {
        charCount.style.color = 'var(--text-muted)';
    }
});

// ── Keyboard shortcut (Ctrl+Enter / Cmd+Enter) ────────────────────────
claimInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        analyzeText();
    }
});

// ── Example Chip Handler ───────────────────────────────────────────────
function setExample(btn) {
    const text = btn.getAttribute('data-text');
    claimInput.value = text;
    claimInput.dispatchEvent(new Event('input'));
    claimInput.focus();

    // Visual feedback
    btn.style.background = 'rgba(99, 102, 241, 0.15)';
    btn.style.borderColor = 'var(--accent-primary)';
    setTimeout(() => {
        btn.style.background = '';
        btn.style.borderColor = '';
    }, 300);
}

// ── Main Analysis Function ─────────────────────────────────────────────
async function analyzeText() {
    const text = claimInput.value.trim();

    if (!text) {
        showError('Please enter a claim or news text to analyze.');
        claimInput.focus();
        return;
    }

    if (text.length < 3) {
        showError('Text is too short. Please enter a meaningful claim.');
        return;
    }

    if (isAnalyzing) return;

    // UI → Loading state
    isAnalyzing = true;
    analyzeBtn.classList.add('loading');
    analyzeBtn.disabled = true;
    hideError();
    resultsSection.style.display = 'none';

    try {
        const response = await fetch(ANALYZE_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        if (response.status === 429) {
            const retryAfter = response.headers.get('Retry-After') || '60';
            showError(`Rate limit exceeded. Please wait ${retryAfter} seconds before trying again.`);
            return;
        }

        if (response.status === 422) {
            const err = await response.json();
            showError('Invalid input: ' + (err.detail?.[0]?.msg || 'Please check your input.'));
            return;
        }

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            showError(err.detail || `Server error (${response.status}). Please try again.`);
            return;
        }

        const data = await response.json();
        currentResult = data;
        renderResults(data);

    } catch (err) {
        console.error('Analysis error:', err);
        if (err.name === 'TypeError' && err.message.includes('fetch')) {
            showError('Cannot reach the server. Please ensure the API is running at ' + API_BASE);
        } else {
            showError('An unexpected error occurred. Please try again.');
        }
    } finally {
        isAnalyzing = false;
        analyzeBtn.classList.remove('loading');
        analyzeBtn.disabled = false;
    }
}

// ── Result Renderer ────────────────────────────────────────────────────
function renderResults(data) {
    // ── Verdict Banner ──────────────────────────────────────────
    const banner = document.getElementById('verdictBanner');
    banner.className = 'verdict-banner ' + data.label.toLowerCase();

    const icons = { Fake: '✗', Real: '✓', Misleading: '⚠' };
    document.getElementById('verdictIcon').textContent = icons[data.label] || '?';
    document.getElementById('verdictLabel').textContent = data.label.toUpperCase();
    document.getElementById('verdictText').textContent =
        data.label === 'Fake'
            ? 'This claim has been identified as fake based on misinformation analysis.'
            : data.label === 'Real'
            ? 'This claim appears credible based on available evidence.'
            : 'This claim is partially true but contains misleading elements.';

    // ── Score Rings ─────────────────────────────────────────────
    animateRing('trustRingFill', 'trustValue', data.trust_score, getScoreColor(data.trust_score));
    animateRing('confidenceRingFill', 'confidenceValue', data.confidence, getConfidenceColor(data.confidence));

    // ── Claim & Sub-claims ──────────────────────────────────────
    document.getElementById('langBadge').textContent = data.language.toUpperCase();
    document.getElementById('extractedClaim').textContent = data.claim;

    const subclaimsList = document.getElementById('subclaimsList');
    subclaimsList.innerHTML = data.sub_claims
        .map((sc) => {
            const statusClass = sc.status.toLowerCase();
            const statusIcon = sc.status === 'True' ? '✓' : sc.status === 'False' ? '✗' : '⚠';
            return `
                <div class="subclaim-item">
                    <span class="subclaim-status ${statusClass}">${statusIcon} ${sc.status}</span>
                    <span class="subclaim-text">${escapeHtml(sc.claim)}</span>
                </div>
            `;
        })
        .join('');

    // ── Explanation ─────────────────────────────────────────────
    document.getElementById('explanationText').textContent = data.explanation;

    // ── Reasoning ───────────────────────────────────────────────
    const reasoningList = document.getElementById('reasoningList');
    reasoningList.innerHTML = data.reasoning
        .map((r) => `<li class="reasoning-item">${escapeHtml(r)}</li>`)
        .join('');

    // ── Fact Sources ────────────────────────────────────────────
    const sourcesList = document.getElementById('sourcesList');
    sourcesList.innerHTML = data.fact_sources
        .map(
            (src) => `
            <div class="source-item">
                <span class="source-trust ${src.trust}">${src.trust}</span>
                <div class="source-info">
                    <div class="source-title">${escapeHtml(src.title)}</div>
                    <div class="source-url">${escapeHtml(src.source)}</div>
                </div>
            </div>
        `
        )
        .join('');

    // ── Bias & XAI ──────────────────────────────────────────────
    const biasIndicators = document.getElementById('biasIndicators');
    const biasScore = data.xai.bias_score;
    const biasColor = biasScore > 0.5 ? 'var(--accent-danger)' : biasScore > 0.2 ? 'var(--accent-warning)' : 'var(--accent-success)';

    biasIndicators.innerHTML = `
        <div class="bias-row">
            <span class="bias-label">Bias Detected</span>
            <span class="bias-value ${data.bias_detected ? 'detected' : 'clear'}">
                ${data.bias_detected ? '● Yes' : '○ No'}
            </span>
        </div>
        <div class="bias-row">
            <span class="bias-label">Bias Score</span>
            <div class="bias-bar-track">
                <div class="bias-bar-fill" style="width: ${biasScore * 100}%; background: ${biasColor};"></div>
            </div>
            <span class="bias-value" style="color: ${biasColor};">${(biasScore * 100).toFixed(1)}%</span>
        </div>
        <div class="bias-row">
            <span class="bias-label">Evidence Available</span>
            <span class="bias-value ${data.xai.lack_of_evidence ? 'detected' : 'clear'}">
                ${data.xai.lack_of_evidence ? '● Lacking' : '○ Present'}
            </span>
        </div>
    `;

    const keywordsCloud = document.getElementById('keywordsCloud');
    keywordsCloud.innerHTML = data.xai.keywords_detected
        .map((kw, i) => `<span class="keyword-tag" style="animation-delay: ${i * 0.05}s">${escapeHtml(kw)}</span>`)
        .join('');

    // ── Evidence ────────────────────────────────────────────────
    const evidenceImage = document.getElementById('evidenceImage');
    evidenceImage.innerHTML = `
        <div class="evidence-image-label">🖼 Image Evidence Suggestion</div>
        <div class="evidence-image-text">${escapeHtml(data.evidence.image_explanation)}</div>
    `;

    const evidenceQueries = document.getElementById('evidenceQueries');
    evidenceQueries.innerHTML = `
        <div class="evidence-queries-label">🔍 Verification Search Queries</div>
        ${data.evidence.video_search_queries
            .map(
                (q) => `
            <a class="query-item" href="https://www.youtube.com/results?search_query=${encodeURIComponent(q)}" target="_blank" rel="noopener noreferrer">
                <span class="query-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                </span>
                ${escapeHtml(q)}
            </a>
        `
            )
            .join('')}
    `;

    // ── Final Verdict ───────────────────────────────────────────
    const sealIcons = { Fake: '🚫', Real: '✅', Misleading: '⚠️' };
    document.getElementById('verdictSeal').textContent = sealIcons[data.label] || '🔍';
    document.getElementById('verdictStatement').textContent = data.final_verdict;

    // ── JSON Output ─────────────────────────────────────────────
    document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
    document.getElementById('jsonOutput').style.display = 'none';
    document.getElementById('jsonToggleBtn').querySelector('span').textContent = 'View Raw JSON';

    // ── Show Results ────────────────────────────────────────────
    resultsSection.style.display = 'block';

    // Smooth scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// ── Ring Animation ─────────────────────────────────────────────────────
function animateRing(fillId, valueId, targetValue, color) {
    const circumference = 2 * Math.PI * 42; // r=42
    const offset = circumference - (targetValue / 100) * circumference;

    const fillEl = document.getElementById(fillId);
    const valueEl = document.getElementById(valueId);

    fillEl.style.stroke = color;
    fillEl.style.strokeDasharray = circumference;

    // Reset then animate
    fillEl.style.transition = 'none';
    fillEl.style.strokeDashoffset = circumference;

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            fillEl.style.transition = 'stroke-dashoffset 1.5s cubic-bezier(0.4, 0, 0.2, 1)';
            fillEl.style.strokeDashoffset = offset;
        });
    });

    // Animate number
    animateCounter(valueEl, 0, targetValue, 1200);
}

function animateCounter(element, start, end, duration) {
    const startTime = performance.now();

    function tick(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = Math.round(start + (end - start) * eased);
        element.textContent = current;

        if (progress < 1) {
            requestAnimationFrame(tick);
        }
    }

    requestAnimationFrame(tick);
}

// ── Color Utilities ────────────────────────────────────────────────────
function getScoreColor(score) {
    if (score >= 80) return '#10b981'; // Success green
    if (score >= 50) return '#f59e0b'; // Warning amber
    return '#ef4444'; // Danger red
}

function getConfidenceColor(conf) {
    if (conf >= 70) return '#6366f1'; // Brand indigo
    if (conf >= 40) return '#06b6d4'; // Cyan
    return '#a855f7'; // Purple
}

// ── JSON Toggle ────────────────────────────────────────────────────────
function toggleJSON() {
    const output = document.getElementById('jsonOutput');
    const btn = document.getElementById('jsonToggleBtn');
    const isVisible = output.style.display !== 'none';

    output.style.display = isVisible ? 'none' : 'block';
    btn.querySelector('span').textContent = isVisible ? 'View Raw JSON' : 'Hide Raw JSON';
}

// ── Error Handling ─────────────────────────────────────────────────────
function showError(message) {
    errorText.textContent = message;
    errorBanner.style.display = 'flex';

    setTimeout(() => {
        errorBanner.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 50);
}

function hideError() {
    errorBanner.style.display = 'none';
}

// ── HTML Escaping ──────────────────────────────────────────────────────
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ── Server Health Polling ──────────────────────────────────────────────
async function checkServerHealth() {
    const dot = serverStatus.querySelector('.status-dot');
    const text = serverStatus.querySelector('.status-text');

    try {
        const resp = await fetch(HEALTH_ENDPOINT, { signal: AbortSignal.timeout(3000) });
        if (resp.ok) {
            dot.classList.remove('disconnected');
            text.textContent = 'Connected';
        } else {
            dot.classList.add('disconnected');
            text.textContent = 'Degraded';
        }
    } catch {
        dot.classList.add('disconnected');
        text.textContent = 'Offline';
    }
}

// Poll every 15 seconds
checkServerHealth();
setInterval(checkServerHealth, 15000);
