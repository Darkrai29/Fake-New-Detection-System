/**
 * Fake News Detection & Linguistic Analysis System - Client JavaScript
 * Handles metric charts, confusion matrices, live API prediction,
 * sample loader, and linguistic report rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
    initInputCounters();
    initSampleButtons();
    initPredictionForm();
});

// Curated benchmark samples from the dataset for 1-click testing
const SAMPLES = {
    fake1: {
        title: "SHOCKING BOMBSHELL: Secret FBI Memos Expose Massive Media Conspiracy To Rig General Election",
        text: "You will not believe what was just leaked from deep state insiders! Explosive undercover documents reveal an unprecedented conspiracy orchestrated by corrupt mainstream media moguls and political elites to deceive the American public. Uncovered emails demonstrate that corporate journalists were secretly colluding with campaign chiefs to fabricate poll figures, censor dissident voices, and manipulate election results. Wake up people, the entire system is completely rigged and they are terrified that the smoking gun has finally been exposed to the world!"
    },
    fake2: {
        title: "CONFIRMED HOAX: Top Scientists Busted Admitting Climate Data Was Completely Fabricated For Billions In Subsidies",
        text: "A devastating scandal has rocked the global scientific establishment after whistleblowers uncovered confidential audio recordings proving researchers knowingly faked climate disaster projections. According to insider transcripts, researchers deliberately altered temperature readings to trigger public panic and secure trillions in government grants. Corporate news networks have implemented a total blackout on this horrifying revelation, desperate to prevent citizens from discovering the truth about this catastrophic deception."
    },
    real1: {
        title: "Kerry to visit Paris in gesture of sympathy after attacks",
        text: "WASHINGTON (Reuters) - U.S. Secretary of State John Kerry will travel to Paris this week in a gesture of sympathy and solidarity with France following last week's deadly Islamist attacks, the State Department announced on Monday. Kerry, who was criticized in some media for not attending a massive solidarity march in Paris on Sunday, will hold talks with French Foreign Minister Laurent Fabius and President Francois Hollande on Friday, department spokesman John Kirby confirmed during a press briefing. Kirby stated that the United States remains firmly committed to working with European allies to counter violent extremism and enhance intelligence sharing across international borders."
    },
    real2: {
        title: "Federal Reserve signals steady interest rates amid balanced economic growth",
        text: "WASHINGTON (Associated Press) - Federal Reserve officials indicated Wednesday that benchmark interest rates are likely to remain unchanged in the near term as policymakers assess the pace of domestic job growth and moderate inflation. According to the minutes released from the central bank's policy committee meeting, representatives agreed that while consumer spending has shown consistent strength, global market uncertainties warrant a patient monetary approach. Central bank Chairman confirmed in a press conference that future adjustments will depend on comprehensive economic data and labor market reports."
    }
};

/**
 * 1. Textarea Word & Char Counter
 */
function initInputCounters() {
    const textarea = document.getElementById('articleText');
    const counter = document.getElementById('charWordCounter');
    if (!textarea || !counter) return;

    function update() {
        const text = textarea.value.trim();
        const chars = textarea.value.length;
        const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
        counter.textContent = `${words} words • ${chars} chars`;
    }

    textarea.addEventListener('input', update);
}

/**
 * 4. Sample Loaders
 */
function initSampleButtons() {
    const titleInput = document.getElementById('articleTitle');
    const textArea = document.getElementById('articleText');

    function loadSample(key) {
        if (!SAMPLES[key]) return;
        titleInput.value = SAMPLES[key].title;
        textArea.value = SAMPLES[key].text;
        textArea.dispatchEvent(new Event('input'));
    }

    document.getElementById('btnSampleFake1')?.addEventListener('click', () => loadSample('fake1'));
    document.getElementById('btnSampleFake2')?.addEventListener('click', () => loadSample('fake2'));
    document.getElementById('btnSampleReal1')?.addEventListener('click', () => loadSample('real1'));
    document.getElementById('btnSampleReal2')?.addEventListener('click', () => loadSample('real2'));

    document.getElementById('btnClear')?.addEventListener('click', () => {
        titleInput.value = '';
        textArea.value = '';
        textArea.dispatchEvent(new Event('input'));
        document.getElementById('resultsSection').style.display = 'none';
    });
}

/**
 * 5. Handle Live Prediction & Linguistic Report
 */
function initPredictionForm() {
    const form = document.getElementById('predictionForm');
    const btnSubmit = document.getElementById('btnSubmit');
    const resultsSection = document.getElementById('resultsSection');

    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const title = document.getElementById('articleTitle').value.trim();
        const text = document.getElementById('articleText').value.trim();
        const model = document.getElementById('modelSelect').value;

        if (!text && !title) {
            alert('Please enter an article title or body text.');
            return;
        }

        // Loading state
        const originalBtnHtml = btnSubmit.innerHTML;
        btnSubmit.disabled = true;
        btnSubmit.innerHTML = `
            <svg class="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
                <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"></path>
            </svg>
            Analyzing Linguistics...
        `;

        try {
            const resp = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title, text, model })
            });

            const data = await resp.json();

            if (data.status === 'success') {
                renderResults(data);
                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
                alert(`Error: ${data.message}`);
            }
        } catch (err) {
            alert(`Network or Server error: ${err.message}`);
        } finally {
            btnSubmit.disabled = false;
            btnSubmit.innerHTML = originalBtnHtml;
        }
    });
}

/**
 * 6. Render Full Prediction and Linguistic Analysis UI
 */
function renderResults(res) {
    const pred = res.prediction;
    const ling = res.linguistic_analysis;
    const consensus = res.multi_model_consensus;
    const tokens = res.salient_tokens || [];

    const isFake = pred.label === 'FAKE NEWS';

    // 1. Main Verdict Card
    const verdictCard = document.getElementById('mainVerdictCard');
    const verdictBadge = document.getElementById('verdictBadge');
    const evaluatedModelBadge = document.getElementById('evaluatedModelBadge');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');
    const probFakeVal = document.getElementById('probFakeVal');
    const probGenuineVal = document.getElementById('probGenuineVal');

    verdictCard.className = `verdict-card panel ${isFake ? 'verdict-fake' : 'verdict-genuine'}`;
    verdictBadge.className = `verdict-badge ${isFake ? 'fake' : 'genuine'}`;
    verdictBadge.textContent = isFake ? '🚨 FAKE NEWS' : '✅ GENUINE NEWS';

    evaluatedModelBadge.textContent = `Evaluated with: ${pred.model_used} ${pred.is_best_model ? '(👑 Best Model)' : ''}`;
    confidenceValue.textContent = `${pred.confidence_percent}%`;

    confidenceBar.className = `progress-bar-fill ${isFake ? 'fill-fake' : 'fill-genuine'}`;
    confidenceBar.style.width = `${pred.confidence_percent}%`;

    probFakeVal.textContent = `${pred.prob_fake_percent}%`;
    probGenuineVal.textContent = `${pred.prob_genuine_percent}%`;

    // 2. Multi-Model Consensus
    const consensusList = document.getElementById('consensusList');
    consensusList.innerHTML = '';
    for (const [mName, mInfo] of Object.entries(consensus)) {
        const isModelFake = mInfo.label === 'FAKE NEWS';
        const row = document.createElement('div');
        row.className = `consensus-row ${mInfo.is_best ? 'is-best-row' : ''}`;
        row.innerHTML = `
            <div class="consensus-model-name">
                <span>${mName}</span>
                ${mInfo.is_best ? '<span class="badge badge-accent" style="font-size:0.65rem; padding:2px 6px;">Best Model</span>' : ''}
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:0.75rem; color:#94a3b8; font-family:'JetBrains Mono'">${mInfo.confidence}% conf</span>
                <span class="consensus-tag ${isModelFake ? 'tag-fake' : 'tag-genuine'}">${mInfo.label}</span>
            </div>
        `;
        consensusList.appendChild(row);
    }

    // 3. Linguistic Affective Indicators
    document.getElementById('sensationalScore').textContent = `${ling.sensationalism_score} / 100`;
    document.getElementById('sensationalBar').style.width = `${ling.sensationalism_score}%`;

    const sensContainer = document.getElementById('sensationalWordsContainer');
    sensContainer.innerHTML = '';
    if (ling.sensational_words_found && ling.sensational_words_found.length > 0) {
        ling.sensational_words_found.forEach(word => {
            const chip = document.createElement('span');
            chip.className = 'keyword-chip chip-sensational';
            chip.textContent = `⚠ ${word}`;
            sensContainer.appendChild(chip);
        });
    } else {
        sensContainer.innerHTML = '<span class="text-muted text-xs">No sensational triggers detected</span>';
    }

    document.getElementById('attributionScore').textContent = `${ling.attribution_score} / 100`;
    document.getElementById('attributionBar').style.width = `${ling.attribution_score}%`;

    const attrContainer = document.getElementById('attributionWordsContainer');
    attrContainer.innerHTML = '';
    if (ling.attribution_words_found && ling.attribution_words_found.length > 0) {
        ling.attribution_words_found.forEach(word => {
            const chip = document.createElement('span');
            chip.className = 'keyword-chip chip-attribution';
            chip.textContent = `✓ ${word}`;
            attrContainer.appendChild(chip);
        });
    } else {
        attrContainer.innerHTML = '<span class="text-muted text-xs">No journalistic attribution keywords detected</span>';
    }

    // 4. Stylistic Stats Grid
    document.getElementById('statAllCaps').textContent = ling.all_caps_count;
    document.getElementById('statExclamations').textContent = ling.exclamation_count;
    document.getElementById('statPronouns').textContent = `${ling.personal_pronoun_ratio}%`;
    document.getElementById('statDiversity').textContent = `${ling.lexical_diversity}%`;

    // 5. Bullets
    const bulletsContainer = document.getElementById('linguisticBullets');
    bulletsContainer.innerHTML = '';
    if (ling.summary_bullet_points) {
        ling.summary_bullet_points.forEach(bullet => {
            const item = document.createElement('div');
            item.className = 'bullet-item';
            item.innerHTML = `<span class="bullet-dot">›</span><span>${bullet}</span>`;
            bulletsContainer.appendChild(item);
        });
    }

    // 6. Salient Tokens (TF-IDF features)
    const tokensGrid = document.getElementById('salientTokensGrid');
    tokensGrid.innerHTML = '';
    if (tokens.length > 0) {
        tokens.forEach(tok => {
            const isTokFake = tok.leaning === 'Fake News';
            const chip = document.createElement('span');
            chip.className = `salient-chip ${isTokFake ? 'lean-fake' : 'lean-genuine'}`;
            chip.innerHTML = `
                <span>${tok.token}</span>
                <span class="token-impact">(${isTokFake ? '+' : ''}${tok.impact})</span>
            `;
            chip.title = `Leaning: ${tok.leaning} | Model Weight: ${tok.weight} | TF-IDF: ${tok.tfidf}`;
            tokensGrid.appendChild(chip);
        });
    } else {
        tokensGrid.innerHTML = '<span class="text-muted text-xs">No vocabulary overlap with trained dictionary</span>';
    }

    // 7. Syntax Bar
    document.getElementById('statWords').textContent = ling.word_count;
    document.getElementById('statSentences').textContent = ling.sentence_count;
    document.getElementById('statAvgSent').textContent = ling.avg_sentence_len;
}
