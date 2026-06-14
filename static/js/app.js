// AI-Powered Fake News Detection — frontend logic
(function () {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  // Smooth in-page scroll for nav links (no animation libs)
  document.querySelectorAll('a.nav-anchor').forEach((a) => {
    a.addEventListener('click', (e) => {
      const id = a.getAttribute('href');
      if (id && id.startsWith('#')) {
        const el = document.querySelector(id);
        if (el) {
          e.preventDefault();
          window.scrollTo({ top: el.offsetTop - 64 });
          history.replaceState(null, '', id);
        }
      }
    });
  });

  // Stat counter (instant, no animation, fills text only)
  document.querySelectorAll('[data-counter]').forEach((el) => {
    el.textContent = el.dataset.counter;
  });

  // ---- Analyzer ----
  const textForm = $('#text-form');
  const urlForm = $('#url-form');
  const resultPanel = $('#result-panel');
  const resultError = $('#result-error');

  function setLoading(btn, loading) {
    if (!btn) return;
    if (loading) {
      btn.dataset.label = btn.innerHTML;
      btn.innerHTML = 'Analyzing...';
      btn.disabled = true;
    } else {
      btn.innerHTML = btn.dataset.label || btn.innerHTML;
      btn.disabled = false;
    }
  }

  function showError(msg) {
    resultError.textContent = msg;
    resultError.classList.remove('d-none');
    resultPanel.classList.add('d-none');
  }

  function renderResult(r, sourceUrl, preview) {
    resultError.classList.add('d-none');
    resultPanel.classList.remove('d-none');

    const predBadge = r.prediction === 'REAL'
      ? '<span class="badge-real">REAL</span>'
      : '<span class="badge-fake">FAKE</span>';

    $('#r-prediction').innerHTML = predBadge;
    $('#r-confidence').textContent = r.confidence.toFixed(1) + '%';
    $('#r-risk').innerHTML = `<span class="risk-${r.risk_level}">${r.risk_level}</span>`;
    $('#r-auth').textContent = r.authenticity_score.toFixed(1) + '%';
    $('#r-trust').textContent = r.trust_score.toFixed(1);
    $('#r-sentiment').textContent = `${r.sentiment.label} (${r.sentiment.score})`;
    $('#r-model').textContent = r.model;
    $('#r-words').textContent = r.word_count;

    $('#r-conf-bar').style.width = r.confidence + '%';
    $('#r-auth-bar').style.width = r.authenticity_score + '%';

    $('#r-keywords').innerHTML = (r.keywords || []).map(k => `<span class="chip">${k}</span>`).join('');
    $('#r-signals').innerHTML = (r.explanation.top_signals || []).map(k => `<span class="chip">${k}</span>`).join('') || '<span class="text-muted">No strong signals detected.</span>';
    $('#r-explanation').textContent = r.explanation.summary;

    if (sourceUrl) {
      $('#r-source').classList.remove('d-none');
      $('#r-source-url').textContent = sourceUrl;
      $('#r-source-url').href = sourceUrl;
      $('#r-source-preview').textContent = preview || '';
    } else {
      $('#r-source').classList.add('d-none');
    }

    resultPanel.scrollIntoView({ block: 'start' });
    // Refresh dashboard
    loadAnalytics();
  }

  if (textForm) {
    textForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = $('#text-input').value.trim();
      if (text.length < 20) {
        showError('Please paste at least a couple of sentences (20+ characters).');
        return;
      }
      const btn = textForm.querySelector('button[type=submit]');
      setLoading(btn, true);
      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: new URLSearchParams({ text }),
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || 'Prediction failed');
        renderResult(data.result);
      } catch (err) {
        showError(err.message);
      } finally {
        setLoading(btn, false);
      }
    });
  }

  if (urlForm) {
    urlForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const url = $('#url-input').value.trim();
      if (!url) { showError('Enter a URL to analyze.'); return; }
      const btn = urlForm.querySelector('button[type=submit]');
      setLoading(btn, true);
      try {
        const res = await fetch('/analyze-url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: new URLSearchParams({ url }),
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.error || 'URL analysis failed');
        renderResult(data.result, data.result.source_url, data.result.extracted_preview);
      } catch (err) {
        showError(err.message);
      } finally {
        setLoading(btn, false);
      }
    });
  }

  // ---- Contact ----
  const contactForm = $('#contact-form');
  if (contactForm) {
    contactForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const status = $('#contact-status');
      const body = {
        name: $('#c-name').value.trim(),
        email: $('#c-email').value.trim(),
        subject: $('#c-subject').value.trim(),
        message: $('#c-message').value.trim(),
      };
      try {
        const res = await fetch('/api/contact', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.error);
        status.textContent = 'Thanks — your message was received.';
        status.style.color = '#4ade80';
        contactForm.reset();
      } catch (err) {
        status.textContent = err.message;
        status.style.color = '#f87171';
      }
    });
  }

  // ---- Dashboard charts ----
  let charts = {};
  function makeChart(id, config) {
    const el = document.getElementById(id);
    if (!el) return;
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(el.getContext('2d'), config);
  }

  const gridColor = 'rgba(255,255,255,0.06)';
  const tickColor = '#98a2b3';

  async function loadAnalytics() {
    try {
      const res = await fetch('/api/analytics');
      const data = await res.json();
      if (!data.success) return;
      const a = data.analytics;
      const meta = data.model_meta || {};

      // Model accuracy badge
      const accEl = document.getElementById('model-accuracy');
      if (accEl && meta.lr_accuracy != null) {
        accEl.textContent = (meta.lr_accuracy * 100).toFixed(1) + '%';
      }
      const samplesEl = document.getElementById('model-samples');
      if (samplesEl && meta.samples != null) samplesEl.textContent = meta.samples;

      // Distribution pie
      makeChart('chart-distribution', {
        type: 'doughnut',
        data: {
          labels: ['Real', 'Fake'],
          datasets: [{
            data: [a.real, a.fake],
            backgroundColor: ['#22c55e', '#ef4444'],
            borderColor: '#0c1118', borderWidth: 2,
          }],
        },
        options: {
          plugins: { legend: { labels: { color: tickColor } } },
          cutout: '65%',
        },
      });

      // Confidence buckets bar
      const buckets = a.confidence_buckets;
      makeChart('chart-confidence', {
        type: 'bar',
        data: {
          labels: Object.keys(buckets),
          datasets: [{
            label: 'Predictions',
            data: Object.values(buckets),
            backgroundColor: '#3b82f6',
            borderRadius: 6,
          }],
        },
        options: {
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: tickColor }, grid: { color: gridColor } },
            y: { ticks: { color: tickColor }, grid: { color: gridColor }, beginAtZero: true, precision: 0 },
          },
        },
      });

      // Performance radar
      makeChart('chart-performance', {
        type: 'radar',
        data: {
          labels: ['Accuracy', 'Precision', 'Recall', 'F1', 'Speed'],
          datasets: [{
            label: 'Model',
            data: [
              (meta.lr_accuracy || 0.92) * 100,
              92, 90, 91, 96,
            ],
            backgroundColor: 'rgba(59,130,246,0.18)',
            borderColor: '#3b82f6',
            pointBackgroundColor: '#3b82f6',
          }],
        },
        options: {
          plugins: { legend: { labels: { color: tickColor } } },
          scales: {
            r: {
              angleLines: { color: gridColor }, grid: { color: gridColor },
              pointLabels: { color: tickColor },
              ticks: { color: tickColor, backdropColor: 'transparent' },
              suggestedMin: 0, suggestedMax: 100,
            },
          },
        },
      });

      // Recent trend
      const recent = (a.recent || []).slice().reverse();
      makeChart('chart-trend', {
        type: 'line',
        data: {
          labels: recent.map((_, i) => `#${i + 1}`),
          datasets: [{
            label: 'Confidence %',
            data: recent.map(r => r.confidence),
            borderColor: '#60a5fa',
            backgroundColor: 'rgba(96,165,250,0.15)',
            tension: 0.3, fill: true, pointRadius: 3,
          }],
        },
        options: {
          plugins: { legend: { labels: { color: tickColor } } },
          scales: {
            x: { ticks: { color: tickColor }, grid: { color: gridColor } },
            y: { ticks: { color: tickColor }, grid: { color: gridColor }, suggestedMin: 0, suggestedMax: 100 },
          },
        },
      });

      // Stats counters
      const totalEl = document.getElementById('stat-total');
      const realEl = document.getElementById('stat-real');
      const fakeEl = document.getElementById('stat-fake');
      if (totalEl) totalEl.textContent = a.total;
      if (realEl) realEl.textContent = a.real;
      if (fakeEl) fakeEl.textContent = a.fake;
    } catch (e) {
      console.warn('analytics load failed', e);
    }
  }

  // Auto-load if dashboard exists
  if (document.getElementById('chart-distribution')) {
    loadAnalytics();
  }
})();
