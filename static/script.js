(() => {
  const sceneInput = document.getElementById('sceneInput');
  const adInput = document.getElementById('adInput');
  const scenePreview = document.getElementById('scenePreview');
  const adPreview = document.getElementById('adPreview');

  const confSlider = document.getElementById('confSlider');
  const confVal = document.getElementById('confVal');
  const fitShape = document.getElementById('fitShape');
  const blendEdges = document.getElementById('blendEdges');

  const detectBtn = document.getElementById('detectBtn');
  const detectError = document.getElementById('detectError');

  const detectionsPanel = document.getElementById('detectionsPanel');
  const detectPreview = document.getElementById('detectPreview');
  const boxSelect = document.getElementById('boxSelect');
  const compositeError = document.getElementById('compositeError');

  const resultPanel = document.getElementById('resultPanel');
  const resultPreview = document.getElementById('resultPreview');
  const resultFinal = document.getElementById('resultFinal');
  const fallbackNote = document.getElementById('fallbackNote');
  const downloadBtn = document.getElementById('downloadBtn');

  const jobNo = document.getElementById('jobNo');
  const jobStatus = document.getElementById('jobStatus');
  const loadingIndicator = document.getElementById('loadingIndicator');

  let boxes = [];
  // Bumped on every composite request so a slow, stale response can never
  // overwrite the result of a request that started after it (e.g. quickly
  // switching detections).
  let compositeToken = 0;

  jobNo.textContent = String(Date.now()).slice(-6);

  function setLoading(isLoading) {
    loadingIndicator.hidden = !isLoading;
  }

  function setStatus(text) {
    jobStatus.textContent = text;
  }

  function showError(el, message) {
    el.textContent = message;
    el.hidden = false;
  }

  function hideError(el) {
    el.hidden = true;
  }

  function updateDetectAvailability() {
    detectBtn.disabled = !(sceneInput.files[0] && adInput.files[0]);
  }

  function wireUpload(input, previewImg, dropLabel) {
    input.addEventListener('change', () => {
      const file = input.files[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      previewImg.src = url;
      previewImg.hidden = false;
      dropLabel.hidden = true;
      updateDetectAvailability();
      // Reset downstream steps whenever inputs change.
      detectionsPanel.hidden = true;
      resultPanel.hidden = true;
      setStatus('awaiting detection');
    });
  }

  wireUpload(sceneInput, scenePreview, document.querySelector('#sceneFrame .frame__drop'));
  wireUpload(adInput, adPreview, document.querySelector('#adFrame .frame__drop'));

  confSlider.addEventListener('input', () => {
    confVal.textContent = Number(confSlider.value).toFixed(2);
  });

  detectBtn.addEventListener('click', async () => {
    hideError(detectError);
    const sceneFile = sceneInput.files[0];
    if (!sceneFile) return;

    setLoading(true);
    setStatus('detecting…');
    detectBtn.disabled = true;
    resultPanel.hidden = true;

    try {
      const form = new FormData();
      form.append('scene', sceneFile);
      form.append('conf', confSlider.value);

      const res = await fetch('/api/detect', { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Detection failed (${res.status})`);
      }
      const data = await res.json();

      boxes = data.boxes;
      if (boxes.length === 0) {
        showError(detectError, 'No billboard detected above this confidence threshold. Try lowering it, or use a clearer scene photo.');
        detectionsPanel.hidden = true;
        setStatus('no detections');
        return;
      }

      detectPreview.src = data.preview;
      boxSelect.innerHTML = '';
      boxes.forEach((b, i) => {
        const opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = `detection ${i + 1} — confidence ${b.confidence.toFixed(2)} — box (${b.x1},${b.y1})–(${b.x2},${b.y2})`;
        boxSelect.appendChild(opt);
      });
      boxSelect.value = '0';

      detectionsPanel.hidden = false;
      setStatus(`${boxes.length} panel(s) found`);
      detectionsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

      // Show a result immediately for the top detection — no extra click.
      runComposite({ scroll: true });
    } catch (e) {
      showError(detectError, e.message || 'Something went wrong running detection.');
      setStatus('detection error');
    } finally {
      setLoading(false);
      detectBtn.disabled = false;
    }
  });

  // Any of these should immediately re-render the proof with no extra click.
  boxSelect.addEventListener('change', () => runComposite({ scroll: false }));
  fitShape.addEventListener('change', () => runComposite({ scroll: false }));
  blendEdges.addEventListener('change', () => runComposite({ scroll: false }));

  async function runComposite({ scroll }) {
    hideError(compositeError);
    const sceneFile = sceneInput.files[0];
    const adFile = adInput.files[0];
    const chosen = boxes[Number(boxSelect.value)];
    if (!sceneFile || !adFile || !chosen) return;

    const token = ++compositeToken;

    setLoading(true);
    setStatus('compositing…');

    try {
      const form = new FormData();
      form.append('scene', sceneFile);
      form.append('ad', adFile);
      form.append('x1', chosen.x1);
      form.append('y1', chosen.y1);
      form.append('x2', chosen.x2);
      form.append('y2', chosen.y2);
      form.append('fit_shape', fitShape.checked);
      form.append('blend_edges', blendEdges.checked);

      const res = await fetch('/api/composite', { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Compositing failed (${res.status})`);
      }
      const data = await res.json();

      // A newer request has already started — drop this stale response.
      if (token !== compositeToken) return;

      resultPreview.src = data.preview;
      resultFinal.src = data.result;
      fallbackNote.hidden = !(fitShape.checked && data.used_fallback);
      downloadBtn.href = data.result;

      resultPanel.hidden = false;
      setStatus('proof ready');
      if (scroll) {
        resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    } catch (e) {
      if (token !== compositeToken) return;
      showError(compositeError, e.message || 'Something went wrong compositing the ad.');
      setStatus('composite error');
    } finally {
      if (token === compositeToken) setLoading(false);
    }
  }
})();
