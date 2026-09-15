document.addEventListener('DOMContentLoaded', () => {
    // Motor elements
    const slider1  = document.getElementById('motor-1');
    const slider2  = document.getElementById('motor-2');
    const slider3  = document.getElementById('motor-3');
    const val1     = document.getElementById('val-1');
    const val2     = document.getElementById('val-2');
    const val3     = document.getElementById('val-3');
    const resetBtn = document.getElementById('reset-btn');

    // Light elements
    const lightToggle     = document.getElementById('light-toggle');
    const lightColor      = document.getElementById('light-color');
    const lightStatusText = document.getElementById('light-status-text');

    // Mode toggle
    const handTrackToggle = document.getElementById('hand-track-toggle');
    const modeLabel       = document.getElementById('mode-label');
    const modeIcon        = document.getElementById('mode-icon');

    // Hand status card
    const handStatusCard = document.getElementById('hand-status-card');
    const handBadge      = document.getElementById('hand-badge');
    const arM1           = document.getElementById('ar-m1');
    const arM2           = document.getElementById('ar-m2');
    const arM3           = document.getElementById('ar-m3');

    // Status bar
    const statusDot  = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');

    let lightDebounceTimer = null;
    let lastSendTime = 0;
    let throttleTimer = null;
    let statusPoller  = null;

    // ── Helpers ───────────────────────────────────────────────────────────────
    function setStatus(state, text) {
        statusDot.className = `dot ${state}`;
        statusText.textContent = text;
    }

    function setSliderLock(locked) {
        [slider1, slider2, slider3, val1, val2, val3].forEach(el => {
            el.disabled = locked;
        });
        [slider1, slider2, slider3].forEach(el => {
            el.classList.toggle('disabled-track', locked);
        });
        resetBtn.disabled = locked;
        resetBtn.style.opacity = locked ? '0.4' : '1';
    }

    function updateMotorDisplays(angles) {
        if (angles) {
            slider1.value = angles[0]; val1.value = angles[0];
            slider2.value = angles[1]; val2.value = angles[1];
            slider3.value = angles[2]; val3.value = angles[2];
        } else {
            val1.value = slider1.value;
            val2.value = slider2.value;
            val3.value = slider3.value;
        }
    }

    // ── Send angles ───────────────────────────────────────────────────────────
    function sendAngles() {
        const angles = [
            parseFloat(slider1.value),
            parseFloat(slider2.value),
            parseFloat(slider3.value)
        ];
        setStatus('sending', 'Sending…');
        fetch('/api/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ angles })
        })
        .then(r => r.json())
        .then(d => {
            if (d.status === 'success') setStatus('', 'Connected & Synced');
            else setStatus('error', 'Error: ' + d.message);
        })
        .catch(() => setStatus('error', 'Connection failed'));
    }

    // ── Light ─────────────────────────────────────────────────────────────────
    function sendLight() {
        fetch('/api/light', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ on: lightToggle.checked, color: lightColor.value })
        }).catch(() => {});
    }

    function handleLightChange() {
        lightColor.disabled = !lightToggle.checked;
        lightStatusText.textContent = lightToggle.checked ? 'On' : 'Off';
        clearTimeout(lightDebounceTimer);
        lightDebounceTimer = setTimeout(sendLight, 100);
    }

    // ── Mode toggle ───────────────────────────────────────────────────────────
    function setMode(handTrack) {
        fetch('/api/mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hand_track: handTrack })
        }).catch(() => {});

        if (handTrack) {
            modeLabel.textContent = 'Hand Track';
            modeIcon.textContent  = '🤚';
            setSliderLock(true);
            setStatus('sending', 'Hand tracking active');
            handStatusCard.style.display = 'flex';
            startStatusPolling();
        } else {
            modeLabel.textContent = 'Manual';
            modeIcon.textContent  = '🎮';
            setSliderLock(false);
            setStatus('', 'Manual mode');
            handStatusCard.style.display = 'none';
            stopStatusPolling();
            
            // Revert back to default neutral coordinates upon disconnecting hand tracker
            val1.value = 0; slider1.value = 0;
            val2.value = 0; slider2.value = 0;
            val3.value = 0; slider3.value = 0;
            sendAngles();
        }
    }

    handTrackToggle.addEventListener('change', () => setMode(handTrackToggle.checked));

    // ── Status polling (hand-track mode) ─────────────────────────────────────
    function startStatusPolling() {
        if (statusPoller) return;
        statusPoller = setInterval(async () => {
            try {
                const res  = await fetch('/api/status');
                const data = await res.json();
                if (data.hand_detected && data.angles) {
                    handBadge.textContent = 'Hand Detected ✋';
                    handBadge.classList.add('detected');
                    arM1.textContent = data.angles[0].toFixed(0) + '°';
                    arM2.textContent = data.angles[1].toFixed(0) + '°';
                    arM3.textContent = data.angles[2].toFixed(0) + '°';
                    updateMotorDisplays(data.angles);
                    setStatus('', 'Hand tracking · motors synced');
                } else {
                    handBadge.textContent = 'No Hand';
                    handBadge.classList.remove('detected');
                    setStatus('sending', 'Waiting for hand…');
                }
            } catch (e) {
                setStatus('error', 'Poll failed');
            }
        }, 150);
    }

    function stopStatusPolling() {
        clearInterval(statusPoller);
        statusPoller = null;
        handBadge.textContent = 'No Hand';
        handBadge.classList.remove('detected');
        arM1.textContent = arM2.textContent = arM3.textContent = '—';
    }

    // ── Slider & input listeners ──────────────────────────────────────────────
    function handleSliderChange() {
        updateMotorDisplays();
        const now = Date.now();
        if (now - lastSendTime >= 150) {
            sendAngles();
            lastSendTime = now;
        } else {
            clearTimeout(throttleTimer);
            throttleTimer = setTimeout(() => {
                sendAngles();
                lastSendTime = Date.now();
            }, 150);
        }
    }

    function handleInputChange(e) {
        // No extra clamping — slider min/max handles boundaries

        // Update the slider to match (clamped to slider's own range visually,
        // but the RAW typed value is always sent to the Pi).
        slider1.value = val1.value;
        slider2.value = val2.value;
        slider3.value = val3.value;
        // Only send when the user presses Enter — see keydown listeners below.
    }

    function handleInputEnter(e) {
        if (e.key === 'Enter') {
            // Send the raw typed value with NO clamping
            clearTimeout(throttleTimer);
            sendAngles();
            lastSendTime = Date.now();
        }
    }

    slider1.addEventListener('input', handleSliderChange);
    slider2.addEventListener('input', handleSliderChange);
    slider3.addEventListener('input', handleSliderChange);
    val1.addEventListener('change', handleInputChange);
    val2.addEventListener('change', handleInputChange);
    val3.addEventListener('change', handleInputChange);
    val1.addEventListener('keydown', handleInputEnter);
    val2.addEventListener('keydown', handleInputEnter);
    val3.addEventListener('keydown', handleInputEnter);

    resetBtn.addEventListener('click', () => {
        slider1.value = 0; val1.value = 0;
        slider2.value = 0; val2.value = 0;
        slider3.value = 0; val3.value = 0;
        sendAngles();
    });

    lightToggle.addEventListener('change', handleLightChange);
    lightColor.addEventListener('input', handleLightChange);

    // ── Init ─────────────────────────────────────────────────────────────────
    val1.value = 0; slider1.value = 0;
    val2.value = 0; slider2.value = 0;
    val3.value = 0; slider3.value = 0;
    sendAngles();
});
