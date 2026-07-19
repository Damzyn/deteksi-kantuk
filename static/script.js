class DrowsinessDetector {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.stream = null;
        this.isRunning = false;
        this.animationId = null;
        this.lastFrameTime = 0;
        this.frameInterval = 33;
        this.currentAlertType = null;
        this.currentPlayingSource = null;
        this.isPlayingAlarm = false;
        
        this.currentFacingMode = 'user';
        this.currentDeviceId = null;
        this.isMobile = this.detectMobile();
        this.availableCameras = [];
        this.cameraInfo = document.getElementById('cameraInfo');
        
        this.soundMuted = false;
        this.soundBtn = document.getElementById('nightVisionBtn');
        
        this.audioContext = null;
        this.warningBuffer = null;
        this.dangerBuffer = null;
        this.veryDangerousBuffer = null;
        
        this.synth = window.speechSynthesis;
        this.lastSpokenStatus = null;
        this.isSpeaking = false;
        this.voiceQueue = null;
        
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.switchCameraBtn = document.getElementById('switchCameraBtn');
        this.soundBtn = document.getElementById('nightVisionBtn');
        
        this.statusCircle = document.getElementById('statusCircle');
        this.statusMessage = document.getElementById('statusMessage');
        this.statusDescription = document.getElementById('statusDescription');
        this.eyeStateEl = document.getElementById('eyeState');
        this.mouthStateEl = document.getElementById('mouthState');
        this.earValue = document.getElementById('earValue');
        this.marValue = document.getElementById('marValue');
        this.eyeDuration = document.getElementById('eyeDuration');
        this.yawnDuration = document.getElementById('yawnDuration');
        this.statusText = document.getElementById('statusText');
        
        this.cnnPrediction = document.getElementById('cnnPrediction');
        this.cnnConfidence = document.getElementById('cnnConfidence');
        this.confidenceFill = document.getElementById('confidenceFill');
        
        this.fullscreenBtn = document.getElementById('fullscreenBtn');
        this.isFullscreen = false;
        this.pipActive = false;
        this.pipInterval = null;
        this.indicatorTimeout = null;
        
        // ========== WAKE LOCK ==========
        this.wakeLock = null;
        this.wakeLockSupported = false;
        
        if ('wakeLock' in navigator) {
            this.wakeLockSupported = true;
            console.log('✅ Wake Lock supported');
        } else {
            console.log('⚠️ Wake Lock NOT supported');
        }
        
        this.resultImg = document.createElement('img');
        this.resultImg.id = 'resultImg';
        this.resultImg.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 15px;
            z-index: 1;
            pointer-events: none;
            display: block;
        `;
        
        this.tabBtns = document.querySelectorAll('.tab-btn');
        this.tabContents = document.querySelectorAll('.tab-content');
        
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        this.switchCameraBtn.addEventListener('click', () => this.switchCamera());
        this.soundBtn.addEventListener('click', () => this.toggleMute());
        
        if (this.fullscreenBtn) {
            this.fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
        }
        
        this.tabBtns.forEach(btn => {
            btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
        });
        
        // ========== VISIBILITY CHANGE EVENT ==========
        document.addEventListener('visibilitychange', () => {
            console.log('📱 visibilitychange FIRED! hidden:', document.hidden);
            this.handleVisibilityChange();
        });
        
        document.addEventListener('fullscreenchange', () => this.handleFullscreenChange());
        document.addEventListener('webkitfullscreenchange', () => this.handleFullscreenChange());
        
        // ========== DETEKSI PiP CHROME ==========
        document.addEventListener('webkitpresentationchange', () => {
            console.log('📱 webkitpresentationchange FIRED!');
            this.handleVisibilityChange();
        });
        
        document.addEventListener('enterpictureinpicture', () => {
            console.log('📱 enterpictureinpicture FIRED!');
            this.pipActive = true;
            console.log('✅ PiP entered - pipActive =', this.pipActive);
            this.handlePiPEnter();
            this.sendPiPStatus();
        });
        
        document.addEventListener('leavepictureinpicture', () => {
            console.log('📱 leavepictureinpicture FIRED!');
            this.pipActive = false;
            console.log('✅ PiP left - pipActive =', this.pipActive);
            this.handlePiPExit();
            this.sendPiPStatus();
        });
        
        this.initAudio();
        this.checkServer();
        this.enumerateCameras();
        this.setupVideoContainer();
        
        console.log('Device type:', this.isMobile ? 'MOBILE (HP)' : 'DESKTOP (LAPTOP/PC)');
    }
    
    // ========== HANDLE PiP ENTER ==========
    handlePiPEnter() {
        // Di PiP, tampilkan resultImg (sudah di-flip) dan sembunyikan video asli
        if (this.resultImg) {
            this.resultImg.style.display = 'block';
            this.resultImg.style.zIndex = '2';
        }
        this.video.style.display = 'none';
        console.log('✅ PiP: menampilkan resultImg, video asli disembunyikan');
    }
    
    // ========== HANDLE PiP EXIT ==========
    handlePiPExit() {
        // Kembali ke normal
        this.video.style.display = 'block';
        if (this.resultImg) {
            this.resultImg.style.display = 'block';
            this.resultImg.style.zIndex = '1';
        }
        console.log('✅ PiP EXIT: kembali ke normal');
    }
    
    // ========== PiP DETECTION POLLING ==========
    startPiPDetection() {
        this.pipInterval = setInterval(() => {
            const isPipActive = document.pictureInPictureElement !== null;
            
            if (isPipActive && !this.pipActive) {
                this.pipActive = true;
                console.log('✅ PiP DETECTED (polling) - pipActive =', this.pipActive);
                this.handlePiPEnter();
                this.sendPiPStatus();
            } else if (!isPipActive && this.pipActive) {
                this.pipActive = false;
                console.log('✅ PiP EXITED (polling) - pipActive =', this.pipActive);
                this.handlePiPExit();
                this.sendPiPStatus();
            }
        }, 500);
    }
    
    stopPiPDetection() {
        if (this.pipInterval) {
            clearInterval(this.pipInterval);
            this.pipInterval = null;
        }
    }
    
    setupVideoContainer() {
        const container = document.querySelector('#liveTab .video-container');
        
        // Video asli - di bawah
        this.video.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            z-index: 0;
            border-radius: 15px;
            display: block !important;
            transform: scaleX(1) !important;
            -webkit-transform: scaleX(1) !important;
        `;
        
        // Hapus img lama
        const oldImg = container.querySelector('img');
        if (oldImg) oldImg.remove();
        
        // resultImg - di atas video
        this.resultImg.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 15px;
            z-index: 1;
            pointer-events: none;
            display: block;
        `;
        container.appendChild(this.resultImg);
        
        // Canvas - hidden
        this.canvas.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            display: none;
        `;
        
        const overlay = container.querySelector('.video-overlay');
        if (overlay) {
            overlay.style.zIndex = '10';
            overlay.style.pointerEvents = 'none';
        }
        
        console.log('✅ Video container setup complete');
    }
    
    toggleMute() {
        this.soundMuted = !this.soundMuted;
        if (this.soundMuted) {
            this.soundBtn.innerHTML = '🔇 Alarm OFF';
            this.soundBtn.style.background = '#dc3545';
            this.stopAllSounds();
        } else {
            this.soundBtn.innerHTML = '🔊 Alarm ON';
            this.soundBtn.style.background = '#28a745';
        }
        console.log('Alarm muted:', this.soundMuted);
    }
    
    stopAllSounds() {
        if (this.currentPlayingSource) {
            try { this.currentPlayingSource.stop(); } catch(e) {}
            this.currentPlayingSource = null;
        }
        this.isPlayingAlarm = false;
        
        if (this.synth.speaking) {
            this.synth.cancel();
        }
        this.isSpeaking = false;
        this.currentAlertType = null;
    }
    
    playAlert(alertType, status, eyeDuration, yawnDuration) {
        if (this.soundMuted) return;
        
        this.stopAllSounds();
        
        let buffer = null;
        if (alertType === 'very_dangerous') buffer = this.veryDangerousBuffer;
        else if (alertType === 'danger') buffer = this.dangerBuffer;
        else if (alertType === 'warning') buffer = this.warningBuffer;
        else return;
        
        if (!buffer || !this.audioContext) return;
        
        this.currentAlertType = alertType;
        this.isPlayingAlarm = true;
        
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
        
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);
        
        source.onended = () => {
            console.log('✅ Alarm selesai, mulai voice...');
            this.isPlayingAlarm = false;
            this.currentPlayingSource = null;
            
            if (!this.soundMuted && this.lastSpokenStatus !== status) {
                this.speakOnce(status, alertType, eyeDuration, yawnDuration);
            }
        };
        
        source.start();
        this.currentPlayingSource = source;
        console.log('🔊 Alarm started (1x):', alertType);
    }
    
    speak(message) {
        if (this.soundMuted) return;
        
        if (this.isSpeaking) {
            console.log('⏳ Still speaking, skip');
            return;
        }
        
        if (this.synth.speaking) {
            this.synth.cancel();
        }
        
        this.isSpeaking = true;
        
        const utterance = new SpeechSynthesisUtterance(message);
        utterance.lang = 'id-ID';
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        const voices = this.synth.getVoices();
        const indonesianVoice = voices.find(voice => voice.lang.includes('id'));
        if (indonesianVoice) utterance.voice = indonesianVoice;
        
        utterance.onend = () => {
            console.log('✅ Voice finished');
            this.isSpeaking = false;
        };
        
        utterance.onerror = () => {
            console.log('❌ Voice error');
            this.isSpeaking = false;
        };
        
        this.synth.speak(utterance);
        console.log('🔊 Voice:', message);
    }
    
    speakOnce(status, alertType, eyeDuration, yawnDuration) {
        if (this.soundMuted) return;
        if (this.lastSpokenStatus === status) return;
        
        let message = '';
        switch (status) {
            case 'WARNING':
                message = 'Peringatan. Anda terdeteksi menguap. Segera minum kopi atau makan permen untuk menjaga kewaspadaan.';
                break;
            case 'DANGER':
                message = 'Bahaya. Mata Anda tertutup. Segera menepi ke pinggir jalan yang aman.';
                break;
            case 'VERY_DANGEROUS':
                message = 'Bahaya sekali. Mata tertutup dan menguap secara bersamaan. WAJIB menepi SEKARANG juga.';
                break;
            default:
                return;
        }
        
        this.lastSpokenStatus = status;
        this.speak(message);
    }
    
    getStatusDescription(status, eyeDuration, yawnDuration) {
        const statuses = {
            'WARNING': {
                title: '🟡 PERINGATAN - Menguap',
                message: 'Anda terdeteksi menguap. Ini tanda awal kantuk.',
                advice: '☕ Minum kopi atau teh, istirahat sejenak.'
            },
            'DANGER': {
                title: '🔴 BAHAYA - Tutup Mata',
                message: 'Mata Anda tertutup! Anda mengantuk berat.',
                advice: '⚠️ SEGERA MENEPI! Berhenti dan istirahat!'
            },
            'VERY_DANGEROUS': {
                title: '💀 SANGAT BERBAHAYA - Mata Tertutup + Menguap',
                message: 'Mata tertutup DAN menguap! Anda sangat mengantuk.',
                advice: '🛑 WAJIB BERHENTI! Tidur sebentar!'
            }
        };
        return statuses[status] || { title: '', message: '', advice: '' };
    }
    
    updateStatusDescription(status, eyeDuration, yawnDuration) {
        if (!this.statusDescription) return;
        
        if (status === 'WARNING' || status === 'DANGER' || status === 'VERY_DANGEROUS') {
            const desc = this.getStatusDescription(status, eyeDuration, yawnDuration);
            let borderColor = '#ffa500';
            if (status === 'DANGER') borderColor = '#ff0000';
            else if (status === 'VERY_DANGEROUS') borderColor = '#8B0000';
            
            this.statusDescription.style.borderLeft = `4px solid ${borderColor}`;
            this.statusDescription.innerHTML = `
                <div style="font-weight: bold; margin-bottom: 8px; color: ${borderColor};">${desc.title}</div>
                <div style="margin-bottom: 6px;">📢 ${desc.message}</div>
                <div style="font-size: 0.75rem; color: #ffa500;">💡 ${desc.advice}</div>
            `;
            this.statusDescription.style.display = 'block';
        } else {
            this.statusDescription.style.display = 'none';
        }
    }
    
    detectMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }
    
    async enumerateCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            this.availableCameras = devices.filter(device => device.kind === 'videoinput');
            console.log('📷 Available cameras:', this.availableCameras.length);
            return this.availableCameras;
        } catch (error) {
            console.error('Enumerate error:', error);
            return [];
        }
    }
    
    getShortCameraName(label) {
        if (!label) return 'Kamera';
        let name = label;
        name = name.replace('Integrated Camera', 'Kamera Laptop');
        name = name.replace('HD Webcam', 'Webcam');
        name = name.replace('USB Camera', 'USB Webcam');
        name = name.replace('VGA WebCam', 'USB Webcam');
        name = name.replace('OBS Virtual Camera', 'OBS Virtual');
        name = name.replace('Streaming Webcams', 'Streaming Webcam');
        if (name.length > 28) name = name.substring(0, 25) + '...';
        return name;
    }
    
    async showCameraSelectionDialog() {
        await this.enumerateCameras();
        if (this.availableCameras.length === 0) {
            alert('Tidak ada kamera yang ditemukan!');
            return;
        }
        let options = this.availableCameras.map((cam, idx) => {
            return `${idx + 1}. ${this.getShortCameraName(cam.label)}`;
        }).join('\n');
        let choice = prompt(`Pilih Kamera (1-${this.availableCameras.length}):\n\n${options}\n\nMasukkan nomor kamera:`);
        if (choice) {
            let idx = parseInt(choice) - 1;
            if (idx >= 0 && idx < this.availableCameras.length) {
                await this.selectCamera(this.availableCameras[idx].deviceId);
            }
        }
    }
    
    async selectCamera(deviceId) {
        this.currentDeviceId = deviceId;
        if (this.isRunning) await this.restartCamera();
        else await this.startCamera();
    }
    
    async switchCamera() {
        if (this.isMobile) {
            if (this.stream) this.stream.getTracks().forEach(track => track.stop());
            this.currentFacingMode = this.currentFacingMode === 'user' ? 'environment' : 'user';
            await this.startCamera();
        } else {
            await this.showCameraSelectionDialog();
        }
    }
    
    switchTab(tabId) {
        this.tabBtns.forEach(btn => {
            if (btn.dataset.tab === tabId) btn.classList.add('active');
            else btn.classList.remove('active');
        });
        this.tabContents.forEach(content => {
            if (content.id === `${tabId}Tab`) content.classList.add('active');
            else content.classList.remove('active');
        });
        if (this.isRunning) this.stopCamera();
    }
    
    async initAudio() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        try {
            const warningResponse = await fetch('/static/peringatan.wav');
            const warningArrayBuffer = await warningResponse.arrayBuffer();
            this.warningBuffer = await this.audioContext.decodeAudioData(warningArrayBuffer);
            
            const dangerResponse = await fetch('/static/bahaya.wav');
            const dangerArrayBuffer = await dangerResponse.arrayBuffer();
            this.dangerBuffer = await this.audioContext.decodeAudioData(dangerArrayBuffer);
            
            const veryDangerousResponse = await fetch('/static/sangat_bahaya.wav');
            const veryDangerousArrayBuffer = await veryDangerousResponse.arrayBuffer();
            this.veryDangerousBuffer = await this.audioContext.decodeAudioData(veryDangerousArrayBuffer);
            
            const resumeAudio = () => {
                if (this.audioContext && this.audioContext.state === 'suspended') {
                    this.audioContext.resume();
                }
            };
            document.addEventListener('click', resumeAudio);
            document.addEventListener('touchstart', resumeAudio);
            
            const loadVoices = () => {
                const voices = this.synth.getVoices();
                console.log('🎤 Voices loaded:', voices.length);
            };
            this.synth.onvoiceschanged = loadVoices;
            loadVoices();
            
        } catch (error) {
            console.error('Audio load error:', error);
        }
    }
    
    async checkServer() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            console.log('Server:', data);
        } catch (error) {
            console.error('Server error:', error);
        }
    }
    
    async restartCamera() {
        const wasRunning = this.isRunning;
        if (this.stream) this.stream.getTracks().forEach(track => track.stop());
        if (this.animationId) cancelAnimationFrame(this.animationId);
        this.isRunning = false;
        if (wasRunning) await this.startCamera();
    }
    
    // ========== WAKE LOCK FUNCTIONS ==========
    async requestWakeLock() {
        if (!this.wakeLockSupported) return;
        
        try {
            if (!this.wakeLock) {
                this.wakeLock = await navigator.wakeLock.request('screen');
                console.log('✅ Wake Lock activated - screen will not sleep');
                
                document.addEventListener('visibilitychange', () => {
                    if (document.visibilityState === 'visible' && !this.wakeLock) {
                        this.requestWakeLock();
                    }
                });
            }
        } catch (error) {
            console.error('Wake Lock error:', error);
        }
    }
    
    async releaseWakeLock() {
        if (this.wakeLock) {
            try {
                await this.wakeLock.release();
                this.wakeLock = null;
                console.log('✅ Wake Lock released');
            } catch (error) {
                console.error('Release wake lock error:', error);
            }
        }
    }
    
    async startCamera() {
        try {
            this.updateStatus('Meminta akses kamera...', '#ffa500');
            if (this.audioContext && this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
            
            let constraints = null;
            if (this.isMobile) {
                constraints = { video: { facingMode: this.currentFacingMode } };
            } else if (this.currentDeviceId) {
                constraints = { video: { deviceId: { exact: this.currentDeviceId } } };
            } else {
                constraints = { video: true };
            }
            
            try {
                this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (err) {
                this.stream = await navigator.mediaDevices.getUserMedia({ video: true });
            }
            
            const videoTrack = this.stream.getVideoTracks()[0];
            let cameraName = videoTrack.label || 'Kamera';
            cameraName = this.getShortCameraName(cameraName);
            
            this.video.srcObject = this.stream;
            this.video.style.display = 'block !important';
            
            await new Promise((resolve) => { 
                this.video.onloadedmetadata = () => resolve(); 
            });
            await this.video.play();
            
            // Video ditampilkan normal
            this.video.style.transform = 'scaleX(1)';
            this.video.style.webkitTransform = 'scaleX(1)';
            
            if (this.isMobile) {
                const settings = videoTrack.getSettings();
                if (settings.facingMode) {
                    this.currentFacingMode = settings.facingMode;
                    const displayName = this.currentFacingMode === 'user' ? 'Depan (Selfie)' : 'Belakang';
                    this.updateCameraInfo(displayName);
                    this.updateStatus(`✅ Aktif! (${displayName})`, '#00ff00');
                    if (this.statusText) this.statusText.textContent = `AKTIF (${displayName})`;
                } else {
                    this.updateCameraInfo(cameraName);
                    this.updateStatus(`✅ Aktif! (${cameraName})`, '#00ff00');
                }
            } else {
                this.updateCameraInfo(cameraName);
                this.updateStatus(`✅ Aktif! (${cameraName})`, '#00ff00');
            }
            
            this.isRunning = true;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.switchCameraBtn.disabled = false;
            this.soundBtn.disabled = false;
            
            if (this.fullscreenBtn) {
                this.fullscreenBtn.disabled = false;
            }
            
            const videoWidth = this.video.videoWidth || 480;
            const videoHeight = this.video.videoHeight || 360;
            
            this.canvas.width = videoWidth;
            this.canvas.height = videoHeight;
            this.canvas.style.display = 'none';
            
            await this.requestWakeLock();
            
            // ========== START PiP DETECTION ==========
            this.startPiPDetection();
            
            this.processFrame();
            
            if (this.isMobile) {
                this.switchCameraBtn.innerHTML = '🔄 Ganti Kamera';
                this.switchCameraBtn.title = 'Ganti kamera depan/belakang';
            } else {
                this.switchCameraBtn.innerHTML = '📷 Ganti Kamera';
                this.switchCameraBtn.title = 'Pilih kamera (Laptop / Webcam External)';
            }
            
            this.soundMuted = false;
            this.soundBtn.innerHTML = '🔊 Alarm ON';
            this.soundBtn.style.background = '#28a745';
            this.lastSpokenStatus = null;
            this.isSpeaking = false;
            this.isPlayingAlarm = false;
            
        } catch (error) {
            console.error('Camera error:', error);
            let errorMsg = '❌ Gagal akses kamera.\n\n';
            if (error.name === 'NotAllowedError') errorMsg += 'Izinkan akses kamera di browser.';
            else if (error.name === 'NotFoundError') errorMsg += 'Tidak ada kamera yang terdeteksi.';
            else errorMsg += error.message;
            this.updateStatus(errorMsg, '#ff0000');
            alert(errorMsg);
            this.startBtn.disabled = false;
            this.stopBtn.disabled = true;
            this.switchCameraBtn.disabled = true;
            this.soundBtn.disabled = true;
            if (this.fullscreenBtn) {
                this.fullscreenBtn.disabled = true;
            }
        }
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        this.isRunning = false;
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.switchCameraBtn.disabled = false;
        this.soundBtn.disabled = true;
        this.video.srcObject = null;
        this.video.style.display = 'none';
        this.video.style.transform = 'scaleX(1)';
        
        if (this.fullscreenBtn) {
            this.fullscreenBtn.disabled = true;
            this.fullscreenBtn.textContent = '⛶ Fullscreen';
            this.fullscreenBtn.classList.remove('active');
        }
        
        if (this.isFullscreen) {
            this.exitFullscreen();
        }
        
        // ========== STOP PiP DETECTION ==========
        this.stopPiPDetection();
        
        this.releaseWakeLock();
        
        this.stopAllSounds();
        this.updateStatus('Berhenti', '#666');
        this.resultImg.src = '';
        if (this.statusText) this.statusText.textContent = 'BERHENTI';
        this.eyeStateEl.textContent = '---';
        this.mouthStateEl.textContent = '---';
        this.earValue.textContent = '0.00';
        this.marValue.textContent = '0.00';
        this.eyeDuration.textContent = '0.0d';
        this.yawnDuration.textContent = '0.0d';
        if (this.cnnPrediction) {
            this.cnnPrediction.textContent = 'Menunggu wajah...';
            this.cnnConfidence.textContent = '0% keyakinan';
            this.confidenceFill.style.width = '0%';
        }
        if (this.audioContext) this.audioContext.suspend();
    }
    
    updateCameraInfo(camera) {
        if (this.cameraInfo) this.cameraInfo.innerHTML = `📷 Kamera: ${camera}`;
    }
    
    async processFrame() {
        if (!this.isRunning || !this.video.videoWidth) {
            this.animationId = requestAnimationFrame(() => this.processFrame());
            return;
        }
        const now = Date.now();
        if (now - this.lastFrameTime < this.frameInterval) {
            this.animationId = requestAnimationFrame(() => this.processFrame());
            return;
        }
        this.lastFrameTime = now;
        
        // Gambar video ke canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        const imageData = this.canvas.toDataURL('image/jpeg', 0.7);
        
        try {
            const response = await fetch('/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    image: imageData,
                    facing_mode: this.isMobile ? this.currentFacingMode : 'user',
                    pip_mode: this.pipActive
                })
            });
            const data = await response.json();
            if (data.success) {
                // Update resultImg dengan gambar yang sudah di-flip
                this.resultImg.src = 'data:image/jpeg;base64,' + data.processed_image;
                this.resultImg.style.display = 'block';
                
                // Server sudah melakukan flip, tampilkan apa adanya
                this.resultImg.style.transform = 'scaleX(1)';
                this.resultImg.style.webkitTransform = 'scaleX(1)';
                
                // Pastikan z-index sesuai
                if (this.pipActive) {
                    this.resultImg.style.zIndex = '2';
                } else {
                    this.resultImg.style.zIndex = '1';
                }
                
                this.updateUI(data);
                this.handleAlert(data.alert, data.alert_type, data.status, 
                    data.eye_closed_duration || 0, data.yawn_duration || 0);
            }
        } catch (error) {
            console.error('Processing error:', error);
        }
        this.animationId = requestAnimationFrame(() => this.processFrame());
    }
    
    handleAlert(alert, alertType, status, eyeDuration, yawnDuration) {
        if (this.soundMuted) return;
        
        if (status === 'WARNING' || status === 'DANGER' || status === 'VERY_DANGEROUS') {
            if (this.lastSpokenStatus !== status && !this.isPlayingAlarm && !this.isSpeaking) {
                this.playAlert(alertType, status, eyeDuration, yawnDuration);
            }
        } else {
            this.stopAllSounds();
        }
    }
    
    updateUI(data) {
        const statusClass = (data.status || 'unknown').toLowerCase();
        this.statusCircle.className = `status-circle ${statusClass}`;
        
        this.updateStatusDescription(data.status, data.eye_closed_duration || 0, data.yawn_duration || 0);
        
        if (data.status === 'SAFE' || data.status === 'UNKNOWN') {
            this.lastSpokenStatus = null;
            this.stopAllSounds();
        }
        
        this.eyeStateEl.textContent = data.eye_state === 'Closed_Eyes' ? '😴 Tertutup' : '👁️ Terbuka';
        this.mouthStateEl.textContent = data.mouth_state === 'Yawn' ? '😮 Menguap' : '😐 Normal';
        this.earValue.textContent = (data.ear || 0).toFixed(3);
        this.marValue.textContent = (data.mar || 0).toFixed(3);
        this.eyeDuration.textContent = `${(data.eye_closed_duration || 0).toFixed(1)}d`;
        this.yawnDuration.textContent = `${(data.yawn_duration || 0).toFixed(1)}d`;
        
        if (data.cnn_mouth_pred) {
            let mouthText = data.cnn_mouth_pred;
            if (mouthText === 'Yawn') mouthText = 'Menguap';
            else if (mouthText === 'No_yawn') mouthText = 'Tidak Menguap';
            
            this.cnnPrediction.textContent = mouthText;
            this.cnnConfidence.textContent = `${Math.round(data.cnn_mouth_conf || 0)}% keyakinan`;
            this.confidenceFill.style.width = `${Math.min(data.cnn_mouth_conf || 0, 100)}%`;
        } else {
            this.cnnPrediction.textContent = 'Menunggu wajah...';
            this.cnnConfidence.textContent = '0% keyakinan';
            this.confidenceFill.style.width = '0%';
        }
        
        if (this.statusText) {
            let statusText = data.status;
            let displayText = '';
            let displayColor = data.color;
            
            switch (statusText) {
                case 'WARNING':
                    displayText = '🟡 PERINGATAN - Menguap';
                    displayColor = '#ffa500';
                    break;
                case 'DANGER':
                    displayText = '🔴 BAHAYA - Tutup Mata';
                    displayColor = '#ff0000';
                    break;
                case 'VERY_DANGEROUS':
                    displayText = '💀 SANGAT BERBAHAYA - Mata Tertutup + Menguap';
                    displayColor = '#8B0000';
                    break;
                default:
                    displayText = '';
                    displayColor = 'transparent';
                    break;
            }
            
            this.statusText.textContent = displayText;
            this.statusText.style.color = displayColor;
            this.statusText.style.display = displayText ? 'block' : 'none';
            this.statusText.style.fontSize = '0.85rem';
        }
    }
    
    updateStatus(message, color) {
        this.statusMessage.textContent = message;
        this.statusMessage.style.color = color;
    }
    
    async toggleFullscreen() {
        const container = document.querySelector('.video-container');
        
        if (!this.isFullscreen) {
            try {
                if (container.requestFullscreen) {
                    await container.requestFullscreen();
                } else if (container.webkitRequestFullscreen) {
                    await container.webkitRequestFullscreen();
                } else if (container.msRequestFullscreen) {
                    await container.msRequestFullscreen();
                }
                this.isFullscreen = true;
                this.fullscreenBtn.textContent = '⛶ Keluar Fullscreen';
                this.fullscreenBtn.classList.add('active');
                console.log('✅ Fullscreen activated');
            } catch (error) {
                console.error('Fullscreen error:', error);
                alert('Fullscreen gagal. Pastikan Anda mengklik tombol terlebih dahulu.');
            }
        } else {
            await this.exitFullscreen();
        }
    }
    
    async exitFullscreen() {
        try {
            if (document.exitFullscreen) {
                await document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                await document.webkitExitFullscreen();
            } else if (document.msExitFullscreen) {
                await document.msExitFullscreen();
            }
            this.isFullscreen = false;
            this.fullscreenBtn.textContent = '⛶ Fullscreen';
            this.fullscreenBtn.classList.remove('active');
            console.log('✅ Fullscreen exited');
        } catch (error) {
            console.error('Exit fullscreen error:', error);
        }
    }
    
    handleFullscreenChange() {
        if (!document.fullscreenElement && !document.webkitFullscreenElement) {
            if (this.isFullscreen) {
                this.isFullscreen = false;
                this.fullscreenBtn.textContent = '⛶ Fullscreen';
                this.fullscreenBtn.classList.remove('active');
                console.log('Fullscreen exited by user');
            }
        }
    }
    
    handleVisibilityChange() {
        console.log('📱 handleVisibilityChange called - hidden:', document.hidden);
        
        const isPipActive = document.pictureInPictureElement !== null;
        const isHidden = document.hidden;
        
        console.log('📱 isPipActive:', isPipActive);
        console.log('📱 isHidden:', isHidden);
        
        if (isPipActive) {
            this.pipActive = true;
            console.log('✅ PiP mode DETECTED - pipActive =', this.pipActive);
            this.handlePiPEnter();
            this.sendPiPStatus();
        } else {
            if (this.pipActive) {
                this.pipActive = false;
                console.log('✅ PiP mode EXITED - pipActive =', this.pipActive);
                this.handlePiPExit();
                this.sendPiPStatus();
            }
        }
    }
    
    async sendPiPStatus() {
        try {
            await fetch('/pip_status', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    pip_active: this.pipActive,
                    is_fullscreen: this.isFullscreen,
                    is_running: this.isRunning,
                    timestamp: new Date().toISOString()
                })
            });
        } catch (error) {
            console.error('PiP status error:', error);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new DrowsinessDetector();
});