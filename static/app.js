// API Configuration
const API_BASE_URL = window.location.origin;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const enhanceToggle = document.getElementById('enhanceToggle');
const confidenceSlider = document.getElementById('confidenceSlider');
const confidenceValue = document.getElementById('confidenceValue');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const clearBtn = document.getElementById('clearBtn');
const originalImage = document.getElementById('originalImage');
const analyzedCanvas = document.getElementById('analyzedCanvas');

// State
let currentImageFile = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkAPIStatus();
    setupEventListeners();
    setInterval(checkAPIStatus, 30000); // Check status every 30 seconds
});

// Event Listeners
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Confidence slider
    confidenceSlider.addEventListener('input', (e) => {
        const value = (e.target.value / 100).toFixed(2);
        confidenceValue.textContent = value;
    });

    // Clear button
    clearBtn.addEventListener('click', clearResults);
}

// File Handling
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processFile(file);
    } else {
        showError('Please drop a valid image file (JPG, PNG, JPEG)');
    }
}

function processFile(file) {
    if (!file.type.match('image/(jpeg|png|jpg)')) {
        showError('Invalid file type. Please upload JPG, PNG, or JPEG images.');
        return;
    }

    if (file.size > 10 * 1024 * 1024) { // 10MB limit
        showError('File too large. Maximum size is 10MB.');
        return;
    }

    currentImageFile = file;
    
    // Display original image
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImage.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Analyze image
    analyzeImage(file);
}

// API Functions
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        const statusIndicator = document.getElementById('apiStatus');
        const statusText = document.getElementById('apiStatusText');
        const classCount = document.getElementById('classCount');
        
        if (data.status === 'healthy') {
            statusIndicator.className = 'status-indicator online';
            statusText.textContent = 'Online';
            if (data.models.detector.classes) {
                classCount.textContent = Object.keys(data.models.detector.classes).length;
            }
        } else {
            statusIndicator.className = 'status-indicator offline';
            statusText.textContent = 'Offline';
        }
    } catch (error) {
        const statusIndicator = document.getElementById('apiStatus');
        const statusText = document.getElementById('apiStatusText');
        statusIndicator.className = 'status-indicator offline';
        statusText.textContent = 'Error';
        console.error('Status check failed:', error);
    }
}

async function analyzeImage(file) {
    showLoading();
    hideError();
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('enhance', enhanceToggle.checked);
        formData.append('confidence_threshold', confidenceSlider.value / 100);

        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Analysis failed');
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        showError(`Analysis failed: ${error.message}`);
        console.error('Analysis error:', error);
    } finally {
        hideLoading();
    }
}

// Display Functions
function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';
    
    // Draw detections on canvas
    drawDetections(data);
    
    // Update statistics
    updateStatistics(data);
    
    // Display detection list
    displayDetectionList(data.detections);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function drawDetections(data) {
    const img = new Image();
    img.onload = () => {
        const canvas = analyzedCanvas;
        const ctx = canvas.getContext('2d');
        
        // Set canvas size to match image
        canvas.width = img.width;
        canvas.height = img.height;
        
        // Draw original image
        ctx.drawImage(img, 0, 0);
        
        // Draw bounding boxes
        data.detections.forEach((detection, index) => {
            const [x1, y1, x2, y2] = detection.bbox;
            const width = x2 - x1;
            const height = y2 - y1;
            
            // Generate color based on class
            const color = getClassColor(index);
            
            // Draw box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, width, height);
            
            // Draw label background
            const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(1)}%`;
            ctx.font = 'bold 16px Arial';
            const textMetrics = ctx.measureText(label);
            const textHeight = 20;
            
            ctx.fillStyle = color;
            ctx.fillRect(x1, y1 - textHeight - 4, textMetrics.width + 10, textHeight + 4);
            
            // Draw label text
            ctx.fillStyle = 'white';
            ctx.fillText(label, x1 + 5, y1 - 8);
        });
    };
    
    img.src = originalImage.src;
}

function updateStatistics(data) {
    // Total detections
    document.getElementById('totalDetections').textContent = data.detections.length;
    
    // Unique classes
    const uniqueClasses = new Set(data.detections.map(d => d.class_name));
    document.getElementById('uniqueClasses').textContent = uniqueClasses.size;
    
    // Average confidence
    if (data.detections.length > 0) {
        const avgConf = data.detections.reduce((sum, d) => sum + d.confidence, 0) / data.detections.length;
        document.getElementById('avgConfidence').textContent = `${(avgConf * 100).toFixed(1)}%`;
    } else {
        document.getElementById('avgConfidence').textContent = 'N/A';
    }
    
    // Processing time
    const totalTime = (data.processing_time.enhancement + data.processing_time.detection) * 1000;
    document.getElementById('processingTime').textContent = `${totalTime.toFixed(0)}ms`;
}

function displayDetectionList(detections) {
    const listContainer = document.getElementById('detectionsList');
    listContainer.innerHTML = '';
    
    if (detections.length === 0) {
        listContainer.innerHTML = '<p style="text-align: center; color: #7f8c8d;">No objects detected</p>';
        return;
    }
    
    detections.forEach((detection, index) => {
        const item = document.createElement('div');
        item.className = 'detection-item';
        
        const confidence = detection.confidence * 100;
        const confidenceClass = confidence >= 70 ? 'confidence-high' : 
                               confidence >= 40 ? 'confidence-medium' : 'confidence-low';
        
        const [x1, y1, x2, y2] = detection.bbox;
        const width = Math.round(x2 - x1);
        const height = Math.round(y2 - y1);
        
        item.innerHTML = `
            <div class="detection-info">
                <div class="detection-icon" style="background: ${getClassColor(index)}">
                    ${getClassEmoji(detection.class_name)}
                </div>
                <div class="detection-details">
                    <h4>${detection.class_name}</h4>
                    <p>Position: (${Math.round(x1)}, ${Math.round(y1)}) | Size: ${width}×${height}px</p>
                </div>
            </div>
            <div class="confidence-badge ${confidenceClass}">
                ${confidence.toFixed(1)}%
            </div>
        `;
        
        listContainer.appendChild(item);
    });
}

// Helper Functions
function getClassColor(index) {
    const colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
        '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'
    ];
    return colors[index % colors.length];
}

function getClassEmoji(className) {
    const emojiMap = {
        'fish': '🐟',
        'jellyfish': '🪼',
        'penguin': '🐧',
        'puffin': '🐦',
        'shark': '🦈',
        'starfish': '⭐',
        'stingray': '🐡'
    };
    return emojiMap[className] || '🔍';
}

function showLoading() {
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
    setTimeout(() => {
        hideError();
    }, 5000);
}

function hideError() {
    errorMessage.style.display = 'none';
}

function clearResults() {
    resultsSection.style.display = 'none';
    currentImageFile = null;
    fileInput.value = '';
    originalImage.src = '';
    
    const ctx = analyzedCanvas.getContext('2d');
    ctx.clearRect(0, 0, analyzedCanvas.width, analyzedCanvas.height);
    
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
