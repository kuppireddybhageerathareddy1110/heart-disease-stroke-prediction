// ==========================================
// HEART DISEASE AI - ADVANCED JAVASCRIPT
// Production-grade client-side logic
// ==========================================

const API_BASE = 'http://127.0.0.1:8001';

// State management
const state = {
    currentPrediction: null,
    currentPatient: null,
    explanationsLoaded: false
};

// ==========================================
// NAVIGATION
// ==========================================

document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const section = btn.dataset.section;
        showSection(section);
        
        // Update active nav button
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    });
});

function showSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    
    // Show target section
    const targetSection = document.getElementById(`${sectionName}Section`);
    if (targetSection) {
        targetSection.classList.add('active');
    }
}

// ==========================================
// TABS
// ==========================================

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        showTab(tab);
        
        // Update active tab button
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    });
});

function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    
    // Show target tab
    const targetTab = document.getElementById(`${tabName}Tab`);
    if (targetTab) {
        targetTab.classList.add('active');
    }
}

// ==========================================
// FORM SUBMISSION
// ==========================================

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading
    showLoading(true);
    
    try {
        // Collect form data
        const patientData = {
            age: parseFloat(document.getElementById('age').value),
            sex: document.getElementById('sex').value,
            oldpeak: parseFloat(document.getElementById('oldpeak').value),
            chest_pain: document.getElementById('chest_pain').value,
            restingbp_final: parseFloat(document.getElementById('restingbp_final').value),
            chol_final: parseFloat(document.getElementById('chol_final').value),
            maxhr_final: parseFloat(document.getElementById('maxhr_final').value),
            fasting_bs: document.getElementById('fasting_bs').value,
            resting_ecg: document.getElementById('resting_ecg').value,
            exercise_angina: document.getElementById('exercise_angina').value,
            st_slope: document.getElementById('st_slope').value
        };
        
        // Save patient data to state
        state.currentPatient = patientData;
        
        // Make prediction
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patientData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        state.currentPrediction = result;
        
        // Display results
        displayResults(result);
        
        // Load explanations in background
        loadExplanations(patientData);
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error making prediction. Please check if the API is running on http://127.0.0.1:8000');
    } finally {
        showLoading(false);
    }
});

// ==========================================
// DISPLAY RESULTS
// ==========================================

function displayResults(result) {
    // Hide placeholder, show results
    document.getElementById('resultsContent').style.display = 'none';
    document.getElementById('predictionResults').style.display = 'block';
    
    // Risk Badge
    const riskBadge = document.getElementById('riskBadge');
    const riskLevel = result.risk_level.toLowerCase();
    riskBadge.className = `risk-badge ${riskLevel}`;
    
    const riskIcons = {
        'low': '🟢',
        'medium': '🟡',
        'high': '🔴'
    };
    
    const riskTexts = {
        'low': 'Low Risk',
        'medium': 'Medium Risk',
        'high': 'High Risk'
    };
    
    riskBadge.innerHTML = `
        <div style="font-size: 4rem; margin-bottom: 1rem;">${riskIcons[riskLevel]}</div>
        <h2 style="font-size: 2rem; margin-bottom: 0.5rem;">${riskTexts[riskLevel]}</h2>
        <p style="font-size: 1.25rem; color: #4b5563;">${result.prediction}</p>
    `;
    
    // Probability Gauge
    const probability = result.probability * 100;
    document.getElementById('gaugePercent').textContent = probability.toFixed(1);
    
    // Update gauge fill
    const gaugeFill = document.getElementById('gaugeFill');
    gaugeFill.style.setProperty('--gauge-percent', `${probability}%`);
    
    // Confidence Interval
    const ci = result.confidence_interval;
    document.getElementById('confidenceText').textContent = 
        `${(ci.lower * 100).toFixed(1)}% - ${(ci.upper * 100).toFixed(1)}%`;
    
    // Risk Factors
    const riskList = document.getElementById('riskFactorsList');
    riskList.innerHTML = '';
    if (result.risk_factors && result.risk_factors.length > 0) {
        result.risk_factors.forEach(factor => {
            const li = document.createElement('li');
            li.innerHTML = `<i class="fas fa-circle" style="font-size: 0.5rem; color: #ef4444;"></i> ${factor}`;
            riskList.appendChild(li);
        });
    } else {
        riskList.innerHTML = '<li>No significant risk factors identified</li>';
    }
    
    // Protective Factors
    const protectiveList = document.getElementById('protectiveFactorsList');
    protectiveList.innerHTML = '';
    if (result.protective_factors && result.protective_factors.length > 0) {
        result.protective_factors.forEach(factor => {
            const li = document.createElement('li');
            li.innerHTML = `<i class="fas fa-circle" style="font-size: 0.5rem; color: #10b981;"></i> ${factor}`;
            protectiveList.appendChild(li);
        });
    } else {
        protectiveList.innerHTML = '<li>No significant protective factors identified</li>';
    }
    
    // Scroll to results
    document.getElementById('predictionResults').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'nearest' 
    });
}

// ==========================================
// LOAD EXPLANATIONS
// ==========================================

async function loadExplanations(patientData) {
    try {
        // Load waterfall plot
        const waterfallRes = await fetch(`${API_BASE}/explain/waterfall`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patientData)
        });
        
        if (waterfallRes.ok) {
            const waterfallData = await waterfallRes.json();
            const waterfallImg = document.getElementById('waterfallPlot');
            waterfallImg.src = `data:image/png;base64,${waterfallData.plot}`;
            waterfallImg.classList.add('loaded');
            waterfallImg.nextElementSibling.style.display = 'none';
        }
        
        // Load force plot
        const forceRes = await fetch(`${API_BASE}/explain/force`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patientData)
        });
        
        if (forceRes.ok) {
            const forceData = await forceRes.json();
            const forceFrame = document.getElementById('forcePlotFrame');
            forceFrame.srcdoc = forceData.plot;
            forceFrame.nextElementSibling.style.display = 'none';
        }
        
        // Load LIME explanation
        const limeRes = await fetch(`${API_BASE}/explain/lime`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patientData)
        });
        
        if (limeRes.ok) {
            const limeData = await limeRes.json();
            const limeImg = document.getElementById('limePlot');
            limeImg.src = `data:image/png;base64,${limeData.plot}`;
            limeImg.classList.add('loaded');
            limeImg.nextElementSibling.style.display = 'none';
            
            // Display feature weights
            if (limeData.feature_weights) {
                displayLimeWeights(limeData.feature_weights);
            }
        }
        
        state.explanationsLoaded = true;
        
    } catch (error) {
        console.error('Error loading explanations:', error);
    }
}

// ==========================================
// SHOW EXPLANATIONS
// ==========================================

function displayLimeWeights(weights) {
    const container = document.getElementById('limeWeightsContent');
    const section = document.getElementById('limeWeights');
    
    if (!weights || weights.length === 0) {
        section.style.display = 'none';
        return;
    }
    
    section.style.display = 'block';
    
    // Create table
    let html = '<table style="width: 100%; border-collapse: collapse;">';
    html += '<thead><tr style="background: #f3f4f6; border-bottom: 2px solid #e5e7eb;">';
    html += '<th style="padding: 0.75rem; text-align: left;">Feature</th>';
    html += '<th style="padding: 0.75rem; text-align: right;">Weight</th>';
    html += '<th style="padding: 0.75rem; text-align: left;">Impact</th>';
    html += '</tr></thead><tbody>';
    
    weights.forEach((item, idx) => {
        const isPositive = item.weight > 0;
        const color = isPositive ? '#ef4444' : '#10b981';
        const arrow = isPositive ? '↑' : '↓';
        const impact = isPositive ? 'Increases Risk' : 'Decreases Risk';
        
        html += `<tr style="border-bottom: 1px solid #e5e7eb;">`;
        html += `<td style="padding: 0.75rem;">${item.feature}</td>`;
        html += `<td style="padding: 0.75rem; text-align: right; color: ${color}; font-weight: 600;">`;
        html += `${item.weight > 0 ? '+' : ''}${item.weight.toFixed(4)}`;
        html += `</td>`;
        html += `<td style="padding: 0.75rem; color: ${color};">`;
        html += `<span style="margin-right: 0.5rem;">${arrow}</span>${impact}`;
        html += `</td>`;
        html += `</tr>`;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;
}

function showExplanations() {
    showSection('explain');
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('[data-section="explain"]').classList.add('active');
}

// ==========================================
// GLOBAL PLOTS
// ==========================================

async function loadSummaryPlot() {
    showLoading(true);
    try {
        const response = await fetch(`${API_BASE}/explain/summary`);
        const data = await response.json();
        
        const summaryImg = document.getElementById('summaryPlot');
        summaryImg.src = `data:image/png;base64,${data.plot}`;
        summaryImg.style.display = 'block';
        summaryImg.previousElementSibling.style.display = 'none';
    } catch (error) {
        console.error('Error loading summary plot:', error);
        alert('Error loading global summary. Please check if API is running.');
    } finally {
        showLoading(false);
    }
}

async function loadImportancePlot() {
    showLoading(true);
    try {
        const response = await fetch(`${API_BASE}/explain/importance`);
        const data = await response.json();
        
        const importanceImg = document.getElementById('importancePlot');
        importanceImg.src = `data:image/png;base64,${data.plot}`;
        importanceImg.style.display = 'block';
        importanceImg.previousElementSibling.style.display = 'none';
    } catch (error) {
        console.error('Error loading importance plot:', error);
        alert('Error loading feature importance. Please check if API is running.');
    } finally {
        showLoading(false);
    }
}

// ==========================================
// SAMPLE PATIENT
// ==========================================

function loadSamplePatient() {
    // High-risk patient sample
    document.getElementById('age').value = 65;
    document.getElementById('sex').value = 'Male';
    document.getElementById('chest_pain').value = 'Asymptomatic';
    document.getElementById('restingbp_final').value = 160;
    document.getElementById('chol_final').value = 300;
    document.getElementById('fasting_bs').value = 'Yes';
    document.getElementById('resting_ecg').value = 'ST-T Abnormality';
    document.getElementById('maxhr_final').value = 130;
    document.getElementById('exercise_angina').value = 'Yes';
    document.getElementById('oldpeak').value = 2.5;
    document.getElementById('st_slope').value = 'Flat';
    
    // Show notification
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #10b981;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        z-index: 10000;
        animation: slideIn 0.3s;
    `;
    notification.innerHTML = '<i class="fas fa-check-circle"></i> Sample patient data loaded';
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s';
        setTimeout(() => notification.remove(), 300);
    }, 2000);
}

// ==========================================
// DOWNLOAD REPORT
// ==========================================

function downloadReport() {
    if (!state.currentPrediction || !state.currentPatient) {
        alert('No prediction available to download');
        return;
    }
    
    const report = {
        ...state.currentPatient,
        prediction: state.currentPrediction.prediction,
        probability: (state.currentPrediction.probability * 100).toFixed(2) + '%',
        risk_level: state.currentPrediction.risk_level,
        timestamp: new Date().toISOString()
    };
    
    const csv = convertToCSV(report);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `heart_disease_report_${Date.now()}.csv`;
    a.click();
    
    URL.revokeObjectURL(url);
}

function convertToCSV(obj) {
    const headers = Object.keys(obj).join(',');
    const values = Object.values(obj).join(',');
    return `${headers}\n${values}`;
}

// ==========================================
// LOADING OVERLAY
// ==========================================

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (show) {
        overlay.classList.add('active');
    } else {
        overlay.classList.remove('active');
    }
}

// ==========================================
// ANIMATIONS
// ==========================================

const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ==========================================
// INITIALIZE
// ==========================================

console.log('Heart Disease AI - Advanced System Loaded');
console.log('API Endpoint:', API_BASE);
console.log('Ready for predictions!');
