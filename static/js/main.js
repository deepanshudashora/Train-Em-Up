function selectDataset(dataset) {
    fetch('/select_dataset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `dataset=${dataset}`
    })
    .then(response => response.json())
    .then(data => {
        if (data.redirect) {
            window.location.href = data.redirect;
        }
    });
}

function startTraining() {
    const model1Form = document.getElementById('model1-form');
    const model2Form = document.getElementById('model2-form');
    
    if (!model1Form || !model2Form) {
        console.error('Could not find model forms');
        return;
    }
    
    const model1Params = getFormData('model1-form');
    const model2Params = getFormData('model2-form');
    
    // Convert numeric values from strings to numbers
    for (let model of [model1Params, model2Params]) {
        model.epochs = parseInt(model.epochs);
        model.batch_size = parseInt(model.batch_size);
        model.learning_rate = parseFloat(model.learning_rate);
        model.channels = parseInt(model.channels);
        model.dropout_rate = parseFloat(model.dropout_rate);
        
        // Validate epochs
        if (model.epochs < 1 || model.epochs > 10) {
            alert('Number of epochs must be between 1 and 10');
            return;
        }
    }
    
    // Store epochs in localStorage for training page
    localStorage.setItem('model1_epochs', model1Params.epochs);
    localStorage.setItem('model2_epochs', model2Params.epochs);
    
    // Store full configurations in localStorage
    localStorage.setItem('model1_config', JSON.stringify(model1Params));
    localStorage.setItem('model2_config', JSON.stringify(model2Params));
    
    fetch('/start_training', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model1: model1Params,
            model2: model2Params
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'started') {
            window.location.href = '/training';
        }
    })
    .catch(error => {
        console.error('Error starting training:', error);
        alert('Error starting training. Please check the console for details.');
    });
}

function getFormData(formId) {
    const form = document.getElementById(formId);
    const formData = new FormData(form);
    const data = {};
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    return data;
}

let trainingInterval;

function updateTrainingProgress() {
    console.log('Updating training progress...');
    
    // Fetch all required data
    Promise.all([
        fetch('/get_training_progress')
            .then(response => {
                if (!response.ok) throw new Error('Failed to fetch training progress');
                return response.json();
            }),
        fetch('/get_training_status')
            .then(response => {
                if (!response.ok) throw new Error('Failed to fetch training status');
                return response.json();
            }),
        fetch('/get_training_logs')
            .then(response => {
                if (!response.ok) throw new Error('Failed to fetch training logs');
                return response.json();
            })
    ]).then(([progressData, statusData, logsData]) => {
        console.log('Progress Data:', progressData);
        console.log('Status Data:', statusData);
        console.log('Logs Data:', logsData);

        // Process progress data
        for (let model of ['model1', 'model2']) {
            if (progressData[model]) {
                const data = progressData[model];
                updateProgressBar(model.slice(-1), data.progress);
                
                // Update stats display
                document.getElementById(`${model}-epoch`).textContent = data.current_epoch;
                document.getElementById(`${model}-progress`).textContent = Math.round(data.progress);
                document.getElementById(`${model}-loss`).textContent = 
                    `Train: ${data.current_loss.toFixed(4)} | Test: ${data.current_test_loss.toFixed(4)}`;
                document.getElementById(`${model}-accuracy`).textContent = 
                    `Train: ${data.current_accuracy.toFixed(2)}% | Test: ${data.current_test_accuracy.toFixed(2)}%`;
                document.getElementById(`${model}-batch-size`).textContent = 
                    `${data.batch_size} (Effective: ${data.effective_batch_size})`;
            }
        }

        // Update charts if we have new data
        if (statusData && (statusData.model1.accuracy.length > 0 || statusData.model2.accuracy.length > 0)) {
            console.log('Updating charts with new data');
            const chartData = {
                model1: {
                    accuracy: statusData.model1.accuracy || [],
                    test_accuracy: statusData.model1.test_accuracy || [],
                    loss: statusData.model1.loss || [],
                    test_loss: statusData.model1.test_loss || []
                },
                model2: {
                    accuracy: statusData.model2.accuracy || [],
                    test_accuracy: statusData.model2.test_accuracy || [],
                    loss: statusData.model2.loss || [],
                    test_loss: statusData.model2.test_loss || []
                }
            };
            updateCharts(chartData);
        }

        // Update logs
        if (logsData && (logsData.model1.length > 0 || logsData.model2.length > 0)) {
            console.log('Updating training logs');
            updateLogs({
                model1: logsData.model1 || [],
                model2: logsData.model2 || []
            });
        }

        // Check if training is complete
        checkTrainingComplete(progressData);
    }).catch(error => {
        console.error('Error updating training progress:', error);
    });
}

function updateProgressBar(modelNum, progress) {
    const progressBar = document.getElementById(`progress-bar-${modelNum}`);
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
    }
}

function updateLogs(data) {
    function createProgressBar(value, max) {
        const percentage = (value / max) * 100;
        return `
            <div class="log-progress-bar">
                <div class="log-progress-fill" style="width: ${percentage}%"></div>
            </div>
        `;
    }

    function formatLog(log, totalEpochs) {
        const lines = log.split('\n');
        let formattedLog = '';
        
        lines.forEach(line => {
            if (line.includes('Epoch')) {
                const epochNum = parseInt(line.match(/\d+/)[0]);
                formattedLog += `
                    <div class="log-header">
                        ${line}
                        ${createProgressBar(epochNum, totalEpochs)}
                    </div>
                `;
            } else if (line.includes('Training')) {
                formattedLog += `<div class="log-section-header">${line}</div>`;
            } else if (line.includes('Testing')) {
                formattedLog += `<div class="log-section-header">${line}</div>`;
            } else if (line.includes('Loss') || line.includes('Accuracy')) {
                formattedLog += `<div class="log-metrics">${line}</div>`;
            } else {
                formattedLog += `<div class="log-info">${line}</div>`;
            }
        });
        
        return formattedLog;
    }

    // Update Model 1 logs
    const logContainer1 = document.getElementById('log-container-1');
    if (logContainer1 && Array.isArray(data.model1)) {
        const totalEpochs = parseInt(document.getElementById('model1-total-epochs').textContent);
        let logContent = '';
        
        data.model1.forEach(log => {
            logContent += `
                <div class="epoch-log">
                    ${formatLog(log, totalEpochs)}
                </div>
            `;
        });
        
        logContainer1.innerHTML = logContent;
        logContainer1.scrollTop = logContainer1.scrollHeight;
    }
    
    // Update Model 2 logs
    const logContainer2 = document.getElementById('log-container-2');
    if (logContainer2 && Array.isArray(data.model2)) {
        const totalEpochs = parseInt(document.getElementById('model2-total-epochs').textContent);
        let logContent = '';
        
        data.model2.forEach(log => {
            logContent += `
                <div class="epoch-log">
                    ${formatLog(log, totalEpochs)}
                </div>
            `;
        });
        
        logContainer2.innerHTML = logContent;
        logContainer2.scrollTop = logContainer2.scrollHeight;
    }
}

function updateCharts(data) {
    const chartConfig = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            family: 'Inter, sans-serif',
            color: '#ffffff',
            size: 12
        },
        margin: { t: 30, r: 15, b: 40, l: 60 },
        showlegend: true,
        legend: {
            bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#8b95a5' },
            orientation: 'h',
            y: -0.2
        },
        xaxis: {
            gridcolor: 'rgba(255,255,255,0.1)',
            zerolinecolor: 'rgba(255,255,255,0.1)',
            tickfont: { color: '#8b95a5' },
            showgrid: true,
            gridwidth: 1
        },
        yaxis: {
            gridcolor: 'rgba(255,255,255,0.1)',
            zerolinecolor: 'rgba(255,255,255,0.1)',
            tickfont: { color: '#8b95a5' },
            showgrid: true,
            gridwidth: 1
        }
    };

    // Create traces for accuracy plot with both training and testing
    const accuracyTraces = [
        {
            x: Array.from({length: data.model1.accuracy.length}, (_, i) => i + 1),
            y: data.model1.accuracy,
            name: 'Model 1 Training Accuracy',
            type: 'scatter',
            line: { color: '#4a90e2', width: 2 }
        },
        {
            x: Array.from({length: data.model1.test_accuracy.length}, (_, i) => i + 1),
            y: data.model1.test_accuracy,
            name: 'Model 1 Testing Accuracy',
            type: 'scatter',
            line: { color: '#4a90e2', width: 2, dash: 'dot' }
        },
        {
            x: Array.from({length: data.model2.accuracy.length}, (_, i) => i + 1),
            y: data.model2.accuracy,
            name: 'Model 2 Training Accuracy',
            type: 'scatter',
            line: { color: '#2ecc71', width: 2 }
        },
        {
            x: Array.from({length: data.model2.test_accuracy.length}, (_, i) => i + 1),
            y: data.model2.test_accuracy,
            name: 'Model 2 Testing Accuracy',
            type: 'scatter',
            line: { color: '#2ecc71', width: 2, dash: 'dot' }
        }
    ];

    // Create traces for loss plot with both training and testing
    const lossTraces = [
        {
            x: Array.from({length: data.model1.loss.length}, (_, i) => i + 1),
            y: data.model1.loss,
            name: 'Model 1 Training Loss',
            type: 'scatter',
            line: { color: '#4a90e2', width: 2 }
        },
        {
            x: Array.from({length: data.model1.test_loss.length}, (_, i) => i + 1),
            y: data.model1.test_loss,
            name: 'Model 1 Testing Loss',
            type: 'scatter',
            line: { color: '#4a90e2', width: 2, dash: 'dot' }
        },
        {
            x: Array.from({length: data.model2.loss.length}, (_, i) => i + 1),
            y: data.model2.loss,
            name: 'Model 2 Training Loss    ',
            type: 'scatter',
            line: { color: '#2ecc71', width: 2 }
        },
        {
            x: Array.from({length: data.model2.test_loss.length}, (_, i) => i + 1),
            y: data.model2.test_loss,
            name: 'Model 2 Testing Loss',
            type: 'scatter',
            line: { color: '#2ecc71', width: 2, dash: 'dot' }
        }
    ];

    // Plot both charts with all metrics
    Plotly.react('accuracy-chart', accuracyTraces, {
        ...chartConfig,
        title: {
            text: 'Training & Testing Accuracy for both the Models',
            font: { color: '#ffffff', size: 20 }
        },
        yaxis: { ...chartConfig.yaxis, title: { text: 'Accuracy (%)', font: { color: '#8b95a5' } } }
    });

    Plotly.react('loss-chart', lossTraces, {
        ...chartConfig,
        title: {
            text: 'Training & Testing Loss for both the Models',
            font: { color: '#ffffff', size: 20 }
        },
        yaxis: { ...chartConfig.yaxis, title: { text: 'Loss', font: { color: '#8b95a5' } } }
    });
}

// Start progress monitoring when on training page
if (window.location.pathname === '/training') {
    // Set the total epochs from localStorage
    document.getElementById('model1-total-epochs').textContent = 
        localStorage.getItem('model1_epochs') || '5';
    document.getElementById('model2-total-epochs').textContent = 
        localStorage.getItem('model2_epochs') || '5';
    
    // Initialize empty charts
    const emptyData = {
        model1: { 
            accuracy: [], 
            loss: [], 
            test_accuracy: [], 
            test_loss: [] 
        },
        model2: { 
            accuracy: [], 
            loss: [], 
            test_accuracy: [], 
            test_loss: [] 
        }
    };
    
    // Initial chart creation
    updateCharts(emptyData);
    
    // Start polling with a longer interval
    trainingInterval = setInterval(updateTrainingProgress, 2000);
}

// Add this function to check if training is complete
function checkTrainingComplete(data) {
    const model1Complete = data.model1.current_epoch === data.model1.total_epochs && data.model1.progress >= 100;
    const model2Complete = data.model2.current_epoch === data.model2.total_epochs && data.model2.progress >= 100;
    
    if (model1Complete && model2Complete) {
        showPredictionButton();
        clearInterval(trainingInterval);  // Stop polling for updates
    }
}

// Update the showPredictionButton function
function showPredictionButton() {
    // Check if button already exists
    if (document.querySelector('.prediction-button-container')) {
        return;  // If button exists, don't add another one
    }

    const container = document.querySelector('.container');
    const buttonDiv = document.createElement('div');
    buttonDiv.className = 'prediction-button-container';
    buttonDiv.innerHTML = `
        <button class="prediction-btn" onclick="window.location.href='/prediction'">
            View Model Predictions
        </button>
    `;
    container.appendChild(buttonDiv);
} 