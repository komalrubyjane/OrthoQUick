document.addEventListener('DOMContentLoaded', function () {
    // Load history on page load
    loadHistory();

    // Attach event listener to the analysis form submission
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function (event) {
            event.preventDefault();
            uploadImage(event);
        });
    }

    // Dark Mode Toggle
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', function () {
            document.body.classList.toggle('dark-mode');
            this.textContent = document.body.classList.contains('dark-mode') ? 'Light Mode' : 'Dark Mode';
        });
    }

    // Scroll animations with Intersection Observer
    const sections = document.querySelectorAll('.hero-section, .history-section');
    const observerOptions = {
        threshold: 0.2
    };
    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
                const skeletonImg = entry.target.querySelector('[data-pop-animation] .skeleton-img');
                if (skeletonImg) {
                    animateSkeleton(skeletonImg);
                }
            }
        });
    }, observerOptions);
    sections.forEach(section => observer.observe(section));

    // Trigger initial skeleton animation
    document.querySelectorAll('[data-pop-animation] .skeleton-img').forEach(img => {
        animateSkeleton(img);
    });

    // Create animated background
    createAnimatedBackground();
});

// Function to load history
async function loadHistory() {
    try {
        const response = await fetch('/get_history');
        const data = await response.json();
        if (data.status === 'success') {
            const historyList = document.getElementById('historyList');
            historyList.innerHTML = '';

            data.history.forEach(entry => {
                const card = document.createElement('div');
                card.className = 'col-md-4'; // Bootstrap column class
                card.innerHTML = `
                    <div class="history-card h-100">
                        <div class="card-body">
                            <p class="card-text"><strong>Timestamp:</strong> ${entry.timestamp}</p>
                            <p class="card-text"><strong>Category:</strong> ${entry.category || entry.current_category || 'N/A'}</p>
                            <p class="card-text"><strong>Bones Present:</strong> ${entry.bones_present ? entry.bones_present.join(', ') : 'N/A'}</p>
                            <p class="card-text"><strong>Fracture Detected:</strong> ${entry.fracture_detected !== undefined ? (entry.fracture_detected ? 'Yes' : 'No') : 'N/A'}</p>
                            ${entry.fracture_detected ? `
                                <p class="card-text"><strong>Fractured Bone:</strong> ${entry.fractured_bone || 'N/A'}</p>
                                <p class="card-text"><strong>Fracture Type:</strong> ${entry.fracture_type || 'N/A'}</p>
                                <p class="card-text"><strong>Fracture Coordinates:</strong> (${entry.fracture_coordinates ? entry.fracture_coordinates.join(', ') : 'N/A'})</p>
                                <p class="card-text"><strong>Severity:</strong> ${entry.severity || 'N/A'}% (${entry.severity_grade || 'N/A'})</p>
                            ` : ''}
                            ${entry.initial_fractured_bone ? `
                                <p class="card-text"><strong>Initial Fractured Bone:</strong> ${entry.initial_fractured_bone}</p>
                                <p class="card-text"><strong>Initial Fracture Type:</strong> ${entry.initial_fracture_type}</p>
                                <p class="card-text"><strong>Initial Fracture Coordinates:</strong> (${entry.initial_fracture_coordinates.join(', ')})</p>
                                <p class="card-text"><strong>Initial Severity:</strong> ${entry.initial_severity}% (${entry.initial_severity_grade})</p>
                                <img src="${entry.initial_annotated_image_path}?${new Date().getTime()}" class="img-fluid rounded mt-2" style="max-width: 100%;" onerror="console.error('Failed to load initial annotated image: ${entry.initial_annotated_image_path}')">
                            ` : ''}
                            ${entry.current_fractured_bone ? `
                                <p class="card-text"><strong>Current Fractured Bone:</strong> ${entry.current_fractured_bone}</p>
                                <p class="card-text"><strong>Current Fracture Type:</strong> ${entry.current_fracture_type}</p>
                                <p class="card-text"><strong>Current Fracture Coordinates:</strong> (${entry.current_fracture_coordinates.join(', ')})</p>
                                <p class="card-text"><strong>Current Severity:</strong> ${entry.current_severity}% (${entry.current_severity_grade})</p>
                            ` : ''}
                            <p class="card-text"><strong>Bone Health:</strong> ${entry.current_bone_health?.toFixed(2) || entry.bone_health?.toFixed(2) || 'N/A'}</p>
                            <p class="card-text"><strong>Confidence:</strong> ${entry.confidence?.toFixed(2) || 'N/A'}%</p>
                            ${entry.low_confidence_warning ? '<p class="text-warning"><strong>Warning:</strong> Low confidence prediction (&lt; 60%)</p>' : ''}
                            ${entry.improvement_percentage !== undefined ? `
                                <p class="card-text"><strong>Improvement:</strong> ${entry.improvement_percentage.toFixed(2)}%</p>
                                <p class="card-text"><strong>Healing Stage:</strong> ${entry.healing_stage}</p>
                            ` : ''}
                            <p class="card-text"><strong>Description:</strong> ${entry.current_image_description || entry.image_description || entry.description || 'N/A'}</p>
                            <img src="${entry.current_annotated_image_path || entry.annotated_image_path}?${new Date().getTime()}" alt="Annotated X-ray Image" class="img-fluid rounded mt-2" style="max-width: 100%;" onerror="console.error('Failed to load annotated image: ${entry.current_annotated_image_path || entry.annotated_image_path}')">
                            <form id="feedback-form-${entry.id}" class="mt-3">
                                <input type="hidden" name="history_id" value="${entry.id}">
                                <select name="rating" class="form-control mb-2" required>
                                    ${Array.from({ length: 5 }, (_, i) => i + 1).map(i => `<option value="${i}">${i} Star${i > 1 ? 's' : ''}</option>`).join('')}
                                </select>
                                <textarea name="comments" class="form-control mb-2" placeholder="Add comments (optional)"></textarea>
                                <button type="submit" class="btn btn-primary btn-sm">Submit Feedback</button>
                            </form>
                        </div>
                    </div>
                `;
                historyList.appendChild(card);

                // Attach feedback submission handler
                const feedbackForm = document.getElementById(`feedback-form-${entry.id}`);
                feedbackForm.addEventListener('submit', async function (e) {
                    e.preventDefault();
                    const formData = new FormData(this);
                    try {
                        const response = await fetch('/submit_feedback', {
                            method: 'POST',
                            body: new URLSearchParams(formData)
                        });
                        const result = await response.json();
                        alert(result.message);
                    } catch (error) {
                        console.error('Feedback submission error:', error);
                        alert('Error submitting feedback: ' + error.message);
                    }
                });
            });
        } else {
            console.error('Failed to load history:', data.error);
            document.getElementById('historyList').innerHTML = '<p class="text-danger">Failed to load history.</p>';
        }
    } catch (error) {
        console.error('Error loading history:', error);
        document.getElementById('historyList').innerHTML = '<p class="text-danger">Error loading history.</p>';
    }
}

// Function to upload and analyze image
async function uploadImage(event) {
    event.preventDefault();

    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select an image to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/analysis', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        if (result.status === 'success') {
            console.log('Analysis successful:', result);

            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '';

            const card = document.createElement('div');
            card.className = 'card mt-4';
            card.innerHTML = `
                <div class="card-body">
                    <p class="card-text"><strong>Predicted Bone:</strong> ${result.category || 'N/A'}</p>
                    <p class="card-text"><strong>Confidence:</strong> ${result.confidence?.toFixed(2) || 'N/A'}%</p>
                    ${result.low_confidence_warning ? '<p class="text-danger"><strong>Warning:</strong> Low confidence in classification</p>' : ''}
                    <p class="card-text"><strong>Bones Present:</strong> ${result.bones_present ? result.bones_present.join(', ') : 'N/A'}</p>
                    <p class="card-text"><strong>Fracture Detected:</strong> ${result.fracture_detected !== undefined ? (result.fracture_detected ? 'Yes' : 'No') : 'N/A'}</p>
                    ${result.fracture_detected ? `
                        <p class="card-text"><strong>Fractured Bone:</strong> ${result.fractured_bone || 'N/A'}</p>
                        <p class="card-text"><strong>Fracture Type:</strong> ${result.fracture_type || 'N/A'}</p>
                        <p class="card-text"><strong>Fracture Coordinates:</strong> (${result.fracture_coordinates ? result.fracture_coordinates.join(', ') : 'N/A'})</p>
                        <p class="card-text"><strong>Severity:</strong> ${result.severity || 'N/A'}% (${result.severity_grade || 'N/A'})</p>
                    ` : ''}
                    <p class="card-text"><strong>Bone Health Score:</strong> ${result.bone_health?.toFixed(2) || 'N/A'}</p>
                    <p class="card-text"><strong>Description:</strong> ${result.image_description || result.description || 'N/A'}</p>
                    <div class="card-text"><strong>Bone Details:</strong>
                        <ul class="list-group list-group-flush">
                            ${result.bones ? result.bones.map(bone => `<li class="list-group-item">${bone.name}: ${bone.description}</li>`).join('') : '<li>No bone details available</li>'}
                        </ul>
                    </div>
                    <img src="${result.annotated_image_path}?${new Date().getTime()}" alt="Annotated X-ray" class="img-fluid rounded mt-2" style="max-width: 400px;" onerror="console.error('Failed to load annotated image: ${result.annotated_image_path}')">
                </div>
            `;
            resultDiv.appendChild(card);

            await loadHistory();
        } else {
            console.error('Analysis failed:', result.error);
            alert('Analysis failed: ' + result.error);
        }
    } catch (error) {
        console.error('Error uploading image:', error);
        alert('Error uploading image: ' + error.message);
    }
}

// Function to retrain the model
async function retrainModel() {
    try {
        const response = await fetch('/retrain_model', {
            method: 'POST'
        });
        const result = await response.json();
        if (result.status === 'success') {
            alert('Model retrained successfully!');
            window.location.reload();
        } else {
            console.error('Model retraining failed:', result.error);
            alert('Model retraining failed: ' + result.error);
        }
    } catch (error) {
        console.error('Error retraining model:', error);
        alert('Error retraining model: ' + error.message);
    }
}

// Skeleton Pop Animation
function animateSkeleton(element) {
    if (!element) return;
    let startTime = null;
    const duration = 1200;

    function animate(currentTime) {
        if (!startTime) startTime = currentTime;
        const progress = Math.min((currentTime - startTime) / duration, 1);
        const easeProgress = easeInOutQuad(progress);

        if (progress < 1 && element.parentNode) {
            element.style.transform = `scale(${0.5 + easeProgress * 0.5}) translateY(${50 - easeProgress * 70}px) rotate(${20 - easeProgress * 40}deg)`;
            element.style.opacity = easeProgress;
            element.style.boxShadow = `0 0 ${20 * easeProgress}px rgba(0, 0, 0, ${0.2 * easeProgress})`;
            requestAnimationFrame(animate);
        } else if (element.parentNode) {
            element.style.transform = 'scale(1) translateY(0) rotate(0deg)';
            element.style.opacity = 1;
            element.style.boxShadow = '0 0 10px rgba(0, 0, 0, 0.1)';
        }
    }

    requestAnimationFrame(animate);
}

function easeInOutQuad(t) {
    return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}

// Animated Background
function createAnimatedBackground() {
    const background = document.createElement('div');
    background.className = 'background-skeletons';
    for (let i = 0; i < 3; i++) {
        const skeleton = document.createElement('div');
        skeleton.className = 'background-skeleton';
        background.appendChild(skeleton);
    }
    document.body.appendChild(background);
}