$(document).ready(function() {
    function updateStatus(message) {
        $('#status').text(message);
    }

    // Fetch and display history on page load
    function loadHistory() {
        $.ajax({
            url: '/get_history',
            type: 'GET',
            success: function(response) {
                if (response.status === 'error') {
                    $('#history-result').html(`<p class="text-danger">Error: ${response.error}</p>`);
                    console.error('History fetch error:', response.error);
                    return;
                }
                let html = '';
                response.history.forEach(entry => {
                    html += '<div class="col-md-4">';
                    html += '<div class="history-card">';
                    html += `<p><strong>Timestamp:</strong> ${entry.timestamp}</p>`;
                    html += `<p><strong>Category:</strong> ${entry.category || entry.current_category}</p>`;
                    html += `<p><strong>Bones Present:</strong> ${entry.bones_present ? entry.bones_present.join(', ') : 'N/A'}</p>`;
                    html += `<p><strong>Fracture Detected:</strong> ${entry.fracture_detected !== undefined ? (entry.fracture_detected ? 'Yes' : 'No') : 'N/A'}</p>`;
                    if (entry.fracture_detected) {
                        html += `<p><strong>Fractured Bone:</strong> ${entry.fractured_bone}</p>`;
                        html += `<p><strong>Fracture Type:</strong> ${entry.fracture_type}</p>`;
                        html += `<p><strong>Fracture Coordinates:</strong> (${entry.fracture_coordinates[0]}, ${entry.fracture_coordinates[1]})</p>`;
                        html += `<p><strong>Severity:</strong> ${entry.severity}% (${entry.severity_grade})</p>`;
                    }
                    if (entry.initial_fractured_bone) {
                        html += `<p><strong>Initial Fractured Bone:</strong> ${entry.initial_fractured_bone}</p>`;
                        html += `<p><strong>Initial Fracture Type:</strong> ${entry.initial_fracture_type}</p>`;
                        html += `<p><strong>Initial Fracture Coordinates:</strong> (${entry.initial_fracture_coordinates[0]}, ${entry.initial_fracture_coordinates[1]})</p>`;
                        html += `<p><strong>Initial Severity:</strong> ${entry.initial_severity}% (${entry.initial_severity_grade})</p>`;
                        html += `<img src="${entry.initial_annotated_image_path}?${new Date().getTime()}" class="img-fluid rounded mt-2" style="max-width: 100%;" onerror="console.log('Failed to load initial annotated image: ${entry.initial_annotated_image_path}')">`;
                    }
                    if (entry.current_fractured_bone) {
                        html += `<p><strong>Current Fractured Bone:</strong> ${entry.current_fractured_bone}</p>`;
                        html += `<p><strong>Current Fracture Type:</strong> ${entry.current_fracture_type}</p>`;
                        html += `<p><strong>Current Fracture Coordinates:</strong> (${entry.current_fracture_coordinates[0]}, ${entry.current_fracture_coordinates[1]})</p>`;
                        html += `<p><strong>Current Severity:</strong> ${entry.current_severity}% (${entry.current_severity_grade})</p>`;
                    }
                    html += `<p><strong>Bone Health:</strong> ${entry.current_bone_health?.toFixed(2) || entry.bone_health?.toFixed(2)}</p>`;
                    html += `<p><strong>Confidence:</strong> ${entry.confidence?.toFixed(2) || 'N/A'}%</p>`;
                    if (entry.low_confidence_warning) {
                        html += `<p class="text-warning">Warning: Low confidence prediction (< 60%)</p>`;
                    }
                    if (entry.improvement_percentage !== undefined) {
                        html += `<p><strong>Improvement:</strong> ${entry.improvement_percentage.toFixed(2)}%</p>`;
                        html += `<p><strong>Healing Stage:</strong> ${entry.healing_stage}</p>`;
                    }
                    html += `<p><strong>Description:</strong> ${entry.current_image_description || entry.image_description || 'N/A'}</p>`;
                    html += `<img src="${entry.current_annotated_image_path || entry.annotated_image_path}?${new Date().getTime()}" class="img-fluid rounded mt-2" style="max-width: 100%;" onerror="console.log('Failed to load annotated image: ${entry.current_annotated_image_path || entry.annotated_image_path}')">`;
                    html += '<form id="feedback-form-' + entry.id + '" class="mt-3">';
                    html += '<input type="hidden" name="history_id" value="' + entry.id + '">';
                    html += '<select name="rating" class="form-control mb-2" required>';
                    for (let i = 1; i <= 5; i++) {
                        html += `<option value="${i}">${i} Star${i > 1 ? 's' : ''}</option>`;
                    }
                    html += '</select>';
                    html += '<textarea name="comments" class="form-control mb-2" placeholder="Add comments (optional)"></textarea>';
                    html += '<button type="submit" class="btn btn-primary btn-sm">Submit Feedback</button>';
                    html += '</form>';
                    html += '</div>';
                    html += '</div>';
                });
                $('#history-result').html(html);

                // Attach feedback submission handlers
                response.history.forEach(entry => {
                    $('#feedback-form-' + entry.id).submit(function(e) {
                        e.preventDefault();
                        const formData = $(this).serialize();
                        $.ajax({
                            url: '/submit_feedback',
                            type: 'POST',
                            data: formData,
                            success: function(response) {
                                alert(response.message);
                            },
                            error: function(xhr, status, error) {
                                console.error('Feedback submission error:', status, error, xhr.responseText);
                                alert('Error submitting feedback: ' + error);
                            }
                        });
                    });
                });
            },
            error: function(xhr, status, error) {
                console.error('History fetch error:', status, error, xhr.responseText);
                $('#history-result').html('<p class="text-danger">Failed to load history.</p>');
            }
        });
    }
    loadHistory();

    // Dark Mode Toggle
    $('#dark-mode-toggle').click(function() {
        $('body').toggleClass('dark-mode');
        $(this).text($('body').hasClass('dark-mode') ? 'Light Mode' : 'Dark Mode');
    });

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

    // Trigger initial skeleton animation
    document.querySelectorAll('[data-pop-animation] .skeleton-img').forEach(img => {
        animateSkeleton(img);
    });

    // Handle analysis form submission
    $('#analysis-form').submit(function(e) {
        e.preventDefault();
        let formData = new FormData(this);
        const fileInput = $('#image')[0].files[0];
        if (!fileInput) {
            $('#prediction-result').html('<p class="text-danger">No image selected for analysis.</p>');
            updateStatus('No image selected');
            return;
        }

        updateStatus('Analyzing...');
        $.ajax({
            url: '/analysis',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.status === 'error') {
                    $('#prediction-result').html(`<p class="text-danger">Error: ${response.error}</p>`);
                    updateStatus('Analysis failed');
                    console.error('Analysis error:', response.error);
                } else {
                    let html = `<h5 class="mt-3">Predicted Category: ${response.category}</h5>`;
                    html += `<p>Bones Present: ${response.bones_present.join(', ')}</p>`;
                    html += `<p>Confidence: ${response.confidence.toFixed(2)}%</p>`;
                    html += `<p>Fracture Detected: ${response.fracture_detected ? 'Yes' : 'No'}</p>`;
                    if (response.fracture_detected) {
                        html += `<p>Fractured Bone: ${response.fractured_bone}</p>`;
                        html += `<p>Fracture Type: ${response.fracture_type}</p>`;
                        html += `<p>Fracture Coordinates: (${response.fracture_coordinates[0]}, ${response.fracture_coordinates[1]})</p>`;
                        html += `<p>Severity: ${response.severity}% (${response.severity_grade})</p>`;
                    }
                    html += `<p>Bone Health Score: ${response.bone_health.toFixed(2)}</p>`;
                    if (response.low_confidence_warning) {
                        html += `<p class="text-warning">Warning: Low confidence prediction (< 60%)</p>`;
                    }
                    html += `<p><strong>Description:</strong> ${response.image_description}</p>`;
                    html += `<img src="${response.annotated_image_path}?${new Date().getTime()}" class="img-fluid rounded mt-2" style="max-width: 400px;" onerror="console.log('Failed to load annotated image: ${response.annotated_image_path}')">`;
                    html += `<h6 class="mt-3">Bones in this region:</h6><ul>`;
                    response.bones.forEach(bone => {
                        html += `<li>${bone.name}: ${bone.description}</li>`;
                    });
                    html += '</ul>';
                    $('#prediction-result').html(html);
                    updateStatus('Analysis completed');
                    loadHistory();
                }
            },
            error: function(xhr, status, error) {
                console.error('Analysis error:', status, error, xhr.responseText);
                $('#prediction-result').html(`<p class="text-danger">Error during analysis: ${status} - ${error}</p>`);
                updateStatus('Error');
            }
        });
    });

    // Handle difference form submission
    $('#difference-form').submit(function(e) {
        e.preventDefault();
        let formData = new FormData(this);
        const beforeInput = $('#before_image')[0].files[0];
        const afterInput = $('#after_image')[0].files[0];
        if (!beforeInput || !afterInput) {
            $('#prediction-result').html('<p class="text-danger">Both before and after images are required.</p>');
            updateStatus('No images selected');
            return;
        }

        updateStatus('Comparing...');
        $.ajax({
            url: '/difference',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.status === 'error') {
                    $('#prediction-result').html(`<p class="text-danger">Error: ${response.error}</p>`);
                    updateStatus('Comparison failed');
                    console.error('Difference error:', response.error);
                } else {
                    let html = `<h5 class="mt-3">Difference Analysis</h5>`;
                    html += `<p><strong>Category:</strong> ${response.category}</p>`;
                    html += `<p><strong>Bones Present:</strong> ${response.bones_present.join(', ')}</p>`;
                    html += `<p><strong>Initial Fractured Bone:</strong> ${response.initial_fractured_bone}</p>`;
                    html += `<p><strong>Initial Fracture Type:</strong> ${response.initial_fracture_type}</p>`;
                    html += `<p><strong>Initial Fracture Coordinates:</strong> (${response.initial_fracture_coordinates[0]}, ${response.initial_fracture_coordinates[1]})</p>`;
                    html += `<p><strong>Initial Severity:</strong> ${response.initial_severity}% (${response.initial_severity_grade})</p>`;
                    html += `<p><strong>Current Fractured Bone:</strong> ${response.current_fractured_bone}</p>`;
                    html += `<p><strong>Current Fracture Type:</strong> ${response.current_fracture_type}</p>`;
                    html += `<p><strong>Current Fracture Coordinates:</strong> (${response.current_fracture_coordinates[0]}, ${response.current_fracture_coordinates[1]})</p>`;
                    html += `<p><strong>Current Severity:</strong> ${response.current_severity}% (${response.current_severity_grade})</p>`;
                    html += `<p><strong>Initial Bone Health:</strong> ${response.initial_bone_health.toFixed(2)}</p>`;
                    html += `<p><strong>Current Bone Health:</strong> ${response.current_bone_health.toFixed(2)}</p>`;
                    if (response.low_confidence_warning) {
                        html += `<p class="text-warning">Warning: Low confidence prediction (< 60%)</p>`;
                    }
                    html += `<p><strong>Improvement:</strong> ${response.improvement_percentage.toFixed(2)}%</p>`;
                    html += `<p><strong>Healing Stage:</strong> ${response.healing_stage}</p>`;
                    html += `<p><strong>Initial Description:</strong> ${response.initial_image_description}</p>`;
                    html += `<p><strong>Current Description:</strong> ${response.current_image_description}</p>`;
                    html += `<img src="${response.initial_annotated_image_path}?${new Date().getTime()}" class="img-fluid rounded mt-2" style="max-width: 400px;" onerror="console.log('Failed to load initial annotated image: ${response.initial_annotated_image_path}')">`;
                    html += `<img src="${response.current_annotated_image_path}?${new Date().getTime()}" class="img-fluid rounded mt-2" style="max-width: 400px;" onerror="console.log('Failed to load current annotated image: ${response.current_annotated_image_path}')">`;
                    html += `<h6 class="mt-3">Bones in this region:</h6><ul>`;
                    response.bones.forEach(bone => {
                        html += `<li>${bone.name}: ${bone.description}</li>`;
                    });
                    html += '</ul>';
                    $('#prediction-result').html(html);
                    updateStatus('Comparison completed');
                    loadHistory();
                }
            },
            error: function(xhr, status, error) {
                console.error('Difference error:', status, error, xhr.responseText);
                $('#prediction-result').html(`<p class="text-danger">Error during comparison: ${status} - ${error}</p>`);
                updateStatus('Error');
            }
        });
    });

    // Animated Background
    function createAnimatedBackground() {
        const background = $('<div class="background-skeletons"></div>').appendTo('body');
        for (let i = 0; i < 3; i++) {
            $('<div class="background-skeleton"></div>').appendTo(background);
        }
    }
    createAnimatedBackground();
});