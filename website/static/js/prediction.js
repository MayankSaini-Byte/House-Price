/**
 * Prediction Form Handler — Sends data to the Flask API and displays results.
 */
(function () {
    'use strict';

    const form = document.getElementById('prediction-form');
    const resultEmpty = document.getElementById('result-empty');
    const resultContent = document.getElementById('result-content');
    const resultError = document.getElementById('result-error');
    const resultPrice = document.getElementById('result-price');
    const resultConfidenceBar = document.getElementById('result-confidence-bar');
    const resultConfidenceText = document.getElementById('result-confidence-text');
    const resultSummary = document.getElementById('result-summary');
    const errorMessage = document.getElementById('error-message');

    if (!form) return;

    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        // Collect form data
        const formData = new FormData(form);
        const data = {};
        formData.forEach(function (value, key) {
            data[key] = value;
        });

        // Add default values for features not in the form
        // These are needed by the pipeline but have reasonable defaults
        data['YrSold'] = data['YrSold'] || '2008';
        data['BsmtFullBath'] = data['BsmtFullBath'] || '0';
        data['BsmtHalfBath'] = data['BsmtHalfBath'] || '0';
        data['WoodDeckSF'] = data['WoodDeckSF'] || '0';
        data['OpenPorchSF'] = data['OpenPorchSF'] || '0';
        data['EnclosedPorch'] = data['EnclosedPorch'] || '0';
        data['3SsnPorch'] = data['3SsnPorch'] || '0';
        data['ScreenPorch'] = data['ScreenPorch'] || '0';
        data['LotFrontage'] = data['LotFrontage'] || '65';
        data['MSSubClass'] = data['MSSubClass'] || '20';
        data['1stFlrSF'] = data['GrLivArea'] || '1500';
        data['2ndFlrSF'] = data['2ndFlrSF'] || '0';

        // Show loading state
        const submitBtn = form.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation: spin 1s linear infinite;"><circle cx="12" cy="12" r="10" stroke-dasharray="30 70"/></svg> Analyzing...';
        submitBtn.disabled = true;

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                showResult(result, data);
            } else {
                showError(result.error || 'An unexpected error occurred.');
            }
        } catch (err) {
            showError('Failed to connect to the prediction server. Please try again.');
        } finally {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    });

    function showResult(result, inputData) {
        // Hide empty/error, show result
        resultEmpty.style.display = 'none';
        resultError.style.display = 'none';
        resultContent.style.display = 'block';

        // Animate price
        animatePrice(result.predicted_price);

        // Set confidence
        const confidence = result.confidence || 90;
        resultConfidenceBar.style.width = confidence + '%';
        resultConfidenceText.textContent = confidence + '%';

        // Build input summary
        const summaryFields = [
            { label: 'Overall Quality', key: 'OverallQual' },
            { label: 'Living Area', key: 'GrLivArea', suffix: ' sq ft' },
            { label: 'Basement Area', key: 'TotalBsmtSF', suffix: ' sq ft' },
            { label: 'Garage Area', key: 'GarageArea', suffix: ' sq ft' },
            { label: 'Year Built', key: 'YearBuilt' },
            { label: 'Bedrooms', key: 'BedroomAbvGr' },
            { label: 'Full Baths', key: 'FullBath' },
            { label: 'Lot Area', key: 'LotArea', suffix: ' sq ft' },
        ];

        let summaryHTML = '<h4 style="font-size: var(--text-sm); font-weight: var(--weight-semibold); margin-bottom: var(--space-4);">Input Summary</h4>';
        summaryFields.forEach(function (field) {
            const value = inputData[field.key] || '—';
            const suffix = field.suffix || '';
            summaryHTML += '<div class="predict-result__row">';
            summaryHTML += '<span class="predict-result__label">' + field.label + '</span>';
            summaryHTML += '<span class="predict-result__value">' + Number(value).toLocaleString() + suffix + '</span>';
            summaryHTML += '</div>';
        });

        resultSummary.innerHTML = summaryHTML;

        // Scroll to result on mobile
        if (window.innerWidth < 1024) {
            resultContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    function showError(message) {
        resultEmpty.style.display = 'none';
        resultContent.style.display = 'none';
        resultError.style.display = 'block';
        errorMessage.textContent = message;
    }

    function animatePrice(targetPrice) {
        const duration = 800;
        const startTime = performance.now();
        const startPrice = 0;

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const currentPrice = startPrice + (targetPrice - startPrice) * eased;

            resultPrice.textContent = '$' + Math.round(currentPrice).toLocaleString();

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    }

    // Add spin animation CSS
    const style = document.createElement('style');
    style.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
    document.head.appendChild(style);

})();
