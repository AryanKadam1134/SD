document.addEventListener('DOMContentLoaded', function() {
    const resultDiv = document.getElementById('result');
    
    function updateStressLevel() {
        fetch('/stress_level')
            .then(response => response.json())
            .then(data => {
                resultDiv.innerHTML = `
                    <div class="stress-level">
                        Stress Level: ${(data.stress_level * 100).toFixed(1)}%
                    </div>
                    <div class="recommendation">
                        ${data.recommendation}
                    </div>
                `;
            })
            .catch(error => console.error('Error:', error));
    }
    
    // Update stress level every 5 seconds
    setInterval(updateStressLevel, 5000);
});