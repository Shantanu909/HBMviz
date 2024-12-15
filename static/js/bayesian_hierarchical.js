document.addEventListener('DOMContentLoaded', () => {
    const slider = document.querySelector("#mu");
    const valueDisplay = document.querySelector("#mu_value");

    // Set default sigma value (you can add a slider for sigma if needed)
    const sigma = 1;

    // Update displayed value of slider
    slider.addEventListener("input", () => {
        valueDisplay.textContent = slider.value;
        updatePlot();
    });

    const updatePlot = () => {
        const muValue = slider.value;
        const data = {
            mu: muValue,  // Send the current mu value
            sigma: sigma  // Default sigma value (can be dynamically added if needed)
        };

        const url = window.location.pathname;

        // Send the data to the server to regenerate images based on new mu value
        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(data => {
            // Update the images with the new data (base64 encoded images from server)
            const noPoolingImg = document.querySelector("#no-pooling img");
            const partialPoolingImg = document.querySelector("#partial-pooling img");
            const completePoolingImg = document.querySelector("#complete-pooling img");

            noPoolingImg.src = `data:image/png;base64,${data.img_str_no_pooling}`;
            partialPoolingImg.src = `data:image/png;base64,${data.img_str_partial_pooling}`;
            completePoolingImg.src = `data:image/png;base64,${data.img_str_complete_pooling}`;
        });
    };

    // Initial plot update
    updatePlot();
});
