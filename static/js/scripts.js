document.addEventListener('DOMContentLoaded', () => {
    const sliders = document.querySelectorAll("input[type=range]");
    const values = document.querySelectorAll("span");

    sliders.forEach((slider, index) => {
        slider.addEventListener("input", () => {
            values[index].textContent = slider.value;
            updatePlot();
        });
    });

    const updatePlot = () => {
        const data = {};
        
        sliders.forEach((slider) => {
            data[slider.id] = slider.value;
        });

        const url = window.location.pathname;
        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(data => {
            const img = document.querySelector("#interactive-plot img");
            img.src = `data:image/png;base64,${data.img_str}`;
        });
    };

    updatePlot();  // Initial plot update
});

