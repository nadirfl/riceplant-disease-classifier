function checkFiles(files) {
    console.log(files);

    if (files.length != 1) {
        alert("Bitte genau eine Datei hochladen.")
        return;
    }

    const fileSize = files[0].size / 1024 / 1024; // in MiB
    if (fileSize > 10) {
        alert("Datei zu gross (max. 10Mb)");
        return;
    }

    answerPart.style.visibility = "visible";
    const file = files[0];

    // Preview
    if (file) {
        preview.src = URL.createObjectURL(files[0])
    }

    // Upload
    const formData = new FormData();
    for (const name in files) {
        formData.append("image", files[name]);
    }

    fetch('/analyze', {
        method: 'POST',
        body: formData
    }).then(
        response => {
            console.log(response)
            return response.json()
        }
    ).then(
        data => {
            console.log(data)
            answer.innerHTML = JSON.stringify(data, null, 2)
            updateChart(data)
        }
    ).catch(
        error => console.log(error)
    );

}

function updateChart(data) {
    const ctx = document.getElementById('myChart').getContext('2d');
    if (window.bar != undefined) {
        window.bar.destroy();
    }
    window.bar = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(item => item.className),
            datasets: [{
                label: 'Wahrscheinlichkeit',
                data: data.map(item => item.probability),
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}