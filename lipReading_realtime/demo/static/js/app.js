document.addEventListener('DOMContentLoaded', () => {
    const socket = io.connect('http://' + document.domain + ':' + location.port);
    const video = document.getElementById('video');

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                video.srcObject = stream;
            })
            .catch(function (error) {
                console.log("Something went wrong!");
            });
    }

    video.addEventListener('play', function () {
        const canvas = document.createElement('canvas');
        canvas.width = video.width;
        canvas.height = video.height;
        const context = canvas.getContext('2d');

        setInterval(() => {
            context.drawImage(video, 0, 0, video.width, video.height);
            const dataURL = canvas.toDataURL('image/jpeg');
            const base64Data = dataURL.split(',')[1];

            socket.emit('image', base64Data);
        }, 1000 / 30); // Send image 30 times per second

        socket.on('response_back', function (data) {
            console.log(data.status);
        });
    });
});
