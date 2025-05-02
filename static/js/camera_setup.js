/**
 * 初始化使用者攝影機設定
 */
function initializeCamera() {
    const userCamera = document.getElementById('userCamera');
    const userCameraDesktop = document.getElementById('userCameraDesktop');
    const cameraError = document.getElementById('cameraError');
    const cameraErrorDesktop = document.getElementById('cameraErrorDesktop');

    // 檢查瀏覽器是否支援 getUserMedia
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" } // "user" 表示前置攝影機，"environment" 表示後置攝影機
        })
        .then(function(stream) {
            // 將視訊流同時應用到手機版和桌面版的 video 元素 (如果存在)
            if (userCamera) {
                userCamera.srcObject = stream;
                userCamera.play(); // 確保開始播放
            }
            if (userCameraDesktop) {
                userCameraDesktop.srcObject = stream;
                userCameraDesktop.play(); // 確保開始播放
            }
        })
        .catch(function(error) {
            console.error("無法訪問攝影機:", error);
            // 在手機版和桌面版的錯誤區塊顯示錯誤訊息 (如果存在)
            const errorMsg = "無法訪問攝影機，請檢查瀏覽器權限設置。";
            if (cameraError) {
                cameraError.innerText = errorMsg;
                cameraError.style.display = "block";
            }
            if (cameraErrorDesktop) {
                cameraErrorDesktop.innerText = errorMsg;
                cameraErrorDesktop.style.display = "block";
            }
        });
    } else {
        console.warn("瀏覽器不支援 navigator.mediaDevices.getUserMedia API");
        const errorMsg = "您的瀏覽器不支援視訊功能。";
         if (cameraError) {
             cameraError.innerText = errorMsg;
             cameraError.style.display = "block";
         }
         if (cameraErrorDesktop) {
             cameraErrorDesktop.innerText = errorMsg;
             cameraErrorDesktop.style.display = "block";
         }
    }
}