/**
 * 根據 header 高度調整 main 區塊的上邊距
 */
function adjustMainMargin() {
    const header = document.querySelector('header');
    if (!header) return; // 防禦性檢查
    const main = document.querySelector('main');
    if (!main) return; // 防禦性檢查
    main.style.marginTop = `${header.offsetHeight}px`;
}

/**
 * 顯示 IP 受限的彈出視窗及倒數計時
 * @param {number} remainingTime - 剩餘的限制秒數
 */
function showRestrictedModal(remainingTime) {
    const modalElement = document.getElementById("ipRestrictedModal");
    const remainingTimeSpan = document.getElementById("remainingTime");
    if (!modalElement || !remainingTimeSpan) {
        console.error("無法找到 IP 限制 Modal 或時間顯示元素");
        return;
    }

    // 確保 Bootstrap Modal 已載入
    if (typeof bootstrap === 'undefined' || typeof bootstrap.Modal === 'undefined') {
         console.error("Bootstrap Modal 未載入");
         return;
    }

    const modal = new bootstrap.Modal(modalElement, {
        backdrop: 'static', // 防止點擊背景關閉
        keyboard: false     // 防止用鍵盤關閉
    });

    let timeLeft = remainingTime;
    remainingTimeSpan.innerText = timeLeft;

    modal.show();

    // 清除可能存在的舊計時器
    if (window.ipRestrictionCountdownInterval) {
        clearInterval(window.ipRestrictionCountdownInterval);
    }

    window.ipRestrictionCountdownInterval = setInterval(() => {
        timeLeft--;
        remainingTimeSpan.innerText = timeLeft;
        if (timeLeft <= 0) {
            clearInterval(window.ipRestrictionCountdownInterval);
            window.ipRestrictionCountdownInterval = null; // 清除標記
            modal.hide();
            // 可選：重新啟用表單 (檢查 checkInput 是否存在)
            if (typeof checkInput === 'function') {
               checkInput();
            }
        }
    }, 1000);
}