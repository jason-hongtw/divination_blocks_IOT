/**
 * 檢查使用者 IP 是否受到限制，並在需要時顯示限制 Modal。
 * 需要確保 showRestrictedModal 函數已在之前載入 (例如從 ui_helpers.js)。
 */
function checkIpRestriction() {
    // 檢查 showRestrictedModal 是否可用
    if (typeof showRestrictedModal !== 'function') {
        console.error("showRestrictedModal function is not available. Make sure ui_helpers.js is loaded before api_calls.js.");
        return;
    }

    fetch("/check_ip_restriction", {
        method: "GET",
        headers: { "Content-Type": "application/json" }
    })
    .then(response => {
        // 增加對非 JSON 回應或網路錯誤的處理
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("IP 限制檢查:", data); // 調試用
        if (data.status === "BLOCKED" && data.remaining_time) {
            showRestrictedModal(data.remaining_time); // 呼叫來自 ui_helpers.js 的函數
        } else if (data.status !== "OK") {
            console.warn("IP 限制檢查回應狀態未知:", data.status);
        }
    })
    .catch(error => {
        console.error("檢查 IP 限制時發生錯誤:", error);
        // 這裡可以考慮是否要通知使用者檢查失敗
        // 例如： alert("無法檢查伺服器狀態，請稍後再試。");
    });
}