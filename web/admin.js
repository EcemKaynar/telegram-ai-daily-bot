const sections = document.querySelectorAll(".panel-section");
const navButtons = document.querySelectorAll(".nav-button");
const refreshButton = document.getElementById("refreshButton");

let adminData = null;

function showSection(sectionId) {
    sections.forEach(section => {
        section.classList.remove("active");
    });

    navButtons.forEach(button => {
        button.classList.remove("active");
    });

    document.getElementById(sectionId).classList.add("active");

    const activeButton = document.querySelector(`[data-section="${sectionId}"]`);

    if (activeButton) {
        activeButton.classList.add("active");
    }
}

function text(value) {
    if (value === null || value === undefined || value === "") {
        return "-";
    }

    return String(value);
}

function shorten(value, maxLength = 500) {
    const valueText = text(value);

    if (valueText.length <= maxLength) {
        return valueText;
    }

    return valueText.slice(0, maxLength) + "...";
}

async function loadAdminData() {
    try {
        const response = await fetch("/admin-data");

        if (!response.ok) {
            throw new Error("Admin data alınamadı: " + response.status);
        }

        adminData = await response.json();

        renderOverview(adminData.overview);
        renderSessions(adminData.sessions);
        renderServiceLogs(adminData.service_logs);
        renderRagLogs(adminData.rag_logs);
        renderEvaluation(adminData.evaluation_summary, adminData.evaluation_results);

    } catch (error) {
        console.error(error);
        alert("Admin verileri alınırken hata oluştu.");
    }
}

function renderOverview(overview) {
    document.getElementById("sessionsCount").textContent = overview.sessions_count || 0;
    document.getElementById("interactionsCount").textContent = overview.interactions_count || 0;
    document.getElementById("serviceLogsCount").textContent = overview.service_logs_count || 0;
    document.getElementById("ragLogsCount").textContent = overview.rag_logs_count || 0;
    document.getElementById("evaluationAverage").textContent = (overview.evaluation_average_success || 0) + "%";
}

function renderSessions(sessions) {
    const sessionList = document.getElementById("sessionList");
    sessionList.innerHTML = "";

    if (!sessions || sessions.length === 0) {
        sessionList.innerHTML = "<div class='session-item'>Henüz session yok.</div>";
        return;
    }

    sessions.forEach(session => {
        const item = document.createElement("div");
        item.className = "session-item";

        item.innerHTML = `
            <strong>${text(session.username)}</strong>
            <small>Session: ${text(session.session_id)}</small>
            <small>Mesaj: ${text(session.message_count)}</small>
            <small>Son aktivite: ${text(session.last_activity_at)}</small>
        `;

        item.addEventListener("click", () => {
            loadSessionDetail(session.session_id);
        });

        sessionList.appendChild(item);
    });
}

async function loadSessionDetail(sessionId) {
    try {
        const response = await fetch(`/admin-data/sessions/${encodeURIComponent(sessionId)}`);

        if (!response.ok) {
            throw new Error("Session detayı alınamadı.");
        }

        const data = await response.json();

        renderSessionDetail(data);

    } catch (error) {
        console.error(error);
        alert("Session detayı alınırken hata oluştu.");
    }
}

function renderSessionDetail(data) {
    const container = document.getElementById("sessionDetail");
    container.classList.remove("empty");

    const session = data.session;
    const interactions = data.interactions || [];

    let html = "";

    if (session) {
        html += `
            <div class="session-meta">
                <p><strong>Session:</strong> ${text(session.session_id)}</p>
                <p><strong>Kullanıcı:</strong> ${text(session.username)}</p>
                <p><strong>Durum:</strong> ${text(session.status)}</p>
                <p><strong>Özet:</strong> ${text(session.summary)}</p>
            </div>
            <hr />
        `;
    }

    if (interactions.length === 0) {
        html += "<p>Bu session için mesaj bulunamadı.</p>";
    } else {
        interactions.forEach(item => {
            html += `
                <div class="message-row">
                    <div class="question">
                        <strong>Kullanıcı:</strong><br />
                        ${text(item.question)}
                    </div>

                    <div class="answer">
                        <strong>Bot:</strong><br />
                        ${text(item.answer)}
                    </div>

                    <small>${text(item.created_at)}</small>
                </div>
            `;
        });
    }

    container.innerHTML = html;
}

function renderServiceLogs(logs) {
    const tbody = document.getElementById("serviceLogsTable");
    tbody.innerHTML = "";

    if (!logs || logs.length === 0) {
        tbody.innerHTML = "<tr><td colspan='5'>Tool/service log bulunamadı.</td></tr>";
        return;
    }

    logs.forEach(log => {
        const row = document.createElement("tr");

        row.innerHTML = `
            <td>${text(log.created_at)}</td>
            <td>${shorten(log.session_id, 80)}</td>
            <td>${text(log.service_name)}</td>
            <td>${shorten(log.request_data, 400)}</td>
            <td>${shorten(log.response_data, 400)}</td>
        `;

        tbody.appendChild(row);
    });
}

function renderRagLogs(logs) {
    const tbody = document.getElementById("ragLogsTable");
    tbody.innerHTML = "";

    if (!logs || logs.length === 0) {
        tbody.innerHTML = "<tr><td colspan='5'>RAG log bulunamadı.</td></tr>";
        return;
    }

    logs.forEach(log => {
        const row = document.createElement("tr");

        row.innerHTML = `
            <td>${text(log.created_at)}</td>
            <td>${shorten(log.session_id, 80)}</td>
            <td>${shorten(log.question, 300)}</td>
            <td>${shorten(log.source_files, 400)}</td>
            <td>${shorten(log.matched_text, 500)}</td>
        `;

        tbody.appendChild(row);
    });
}

function renderEvaluation(summary, results) {
    document.getElementById("evalTotal").textContent = summary.total_tests || 0;
    document.getElementById("evalPass").textContent = summary.passed_tests || 0;
    document.getElementById("evalFail").textContent = summary.failed_tests || 0;
    document.getElementById("evalAvg").textContent = (summary.average_success || 0) + "%";

    const tbody = document.getElementById("evaluationTable");
    tbody.innerHTML = "";

    if (!results || results.length === 0) {
        tbody.innerHTML = "<tr><td colspan='7'>Evaluation sonucu bulunamadı.</td></tr>";
        return;
    }

    results.forEach(result => {
        const row = document.createElement("tr");
        const status = text(result.status);
        const statusClass = status === "PASS" ? "pass" : "fail";

        row.innerHTML = `
            <td>${text(result.test_id)}</td>
            <td>${text(result.category)}</td>
            <td>${shorten(result.input, 250)}</td>
            <td>${text(result.score)}/${text(result.max_score)} - ${text(result.success_rate)}%</td>
            <td><span class="badge ${statusClass}">${status}</span></td>
            <td>${shorten(result.answer, 500)}</td>
            <td>${shorten(result.notes, 500)}</td>
        `;

        tbody.appendChild(row);
    });
}

navButtons.forEach(button => {
    button.addEventListener("click", () => {
        showSection(button.dataset.section);
    });
});

refreshButton.addEventListener("click", loadAdminData);

loadAdminData();