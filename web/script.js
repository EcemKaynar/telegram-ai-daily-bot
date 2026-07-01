const chatMessages = document.getElementById("chatMessages");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const recordButton = document.getElementById("recordButton");
const statusText = document.getElementById("statusText");

const WEB_USER_ID = Date.now();
const WEB_USERNAME = "web_user";
const WEB_FIRST_NAME = "Web";

let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;

function addMessage(text, type = "bot") {
    const messageElement = document.createElement("div");
    messageElement.classList.add("message", type);
    messageElement.textContent = text;

    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setStatus(text) {
    statusText.textContent = text || "";
}

async function sendTextMessage() {
    const message = messageInput.value.trim();

    if (!message) {
        return;
    }

    addMessage(message, "user");
    messageInput.value = "";

    setStatus("Cevap bekleniyor...");

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                user_id: WEB_USER_ID,
                username: WEB_USERNAME,
                first_name: WEB_FIRST_NAME,
                message: message
            })
        });

        if (!response.ok) {
            throw new Error("API hatası: " + response.status);
        }

        const data = await response.json();

        addMessage(data.answer || "Cevap alınamadı.", "bot");

        if (data.source_type) {
            console.log("source_type:", data.source_type);
            console.log("sources:", data.sources);
        }

    } catch (error) {
        console.error(error);
        addMessage("API'ye bağlanırken bir hata oluştu. Lütfen tekrar dene.", "bot");

    } finally {
        setStatus("");
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: true
        });

        recordedChunks = [];

        mediaRecorder = new MediaRecorder(stream, {
            mimeType: "audio/webm"
        });

        mediaRecorder.ondataavailable = function (event) {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async function () {
            stream.getTracks().forEach(track => track.stop());

            const audioBlob = new Blob(recordedChunks, {
                type: "audio/webm"
            });

            await sendVoiceMessage(audioBlob);
        };

        mediaRecorder.start();
        isRecording = true;

        recordButton.classList.add("recording");
        recordButton.textContent = "⏹️";
        setStatus("Kayıt alınıyor... Durdurmak için tekrar bas.");

    } catch (error) {
        console.error(error);
        addMessage("Mikrofon izni alınamadı veya kayıt başlatılamadı.", "bot");
        setStatus("");
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;

        recordButton.classList.remove("recording");
        recordButton.textContent = "🎙️";
        setStatus("Ses işleniyor...");
    }
}

async function sendVoiceMessage(audioBlob) {
    addMessage("Sesli mesaj gönderildi.", "user");

    const formData = new FormData();
    formData.append("user_id", String(WEB_USER_ID));
    formData.append("username", WEB_USERNAME);
    formData.append("first_name", WEB_FIRST_NAME);
    formData.append("language", "tr-TR");
    formData.append("file", audioBlob, "voice.webm");

    try {
        const response = await fetch("/voice", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("API hatası: " + response.status);
        }

        const data = await response.json();

        let text = "";

        if (data.transcript) {
            text += "Seni şöyle anladım:\n" + data.transcript + "\n\n";
        }

        text += data.answer || "Cevap alınamadı.";

        addMessage(text, "bot");

        if (data.audio_base64) {
            playBase64Audio(data.audio_base64);
        }

    } catch (error) {
        console.error(error);
        addMessage("Sesli mesaj API'ye gönderilirken hata oluştu.", "bot");

    } finally {
        setStatus("");
    }
}

function playBase64Audio(base64Audio) {
    const audio = new Audio("data:audio/ogg;base64," + base64Audio);
    audio.play().catch(error => {
        console.error("Ses oynatma hatası:", error);
    });
}

sendButton.addEventListener("click", sendTextMessage);

messageInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
        sendTextMessage();
    }
});

recordButton.addEventListener("click", function () {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
});