let selectedTask = 'task1';  // Default task

function selectTask(task) {
    if (task === 'task1') {
        document.getElementById('task1Btn').classList.add('active');
        document.getElementById('task2Btn').classList.remove('active');
    } else {
        document.getElementById('task2Btn').classList.add('active');
        document.getElementById('task1Btn').classList.remove('active');
    }
    selectedTask = task;
}

async function processText() {
    var inputTitle = document.getElementById('titleText').value;
    var inputText = document.getElementById('inputText').value;
    if(inputText.length + inputTitle.length >= 3200) {
        alert("The maximum amount of words is 600");
        return
    }
    document.getElementById('outputText').textContent = '';

    // Disable the button and show loading overlay
    const btn = document.querySelector("button");
    btn.disabled = true;
    document.getElementById("loadingOverlay").style.display = "flex";

    let prompt = ""
    prompt += "Give me each and every single mistake this essay has. Afterwards, give an overall summary of "
    prompt += "how good it is for IELTS task 2. Afterwards, give an estimate score for this essay. Afterwards, address all IELTS writing task 2 band descriptors for this essay. "
    prompt += "Afterwards, rewrite the essay with all possible improvements to score 9 in IELTS writing task 2."
    prompt += "\n\nThe prompt is: " + inputTitle + "\n\nThe essay is: " + inputText;
    console.log(prompt);
    
    try {
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer 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`
            },
            body: JSON.stringify({
                model: "gpt-4",
                messages: [{role: "system", content: "You are a world-class IELTS writing task 2 teacher."}, { role: "user", content: prompt }],
                max_tokens: 1000,
            })
        });

        // Enable the button and hide loading overlay
        btn.disabled = false;
        document.getElementById("loadingOverlay").style.display = "none";

        const data = await response.json();
        console.log(data)
        const correctedText = data.choices && data.choices[0] && data.choices[0].message.content;
        if (correctedText) {
            typeText(correctedText, document.getElementById('outputText'));
        } else {
            throw new Error("Unable to process essay.");
        }
    } catch (error) {
        // Enable the button and hide loading overlay in case of an error
        btn.disabled = false;
        document.getElementById("loadingOverlay").style.display = "none";

        console.error("Error processing essay:", error);
    }
}

function typeText(text, element, callback) {
    let i = 0;
    text = text.replace(/\n/g, '<br>'); // Replace newline characters with <br> tags
    const interval = setInterval(function() {
        if (text.charAt(i) === '<') {
            // If the current character is the start of a <br> tag, add the whole tag
            element.innerHTML += text.substring(i, i+4);
            i += 4;
        } else {
            element.innerHTML += text.charAt(i);
            i++;
        }
        if (i >= text.length) {
            clearInterval(interval);
            if (callback) callback();
        }
    }, 10); // 25ms delay between each character
}
