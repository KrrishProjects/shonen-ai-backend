import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fetch from "node-fetch";

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json({ limit: "25mb" }));

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const PORT = process.env.PORT || 3000;

const GEMINI_MODELS = [
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
  "gemini-2.0-flash"
];

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    message: "Shonen AI backend is running with context memory v2"
  });
});

app.post("/chat", async (req, res) => {
  try {
    const { prompt, imageBase64, history } = req.body;

    if (!GEMINI_API_KEY) {
      return res.status(500).json({
        error: "GEMINI_API_KEY is missing"
      });
    }

    if (!prompt && !imageBase64) {
      return res.status(400).json({
        error: "Prompt or image is required"
      });
    }

    let conversationText = "";

    if (Array.isArray(history) && history.length > 0) {
      conversationText +=
        "You are Shonen AI. Continue the conversation using the context below.\n\n";
      conversationText += "Conversation so far:\n";

      for (const item of history) {
        if (!item || !item.text) continue;

        const speaker = item.role === "model" ? "Assistant" : "User";
        conversationText += `${speaker}: ${item.text}\n`;
      }

      conversationText += "\n";
    }

    conversationText += `User's latest message: ${
      prompt || "Describe this image."
    }\n\n`;

    conversationText +=
      "Important: If the user says he, she, her, his, him, it, this, that, or more, use the previous conversation to understand who or what they mean. Do not ask for clarification if the context already makes it clear.";

    const parts = [
      {
        text: conversationText
      }
    ];

    if (imageBase64) {
      parts.push({
        inline_data: {
          mime_type: "image/jpeg",
          data: imageBase64
        }
      });
    }

    let finalData = null;
    let finalStatus = 500;
    let finalModel = null;
    let success = false;

    for (const model of GEMINI_MODELS) {
      const geminiResponse = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GEMINI_API_KEY}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            contents: [
              {
                role: "user",
                parts
              }
            ]
          })
        }
      );

      finalData = await geminiResponse.json();
      finalStatus = geminiResponse.status;
      finalModel = model;

      if (geminiResponse.ok) {
        success = true;
        break;
      }

      console.log(
        `Model failed: ${model}`,
        finalStatus,
        finalData?.error?.message
      );
    }

    if (!success) {
      return res.status(finalStatus).json({
        error: finalData,
        modelTriedLast: finalModel
      });
    }

    const text =
      finalData?.candidates?.[0]?.content?.parts?.[0]?.text ||
      "No response received.";

    return res.json({
      reply: text,
      model: finalModel
    });
  } catch (error) {
    return res.status(500).json({
      error: error.message || "Server error"
    });
  }
});

app.post("/chat-stream", async (req, res) => {
  try {
    const { prompt, imageBase64, history } = req.body;

    if (!GEMINI_API_KEY) {
      res.writeHead(500, {
        "Content-Type": "text/plain",
      });
      res.end("GEMINI_API_KEY is missing");
      return;
    }

    if (!prompt && !imageBase64) {
      res.writeHead(400, {
        "Content-Type": "text/plain",
      });
      res.end("Prompt or image is required");
      return;
    }

    let conversationText = "";

    if (Array.isArray(history) && history.length > 0) {
      conversationText +=
        "You are Shonen AI. Continue the conversation using the context below.\n\n";
      conversationText += "Conversation so far:\n";

      for (const item of history) {
        if (!item || !item.text) continue;

        const speaker = item.role === "model" ? "Assistant" : "User";
        conversationText += `${speaker}: ${item.text}\n`;
      }

      conversationText += "\n";
    }

    conversationText += `User's latest message: ${
      prompt || "Describe this image."
    }\n\n`;

    conversationText +=
      "Important: If the user says he, she, her, his, him, it, this, that, or more, use the previous conversation to understand who or what they mean. Do not ask for clarification if the context already makes it clear.";

    const parts = [
      {
        text: conversationText,
      },
    ];

    if (imageBase64) {
      parts.push({
        inline_data: {
          mime_type: "image/jpeg",
          data: imageBase64,
        },
      });
    }

    res.writeHead(200, {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    });

    const model = "gemini-2.5-flash";

    const geminiResponse = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse&key=${GEMINI_API_KEY}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          contents: [
            {
              role: "user",
              parts,
            },
          ],
        }),
      }
    );

    if (!geminiResponse.ok) {
      const errorText = await geminiResponse.text();
      res.write(`Error: ${geminiResponse.status}\n${errorText}`);
      res.end();
      return;
    }

    let buffer = "";

    geminiResponse.body.on("data", (chunk) => {
      buffer += chunk.toString();

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;

        const jsonText = line.replace("data: ", "").trim();

        if (!jsonText || jsonText === "[DONE]") continue;

        try {
          const parsed = JSON.parse(jsonText);
          const text =
            parsed?.candidates?.[0]?.content?.parts?.[0]?.text || "";

          if (text) {
            res.write(text);
          }
        } catch (e) {
          // Ignore partial malformed chunks
        }
      }
    });

    geminiResponse.body.on("end", () => {
      res.end();
    });

    geminiResponse.body.on("error", (error) => {
      res.write(`\nStream error: ${error.message}`);
      res.end();
    });
  } catch (error) {
    res.write(`Server error: ${error.message || "Unknown error"}`);
    res.end();
  }
});

app.listen(PORT, () => {
  console.log(`Shonen AI backend running on port ${PORT}`);
});
