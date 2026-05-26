const Razorpay = require("razorpay");
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fetch from "node-fetch";
import rateLimit from "express-rate-limit";
import admin from "firebase-admin";

dotenv.config();

const app = express();

const razorpay = new Razorpay({
  key_id: process.env.RAZORPAY_KEY_ID,
  key_secret: process.env.RAZORPAY_KEY_SECRET,
});

const SUBSCRIPTION_PLANS = {
  premium_monthly: {
    plan: "premium",
    label: "Shonen AI Premium Monthly",
    amount: 9900, // ₹99 in paise
    currency: "INR",
    days: 30,
  },
};


app.use(cors());
app.use(express.json({ limit: "25mb" }));

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const PORT = process.env.PORT || 3000;

const GEMINI_MODELS = [
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
  "gemini-2.0-flash"
];

const STREAM_MODEL = "gemini-2.5-flash-lite";

function initFirebaseAdmin() {
  if (admin.apps.length > 0) return;

  const base64ServiceAccount = process.env.FIREBASE_SERVICE_ACCOUNT_BASE64;

  if (base64ServiceAccount) {
    const serviceAccountJson = Buffer.from(
      base64ServiceAccount,
      "base64"
    ).toString("utf8");

    const serviceAccount = JSON.parse(serviceAccountJson);

    admin.initializeApp({
      credential: admin.credential.cert(serviceAccount)
    });

    console.log("Firebase Admin initialized using service account.");
    return;
  }

  admin.initializeApp();
  console.log("Firebase Admin initialized using default credentials.");
}

initFirebaseAdmin();

const aiLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 30,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    error: "Too many requests. Please wait a minute and try again."
  }
});

async function verifyFirebaseUser(req, res, next) {
  try {
    const authHeader = req.headers.authorization || "";

    if (!authHeader.startsWith("Bearer ")) {
      return res.status(401).json({
        error: "Unauthorized. Missing Firebase login token."
      });
    }

    const idToken = authHeader.replace("Bearer ", "").trim();

    if (!idToken) {
      return res.status(401).json({
        error: "Unauthorized. Empty Firebase login token."
      });
    }

    const decodedToken = await admin.auth().verifyIdToken(idToken);

    req.user = {
      uid: decodedToken.uid,
      email: decodedToken.email || null
    };

    next();
  } catch (error) {
    return res.status(401).json({
      error: "Unauthorized. Invalid or expired Firebase login token."
    });
  }
}

function buildConversationText(prompt, history) {
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

  return conversationText;
}

function formatGeminiError(status, data) {
  const message =
    data?.error?.message ||
    data?.message ||
    "Something went wrong while contacting Shonen AI.";

  if (status === 429) {
    return "Shonen AI is temporarily out of AI quota or receiving too many requests. Please try again after some time.";
  }

  if (status === 503) {
    return "Shonen AI model is busy right now. Please try again in a moment.";
  }

  if (status === 401 || status === 403) {
    return "Shonen AI backend is not authorized to use the AI model. Please check the API key.";
  }

  return message;
}

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    message: "Shonen AI backend is running with production safety v1"
  });
});

app.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    app: "Shonen AI",
    timestamp: new Date().toISOString()
  });
});

app.get("/secure-health", verifyFirebaseUser, (req, res) => {
  res.json({
    status: "authenticated",
    uid: req.user.uid,
    email: req.user.email
  });
});

app.post("/chat", aiLimiter, verifyFirebaseUser, async (req, res) => {
  try {
    const { prompt, imageBase64, history } = req.body;

    if (!GEMINI_API_KEY) {
      return res.status(500).json({
        error: "GEMINI_API_KEY is missing."
      });
    }

    if (!prompt && !imageBase64) {
      return res.status(400).json({
        error: "Prompt or image is required."
      });
    }

    const parts = [
      {
        text: buildConversationText(prompt, history)
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
        error: formatGeminiError(finalStatus, finalData),
        rawError: finalData,
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

app.post("/chat-stream", aiLimiter, verifyFirebaseUser, async (req, res) => {
  try {
    const { prompt, imageBase64, history } = req.body;

    if (!GEMINI_API_KEY) {
      res.writeHead(500, {
        "Content-Type": "text/plain; charset=utf-8"
      });
      res.end("GEMINI_API_KEY is missing.");
      return;
    }

    if (!prompt && !imageBase64) {
      res.writeHead(400, {
        "Content-Type": "text/plain; charset=utf-8"
      });
      res.end("Prompt or image is required.");
      return;
    }

    const parts = [
      {
        text: buildConversationText(prompt, history)
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

    res.writeHead(200, {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no"
    });

    const geminiResponse = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${STREAM_MODEL}:streamGenerateContent?alt=sse&key=${GEMINI_API_KEY}`,
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

    if (!geminiResponse.ok) {
      const errorText = await geminiResponse.text();

      try {
        const parsed = JSON.parse(errorText);
        res.write(formatGeminiError(geminiResponse.status, parsed));
      } catch {
        res.write(`Error: ${geminiResponse.status}\n${errorText}`);
      }

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
        } catch {
          // Ignore partial malformed chunks.
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


app.post("/create-razorpay-order", async (req, res) => {
  try {
    const uid = req.user?.uid || req.body?.uid;
    const email = req.user?.email || req.body?.email || "";

    const planId = req.body?.planId || "premium_monthly";
    const selectedPlan = SUBSCRIPTION_PLANS[planId];

    if (!selectedPlan) {
      return res.status(400).json({
        error: "Invalid subscription plan",
      });
    }

    if (!process.env.RAZORPAY_KEY_ID || !process.env.RAZORPAY_KEY_SECRET) {
      return res.status(500).json({
        error: "Razorpay keys are not configured on backend",
      });
    }

    const receipt = `shonen_${Date.now()}`.slice(0, 40);

    const order = await razorpay.orders.create({
      amount: selectedPlan.amount,
      currency: selectedPlan.currency,
      receipt,
      notes: {
        uid: uid || "unknown",
        email,
        planId,
        plan: selectedPlan.plan,
      },
    });

    return res.json({
      success: true,
      keyId: process.env.RAZORPAY_KEY_ID,
      orderId: order.id,
      amount: order.amount,
      currency: order.currency,
      planId,
      plan: selectedPlan.plan,
      label: selectedPlan.label,
      days: selectedPlan.days,
    });
  } catch (error) {
    console.error("Create Razorpay order error:", error);
    return res.status(500).json({
      error: "Failed to create Razorpay order",
      details: error?.message || String(error),
    });
  }
});


app.listen(PORT, () => {
  console.log(`Shonen AI backend running on port ${PORT}`);
});