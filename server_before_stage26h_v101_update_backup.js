import Razorpay from "razorpay";
import crypto from "crypto";
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
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY;
const TOGETHER_API_KEY = process.env.TOGETHER_API_KEY;
const CEREBRAS_API_KEY = process.env.CEREBRAS_API_KEY;

const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.1-8b-instant";
const MISTRAL_MODEL = process.env.MISTRAL_MODEL || "mistral-small-latest";
const TOGETHER_MODEL =
  process.env.TOGETHER_MODEL || "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free";
const CEREBRAS_MODEL = process.env.CEREBRAS_MODEL || "llama-3.3-70b";
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


function isRetryableAiStatus(status) {
  return [408, 429, 500, 502, 503, 504].includes(status);
}

function extractOpenAiText(data) {
  return (
    data?.choices?.[0]?.message?.content ||
    data?.choices?.[0]?.delta?.content ||
    ""
  );
}

async function callOpenAiCompatibleProvider({
  provider,
  apiKey,
  url,
  model,
  promptText,
}) {
  if (!apiKey) {
    return {
      ok: false,
      skipped: true,
      provider,
      model,
      status: 0,
      error: "Missing API key",
    };
  }

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [
          {
            role: "system",
            content:
              "You are Shonen AI, a helpful productivity AI assistant for students, creators, small businesses, and career users.",
          },
          {
            role: "user",
            content: promptText,
          },
        ],
        temperature: 0.7,
      }),
    });

    const data = await response.json().catch(() => ({}));
    const text = extractOpenAiText(data);

    if (!response.ok || !text) {
      return {
        ok: false,
        provider,
        model,
        status: response.status,
        error:
          data?.error?.message ||
          data?.message ||
          `Provider ${provider} failed.`,
        rawError: data,
      };
    }

    return {
      ok: true,
      provider,
      model,
      status: response.status,
      text,
    };
  } catch (error) {
    return {
      ok: false,
      provider,
      model,
      status: 500,
      error: error.message || `${provider} request failed.`,
    };
  }
}

async function callGeminiProvider(parts) {
  if (process.env.FORCE_SKIP_GEMINI === "true") {
    return {
      ok: false,
      provider: "gemini",
      model: "skipped_for_test",
      status: 503,
      error: "Gemini skipped because FORCE_SKIP_GEMINI=true.",
    };
  }

  if (!GEMINI_API_KEY) {
    return {
      ok: false,
      provider: "gemini",
      model: null,
      status: 500,
      error: "GEMINI_API_KEY is missing.",
    };
  }

  let finalData = null;
  let finalStatus = 500;
  let finalModel = null;

  for (const model of GEMINI_MODELS) {
    try {
      const geminiResponse = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${GEMINI_API_KEY}`,
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

      finalData = await geminiResponse.json().catch(() => ({}));
      finalStatus = geminiResponse.status;
      finalModel = model;

      if (geminiResponse.ok) {
        const text =
          finalData?.candidates?.[0]?.content?.parts?.[0]?.text ||
          "No response received.";

        return {
          ok: true,
          provider: "gemini",
          model,
          status: geminiResponse.status,
          text,
        };
      }

      console.log(
        `Gemini model failed: ${model}`,
        finalStatus,
        finalData?.error?.message
      );

      if (!isRetryableAiStatus(finalStatus)) {
        break;
      }
    } catch (error) {
      finalStatus = 500;
      finalData = { error: { message: error.message } };
      finalModel = model;
      console.log(`Gemini model crashed: ${model}`, error.message);
    }
  }

  return {
    ok: false,
    provider: "gemini",
    model: finalModel,
    status: finalStatus,
    error: formatGeminiError(finalStatus, finalData),
    rawError: finalData,
  };
}

async function generateAiReplyWithFallback({ promptText, parts, hasImage }) {
  const failures = [];

  const geminiResult = await callGeminiProvider(parts);

  if (geminiResult.ok) {
    return geminiResult;
  }

  failures.push(geminiResult);

  if (hasImage) {
    throw {
      status: geminiResult.status || 500,
      message: geminiResult.error || "Image AI request failed.",
      failures,
    };
  }

  const providers = [
    {
      provider: "groq",
      apiKey: GROQ_API_KEY,
      url: "https://api.groq.com/openai/v1/chat/completions",
      model: GROQ_MODEL,
    },
    {
      provider: "mistral",
      apiKey: MISTRAL_API_KEY,
      url: "https://api.mistral.ai/v1/chat/completions",
      model: MISTRAL_MODEL,
    },
    {
      provider: "together",
      apiKey: TOGETHER_API_KEY,
      url: "https://api.together.xyz/v1/chat/completions",
      model: TOGETHER_MODEL,
    },
    {
      provider: "cerebras",
      apiKey: CEREBRAS_API_KEY,
      url: "https://api.cerebras.ai/v1/chat/completions",
      model: CEREBRAS_MODEL,
    },
  ];

  for (const providerConfig of providers) {
    const result = await callOpenAiCompatibleProvider({
      ...providerConfig,
      promptText,
    });

    if (result.skipped) {
      continue;
    }

    if (result.ok) {
      return result;
    }

    failures.push(result);

    console.log(
      `AI provider failed: ${result.provider}`,
      result.status,
      result.error
    );
  }

  throw {
    status: 503,
    message:
      "Shonen AI is busy right now. Please try again in a few minutes.",
    failures,
  };
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

app.get("/ai-router-test", async (req, res) => {
  try {
    const promptText =
      "Say exactly this sentence only: Shonen AI fallback router test successful.";

    const parts = [
      {
        text: promptText,
      },
    ];

    const result = await generateAiReplyWithFallback({
      promptText,
      parts,
      hasImage: false,
    });

    return res.json({
      success: true,
      provider: result.provider,
      model: result.model,
      reply: result.text,
    });
  } catch (error) {
    console.error("AI router test error:", error);

    return res.status(error.status || 500).json({
      success: false,
      error: error.message || "AI router test failed.",
      failures: error.failures || [],
    });
  }
});

app.get("/app-version", (req, res) => {
  res.json({
    success: true,
    appName: "Shonen AI",
    latestVersion: "0.1.0",
    latestBuild: 1,
    minRequiredBuild: 1,
    forceUpdate: false,
    updateTitle: "Shonen AI is up to date",
    updateMessage: "You are using the latest version of Shonen AI.",
    apkUrl: "https://github.com/KrrishProjects/shonen_ai_website/releases/download/v1.0.0/ShonenAI.apk",
    websiteUrl: "https://krrishprojects.github.io/shonen_ai_website/",
    changelog: [
      "AI Chat",
      "Tools Hub",
      "PDF Tools",
      "Voice Mode",
      "Profile Memory",
      "Usage Limits",
      "Premium payment link flow",
      "Official website link"
    ],
    updatedAt: new Date().toISOString()
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

    if (!prompt && !imageBase64) {
      return res.status(400).json({
        error: "Prompt or image is required.",
      });
    }

    const promptText = buildConversationText(prompt, history);

    const parts = [
      {
        text: promptText,
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

    const result = await generateAiReplyWithFallback({
      promptText,
      parts,
      hasImage: Boolean(imageBase64),
    });

    return res.json({
      reply: result.text,
      provider: result.provider,
      model: result.model,
    });
  } catch (error) {
    console.error("AI router error:", error);

    return res.status(error.status || 500).json({
      error:
        error.message ||
        "Shonen AI is busy right now. Please try again in a few minutes.",
      failures: error.failures || undefined,
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



app.post("/verify-razorpay-payment", async (req, res) => {
  try {
    const {
      razorpay_order_id,
      razorpay_payment_id,
      razorpay_signature,
      planId = "premium_monthly",
      uid,
      email = "",
    } = req.body || {};

    if (!razorpay_order_id || !razorpay_payment_id || !razorpay_signature) {
      return res.status(400).json({
        success: false,
        error: "Missing Razorpay payment verification fields",
      });
    }

    const selectedPlan = SUBSCRIPTION_PLANS[planId];

    if (!selectedPlan) {
      return res.status(400).json({
        success: false,
        error: "Invalid subscription plan",
      });
    }

    const verifiedUid = req.user?.uid || uid;

    if (!verifiedUid) {
      return res.status(401).json({
        success: false,
        error: "User ID missing. Please sign in again.",
      });
    }

    if (!process.env.RAZORPAY_KEY_SECRET) {
      return res.status(500).json({
        success: false,
        error: "Razorpay secret is not configured on backend",
      });
    }

    const body = `${razorpay_order_id}|${razorpay_payment_id}`;

    const expectedSignature = crypto
      .createHmac("sha256", process.env.RAZORPAY_KEY_SECRET)
      .update(body)
      .digest("hex");

    if (expectedSignature !== razorpay_signature) {
      return res.status(400).json({
        success: false,
        error: "Invalid payment signature",
      });
    }

    const now = new Date();
    const premiumUntil = new Date(
      now.getTime() + selectedPlan.days * 24 * 60 * 60 * 1000
    );

    const subscriptionData = {
      plan: selectedPlan.plan,
      planId,
      label: selectedPlan.label,
      amount: selectedPlan.amount,
      currency: selectedPlan.currency,
      premiumUntil: premiumUntil.toISOString(),
      razorpayOrderId: razorpay_order_id,
      razorpayPaymentId: razorpay_payment_id,
      email,
      updatedAt: new Date().toISOString(),
    };

    await db
      .collection("users")
      .doc(verifiedUid)
      .collection("subscription")
      .doc("main")
      .set(subscriptionData, { merge: true });

    await db
      .collection("users")
      .doc(verifiedUid)
      .collection("usage")
      .doc("daily")
      .set(
        {
          plan: selectedPlan.plan,
          premiumUntil: premiumUntil.toISOString(),
          updatedAt: new Date().toISOString(),
        },
        { merge: true }
      );

    return res.json({
      success: true,
      message: "Payment verified and premium activated",
      plan: selectedPlan.plan,
      planId,
      premiumUntil: premiumUntil.toISOString(),
    });
  } catch (error) {
    console.error("Verify Razorpay payment error:", error);
    return res.status(500).json({
      success: false,
      error: "Failed to verify Razorpay payment",
      details: error?.message || String(error),
    });
  }
});



app.post("/create-razorpay-payment-link", async (req, res) => {
  try {
    const {
      planId = "premium_monthly",
      amount = 9900,
      currency = "INR",
      uid,
      email = "",
      name = "Shonen AI User",
    } = req.body || {};

    if (!uid) {
      return res.status(400).json({
        success: false,
        error: "Missing user uid",
      });
    }

    const paymentLink = await razorpay.paymentLink.create({
      amount,
      currency,
      accept_partial: false,
      description: "Shonen AI Premium Monthly Plan",
      customer: {
        name,
        email,
      },
      notify: {
        sms: false,
        email: true,
      },
      reminder_enable: true,
      notes: {
        uid,
        planId,
        source: "shonen_ai_flutter",
      },
      callback_url: "https://shonen-ai-backend.onrender.com/payment-success",
      callback_method: "get",
    });

    return res.json({
      success: true,
      paymentLinkId: paymentLink.id,
      paymentUrl: paymentLink.short_url,
      short_url: paymentLink.short_url,
    });
  } catch (error) {
    console.error("Create Razorpay payment link error:", error);
    return res.status(500).json({
      success: false,
      error: error.message || "Could not create Razorpay payment link",
    });
  }
});

app.get("/payment-success", async (req, res) => {
  return res.send("Payment completed. You can go back to Shonen AI.");
});


app.listen(PORT, () => {
  console.log(`Shonen AI backend running on port ${PORT}`);
});
