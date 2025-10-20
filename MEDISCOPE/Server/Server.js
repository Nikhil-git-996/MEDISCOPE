// =============================
// MAIN SERVER
// =============================
const FormData = require("form-data");
const express = require("express");
const axios = require("axios");
const mongoose = require("mongoose");
const dotenv = require("dotenv");
const cors = require("cors");
const { Server } = require("socket.io");
const http = require("http");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const multer = require("multer");
const path = require("path");
const fs = require("fs");

dotenv.config();

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

// -----------------------------
// CORS + JSON PARSER
// -----------------------------
app.use(
  cors({
    origin: ["http://localhost:5173", "http://127.0.0.1:5173"],
    credentials: true,
  })
);
app.use(express.json());

// Debug middleware to log all incoming requests
app.use((req, res, next) => {
  console.log(`📥 Incoming request: ${req.method} ${req.path}`);
  next();
});

// -----------------------------
// DATABASE CONNECTION
// -----------------------------
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log("✅ MongoDB Connected"))
  .catch((err) => console.error("❌ Mongo Error:", err.message));

// -----------------------------
// SCHEMAS
// -----------------------------
const userSchema = new mongoose.Schema({
  name: String,
  email: { type: String, unique: true },
  password: String,
});

const historySchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  requestId: String,
  type: String,
  fileName: String,
  rawOutput: Object,
  interpretedOutput: Object,
  status: String,
  createdAt: { type: Date, default: Date.now },
});

const User = mongoose.model("User", userSchema);
const History = mongoose.model("History", historySchema);

// -----------------------------
// AUTH
// -----------------------------
const JWT_SECRET = process.env.JWT_SECRET || "supersecretjwtkey";

const generateToken = (user) =>
  jwt.sign({ id: user._id, name: user.name }, JWT_SECRET, { expiresIn: "1d" });

const authMiddleware = (req, res, next) => {
  console.log("🔑 authMiddleware triggered");
  const token = req.headers.authorization?.split(" ")[1];
  if (!token) {
    console.log("❌ No token provided");
    return res.status(401).json({ error: "Unauthorized" });
  }
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    console.log("✅ Token verified:", decoded);
    next();
  } catch (err) {
    console.log("❌ Invalid token", err.message);
    res.status(401).json({ error: "Invalid token" });
  }
};

// -----------------------------
// FILE UPLOAD CONFIG
// -----------------------------
const UPLOAD_DIR = path.resolve(__dirname, "uploads");
console.log("📂 Upload directory:", UPLOAD_DIR);
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    console.log("📂 Multer saving file to:", UPLOAD_DIR);
    cb(null, UPLOAD_DIR);
  },
  filename: (req, file, cb) => {
    const filename = `${Date.now()}_${file.originalname.replace(/\s+/g, "_")}`;
    console.log("🖊️ Multer filename:", filename);
    cb(null, filename);
  },
});

const upload = multer({ storage });

// -----------------------------
// SAFE POST
// -----------------------------
async function safePost(url, data, headers = {}) {
  try {
    console.log(`🌐 Making POST request to: ${url}`);
    const res = await axios.post(url, data, { headers, timeout: 60000 });
    console.log(`✅ Response received from ${url}`);
    return res.data;
  } catch (err) {
    console.error("❌ Error calling", url, err.message);
    throw err;
  }
}

// -----------------------------
// AUTH ROUTES
// -----------------------------
app.post("/signup", async (req, res) => {
  console.log("📝 /signup called");
  const { name, email, password } = req.body;
  if (!name || !email || !password)
    return res.status(400).json({ error: "All fields are required" });

  try {
    const existing = await User.findOne({ email });
    if (existing) return res.status(409).json({ error: "User already exists" });

    const hashed = await bcrypt.hash(password, 10);
    const user = await User.create({ name, email, password: hashed });
    const token = generateToken(user);

    console.log("✅ Signup successful for:", email);
    res.json({
      message: "Signup successful",
      token,
      user: { id: user._id, name, email },
    });
  } catch (err) {
    console.error("❌ Signup failed:", err.message);
    res.status(500).json({ error: "Signup failed" });
  }
});

app.post("/login", async (req, res) => {
  console.log("📝 /login called");
  const { email, password } = req.body;
  if (!email || !password)
    return res.status(400).json({ error: "Email and password required" });

  try {
    const user = await User.findOne({ email });
    if (!user) return res.status(404).json({ error: "User not found" });

    const valid = await bcrypt.compare(password, user.password);
    if (!valid) return res.status(401).json({ error: "Invalid credentials" });

    const token = generateToken(user);
    console.log("✅ Login successful for:", email);
    res.json({
      message: "Login successful",
      token,
      user: { id: user._id, name: user.name, email },
    });
  } catch (err) {
    console.error("❌ Login failed:", err.message);
    res.status(500).json({ error: "Login failed" });
  }
});

// -----------------------------
// PROCESS UPLOAD & PREDICTION
// -----------------------------
app.post(
  "/process",
  authMiddleware,
  upload.single("file"),
  async (req, res) => {
    console.log("🔥 [PROCESS] Endpoint hit!");
    console.log("👤 Authenticated user:", req.user);
    console.log("📥 Headers:", req.headers);
    console.log("📥 Body:", req.body);
    console.log("📎 Uploaded file info:", req.file);

    if (!req.file) {
      console.log("❌ No file uploaded");
      return res.status(400).json({ error: "No file uploaded" });
    }

    const { type, language } = req.body;
    if (!["xray", "lab", "labreport"].includes(type)) {
      console.log("❌ Invalid type:", type);
      return res.status(400).json({ error: "Invalid type" });
    }

    const requestId = `REQ-${Date.now()}`;
    const userId = req.user.id;

    console.log(`🟢 Processing request ${requestId} for user ${userId}`);
    io.emit("status", { userId, requestId, step: "started" });

    try {
      let microResponse = null;

      // ======================================
      // 🧠 X-RAY HANDLER
      // ======================================
      if (type === "xray") {
        console.log("🧠 [X-RAY] Preparing to call microservice...");

        const filePath = req.file.path;
        console.log("📂 Reading file from:", filePath);

        const fileBuffer = fs.readFileSync(filePath);
        const base64Image = fileBuffer.toString("base64");
        console.log(
          "🖼️ Converted image to Base64 (first 50 chars):",
          base64Image.slice(0, 50) + "..."
        );

        const payload = {
          payload: {
            image_base64: base64Image,
            body_part: "X-ray",
            age: 30,
            weight: 70,
            symptoms: "unknown",
          },
        };

        const XRAY_URL = process.env.XRAY_URL?.replace(/\/$/, "") || "";
        console.log("🌐 Calling X-ray microservice at:", `${XRAY_URL}/predict`);

        const response = await axios.post(`${XRAY_URL}/predict`, payload, {
          headers: { "Content-Type": "application/json" },
        });

        microResponse = response.data;
        console.log("✅ [X-RAY] Microservice Response:", microResponse);
      }

      // ======================================
      // 🧪 LAB REPORT HANDLER
      // ======================================
      else if (type === "labreport" || type === "lab") {
        console.log("🧪 [LAB] Preparing to call Lab microservice...");

        const LAB_URL = process.env.LAB_URL?.replace(/\/$/, "") || "";
        console.log("🌍 Target URL:", `${LAB_URL}/parse`);
        console.log("📄 Sending file path:", req.file.path);

        try {
          // Using multipart/form-data for reliability
          const formData = new FormData();
          formData.append("file_path", req.file.path);

          console.log("📤 [LAB] Sending file path to microservice...");
          const labResponse = await axios.post(`${LAB_URL}/parse`, formData, {
            headers: formData.getHeaders(),
          });

          microResponse = labResponse.data;
          console.log("✅ [LAB] Microservice Response:", microResponse);
        } catch (err) {
          console.error("❌ [LAB] Microservice Error:", err.message);
          return res
            .status(500)
            .json({ error: "Lab microservice failed", details: err.message });
        }
      }

      // ======================================
      // 🎯 EMIT MICROSERVICE RESULT
      // ======================================
      io.emit("status", {
        userId,
        requestId,
        step: "microservice_complete",
        data: microResponse,
      });

      // ======================================
      // 🧠 INTERPRETER CALL
      // ======================================
      console.log("🧠 [INTERPRETER] Sending data for interpretation...");
      const INTERPRETER_URL =
        process.env.INTERPRETER_URL?.replace(/\/$/, "") || "";

      const formData2 = new FormData();
      formData2.append("username", req.user.name || "User");
      formData2.append("language", language || "english");
      formData2.append("predictions", JSON.stringify(microResponse));

      const interpreted = await safePost(
        `${INTERPRETER_URL}/interpret`,
        formData2,
        formData2.getHeaders()
      );

      console.log("✅ [INTERPRETER] Response:", interpreted);

      // ======================================
      // 💾 SAVE HISTORY
      // ======================================
      await History.create({
        userId,
        requestId,
        type,
        fileName: req.file.filename,
        rawOutput: microResponse,
        interpretedOutput: interpreted,
        status: "completed",
      });

      console.log("💾 History saved successfully!");
      io.emit("completed", { userId, requestId, interpreted });

      res.json({ success: true, requestId, interpreted });
    } catch (err) {
      console.error("❌ [PROCESS ERROR]:", err.message);
      io.emit("failed", { userId, requestId, error: err.message });
      res
        .status(500)
        .json({ error: "Processing failed", message: err.message });
    } finally {
      // ======================================
      // 🧹 CLEANUP
      // ======================================
      if (req.file && fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
        console.log("🗑️ Cleaned up uploaded file:", req.file.path);
      }
      console.log("🏁 [PROCESS] Completed request lifecycle.");
    }
  }
);

// -----------------------------
// HISTORY
// -----------------------------
app.get("/history", authMiddleware, async (req, res) => {
  console.log("📄 /history called for user:", req.user.id);
  const history = await History.find({ userId: req.user.id }).sort({
    createdAt: -1,
  });
  res.json(history);
});

// -----------------------------
// SOCKET.IO
// -----------------------------
io.on("connection", (socket) => {
  console.log("🔌 Frontend connected:", socket.id);
  socket.on("disconnect", () => console.log("❌ Disconnected:", socket.id));
});

// -----------------------------
// START SERVER
// -----------------------------
const PORT = process.env.PORT || 4000;
server.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
