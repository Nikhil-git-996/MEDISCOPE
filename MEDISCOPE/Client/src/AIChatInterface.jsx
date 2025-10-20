import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import {
  FiActivity,
  FiAlertCircle,
  FiCheck,
  FiCopy,
  FiEdit,
  FiFileText,
  FiImage,
  FiLogOut,
  FiMenu,
  FiMessageSquare,
  FiMic,
  FiPause,
  FiPlay,
  FiPlus,
  FiSend,
  FiUpload,
  FiUser,
  FiX,
} from "react-icons/fi";
import { useNavigate } from "react-router-dom";
import Logo from "./assets/Logo.webp";

const AIChatInterface = () => {
  const navigate = useNavigate();
  const welcomeMessages = [
    "Hi there! I'm ready to analyze your medical report whenever you are.",
    "Your AI health assistant is online — upload a lab or X-ray to get started.",
    "Want clarity on your report? I can break it down into simple language.",
    "Upload a file and I'll generate structured, patient-friendly insights.",
    "I'm trained to interpret diagnostics — shall we start with your lab or X-ray?",
    "Health jargon can be confusing — let me make your report easy to understand.",
    "Drop a report here and I'll analyze it step by step for you.",
    "I'm ready! Upload your test results or just ask me a question.",
    "Need help with your report? Upload it and I'll do the heavy lifting.",
    "Your health clarity begins here. Lab or X-ray — I'll explain it all.",
  ];

  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isDeepThinking, setIsDeepThinking] = useState(false);
  const [showFileDropdown, setShowFileDropdown] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [profileDropdownOpen, setProfileDropdownOpen] = useState(false);
  const [isResponsePlaying, setIsResponsePlaying] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const [activeConversation, setActiveConversation] = useState(1);
  const [copiedMessageId, setCopiedMessageId] = useState(null);
  const [conversations, setConversations] = useState([
    {
      id: 1,
      title: "Current Session",
      date: "Active now",
      preview: "Ready to assist with your medical queries",
      messages: [],
    },
  ]);
  const [showAuthWarning, setShowAuthWarning] = useState(false);
  const [userDetails, setUserDetails] = useState(null);

  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const deepThinkingTimeoutRef = useRef(null);
  const authWarningTimeoutRef = useRef(null);

  // Check authentication and fetch user details
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      setShowAuthWarning(true);
      authWarningTimeoutRef.current = setTimeout(() => {
        setShowAuthWarning(false);
        navigate("/login");
      }, 3000);
    } else {
      // Fetch user details from localStorage or API
      const userData = localStorage.getItem("user");
      if (userData) {
        try {
          setUserDetails(JSON.parse(userData));
        } catch (error) {
          console.error("Error parsing user data:", error);
        }
      }
    }

    return () => {
      if (authWarningTimeoutRef.current) {
        clearTimeout(authWarningTimeoutRef.current);
      }
    };
  }, [navigate]);

  // Initialize with random welcome message
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;

    const randomWelcome =
      welcomeMessages[Math.floor(Math.random() * welcomeMessages.length)];
    const welcomeMessage = {
      id: Date.now(),
      text: randomWelcome,
      sender: "ai",
      timestamp: new Date(),
      isWelcome: true,
    };
    setMessages([welcomeMessage]);

    // Load conversations from localStorage
    const savedConversations = localStorage.getItem("medicalConversations");
    if (savedConversations) {
      try {
        const parsedConversations = JSON.parse(savedConversations);
        setConversations(parsedConversations);
        // Set the first conversation as active if none is set
        if (parsedConversations.length > 0) {
          setActiveConversation(parsedConversations[0].id);
          setMessages(parsedConversations[0].messages);
        }
      } catch (error) {
        console.error("Error loading conversations:", error);
      }
    } else {
      // Set initial conversation with welcome message
      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === 1 ? { ...conv, messages: [welcomeMessage] } : conv
        )
      );
    }
  }, []);

  // Save conversations to localStorage whenever they change
  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem(
        "medicalConversations",
        JSON.stringify(conversations)
      );
    }
  }, [conversations]);

  // Enhanced scroll to bottom with better detection
  const scrollToBottom = () => {
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "end",
      });
    }, 100);
  };

  // Improved scroll handling
  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping, isDeepThinking]);

  // Auto-scroll when new messages arrive
  useEffect(() => {
    const chatContainer = chatContainerRef.current;
    if (chatContainer) {
      const shouldScroll =
        chatContainer.scrollTop + chatContainer.clientHeight >=
        chatContainer.scrollHeight - 100;

      if (shouldScroll) {
        scrollToBottom();
      }
    }
  }, [messages.length]);

  const startNewChat = () => {
    const token = localStorage.getItem("token");
    if (!token) {
      setShowAuthWarning(true);
      setTimeout(() => navigate("/login"), 2000);
      return;
    }

    const randomWelcome =
      welcomeMessages[Math.floor(Math.random() * welcomeMessages.length)];
    const welcomeMessage = {
      id: Date.now(),
      text: randomWelcome,
      sender: "ai",
      timestamp: new Date(),
      isWelcome: true,
    };

    const newConversation = {
      id: Date.now(),
      title: "New Session",
      date: new Date().toLocaleDateString(),
      preview: "New conversation started",
      messages: [welcomeMessage],
    };

    setConversations((prev) => [newConversation, ...prev]);
    setActiveConversation(newConversation.id);
    setMessages([welcomeMessage]);
    setShowWelcome(true);
    setSidebarOpen(false);
  };

  const handleSend = async () => {
    const token = localStorage.getItem("token");
    if (!token) {
      setShowAuthWarning(true);
      setTimeout(() => navigate("/login"), 2000);
      return;
    }

    if (inputText.trim() === "") return;

    if (showWelcome) setShowWelcome(false);

    const newMessage = {
      id: Date.now(),
      text: inputText,
      sender: "user",
      timestamp: new Date(),
    };

    // Add user message immediately
    setMessages((prev) => [...prev, newMessage]);
    setInputText("");

    // Simulate thinking/typing indicator
    setIsTyping(true);
    setIsDeepThinking(true);

    try {
      // Show "thinking" placeholder message
      const placeholderId = Date.now() + 1000;
      setMessages((prev) => [
        ...prev,
        {
          id: placeholderId,
          sender: "ai",
          timestamp: new Date(),
          isPlaceholder: true,
        },
      ]);

      const response = await fetch("http://127.0.0.1:5002/interpret", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ type: "chat", query: newMessage.text }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      // Format the response
      const formatResponse = (text) => {
        if (typeof text !== "string") {
          text = JSON.stringify(text, null, 2);
        }

        return text
          .replace(/{|}/g, "")
          .replace(/\\n/g, "\n")
          .replace(/"recommendation":/gi, "\n\n**Recommendation:**\n")
          .replace(/"findings":/gi, "\n\n**Findings:**\n")
          .replace(/"summary":/gi, "\n\n**Summary:**\n")
          .replace(/"diagnosis":/gi, "\n\n**Diagnosis:**\n")
          .replace(/"notes":/gi, "\n\n**Notes:**\n")
          .replace(/"/g, "")
          .replace(/,/g, "")
          .replace(/\*\*/g, "**")
          .replace(/\s+/g, " ")
          .replace(/\s\s+/g, "\n\n")
          .trim();
      };

      const aiText = formatResponse(result.response || result);

      const aiResponse = {
        id: Date.now() + 1,
        text: aiText,
        sender: "ai",
        timestamp: new Date(),
      };

      // Replace placeholder message with actual response
      setMessages((prev) =>
        prev.filter((msg) => !msg.isPlaceholder).concat(aiResponse)
      );

      // Update conversation properly
      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === activeConversation
            ? {
                ...conv,
                messages: [
                  ...conv.messages.filter(
                    (m) => !m.isPlaceholder && m.id !== newMessage.id
                  ),
                  newMessage,
                  aiResponse,
                ],
                preview:
                  inputText.slice(0, 30) + (inputText.length > 30 ? "..." : ""),
                date: "Just now",
                title:
                  inputText.slice(0, 20) +
                    (inputText.length > 20 ? "..." : "") || "Medical Chat",
              }
            : conv
        )
      );
    } catch (err) {
      console.error("Interpreter request failed:", err);

      // Replace placeholder with fallback error message
      setMessages((prev) =>
        prev.map((msg) =>
          msg.isPlaceholder
            ? {
                ...msg,
                text: "⚠️ The doctor is taking longer than expected. Please try again shortly.",
                isPlaceholder: false,
              }
            : msg
        )
      );
    } finally {
      setIsTyping(false);
      setIsDeepThinking(false);
    }
  };

  // Handle file upload
  const handleFileUpload = async (type, file = null) => {
    const token = localStorage.getItem("token");
    if (!token) {
      setShowAuthWarning(true);
      setTimeout(() => navigate("/login"), 2000);
      return;
    }

    if (showWelcome) setShowWelcome(false);

    const uploadMessage = {
      id: Date.now(),
      text: `📤 Uploading ${type} file...`,
      sender: "user",
      timestamp: new Date(),
      isUpload: true,
      uploadType: type,
      fileName: file ? file.name : `medical_${type.toLowerCase()}.pdf`,
    };

    setMessages((prev) => [...prev, uploadMessage]);
    setIsTyping(true);
    setIsDeepThinking(true);

    try {
      if (!file) {
        throw new Error("No file selected");
      }

      const normalizedType = type.replace(/[^a-zA-Z]/g, "").toLowerCase();

      const formData = new FormData();
      formData.append("file", file);
      formData.append("type", normalizedType);
      formData.append("language", "english");

      const response = await fetch("http://127.0.0.1:4000/process", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const result = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(result?.error || "Backend returned an error");
      }

      // Format the response
      const formatResponse = (text) => {
        if (typeof text !== "string") {
          text = JSON.stringify(text, null, 2);
        }

        return text
          .replace(/{|}/g, "")
          .replace(/\\n/g, "\n")
          .replace(/"recommendation":/gi, "\n\n**Recommendation:**\n")
          .replace(/"findings":/gi, "\n\n**Findings:**\n")
          .replace(/"summary":/gi, "\n\n**Summary:**\n")
          .replace(/"diagnosis":/gi, "\n\n**Diagnosis:**\n")
          .replace(/"notes":/gi, "\n\n**Notes:**\n")
          .replace(/"/g, "")
          .replace(/,/g, "")
          .replace(/\*\*/g, "**")
          .replace(/\s+/g, " ")
          .replace(/\s\s+/g, "\n\n")
          .trim();
      };

      const aiResponse = {
        id: Date.now() + 1,
        text: formatResponse(
          result.interpreted?.message || result.interpreted || result
        ),
        sender: "ai",
        timestamp: new Date(),
        isAnalysis: true,
      };

      setMessages((prev) => [...prev, aiResponse]);

      // Update conversation with file upload
      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === activeConversation
            ? {
                ...conv,
                messages: [...conv.messages, uploadMessage, aiResponse],
                preview: `Uploaded ${type}`,
                date: "Just now",
                title: `${type} Analysis`,
              }
            : conv
        )
      );
    } catch (err) {
      console.error("Upload error:", err);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          text: `❌ Upload failed: ${err.message}`,
          sender: "ai",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsTyping(false);
      setIsDeepThinking(false);
    }
  };

  // Handle file selection from UI
  const handleFileSelect = (type) => {
    const token = localStorage.getItem("token");
    if (!token) {
      setShowAuthWarning(true);
      setTimeout(() => navigate("/login"), 2000);
      return;
    }

    setShowFileDropdown(false);

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept =
      type === "Lab Report"
        ? ".pdf,.jpg,.jpeg,.png"
        : ".dcm,.dicom,.png,.jpg,.jpeg";

    fileInput.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        handleFileUpload(type, file);
      }
    };

    fileInput.click();
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const copyToClipboard = async (text, messageId) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (err) {
      console.error("Failed to copy text: ", err);
    }
  };

  const editMessage = (messageId) => {
    const message = messages.find((m) => m.id === messageId);
    if (message) {
      setInputText(message.text);
      setMessages((prev) => prev.filter((m) => m.id !== messageId));

      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === activeConversation
            ? {
                ...conv,
                messages: conv.messages.filter((m) => m.id !== messageId),
              }
            : conv
        )
      );
    }
  };

  const toggleResponsePlay = () => {
    setIsResponsePlaying(!isResponsePlaying);
  };

  const switchConversation = (conversationId) => {
    const token = localStorage.getItem("token");
    if (!token) {
      setShowAuthWarning(true);
      setTimeout(() => navigate("/login"), 2000);
      return;
    }

    const conversation = conversations.find((c) => c.id === conversationId);
    if (conversation) {
      setActiveConversation(conversationId);
      setMessages(conversation.messages);
      setShowWelcome(conversation.messages.some((m) => m.isWelcome));
      setSidebarOpen(false);
    }
  };

  // Handle logout functionality
  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/login");
  };

  // Handle logo click to refresh page
  const handleLogoClick = () => {
    window.location.reload();
  };

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      if (deepThinkingTimeoutRef.current) {
        clearTimeout(deepThinkingTimeoutRef.current);
      }
      if (authWarningTimeoutRef.current) {
        clearTimeout(authWarningTimeoutRef.current);
      }
    };
  }, []);

  // Animation variants
  const bubbleVariants = {
    hidden: { opacity: 0, y: 10, scale: 0.95 },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        type: "spring",
        damping: 30,
        stiffness: 400,
      },
    },
  };

  const welcomeVariants = {
    hidden: { opacity: 0, scale: 0.9 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 30,
        duration: 0.6,
      },
    },
    exit: {
      opacity: 0,
      scale: 0.9,
      transition: { duration: 0.4, ease: "easeOut" },
    },
  };

  const sidebarVariants = {
    open: {
      x: 0,
      transition: {
        type: "spring",
        damping: 35,
        stiffness: 400,
        duration: 0.5,
      },
    },
    closed: {
      x: -320,
      transition: {
        type: "spring",
        damping: 35,
        stiffness: 400,
        duration: 0.4,
      },
    },
  };

  const rippleVariants = {
    initial: { scale: 0, opacity: 0.6 },
    animate: {
      scale: 2,
      opacity: 0,
      transition: {
        duration: 0.7,
        ease: "easeOut",
      },
    },
  };

  const warningVariants = {
    hidden: { opacity: 0, y: -20, scale: 0.9 },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        type: "spring",
        stiffness: 400,
        damping: 30,
      },
    },
    exit: {
      opacity: 0,
      y: -20,
      transition: {
        duration: 0.3,
      },
    },
  };

  const RippleButton = ({
    children,
    onClick,
    className = "",
    disabled = false,
    ...props
  }) => {
    const [ripple, setRipple] = useState(false);
    const [coords, setCoords] = useState({ x: -1, y: -1 });

    const handleClick = (e) => {
      if (disabled) return;

      const rect = e.currentTarget.getBoundingClientRect();
      setCoords({ x: e.clientX - rect.left, y: e.clientY - rect.top });
      setRipple(true);
      setTimeout(() => setRipple(false), 700);
      onClick?.(e);
    };

    return (
      <motion.button
        whileHover={!disabled ? { scale: 1.02 } : {}}
        whileTap={!disabled ? { scale: 0.98 } : {}}
        onClick={handleClick}
        className={`relative overflow-hidden ${
          disabled ? "opacity-50 cursor-not-allowed" : ""
        } ${className}`}
        disabled={disabled}
        {...props}
      >
        {children}
        {ripple && !disabled && (
          <motion.span
            variants={rippleVariants}
            initial="initial"
            animate="animate"
            className="absolute rounded-full bg-white/30"
            style={{
              left: coords.x,
              top: coords.y,
              width: "20px",
              height: "20px",
            }}
          />
        )}
      </motion.button>
    );
  };

  const DeepThinkingAnimation = () => (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="flex justify-start items-center space-x-3 bg-white/90 backdrop-blur-sm border border-slate-200/50 rounded-2xl rounded-bl-none px-4 py-3 shadow-lg max-w-md"
    >
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-400 rounded-full flex items-center justify-center"
      >
        <FiActivity className="w-4 h-4 text-white" />
      </motion.div>
      <div className="flex-1">
        <div className="flex items-center space-x-2 mb-1">
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="w-2 h-2 bg-blue-500 rounded-full"
          />
          <span className="text-sm font-medium text-slate-700">
            Deep Analysis
          </span>
        </div>
        <div className="text-xs text-slate-500">
          Analyzing medical patterns and cross-referencing clinical data...
        </div>
      </div>
    </motion.div>
  );

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-cyan-50/30 relative overflow-hidden">
      {/* Custom scrollbar styles */}
      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(0, 0, 0, 0.05);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(0, 0, 0, 0.2);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(0, 0, 0, 0.4);
        }
      `}</style>

      {/* Authentication Warning */}
      <AnimatePresence>
        {showAuthWarning && (
          <motion.div
            variants={warningVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50 bg-amber-500 text-white px-6 py-3 rounded-xl shadow-lg border border-amber-600/30 backdrop-blur-sm"
          >
            <div className="flex items-center space-x-3">
              <FiAlertCircle className="w-5 h-5" />
              <div>
                <div className="font-semibold">Authentication Required</div>
                <div className="text-sm opacity-90">
                  Redirecting to login...
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <>
            <motion.div
              variants={sidebarVariants}
              initial="closed"
              animate="open"
              exit="closed"
              className="absolute lg:relative inset-y-0 left-0 w-80 bg-white/95 backdrop-blur-xl border-r border-slate-200/60 z-40 shadow-2xl custom-scrollbar overflow-y-auto"
            >
              <div className="p-6 border-b border-slate-200/60">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <img
                      src={Logo}
                      alt="MediScope Logo"
                      className="w-8 h-8 rounded-lg"
                    />
                    <h2 className="font-bold text-slate-800 text-lg">
                      Chat History
                    </h2>
                  </div>
                  <RippleButton
                    onClick={() => setSidebarOpen(false)}
                    className="p-2 hover:bg-slate-100 rounded-lg text-slate-600"
                  >
                    <FiX className="w-5 h-5" />
                  </RippleButton>
                </div>
                <RippleButton
                  onClick={startNewChat}
                  className="w-full bg-gradient-to-r from-blue-500 to-cyan-400 text-white py-3 rounded-xl font-semibold shadow-lg flex items-center justify-center gap-2 hover:shadow-xl transition-all duration-300"
                >
                  <FiPlus className="w-4 h-4" />
                  New Chat
                </RippleButton>
              </div>

              <div className="p-4 space-y-2 max-h-[calc(100vh-200px)] overflow-y-auto custom-scrollbar">
                {conversations.length > 0 ? (
                  conversations.map((conv) => (
                    <RippleButton
                      key={conv.id}
                      onClick={() => switchConversation(conv.id)}
                      className={`w-full text-left p-3 rounded-xl transition-all duration-300 ${
                        activeConversation === conv.id
                          ? "bg-blue-50 border border-blue-200 shadow-sm"
                          : "hover:bg-slate-50"
                      }`}
                    >
                      <div className="font-semibold text-slate-800 text-sm mb-1 flex items-center gap-2">
                        <FiMessageSquare className="w-3 h-3 text-blue-500" />
                        {conv.title}
                      </div>
                      <div className="text-xs text-slate-600 mb-1 truncate">
                        {conv.preview}
                      </div>
                      <div className="text-xs text-slate-500">{conv.date}</div>
                    </RippleButton>
                  ))
                ) : (
                  <div className="text-center py-8 text-slate-500">
                    <FiMessageSquare className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No conversations yet</p>
                    <p className="text-sm">Start a new chat to begin</p>
                  </div>
                )}
              </div>
            </motion.div>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="fixed inset-0 bg-black/20 z-30 lg:hidden"
              onClick={() => setSidebarOpen(false)}
            />
          </>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{
            type: "spring",
            stiffness: 300,
            damping: 30,
            duration: 0.6,
          }}
          className="bg-white/90 backdrop-blur-xl border-b border-slate-200/60 px-4 py-3 z-20 shadow-sm"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <RippleButton
                onClick={() => setSidebarOpen(true)}
                className="p-2 rounded-xl text-slate-600 hover:text-slate-800 hover:bg-slate-100 transition-colors duration-200"
              >
                <FiMenu className="w-5 h-5" />
              </RippleButton>

              <div
                className="flex items-center space-x-3 cursor-pointer"
                onClick={handleLogoClick}
              >
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="flex items-center space-x-3"
                >
                  <img
                    src={Logo}
                    alt="MediScope Logo"
                    className="w-10 h-10 rounded-xl shadow-lg"
                  />
                  <div className="flex flex-col">
                    <h1 className="text-lg font-bold text-slate-800">
                      MediScope
                    </h1>
                    <span className="text-xs text-green-600 font-medium">
                      Your Intelligent scope into medical scans
                    </span>
                  </div>
                </motion.div>
              </div>
            </div>

            {/* Profile Dropdown */}
            <div className="relative">
              <RippleButton
                onClick={() => setProfileDropdownOpen(!profileDropdownOpen)}
                className="w-10 h-10 bg-gradient-to-br from-slate-600 to-slate-700 rounded-xl flex items-center justify-center text-white shadow-lg"
              >
                <FiUser className="w-5 h-5" />
              </RippleButton>

              <AnimatePresence>
                {profileDropdownOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 10, scale: 0.95 }}
                    transition={{ duration: 0.2 }}
                    className="absolute right-0 top-12 bg-white/95 backdrop-blur-xl rounded-xl shadow-lg border border-slate-200/60 py-2 z-50 min-w-[200px]"
                  >
                    {userDetails ? (
                      <div className="px-4 py-3 border-b border-slate-200/60">
                        <div className="font-semibold text-slate-800 text-sm">
                          {userDetails.name || "User"}
                        </div>
                        <div className="text-xs text-slate-600 truncate">
                          {userDetails.email || "user@example.com"}
                        </div>
                      </div>
                    ) : (
                      <div className="px-4 py-3 border-b border-slate-200/60">
                        <div className="font-semibold text-slate-800 text-sm">
                          User
                        </div>
                        <div className="text-xs text-slate-600">Loading...</div>
                      </div>
                    )}

                    <motion.button
                      whileHover={{
                        x: 4,
                        backgroundColor: "rgba(239, 68, 68, 0.05)",
                      }}
                      transition={{ duration: 0.2 }}
                      className="flex items-center space-x-2 px-4 py-3 w-full text-left text-red-600"
                      onClick={handleLogout}
                    >
                      <FiLogOut className="w-4 h-4" />
                      <span className="font-medium text-sm">Logout</span>
                    </motion.button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </motion.header>

        {/* Chat Area */}
        <div
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto px-4 py-6 relative custom-scrollbar"
        >
          <div className="max-w-4xl mx-auto h-full flex flex-col">
            {/* Welcome Message */}
            <AnimatePresence>
              {showWelcome && messages[0]?.isWelcome && (
                <motion.div
                  variants={welcomeVariants}
                  initial="hidden"
                  animate="visible"
                  exit="exit"
                  className="flex justify-center items-center flex-1"
                >
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    transition={{ duration: 0.3 }}
                    className="bg-white/90 backdrop-blur-xl rounded-3xl p-8 shadow-2xl border border-slate-200/50 max-w-md text-center mx-4"
                  >
                    <motion.div
                      animate={{ y: [0, -5, 0] }}
                      transition={{
                        duration: 3,
                        repeat: Infinity,
                        ease: "easeInOut",
                      }}
                      className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg"
                    >
                      <img
                        src={Logo}
                        alt="MediScope Logo"
                        className="w-16 h-16 rounded-2xl"
                      />
                    </motion.div>
                    <h2 className="text-2xl font-bold text-slate-800 mb-2">
                      Medical Interpreter AI
                    </h2>
                    <p className="text-slate-600 leading-relaxed mb-4">
                      {messages[0]?.text}
                    </p>
                    <motion.div
                      animate={{ opacity: [0.5, 1, 0.5] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="w-6 h-1 bg-blue-500 rounded-full mx-auto"
                    />
                  </motion.div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Messages */}
            {!showWelcome && (
              <div className="space-y-4 custom-scrollbar">
                <AnimatePresence mode="popLayout">
                  {messages
                    .filter((msg) => !msg.isWelcome)
                    .map((message) => (
                      <motion.div
                        key={message.id}
                        layout
                        variants={bubbleVariants}
                        initial="hidden"
                        animate="visible"
                        exit="hidden"
                        transition={{ duration: 0.3 }}
                        className={`flex ${
                          message.sender === "user"
                            ? "justify-end"
                            : "justify-start"
                        }`}
                      >
                        {message.isUpload ? (
                          <motion.div
                            whileHover={{ scale: 1.02 }}
                            transition={{ duration: 0.2 }}
                            className={`inline-flex items-center space-x-3 px-4 py-3 rounded-2xl shadow-lg border-2 ${
                              message.uploadType === "Lab Report"
                                ? "bg-green-50 border-green-200 text-green-800"
                                : "bg-blue-50 border-blue-200 text-blue-800"
                            }`}
                          >
                            {message.uploadType === "Lab Report" ? (
                              <FiFileText className="w-5 h-5 text-green-600" />
                            ) : (
                              <FiImage className="w-5 h-5 text-blue-600" />
                            )}
                            <div>
                              <div className="font-semibold">
                                {message.text}
                              </div>
                              <div className="text-xs opacity-70">
                                {message.fileName}
                              </div>
                            </div>
                          </motion.div>
                        ) : (
                          <motion.div
                            whileHover={{ scale: 1.01 }}
                            transition={{ duration: 0.2 }}
                            className={`relative max-w-[85%] lg:max-w-[70%] rounded-2xl px-4 py-3 shadow-lg ${
                              message.sender === "user"
                                ? "bg-gradient-to-br from-blue-500 to-cyan-400 text-white rounded-br-none"
                                : "bg-white/90 backdrop-blur-sm border border-slate-200/50 rounded-bl-none"
                            }`}
                          >
                            <p className="leading-relaxed whitespace-pre-wrap">
                              {message.text}
                            </p>
                            <span
                              className={`text-xs mt-2 block ${
                                message.sender === "user"
                                  ? "text-blue-100"
                                  : "text-slate-500"
                              }`}
                            >
                              {message.timestamp.toLocaleTimeString([], {
                                hour: "2-digit",
                                minute: "2-digit",
                              })}
                            </span>

                            {/* Message Actions */}
                            <motion.div
                              initial={{ opacity: 0 }}
                              whileHover={{ opacity: 1 }}
                              transition={{ duration: 0.2 }}
                              className={`absolute -bottom-8 flex space-x-1 ${
                                message.sender === "user" ? "right-0" : "left-0"
                              }`}
                            >
                              {message.sender === "user" && (
                                <RippleButton
                                  onClick={() => editMessage(message.id)}
                                  className="p-1.5 bg-slate-700 rounded text-white shadow hover:bg-slate-600 transition-colors duration-200"
                                  title="Edit message"
                                >
                                  <FiEdit className="w-3 h-3" />
                                </RippleButton>
                              )}
                              <RippleButton
                                onClick={() =>
                                  copyToClipboard(message.text, message.id)
                                }
                                className="p-1.5 bg-slate-700 rounded text-white shadow hover:bg-slate-600 transition-colors duration-200"
                                title="Copy message"
                              >
                                {copiedMessageId === message.id ? (
                                  <FiCheck className="w-3 h-3 text-green-400" />
                                ) : (
                                  <FiCopy className="w-3 h-3" />
                                )}
                              </RippleButton>
                            </motion.div>
                          </motion.div>
                        )}
                      </motion.div>
                    ))}
                </AnimatePresence>

                {/* Deep Thinking Animation */}
                {isDeepThinking && <DeepThinkingAnimation />}

                {/* Typing Indicator */}
                {isTyping && !isDeepThinking && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className="flex justify-start"
                  >
                    <div className="bg-white/90 backdrop-blur-sm border border-slate-200/50 rounded-2xl rounded-bl-none px-4 py-3 shadow-lg">
                      <div className="flex items-center space-x-2">
                        <motion.div
                          className="w-2 h-2 bg-slate-400 rounded-full"
                          animate={{ scale: [1, 1.5, 1] }}
                          transition={{
                            duration: 1,
                            repeat: Infinity,
                            ease: "easeInOut",
                          }}
                        />
                        <motion.div
                          className="w-2 h-2 bg-slate-400 rounded-full"
                          animate={{ scale: [1, 1.5, 1] }}
                          transition={{
                            duration: 1,
                            repeat: Infinity,
                            delay: 0.2,
                            ease: "easeInOut",
                          }}
                        />
                        <motion.div
                          className="w-2 h-2 bg-slate-400 rounded-full"
                          animate={{ scale: [1, 1.5, 1] }}
                          transition={{
                            duration: 1,
                            repeat: Infinity,
                            delay: 0.4,
                            ease: "easeInOut",
                          }}
                        />
                        <span className="text-sm text-slate-600 ml-2">
                          Analyzing...
                        </span>
                      </div>
                    </div>
                  </motion.div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="bg-transparent px-4 py-3 border-t border-slate-200/30"
        >
          <div className="max-w-4xl mx-auto text-center">
            <p className="text-xs text-slate-500/80">
              ⚠️ MediScope can make mistakes. Always verify important medical
              information with healthcare professionals.
            </p>
          </div>
        </motion.footer>

        {/* File Upload Dropdown */}
        <AnimatePresence>
          {showFileDropdown && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.95 }}
              transition={{ duration: 0.2 }}
              className="absolute bottom-24 left-4 z-30"
            >
              <div className="bg-white/95 backdrop-blur-xl rounded-xl shadow-lg border border-slate-200/60 py-2">
                {[
                  {
                    type: "Lab Report",
                    icon: FiFileText,
                    color: "green",
                    desc: "PDF, JPG, PNG files",
                  },
                  {
                    type: "X-ray",
                    icon: FiImage,
                    color: "blue",
                    desc: "DICOM, JPG, PNG files",
                  },
                ].map((option) => (
                  <RippleButton
                    key={option.type}
                    onClick={() => handleFileSelect(option.type)}
                    className="flex items-center space-x-3 px-4 py-3 w-full text-left hover:bg-slate-50/50 transition-colors duration-200 group"
                  >
                    <option.icon
                      className={`w-5 h-5 text-${option.color}-600 group-hover:scale-110 transition-transform duration-200`}
                    />
                    <div className="flex-1">
                      <div className={`font-medium text-${option.color}-800`}>
                        {option.type}
                      </div>
                      <div className={`text-xs text-${option.color}-600`}>
                        {option.desc}
                      </div>
                    </div>
                  </RippleButton>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Input Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{
            type: "spring",
            stiffness: 300,
            damping: 30,
            duration: 0.5,
          }}
          className="bg-white/90 backdrop-blur-xl border-t border-slate-200/60 px-4 py-4 z-20"
        >
          <div className="max-w-4xl mx-auto flex items-end space-x-3">
            {/* File Upload Button */}
            <RippleButton
              onClick={() => setShowFileDropdown(!showFileDropdown)}
              className={`p-3 rounded-xl transition-all duration-200 ${
                showFileDropdown
                  ? "bg-blue-100 text-blue-600"
                  : "text-slate-600 hover:text-slate-800 hover:bg-slate-100"
              }`}
            >
              <FiUpload className="w-5 h-5" />
            </RippleButton>

            {/* Text Input */}
            <div className="flex-1 relative">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Describe your symptoms, ask a medical question, or upload a report..."
                className="w-full bg-white/80 backdrop-blur-sm border border-slate-300/50 rounded-xl px-4 py-3 pr-12 resize-none focus:outline-none focus:ring-2 focus:ring-blue-400/30 focus:border-blue-400 text-slate-800 placeholder-slate-500 transition-all duration-300 custom-scrollbar"
                rows="1"
                style={{
                  minHeight: "48px",
                  maxHeight: "120px",
                }}
                onInput={(e) => {
                  e.target.style.height = "auto";
                  e.target.style.height = e.target.scrollHeight + "px";
                }}
              />

              {inputText && (
                <RippleButton
                  onClick={() => setInputText("")}
                  className="absolute right-14 top-3 p-1 text-slate-400 hover:text-slate-600 transition-colors duration-200"
                >
                  <FiX className="w-4 h-4" />
                </RippleButton>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex items-center space-x-2">
              {/* Play/Pause Button during response */}
              {(isTyping || isDeepThinking) && (
                <RippleButton
                  onClick={toggleResponsePlay}
                  className="p-3 text-slate-600 hover:text-slate-800 hover:bg-slate-100 rounded-xl transition-colors duration-200"
                >
                  {isResponsePlaying ? (
                    <FiPause className="w-5 h-5" />
                  ) : (
                    <FiPlay className="w-5 h-5" />
                  )}
                </RippleButton>
              )}

              <RippleButton className="p-3 text-slate-600 hover:text-slate-800 hover:bg-slate-100 rounded-xl transition-colors duration-200">
                <FiMic className="w-5 h-5" />
              </RippleButton>

              <RippleButton
                onClick={handleSend}
                disabled={!inputText.trim()}
                className={`p-3 rounded-xl transition-all duration-300 ${
                  inputText.trim()
                    ? "bg-gradient-to-r from-blue-500 to-cyan-400 text-white shadow-lg hover:shadow-xl"
                    : "bg-slate-200 text-slate-400 cursor-not-allowed"
                }`}
              >
                <FiSend className="w-5 h-5" />
              </RippleButton>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Close dropdowns when clicking outside */}
      {(showFileDropdown || profileDropdownOpen) && (
        <div
          className="fixed inset-0 z-10"
          onClick={() => {
            setShowFileDropdown(false);
            setProfileDropdownOpen(false);
          }}
        />
      )}
    </div>
  );
};

export default AIChatInterface;
