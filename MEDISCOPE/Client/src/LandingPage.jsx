import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence, useInView } from "framer-motion";
import { useNavigate } from "react-router-dom";
import Logo from "./assets/Logo.webp";
// ========================
// Professional Navbar
// ========================
const Navbar = () => {
  const navigate = useNavigate();
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const menuItems = [
    { name: "How It Works", href: "#how-it-works" },
    { name: "Features", href: "#features" },
    { name: "About Us", href: "#about" },
    { name: "Testimonials", href: "#testimonials" }
  ];

  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className={`fixed w-full z-50 transition-all duration-500 ${
        isScrolled
          ? "bg-white/95 backdrop-blur-xl shadow-lg py-2"
          : "bg-transparent py-4"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center">
          {/* Logo */}
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="flex items-center space-x-3"
          >
            <img
                                  src={Logo}
                                  alt="MediScope Logo"
                                  className="w-8 h-8 rounded-lg"
                                />
            <span className="text-2xl font-bold text-gray-800">MediScope</span>
          </motion.div>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center space-x-8">
            {menuItems.map((item, index) => (
              <motion.a
                key={item.name}
                href={item.href}
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * index }}
                whileHover={{ y: -2, color: "#0ea5e9" }}
                className="text-gray-600 hover:text-blue-600 font-medium transition-colors"
              >
                {item.name}
              </motion.a>
            ))}
           <motion.button
  whileHover={{ scale: 1.05, boxShadow: "0 10px 25px rgba(14, 165, 233, 0.15)" }}
  whileTap={{ scale: 0.95 }}
  className="px-6 py-2 bg-gradient-to-r from-blue-600 to-teal-500 text-white rounded-lg font-semibold shadow-md"
  onClick={() => navigate("/ChatInterface")}  // <-- navigate to your chat interface route
>
  Try Now
</motion.button>

          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="md:hidden p-2 rounded-lg text-gray-600"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {isMobileMenuOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>

        {/* Mobile Menu */}
        <AnimatePresence>
          {isMobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden bg-white/95 backdrop-blur-lg rounded-lg mt-2 p-4 space-y-3"
            >
              {menuItems.map((item) => (
                <a
                  key={item.name}
                  href={item.href}
                  className="block py-2 px-4 rounded-lg text-gray-600 hover:text-blue-600 hover:bg-blue-50 font-medium"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  {item.name}
                </a>
              ))}
              <button className="w-full mt-2 py-3 bg-gradient-to-r from-blue-600 to-teal-500 text-white rounded-lg font-semibold">
                Try Now
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.nav>
  );
};

// ========================
// Futuristic Healthcare CTA Banner Illustration
// ========================
const HealthcareCTAIllustration = () => {
  return (
    <div className="relative w-full max-w-6xl mx-auto h-96 bg-gradient-to-br from-white to-blue-50 rounded-3xl shadow-2xl overflow-hidden border border-blue-100">
      {/* Medical Grid Pattern */}
      <div className="absolute inset-0 opacity-10">
        <svg className="w-full h-full" fill="none">
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M40 0L0 0 0 40" fill="none" stroke="currentColor" strokeWidth="1" />
          </pattern>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>

      {/* Gradient Blobs */}
      <motion.div
        animate={{
          x: [0, 30, 0],
          y: [0, -20, 0],
          scale: [1, 1.1, 1]
        }}
        transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
        className="absolute top-10 left-10 w-64 h-64 bg-gradient-to-r from-blue-200/30 to-teal-200/20 rounded-full blur-2xl"
      />
      <motion.div
        animate={{
          x: [0, -40, 0],
          y: [0, 30, 0],
          scale: [1.2, 1, 1.2]
        }}
        transition={{ duration: 25, repeat: Infinity, ease: "easeInOut" }}
        className="absolute bottom-10 right-10 w-80 h-80 bg-gradient-to-r from-teal-200/20 to-green-200/20 rounded-full blur-2xl"
      />

      {/* Neural Network Lines */}
      <svg className="absolute inset-0 w-full h-full" fill="none">
        <motion.path
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 3, repeat: Infinity, repeatType: "reverse" }}
          d="M100,100 Q300,50 500,150 T900,100"
          stroke="url(#neuralGradient)"
          strokeWidth="2"
          strokeLinecap="round"
          fill="none"
        />
        <motion.path
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 3, delay: 0.5, repeat: Infinity, repeatType: "reverse" }}
          d="M150,200 Q350,100 550,250 T850,200"
          stroke="url(#neuralGradient)"
          strokeWidth="2"
          strokeLinecap="round"
          fill="none"
        />
        <defs>
          <linearGradient id="neuralGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#0ea5e9" stopOpacity="0" />
            <stop offset="50%" stopColor="#0ea5e9" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0" />
          </linearGradient>
        </defs>
      </svg>

      {/* Content Container */}
      <div className="relative z-10 flex flex-col lg:flex-row items-center justify-between h-full p-8 lg:p-12">
        {/* Left Side - Patient Interface */}
        <div className="flex-1 max-w-md">
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-blue-100"
          >
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-3 h-3 rounded-full bg-red-400"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
              <div className="w-3 h-3 rounded-full bg-green-400"></div>
            </div>

            <div className="space-y-4">
              <div className="text-sm text-gray-600">Medical Report Upload</div>

              <motion.div
                whileHover={{ scale: 1.02 }}
                className="bg-blue-50 rounded-xl p-4 border-2 border-dashed border-blue-200 cursor-pointer group"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-semibold text-blue-900">report_medical.pdf</div>
                    <div className="text-xs text-blue-600">2.4 MB • Ready to upload</div>
                  </div>
                  <motion.div
                    whileHover={{ rotate: 15 }}
                    className="text-blue-500"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </motion.div>
                </div>
              </motion.div>

              {/* Upload Button with Hand Cursor */}
              <motion.button
                whileHover={{
                  scale: 1.05,
                  boxShadow: "0 10px 30px rgba(14, 165, 233, 0.3)"
                }}
                whileTap={{ scale: 0.95 }}
                className="w-full py-3 bg-gradient-to-r from-blue-600 to-teal-500 text-white rounded-xl font-semibold shadow-lg relative overflow-hidden group"
              >
                <motion.div
                  className="absolute inset-0 bg-white/20"
                  initial={{ x: "-100%" }}
                  whileHover={{ x: "100%" }}
                  transition={{ duration: 0.6 }}
                />
                <span className="relative">Analyze Report</span>

                {/* Animated Hand Indicator */}
                <motion.div
                  animate={{ y: [0, -5, 0] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                  className="absolute -top-2 -right-2"
                >
                  <svg className="w-5 h-5 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v3.586L7.707 9.293a1 1 0 00-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 10.586V7z" clipRule="evenodd" />
                  </svg>
                </motion.div>
              </motion.button>
            </div>
          </motion.div>
        </div>

        {/* AI Assistant Visualization */}
        <div className="flex-1 flex justify-center lg:justify-end mt-8 lg:mt-0">
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, delay: 0.5 }}
            className="relative"
          >
            {/* AI Brain Core */}
            <motion.div
              animate={{
                rotate: [0, 5, -5, 0],
                scale: [1, 1.05, 1]
              }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              className="w-32 h-32 bg-gradient-to-br from-blue-500 to-teal-400 rounded-full flex items-center justify-center shadow-2xl"
            >
              <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </motion.div>

            {/* Holographic Rings */}
            <motion.div
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.3, 0.6, 0.3]
              }}
              transition={{ duration: 3, repeat: Infinity }}
              className="absolute inset-0 border-4 border-blue-300 rounded-full blur-sm"
            />
            <motion.div
              animate={{
                scale: [1, 1.4, 1],
                opacity: [0.2, 0.4, 0.2]
              }}
              transition={{ duration: 4, repeat: Infinity, delay: 1 }}
              className="absolute inset-0 border-4 border-teal-300 rounded-full blur-sm"
            />

            {/* Data Points */}
            {[0, 45, 90, 135, 180, 225, 270, 315].map((angle, index) => (
              <motion.div
                key={index}
                animate={{
                  scale: [1, 1.5, 1],
                  opacity: [0.5, 1, 0.5]
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  delay: index * 0.2
                }}
                className="absolute w-3 h-3 bg-green-400 rounded-full"
                style={{
                  left: `${50 + 40 * Math.cos((angle * Math.PI) / 180)}%`,
                  top: `${50 + 40 * Math.sin((angle * Math.PI) / 180)}%`,
                  transform: 'translate(-50%, -50%)'
                }}
              />
            ))}

            {/* Connection Lines */}
            <svg className="absolute inset-0 w-full h-full">
              <motion.line
                x1="50%"
                y1="50%"
                x2="20%"
                y2="30%"
                stroke="url(#pulseGradient)"
                strokeWidth="2"
                strokeLinecap="round"
              >
                <animate attributeName="stroke-dasharray" values="0 100; 100 0" dur="2s" repeatCount="indefinite" />
              </motion.line>
              <defs>
                <linearGradient id="pulseGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#0ea5e9" />
                  <stop offset="100%" stopColor="#10b981" />
                </linearGradient>
              </defs>
            </svg>
          </motion.div>
        </div>
      </div>

      {/* Floating Text */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, delay: 1 }}
        className="absolute bottom-6 left-1/2 transform -translate-x-1/2 text-center"
      >
        <p className="text-sm text-gray-600 font-medium">
          Secure AI-powered medical analysis • HIPAA Compliant • Real-time Insights
        </p>
      </motion.div>
    </div>
  );
};

// ========================
// Professional Hero Section
// ========================
const HeroSection = () => {
  return (
    <section className="min-h-screen flex items-center justify-center px-4 sm:px-6 lg:px-8 pt-20 bg-gradient-to-br from-blue-50 via-white to-teal-50">
      <div className="max-w-7xl mx-auto text-center">
        <motion.h1
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-4xl sm:text-6xl lg:text-7xl font-bold mb-6 leading-tight"
        >
          <span className="text-gray-800">Understand Your Health</span>
          <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-teal-600">
            Instantly with AI
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.8 }}
          className="text-xl sm:text-2xl text-gray-600 mb-12 max-w-3xl mx-auto"
        >
          Upload your lab results or medical scans and receive clear, actionable health insights powered by advanced artificial intelligence
        </motion.p>

        <HealthcareCTAIllustration />
      </div>
    </section>
  );
};

// ========================
// How It Works Section
// ========================
const HowItWorksSection = () => {
  const steps = [
    {
      number: "01",
      title: "Upload Medical Reports",
      description: "Securely upload your lab results, medical scans, or health reports in any format",
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
      )
    },
    {
      number: "02",
      title: "AI Analysis & Processing",
      description: "Our advanced CNN models analyze your data with 97% accuracy in real-time",
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      )
    },
    {
      number: "03",
      title: "Get Clear Insights",
      description: "Receive easy-to-understand reports with actionable recommendations",
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      )
    },
    {
      number: "04",
      title: "Consult Healthcare Providers",
      description: "Share results with your doctor for informed discussions about your health",
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a1.994 1.994 0 01-1.414-.586m0 0L11 14h4a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2v4l.586-.586z" />
        </svg>
      )
    }
  ];

  return (
    <section id="how-it-works" className="py-20 px-4 sm:px-6 lg:px-8 bg-white">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl sm:text-5xl font-bold mb-4">How MedAI Works</h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Four simple steps to transform complex medical data into clear, actionable health insights
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Steps */}
          <div className="space-y-8">
            {steps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.2 }}
                className="flex items-start space-x-6 p-6 bg-gradient-to-r from-white to-blue-50 rounded-2xl shadow-lg border border-blue-100 hover:shadow-xl transition-shadow"
              >
                <div className="flex-shrink-0 w-16 h-16 bg-gradient-to-br from-blue-600 to-teal-500 rounded-2xl flex items-center justify-center text-white">
                  {step.icon}
                </div>
                <div>
                  <div className="text-sm font-semibold text-blue-600 mb-1">{step.number}</div>
                  <h3 className="text-xl font-bold mb-2 text-gray-800">{step.title}</h3>
                  <p className="text-gray-600">{step.description}</p>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Visualization */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            className="relative h-96 bg-gradient-to-br from-blue-50 to-teal-50 rounded-3xl p-8 border border-blue-200"
          >
            {/* Animated Process Flow */}
            <div className="relative h-full">
              {/* Connection Lines */}
              <svg className="absolute inset-0 w-full h-full">
                <motion.path
                  initial={{ pathLength: 0 }}
                  whileInView={{ pathLength: 1 }}
                  transition={{ duration: 2 }}
                  d="M20,80 Q200,80 200,200 Q200,320 380,320"
                  stroke="url(#processGradient)"
                  strokeWidth="3"
                  strokeLinecap="round"
                  fill="none"
                  strokeDasharray="5,5"
                />
                <defs>
                  <linearGradient id="processGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#0ea5e9" />
                    <stop offset="100%" stopColor="#10b981" />
                  </linearGradient>
                </defs>
              </svg>

              {/* Step Points */}
              {[
                { x: 20, y: 80, label: "Upload" },
                { x: 200, y: 80, label: "Process" },
                { x: 200, y: 200, label: "Analyze" },
                { x: 380, y: 320, label: "Results" }
              ].map((point, index) => (
                <motion.div
                  key={index}
                  initial={{ scale: 0 }}
                  whileInView={{ scale: 1 }}
                  transition={{ delay: index * 0.3 }}
                  className="absolute w-8 h-8 bg-gradient-to-r from-blue-600 to-teal-500 rounded-full flex items-center justify-center text-white font-bold text-sm"
                  style={{ left: point.x, top: point.y, transform: 'translate(-50%, -50%)' }}
                >
                  {index + 1}
                </motion.div>
              ))}
            </div>

            {/* Floating Elements */}
            <motion.div
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 3, repeat: Infinity }}
              className="absolute top-4 left-4 w-16 h-16 bg-white/80 rounded-xl shadow-lg border border-blue-200 flex items-center justify-center"
            >
              <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </motion.div>

            <motion.div
              animate={{ y: [0, 10, 0] }}
              transition={{ duration: 3, repeat: Infinity, delay: 1 }}
              className="absolute bottom-4 right-4 w-20 h-20 bg-white/80 rounded-2xl shadow-lg border border-teal-200 flex items-center justify-center"
            >
              <svg className="w-10 h-10 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

// ========================
// Professional Features Section
// ========================
const FeaturesSection = () => {
  const features = [
    {
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      ),
      title: "Plain Language Reports",
      description: "Complex medical terminology translated into easy-to-understand insights"
    },
    {
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      ),
      title: "Instant Analysis",
      description: "Receive comprehensive analysis of your medical reports in seconds"
    },
    {
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
        </svg>
      ),
      title: "Enterprise Security",
      description: "HIPAA compliant with military-grade encryption for your data"
    }
  ];

  return (
    <section id="features" className="py-20 px-4 sm:px-6 lg:px-8 bg-white">
      <div className="max-w-6xl mx-auto">
        <motion.h2
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          className="text-4xl sm:text-5xl font-bold text-center mb-16"
        >
          Advanced Medical AI Technology
        </motion.h2>

        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 60 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.2 }}
              whileHover={{ y: -10, scale: 1.02 }}
              className="bg-gradient-to-br from-white to-blue-50 rounded-2xl p-8 shadow-lg border border-blue-100"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-blue-600 to-teal-500 rounded-2xl flex items-center justify-center text-white mb-6">
                {feature.icon}
              </div>
              <h3 className="text-2xl font-semibold mb-3 text-gray-800">{feature.title}</h3>
              <p className="text-gray-600 leading-relaxed">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

// ========================
// About Us Section with CNN Model Highlight
// ========================
const AboutUsSection = () => {
  const [accuracy, setAccuracy] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAccuracy(97);
    }, 500);
    return () => clearTimeout(timer);
  }, []);

  const processSteps = [
    {
      step: "01",
      title: "Data Collection & Preprocessing",
      description: "We gather and anonymize medical data from trusted healthcare partners, ensuring privacy and quality."
    },
    {
      step: "02",
      title: "CNN Model Training",
      description: "Our convolutional neural networks are trained on millions of medical images and reports for optimal accuracy."
    },
    {
      step: "03",
      title: "Validation & Testing",
      description: "Each model undergoes rigorous testing against real-world medical cases to ensure reliability."
    },
    {
      step: "04",
      title: "Deployment & Monitoring",
      description: "Continuous monitoring and updates ensure our AI maintains peak performance in clinical settings."
    }
  ];

  return (
    <section id="about" className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-gray-50 to-blue-50">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl sm:text-5xl font-bold mb-4">About MedAI</h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Pioneering the future of healthcare with cutting-edge artificial intelligence and deep learning technologies
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Column - Process Steps */}
          <div className="space-y-8">
            <motion.h3
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              className="text-3xl font-bold text-gray-800 mb-6"
            >
              How We Work
            </motion.h3>

            {processSteps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.2 }}
                className="flex items-start space-x-4 p-6 bg-white rounded-2xl shadow-lg border border-blue-100"
              >
                <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-teal-500 rounded-xl flex items-center justify-center text-white font-bold text-lg">
                  {step.step}
                </div>
                <div>
                  <h4 className="text-xl font-semibold mb-2 text-gray-800">{step.title}</h4>
                  <p className="text-gray-600">{step.description}</p>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Right Column - CNN Model Highlight */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            className="bg-white rounded-3xl p-8 shadow-2xl border border-blue-100"
          >
            <div className="text-center mb-8">
              <h3 className="text-3xl font-bold mb-4 text-gray-800">Our 97% Accurate CNN Model</h3>
              <p className="text-gray-600">
                Powered by state-of-the-art convolutional neural networks trained on extensive medical datasets
              </p>
            </div>

            {/* Accuracy Visualization */}
            <div className="mb-8">
              <div className="flex justify-between items-center mb-4">
                <span className="text-lg font-semibold text-gray-700">Model Accuracy</span>
                <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-teal-500 bg-clip-text text-transparent">
                  {accuracy}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-4">
                <motion.div
                  initial={{ width: 0 }}
                  whileInView={{ width: "97%" }}
                  transition={{ duration: 2, ease: "easeOut" }}
                  className="h-4 rounded-full bg-gradient-to-r from-blue-600 to-teal-500"
                />
              </div>
            </div>

            {/* Model Performance Metrics */}
            <div className="grid grid-cols-2 gap-4 mb-8">
              <div className="text-center p-4 bg-blue-50 rounded-xl">
                <div className="text-2xl font-bold text-blue-600">99.2%</div>
                <div className="text-sm text-gray-600">Sensitivity</div>
              </div>
              <div className="text-center p-4 bg-teal-50 rounded-xl">
                <div className="text-2xl font-bold text-teal-600">95.8%</div>
                <div className="text-sm text-gray-600">Specificity</div>
              </div>
              <div className="text-center p-4 bg-green-50 rounded-xl">
                <div className="text-2xl font-bold text-green-600">98.1%</div>
                <div className="text-sm text-gray-600">Precision</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-xl">
                <div className="text-2xl font-bold text-purple-600">96.5%</div>
                <div className="text-sm text-gray-600">Recall</div>
              </div>
            </div>


          </motion.div>
        </div>

        {/* Mission Statement */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          className="text-center mt-16 p-8 bg-gradient-to-r from-blue-600 to-teal-500 rounded-3xl text-white"
        >
          <h3 className="text-3xl font-bold mb-4">Our Mission</h3>
          <p className="text-xl opacity-90 max-w-4xl mx-auto">
            "To democratize access to medical expertise through artificial intelligence, making accurate health insights available to everyone, everywhere."
          </p>
        </motion.div>
      </div>
    </section>
  );
};

// ========================
// Testimonials Section
// ========================
const TestimonialsSection = () => {
  const testimonials = [
    {
      name: "Dr. Sarah Chen",
      role: "Cardiologist, Mayo Clinic",
      image: "👩‍⚕️",
      content: "MedAI has revolutionized how we process patient reports. The accuracy is remarkable and it saves us hours of analysis time.",
      rating: 5
    },
    {
      name: "Michael Rodriguez",
      role: "Patient",
      image: "👨‍💼",
      content: "As someone without medical background, MedAI helped me understand my lab results and have informed discussions with my doctor.",
      rating: 5
    },
    {
      name: "Dr. James Wilson",
      role: "Radiology Director",
      image: "👨‍⚕️",
      content: "The CNN model's 97% accuracy in detecting anomalies is impressive. It's become an essential tool in our diagnostic workflow.",
      rating: 5
    }
  ];

  return (
    <section id="testimonials" className="py-20 px-4 sm:px-6 lg:px-8 bg-white">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl sm:text-5xl font-bold mb-4">What Our Users Say</h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Trusted by healthcare professionals and patients worldwide
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 60 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.2 }}
              whileHover={{ y: -10, scale: 1.02 }}
              className="bg-gradient-to-br from-white to-blue-50 rounded-2xl p-8 shadow-lg border border-blue-100"
            >
              {/* Rating Stars */}
              <div className="flex mb-4">
                {[...Array(testimonial.rating)].map((_, i) => (
                  <svg key={i} className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                  </svg>
                ))}
              </div>

              {/* Testimonial Content */}
              <p className="text-gray-700 mb-6 italic">"{testimonial.content}"</p>

              {/* Author */}
              <div className="flex items-center">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-teal-500 rounded-full flex items-center justify-center text-2xl mr-4">
                  {testimonial.image}
                </div>
                <div>
                  <div className="font-semibold text-gray-800">{testimonial.name}</div>
                  <div className="text-sm text-gray-600">{testimonial.role}</div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-8 mt-16"
        >
          {[
            { number: "50K+", label: "Reports Analyzed" },
            { number: "97%", label: "Accuracy Rate" },
            { number: "500+", label: "Healthcare Partners" },
            { number: "4.9/5", label: "User Rating" }
          ].map((stat, index) => (
            <div key={index} className="text-center">
              <div className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-teal-500 bg-clip-text text-transparent mb-2">
                {stat.number}
              </div>
              <div className="text-gray-600">{stat.label}</div>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

// ========================
// Footer Section
// ========================
const Footer = () => {
  return (
    <footer className="bg-gradient-to-br from-gray-900 to-blue-900 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid md:grid-cols-4 gap-8">
          {/* Company Info */}
          <div className="md:col-span-2">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-600 to-teal-500 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <span className="text-2xl font-bold">MedAI</span>
            </div>
            <p className="text-gray-300 mb-6 max-w-md">
              Transforming healthcare through artificial intelligence. Making medical expertise accessible to everyone.
            </p>
            <div className="flex space-x-4">
              {['twitter', 'linkedin', 'github', 'facebook'].map((social) => (
                <motion.a
                  key={social}
                  whileHover={{ scale: 1.1, y: -2 }}
                  href="#"
                  className="w-10 h-10 bg-white/10 rounded-lg flex items-center justify-center hover:bg-white/20 transition-colors"
                >
                  <span className="text-lg">📱</span>
                </motion.a>
              ))}
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              {['How It Works', 'Features', 'About Us', 'Demo', 'Testimonials'].map((link) => (
                <li key={link}>
                  <a href={`#${link.toLowerCase().replace(' ', '-')}`} className="text-gray-300 hover:text-white transition-colors">
                    {link}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Contact Info */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Contact</h3>
            <ul className="space-y-2 text-gray-300">
              <li>📧 hello@medai.com</li>
              <li>📞 +1 (555) 123-4567</li>
              <li>📍 123 Healthcare Ave, Medical City</li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-white/10 mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-300 text-sm">
            © 2024 MedAI. All rights reserved. HIPAA Compliant.
          </p>
          <div className="flex space-x-6 mt-4 md:mt-0">
            <a href="#" className="text-gray-300 hover:text-white text-sm">Privacy Policy</a>
            <a href="#" className="text-gray-300 hover:text-white text-sm">Terms of Service</a>
            <a href="#" className="text-gray-300 hover:text-white text-sm">Cookie Policy</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

// ========================
// Main Landing Page
// ========================
const MedicalAILandingPage = () => {
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <HeroSection />
      <HowItWorksSection />
      <FeaturesSection />
      <AboutUsSection />
      <TestimonialsSection />
      <Footer />
    </div>
  );
};

export default MedicalAILandingPage;