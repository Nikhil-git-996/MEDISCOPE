import { Routes, Route } from "react-router-dom";
import LoginPage from "./LoginPage";
import SignupPage from "./SignupPage";
import AIChatInterface from "./AIChatInterface";
import LandingPage from "./LandingPage"

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage/>} />
      <Route path="/signup" element={<SignupPage />} />
      <Route path="/Login" element={<LoginPage/>} />
      <Route path="/ChatInterface" element={<AIChatInterface/>} />
    </Routes>
  );
}

export default App;
