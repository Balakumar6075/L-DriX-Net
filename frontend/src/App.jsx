import { useState } from "react";
import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";
import Dashboard from "./pages/Dashboard";
import Prediction from "./pages/Prediction";
import About from "./pages/About";
import "./App.css";

function App() {
  const [activePage, setActivePage] = useState("dashboard");

  const renderPage = () => {
    switch (activePage) {
      case "dashboard":
        return <Dashboard />;

      case "prediction":
        return <Prediction />;

      case "about":
        return <About />;

      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="app">
      <Sidebar
        activePage={activePage}
        setActivePage={setActivePage}
      />

      <div className="main-area">
        <Navbar />

        <main className="content">
          {renderPage()}
        </main>
      </div>
    </div>
  );
}

export default App;