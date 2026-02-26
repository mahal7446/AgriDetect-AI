import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import ReactGA from "react-ga4";

ReactGA.initialize("G-V5JEBWT8RL");
createRoot(document.getElementById("root")!).render(<App />);
