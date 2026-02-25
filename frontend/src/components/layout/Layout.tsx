import { Outlet } from "react-router-dom";
import { Navbar } from "./Navbar";
import { ChatbotFAB } from "@/components/chat/ChatbotFAB";

export const Layout = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto px-4 py-6">
        <Outlet />
      </main>
      <ChatbotFAB />
    </div>
  );
};
