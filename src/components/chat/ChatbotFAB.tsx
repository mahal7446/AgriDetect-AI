import { useState, useEffect, useRef } from "react";
import { MessageCircle, X, Mic, Send, Volume2, Loader2, AlertCircle, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { useTranslation } from "react-i18next";
import { useChatContext } from "@/contexts/ChatContext";
import { useChatbot } from "@/hooks/useChatbot";
import { useVoiceInput } from "@/hooks/useVoiceInput";
import { useVoiceOutput } from "@/hooks/useVoiceOutput";

export const ChatbotFAB = () => {
  const { i18n } = useTranslation();
  const { context, pendingMessage, clearPendingMessage } = useChatContext();

  const [isOpen, setIsOpen] = useState(false);
  const [message, setMessage] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Chatbot hook
  const { messages, isLoading, sendMessage, initializeChat } = useChatbot();

  // Voice hooks
  const {
    isListening,
    transcript,
    startListening,
    stopListening,
    resetTranscript,
    isSupported: voiceInputSupported,
  } = useVoiceInput(i18n.language);

  const { speak, stop, isSpeaking } = useVoiceOutput(i18n.language);

  // Initialize chat when opened
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      // Initialize with context if available, otherwise general greeting
      initializeChat(context || undefined);
    }
  }, [isOpen, messages.length, context, initializeChat]);

  // Handle pending message from "Learn More" button - pre-fill input instead of auto-send
  useEffect(() => {
    if (pendingMessage) {
      setIsOpen(true);
      // Pre-fill the message in input field for user to review and send
      setMessage(pendingMessage);
      clearPendingMessage();
    }
  }, [pendingMessage, clearPendingMessage]);

  // Update message when voice transcript changes
  useEffect(() => {
    if (transcript) {
      setMessage(transcript);
    }
  }, [transcript]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle send message
  const handleSend = async () => {
    if (!message.trim()) return;

    await sendMessage(message);
    setMessage("");
    resetTranscript();
  };

  // Handle voice input toggle (auto-send after speaking)
  const toggleVoice = () => {
    if (isListening) {
      stopListening();
      // Auto-send if there's transcript
      if (transcript && transcript.trim()) {
        setTimeout(() => {
          handleSend();
        }, 500);
      }
    } else {
      setMessage(""); // Clear previous message
      resetTranscript();
      startListening();
    }
  };

  // Handle voice output for bot messages
  const handleSpeak = (text: string) => {
    if (isSpeaking) {
      stop();
    } else {
      speak(text);
    }
  };

  return (
    <>
      {/* Chat Window */}
      {isOpen && (
        <div className="fixed bottom-24 right-4 md:right-8 w-[calc(100vw-2rem)] md:w-96 max-h-[70vh] bg-card rounded-2xl shadow-2xl border border-border animate-scale-in z-50 flex flex-col overflow-hidden">
          {/* Header */}
          <div className="gradient-primary p-4 text-primary-foreground">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary-foreground/20 rounded-full">
                  <MessageCircle className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-semibold">AgriDetect Assistant</h3>
                  <p className="text-xs opacity-80">AI-powered agricultural advisor</p>
                </div>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="text-primary-foreground hover:bg-primary-foreground/20 relative z-10"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIsOpen(false);
                }}
              >
                <X className="w-5 h-5" />
              </Button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3 min-h-[200px] max-h-[300px]">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={cn(
                  "flex",
                  msg.isUser ? "justify-end" : "justify-start"
                )}
              >
                <div
                  className={cn(
                    "max-w-[80%] p-3 rounded-2xl text-sm",
                    msg.isUser
                      ? "bg-primary text-primary-foreground rounded-br-md"
                      : "bg-muted text-foreground rounded-bl-md"
                  )}
                >
                  <p className="whitespace-pre-wrap">{msg.text}</p>
                  {!msg.isUser && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className={cn(
                        "w-6 h-6 ml-2 inline-flex hover:bg-background/20",
                        isSpeaking && "text-primary animate-pulse"
                      )}
                      onClick={() => handleSpeak(msg.text)}
                    >
                      {isSpeaking ? <Square className="w-3 h-3 fill-current" /> : <Volume2 className="w-3 h-3" />}
                    </Button>
                  )}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-muted text-foreground p-3 rounded-2xl rounded-bl-md flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Thinking...</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t border-border bg-muted/30">
            <div className="flex gap-2">
              {voiceInputSupported && (
                <Button
                  variant={isListening ? "default" : "outline"}
                  size="icon"
                  onClick={toggleVoice}
                  disabled={isLoading}
                  className={cn(
                    "shrink-0",
                    isListening && "animate-pulse bg-red-500 hover:bg-red-600"
                  )}
                  title="Click to speak, will auto-send"
                >
                  <Mic className="w-4 h-4" />
                </Button>
              )}
              <Input
                placeholder={isListening ? "Listening..." : "Ask about farming, diseases, treatment..."}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                disabled={isLoading}
                className="flex-1"
              />
              <Button
                size="icon"
                onClick={handleSend}
                disabled={!message.trim() || isLoading}
                className="shrink-0"
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground mt-2 text-center">
              🎤 {voiceInputSupported ? "Click mic, speak, auto-sends • 🔊 Voice replies" : "Voice not supported in this browser"}
            </p>
          </div>
        </div >
      )}

      {/* FAB Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "fixed bottom-6 right-4 md:right-8 z-50 p-4 rounded-full shadow-lg transition-all duration-300 group",
          "gradient-primary hover:scale-110",
          isOpen && "rotate-90"
        )}
      >
        <div className="absolute inset-0 rounded-full bg-primary/30 animate-pulse-ring" />
        <MessageCircle className="w-6 h-6 text-primary-foreground relative z-10" />

        {/* Tooltip */}
        {!isOpen && (
          <div className="absolute right-full mr-3 top-1/2 -translate-y-1/2 bg-card text-foreground px-3 py-2 rounded-lg shadow-lg text-sm whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
            Ask AI Assistant 🤖
          </div>
        )}
      </button>
    </>
  );
};
