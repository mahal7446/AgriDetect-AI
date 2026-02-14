import { createContext, useContext, useState, ReactNode, useCallback, useMemo } from 'react';

/**
 * Context data structure for AI chatbot
 */
export interface ChatContextData {
    crop: string;
    disease: string;
    confidence: number;
    location?: string;
    severity?: string;
    imageUrl?: string;
}

interface ChatContextType {
    context: ChatContextData | null;
    setContext: (context: ChatContextData) => void;
    clearContext: () => void;
    pendingMessage: string | null;
    triggerChat: (message: string) => void;
    clearPendingMessage: () => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

interface ChatProviderProps {
    children: ReactNode;
}

/**
 * Provider component for chatbot context
 * Manages disease prediction data to be shared with chatbot
 */
export const ChatProvider = ({ children }: ChatProviderProps) => {
    const [context, setContextState] = useState<ChatContextData | null>(null);
    const [pendingMessage, setPendingMessage] = useState<string | null>(null);

    const setContext = useCallback((newContext: ChatContextData) => {
        setContextState(newContext);
        console.log('[ChatContext] Context updated:', newContext);
    }, []);

    const clearContext = useCallback(() => {
        setContextState(null);
        console.log('[ChatContext] Context cleared');
    }, []);

    const triggerChat = useCallback((message: string) => {
        setPendingMessage(message);
        console.log('[ChatContext] Triggering chat with message:', message);
    }, []);

    const clearPendingMessage = useCallback(() => {
        setPendingMessage(null);
    }, []);

    const value = useMemo(() => ({
        context,
        setContext,
        clearContext,
        pendingMessage,
        triggerChat,
        clearPendingMessage
    }), [context, setContext, clearContext, pendingMessage, triggerChat, clearPendingMessage]);

    return (
        <ChatContext.Provider value={value}>
            {children}
        </ChatContext.Provider>
    );
};

/**
 * Hook to access chatbot context
 * Must be used within ChatProvider
 */
export const useChatContext = () => {
    const context = useContext(ChatContext);
    if (context === undefined) {
        throw new Error('useChatContext must be used within ChatProvider');
    }
    return context;
};
