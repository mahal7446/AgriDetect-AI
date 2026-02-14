import { useState, useCallback, useRef, useEffect } from 'react';

interface UseVoiceInputReturn {
    isListening: boolean;
    transcript: string;
    startListening: () => void;
    stopListening: () => void;
    resetTranscript: () => void;
    isSupported: boolean;
    error: string | null;
}

/**
 * Custom hook for voice input using Web Speech API
 * Converts speech to text with language support
 */
export const useVoiceInput = (language: string = 'en-US'): UseVoiceInputReturn => {
    const [isListening, setIsListening] = useState(false);
    const [transcript, setTranscript] = useState('');
    const [error, setError] = useState<string | null>(null);
    const recognitionRef = useRef<SpeechRecognition | null>(null);

    // Check if browser supports speech recognition
    const isSupported = 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window;

    // Language code mapping
    const languageMap: Record<string, string> = {
        'en': 'en-US',
        'hi': 'hi-IN',
        'kn': 'kn-IN',
        'te': 'te-IN',
        'ta': 'ta-IN',
        'bn': 'bn-IN',
    };

    /**
     * Initialize speech recognition
     */
    useEffect(() => {
        if (!isSupported) {
            setError('Speech recognition not supported in this browser');
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();

        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = languageMap[language] || language;

        recognition.onstart = () => {
            setIsListening(true);
            setError(null);
            console.log('[VoiceInput] Started listening');
        };

        recognition.onresult = (event: SpeechRecognitionEvent) => {
            const current = event.resultIndex;
            const transcriptText = event.results[current][0].transcript;

            if (event.results[current].isFinal) {
                setTranscript(transcriptText);
                console.log('[VoiceInput] Final transcript:', transcriptText);
            }
        };

        recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
            console.error('[VoiceInput] Error:', event.error);

            if (event.error === 'no-speech') {
                setError('No speech detected. Please try again.');
            } else if (event.error === 'not-allowed') {
                setError('Microphone access denied. Please allow microphone access.');
            } else {
                setError(`Speech recognition error: ${event.error}`);
            }

            setIsListening(false);
        };

        recognition.onend = () => {
            setIsListening(false);
            console.log('[VoiceInput] Stopped listening');
        };

        recognitionRef.current = recognition;

        return () => {
            if (recognitionRef.current) {
                recognitionRef.current.stop();
            }
        };
    }, [language, isSupported, languageMap]);

    /**
     * Start listening for speech
     */
    const startListening = useCallback(() => {
        if (!isSupported) {
            setError('Speech recognition not supported');
            return;
        }

        if (!recognitionRef.current) {
            setError('Speech recognition not initialized');
            return;
        }

        try {
            setTranscript('');
            setError(null);
            recognitionRef.current.start();
        } catch (err) {
            console.error('[VoiceInput] Start error:', err);
            setError('Failed to start voice recognition');
        }
    }, [isSupported]);

    /**
     * Stop listening
     */
    const stopListening = useCallback(() => {
        if (recognitionRef.current && isListening) {
            recognitionRef.current.stop();
        }
    }, [isListening]);

    /**
     * Reset transcript
     */
    const resetTranscript = useCallback(() => {
        setTranscript('');
        setError(null);
    }, []);

    return {
        isListening,
        transcript,
        startListening,
        stopListening,
        resetTranscript,
        isSupported,
        error,
    };
};
