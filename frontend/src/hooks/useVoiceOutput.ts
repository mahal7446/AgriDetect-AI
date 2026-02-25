import { useState, useCallback, useEffect } from 'react';

interface UseVoiceOutputReturn {
    speak: (text: string) => void;
    stop: () => void;
    isSpeaking: boolean;
    isSupported: boolean;
}

/**
 * Custom hook for voice output using Web Speech Synthesis API
 * Converts text to speech with language support
 */
export const useVoiceOutput = (language: string = 'en-US'): UseVoiceOutputReturn => {
    const [isSpeaking, setIsSpeaking] = useState(false);
    const isSupported = 'speechSynthesis' in window;

    // Language code mapping for voices
    const languageMap: Record<string, string> = {
        'en': 'en-US',
        'hi': 'hi-IN',
        'kn': 'kn-IN',
        'te': 'te-IN',
        'ta': 'ta-IN',
        'bn': 'bn-IN',
    };

    /**
     * Monitor speech synthesis state
     */
    useEffect(() => {
        if (!isSupported) return;

        const checkSpeaking = () => {
            setIsSpeaking(window.speechSynthesis.speaking);
        };

        const interval = setInterval(checkSpeaking, 100);

        return () => {
            clearInterval(interval);
            window.speechSynthesis.cancel();
        };
    }, [isSupported]);

    /**
     * Get appropriate voice for language
     */
    const getVoice = useCallback((lang: string): SpeechSynthesisVoice | null => {
        const voices = window.speechSynthesis.getVoices();
        const targetLang = languageMap[lang] || lang;

        // Try to find exact language match
        let voice = voices.find(v => v.lang === targetLang);

        // Fallback to language prefix match (e.g., 'en' matches 'en-US')
        if (!voice) {
            const langPrefix = targetLang.split('-')[0];
            voice = voices.find(v => v.lang.startsWith(langPrefix));
        }

        // Fallback to default voice
        if (!voice) {
            voice = voices.find(v => v.default) || voices[0];
        }

        return voice || null;
    }, [languageMap]);

    /**
     * Speak text with appropriate voice
     */
    const speak = useCallback((text: string) => {
        if (!isSupported) {
            console.warn('[VoiceOutput] Speech synthesis not supported');
            return;
        }

        // Cancel any ongoing speech
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = languageMap[language] || language;

        // Set voice
        const voice = getVoice(language);
        if (voice) {
            utterance.voice = voice;
        }

        // Set voice parameters
        utterance.rate = 0.9; // Slightly slower for clarity
        utterance.pitch = 1.0;
        utterance.volume = 1.0;

        utterance.onstart = () => {
            setIsSpeaking(true);
            console.log('[VoiceOutput] Started speaking');
        };

        utterance.onend = () => {
            setIsSpeaking(false);
            console.log('[VoiceOutput] Finished speaking');
        };

        utterance.onerror = (event) => {
            console.error('[VoiceOutput] Error:', event.error);
            setIsSpeaking(false);
        };

        window.speechSynthesis.speak(utterance);
    }, [language, languageMap, getVoice, isSupported]);

    /**
     * Stop current speech
     */
    const stop = useCallback(() => {
        if (isSupported) {
            window.speechSynthesis.cancel();
            setIsSpeaking(false);
        }
    }, [isSupported]);

    return {
        speak,
        stop,
        isSpeaking,
        isSupported,
    };
};
