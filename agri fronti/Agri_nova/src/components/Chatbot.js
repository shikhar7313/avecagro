import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
    Send,
    MessageCircle,
    History,
    Bot,
    User,
    Trash2,
    RefreshCw,
    Mic,
    MicOff,
    Sparkles,
    Plus,
    Loader2
} from 'lucide-react';

const STORAGE_KEY = 'agrinova.chat.threads';
const ACTIVE_THREAD_KEY = 'agrinova.chat.activeThread';

const generateId = () => {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
        return crypto.randomUUID();
    }
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const createBotGreeting = () => ({
    id: generateId(),
    type: 'bot',
    message: 'Hello! I am your AgriNova AI assistant. Ask me anything about your farm, inputs, or upcoming decisions.',
    timestamp: new Date().toISOString()
});

const deriveSummary = (messages = []) => {
    if (!messages.length) return 'Ask anything about your farm.';
    const last = messages[messages.length - 1];
    if (!last?.message) return 'Ask anything about your farm.';
    return last.message.length > 90 ? `${last.message.slice(0, 87)}…` : last.message;
};

const deriveTitle = (messages = [], fallback = 'New Conversation') => {
    const firstUserMessage = messages.find(msg => msg.type === 'user');
    if (!firstUserMessage?.message) return fallback;
    const cleaned = firstUserMessage.message.trim().replace(/\s+/g, ' ');
    if (!cleaned) return fallback;
    return cleaned.length > 32 ? `${cleaned.slice(0, 29)}…` : cleaned;
};

const createThread = (label = 'New Conversation') => {
    const messages = [createBotGreeting()];
    return {
        id: generateId(),
        title: label,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        summary: deriveSummary(messages),
        messages
    };
};

const loadThreads = () => {
    if (typeof window === 'undefined') return [createThread()];
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) return [createThread()];
    try {
        const parsed = JSON.parse(stored);
        if (!Array.isArray(parsed) || !parsed.length) {
            return [createThread()];
        }
        return parsed.map(thread => {
            const messageList = Array.isArray(thread.messages) && thread.messages.length ? thread.messages : [createBotGreeting()];
            return {
                ...thread,
                id: thread.id || generateId(),
                title: thread.title || deriveTitle(messageList),
                summary: thread.summary || deriveSummary(messageList),
                messages: messageList
            };
        });
    } catch (error) {
        console.error('Failed to parse stored chat threads', error);
        return [createThread()];
    }
};

const formatDisplayDate = isoDate => {
    if (!isoDate) return '—';
    const date = new Date(isoDate);
    if (Number.isNaN(date.getTime())) return isoDate;
    return date.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' });
};

const formatDisplayTime = isoDate => {
    if (!isoDate) return '';
    const date = new Date(isoDate);
    if (Number.isNaN(date.getTime())) return isoDate;
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

const formatRelativeTime = isoDate => {
    if (!isoDate) return 'just now';
    const date = new Date(isoDate);
    if (Number.isNaN(date.getTime())) return isoDate;
    const diffMs = Date.now() - date.getTime();
    if (diffMs < 60 * 1000) return 'just now';
    if (diffMs < 60 * 60 * 1000) return `${Math.floor(diffMs / (60 * 1000))} min ago`;
    if (diffMs < 24 * 60 * 60 * 1000) return `${Math.floor(diffMs / (60 * 60 * 1000))} hr ago`;
    return date.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' });
};

// Reusable Voice Input Button Component
const VoiceInputButton = ({ onVoiceResult, onVoiceSend }) => {
    const [isListening, setIsListening] = useState(false);
    const [recognition, setRecognition] = useState(null);
    const transcriptRef = useRef(''); // store interim / final transcript until stop

    useEffect(() => {
        if (!('SpeechRecognition' in window) && !('webkitSpeechRecognition' in window)) {
            console.warn('Speech Recognition API not supported in this browser.');
            return;
        }

        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        const rec = new SpeechRec();
        rec.lang = 'en-US';
        rec.interimResults = false;
        rec.maxAlternatives = 1;

        rec.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            transcriptRef.current = transcript;
            onVoiceResult(transcript);
        };

        rec.onerror = (event) => {
            console.error('Speech recognition error: ', event.error);
            setIsListening(false);
        };

        rec.onend = () => {
            setIsListening(false);
        };

        setRecognition(rec);

        return () => {
            try {
                rec.onresult = null;
                rec.onerror = null;
                rec.onend = null;
            } catch (err) {}
        };
    }, [onVoiceResult, onVoiceSend]);

    const toggleListening = () => {
        if (!recognition) return;
        if (isListening) {
            recognition.stop();
            setIsListening(false);
            if (transcriptRef.current && transcriptRef.current.trim()) {
                onVoiceSend(transcriptRef.current);
                transcriptRef.current = '';
            }
        } else {
            transcriptRef.current = '';
            recognition.start();
            setIsListening(true);
        }
    };

    return (
        <button
            type="button"
            onClick={toggleListening}
            className={`px-4 py-3 rounded-lg transition-colors ${
                isListening ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
            } text-white flex items-center`}
            title={isListening ? 'Stop Listening & Send' : 'Start Voice Input'}
        >
            {isListening ? <MicOff className="w-5 h-5 mr-2" /> : <Mic className="w-5 h-5 mr-2" />}
            {isListening ? 'Listening...' : 'Voice'}
        </button>
    );
};

// Main Chatbot Component
const Chatbot = () => {
    const [threads, setThreads] = useState(() => loadThreads());
    const [activeThreadId, setActiveThreadId] = useState(() => {
        if (typeof window === 'undefined') return null;
        return window.localStorage.getItem(ACTIVE_THREAD_KEY);
    });
    const [inputMessage, setInputMessage] = useState('');
    const [historySearch, setHistorySearch] = useState('');
    const [isTyping, setIsTyping] = useState(false);

    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);
    const isVoiceInput = useRef(false);

    useEffect(() => {
        if (typeof window === 'undefined') return;
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(threads));
    }, [threads]);

    useEffect(() => {
        if (typeof window === 'undefined' || !activeThreadId) return;
        window.localStorage.setItem(ACTIVE_THREAD_KEY, activeThreadId);
    }, [activeThreadId]);

    useEffect(() => {
        if (!threads.length) {
            const fallback = [createThread()];
            setThreads(fallback);
            setActiveThreadId(fallback[0].id);
            return;
        }
        if (!activeThreadId || !threads.some(thread => thread.id === activeThreadId)) {
            setActiveThreadId(threads[0].id);
        }
    }, [threads, activeThreadId]);

    const activeThread = useMemo(() => threads.find(thread => thread.id === activeThreadId) || threads[0], [threads, activeThreadId]);
    const messages = activeThread?.messages ?? [];

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Generate TTS on backend and play the returned MP3 URL
    const generateAndPlayTTS = async (text, lang) => {
        if (!text) return;
        try {
            const resp = await fetch('http://127.0.0.1:5001/generate_tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, lang: lang || 'en' })
            });
            if (!resp.ok) throw new Error('TTS generation failed');
            const j = await resp.json();
            if (j.audio_url) {
                const audio = new Audio(j.audio_url);
                await audio.play().catch(err => console.error('Audio play failed', err));
            }
        } catch (err) {
            console.error('TTS error:', err);
            // fallback to browser TTS
            const utterance = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.speak(utterance);
        }
    };

    const handleSendMessage = async (e, voiceInputMessage) => {
        if (e?.preventDefault) e.preventDefault();
        const messageToSend = voiceInputMessage || inputMessage;
        if (!messageToSend.trim()) return;

        // if this was a manual send, reset voice flag
        if (!voiceInputMessage) {
            isVoiceInput.current = false;
        }

        const userMessage = {
            id: generateId(),
            type: 'user',
            message: messageToSend.trim(),
            timestamp: new Date().toISOString()
        };

        const appendToActiveThread = newMessages => {
            setThreads(prev => prev.map(thread => {
                if (thread.id !== (activeThread?.id || activeThreadId)) return thread;
                const updatedMessages = typeof newMessages === 'function' ? newMessages(thread.messages) : newMessages;
                return {
                    ...thread,
                    messages: updatedMessages,
                    updatedAt: new Date().toISOString(),
                    title: deriveTitle(updatedMessages, thread.title),
                    summary: deriveSummary(updatedMessages)
                };
            }));
        };

        appendToActiveThread(prev => [...prev, userMessage]);
        setInputMessage('');
        setIsTyping(true);

        try {
            const response = await fetch('http://127.0.0.1:5001/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: messageToSend })
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();

            const botResponse = data.answer || "Sorry, I couldn't understand that.";

            const botMessage = {
                id: generateId(),
                type: 'bot',
                message: botResponse,
                timestamp: new Date().toISOString(),
                target_lang: data.target_lang || ''
            };

            appendToActiveThread(prev => [...prev, botMessage]);

            if (isVoiceInput.current) {
                const utterance = new SpeechSynthesisUtterance(botResponse);
                window.speechSynthesis.speak(utterance);
            }

        } catch (error) {
            console.error('Error fetching bot response:', error);
            const errorMsg = {
                id: generateId(),
                type: 'bot',
                message: "Sorry, there was an error processing your request.",
                timestamp: new Date().toISOString()
            };
            appendToActiveThread(prev => [...prev, errorMsg]);
        } finally {
            setIsTyping(false);
            isVoiceInput.current = false; // ✅ reset after send
        }
    };

    const clearChat = () => {
        setThreads(prev => prev.map(thread => {
            if (thread.id !== (activeThread?.id || activeThreadId)) return thread;
            const resetMessages = [createBotGreeting()];
            return {
                ...thread,
                messages: resetMessages,
                summary: deriveSummary(resetMessages),
                title: 'New Conversation',
                updatedAt: new Date().toISOString()
            };
        }));
    };

    const createNewThread = () => {
        const newThread = createThread(`Conversation #${threads.length + 1}`);
        setThreads(prev => [newThread, ...prev]);
        setActiveThreadId(newThread.id);
        setInputMessage('');
    };

    const deleteThread = (threadId) => {
        setThreads(prev => {
            const filtered = prev.filter(thread => thread.id !== threadId);
            if (!filtered.length) {
                const fallback = createThread();
                setActiveThreadId(fallback.id);
                return [fallback];
            }
            if (threadId === (activeThread?.id || activeThreadId)) {
                setActiveThreadId(filtered[0].id);
            }
            return filtered;
        });
    };

    const resetAllThreads = () => {
        const freshThread = createThread();
        setThreads([freshThread]);
        setActiveThreadId(freshThread.id);
        setInputMessage('');
    };

    const orderedThreads = useMemo(() => {
        const list = [...threads].sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
        if (!historySearch.trim()) return list;
        const search = historySearch.toLowerCase();
        return list.filter(thread => {
            const title = thread.title?.toLowerCase() || '';
            const summary = thread.summary?.toLowerCase() || '';
            return title.includes(search) || summary.includes(search);
        });
    }, [threads, historySearch]);

    return (
        <div className="p-6 space-y-6 min-h-screen">
            <div className="glass-card p-6 flex flex-col gap-2">
                <p className="text-sm uppercase tracking-[0.2em] text-green-300 flex items-center gap-2">
                    <Sparkles className="w-4 h-4" /> Autonomous Farm Copilot
                </p>
                <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
                    <div>
                        <h1 className="text-3xl font-semibold text-white">AgriNova AI Assistant</h1>
                        <p className="text-gray-400 mt-1">Ground intelligence, agronomy best-practices, and weather foresight in one conversation.</p>
                    </div>
                    <div className="flex items-center gap-3 text-sm">
                        <div className="px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-gray-300">
                            <p className="text-xs text-gray-400">Active thread</p>
                            <p className="text-white font-medium">{activeThread?.title || 'New Conversation'}</p>
                        </div>
                        <div className="px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-gray-300">
                            <p className="text-xs text-gray-400">Last updated</p>
                            <p className="text-white font-medium">{formatRelativeTime(activeThread?.updatedAt)}</p>
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-240px)]">
                {/* Chat History Panel */}
                <div className="lg:col-span-1">
                    <div className="glass-card p-4 h-full flex flex-col">
                        <div className="flex items-center justify-between mb-4">
                            <div>
                                <p className="text-sm text-gray-400">Conversations</p>
                                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                                    <History className="w-5 h-5 text-blue-400" />
                                    {threads.length} active
                                </h3>
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    type="button"
                                    onClick={createNewThread}
                                    className="p-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white"
                                    title="Start a fresh chat"
                                >
                                    <Plus className="w-4 h-4" />
                                </button>
                                <button
                                    type="button"
                                    onClick={resetAllThreads}
                                    className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300"
                                    title="Clear all chats"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                            </div>
                        </div>

                        <div className="mb-3">
                            <input
                                type="text"
                                value={historySearch}
                                onChange={(e) => setHistorySearch(e.target.value)}
                                placeholder="Search agronomy, pests, soil…"
                                className="w-full bg-gray-800/60 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500/40"
                            />
                        </div>

                        <div className="flex-1 overflow-y-auto space-y-3 pr-1">
                            {orderedThreads.length === 0 ? (
                                <div className="text-center py-10 text-sm text-gray-500">
                                    Start chatting to build your agronomy memory.
                                </div>
                            ) : (
                                orderedThreads.map(thread => {
                                    const isActive = thread.id === (activeThread?.id || activeThreadId);
                                    return (
                                        <button
                                            key={thread.id}
                                            onClick={() => setActiveThreadId(thread.id)}
                                            className={`w-full text-left rounded-2xl p-3 border transition-all ${
                                                isActive
                                                    ? 'border-green-400/60 bg-green-400/10 shadow-lg shadow-green-900/20'
                                                    : 'border-white/5 bg-white/5 hover:border-white/20'
                                            }`}
                                        >
                                            <div className="flex items-center justify-between gap-2">
                                                <div className="min-w-0">
                                                    <p className="text-sm font-semibold text-white truncate">{thread.title}</p>
                                                    <p className="text-xs text-gray-400">{formatRelativeTime(thread.updatedAt)} · {thread.messages.length} msgs</p>
                                                </div>
                                                <button
                                                    type="button"
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        deleteThread(thread.id);
                                                    }}
                                                    className="text-gray-500 hover:text-red-400"
                                                    title="Delete conversation"
                                                >
                                                    <Trash2 className="w-4 h-4" />
                                                </button>
                                            </div>
                                            <p className="text-xs text-gray-300 mt-2 line-clamp-2">{thread.summary}</p>
                                            <p className="text-[11px] text-gray-500 mt-1">Updated {formatDisplayDate(thread.updatedAt)}</p>
                                        </button>
                                    );
                                })
                            )}
                        </div>

                        <div className="pt-4 border-t border-white/5 mt-2">
                            <button
                                onClick={clearChat}
                                className="w-full bg-green-600 hover:bg-green-500 text-white py-2 px-4 rounded-xl text-sm font-medium flex items-center justify-center gap-2"
                            >
                                <RefreshCw className="w-4 h-4" /> Reset Current Thread
                            </button>
                        </div>
                    </div>
                </div>

                {/* Chat Surface */}
                <div className="lg:col-span-3">
                    <div className="glass-card p-6 h-full flex flex-col">
                        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-4">
                            <div>
                                <p className="text-xs uppercase tracking-[0.3em] text-green-300">Live intelligence</p>
                                <h3 className="text-2xl font-semibold text-white flex items-center gap-3">
                                    <MessageCircle className="w-6 h-6 text-green-400" />
                                    AI Agronomist Desk
                                </h3>
                                <p className="text-sm text-gray-400">Share soil tests, field stress, drone findings, or business constraints.</p>
                            </div>
                            <div className="flex items-center gap-3">
                                <button
                                    onClick={clearChat}
                                    className="p-2 rounded-lg border border-white/10 text-gray-300 hover:text-white"
                                    title="Clear current chat"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                            </div>
                        </div>

                        <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
                            {messages.map(message => (
                                <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div
                                        className={`group relative max-w-xl px-4 py-3 rounded-2xl border ${
                                            message.type === 'user'
                                                ? 'bg-green-500/90 text-white border-green-300/40'
                                                : 'bg-white/5 text-gray-100 border-white/10'
                                        }`}
                                    >
                                        <div className="flex items-start gap-2">
                                            <div className="mt-1">
                                                {message.type === 'user' ? (
                                                    <User className="w-4 h-4" />
                                                ) : (
                                                    <Bot className="w-4 h-4 text-green-200" />
                                                )}
                                            </div>
                                            <div className="flex-1">
                                                <p className="text-sm leading-relaxed whitespace-pre-line">{message.message}</p>
                                                <span className="text-[11px] opacity-70 block mt-2">{formatDisplayTime(message.timestamp)}</span>
                                            </div>
                                        </div>
                                        {message.type === 'bot' && (
                                            <button
                                                type="button"
                                                onClick={() => generateAndPlayTTS(message.message, message.target_lang || 'en')}
                                                className="absolute -bottom-3 right-3 text-xs px-2 py-0.5 rounded-full bg-white/10 text-gray-200 opacity-0 group-hover:opacity-100 transition"
                                                title="Play spoken response"
                                            >
                                                🔊 Listen
                                            </button>
                                        )}
                                    </div>
                                </div>
                            ))}

                            {isTyping && (
                                <div className="flex justify-start">
                                    <div className="flex items-center gap-3 bg-white/5 border border-white/10 px-4 py-2 rounded-2xl text-gray-200 text-sm">
                                        <Loader2 className="w-4 h-4 animate-spin text-green-300" />
                                        Drafting precision guidance…
                                    </div>
                                </div>
                            )}

                            <div ref={messagesEndRef} />
                        </div>

                        <form onSubmit={handleSendMessage} className="space-y-3">
                            <div className="flex flex-col lg:flex-row gap-3">
                                <input
                                    ref={inputRef}
                                    type="text"
                                    value={inputMessage}
                                    onChange={(e) => setInputMessage(e.target.value)}
                                    placeholder="Describe field stress, upload lab numbers, or ask for an action plan…"
                                    className="flex-1 bg-black/40 border border-white/10 rounded-2xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-green-500/40"
                                    disabled={isTyping}
                                />
                                <div className="flex items-center gap-2">
                                    <VoiceInputButton
                                        onVoiceResult={(transcript) => setInputMessage(transcript)}
                                        onVoiceSend={(transcript) => {
                                            isVoiceInput.current = true;
                                            handleSendMessage(null, transcript);
                                        }}
                                    />
                                    <button
                                        type="submit"
                                        disabled={!inputMessage.trim() || isTyping}
                                        className="h-full bg-green-600 hover:bg-green-500 disabled:bg-gray-700 disabled:cursor-not-allowed text-white px-6 py-3 rounded-2xl font-medium flex items-center gap-2"
                                    >
                                        <Send className="w-5 h-5" />
                                        Send
                                    </button>
                                </div>
                            </div>
                            <div className="flex flex-wrap gap-2 text-xs">
                                {['Generate irrigation plan','Explain soil test report','Forecast pest risk','Optimize spray schedule','Estimate market-ready date'].map(prompt => (
                                    <button
                                        key={prompt}
                                        type="button"
                                        onClick={() => setInputMessage(prompt)}
                                        className="px-3 py-1 rounded-full border border-white/10 text-gray-300 hover:text-white hover:border-white/40"
                                    >
                                        {prompt}
                                    </button>
                                ))}
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Chatbot;
