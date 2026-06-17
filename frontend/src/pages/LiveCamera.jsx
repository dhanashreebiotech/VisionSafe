import { useState, useRef, useEffect, useCallback } from 'react';
import { detectMedia } from '../utils/api';
import { Camera, StopCircle, RefreshCw, Zap, AlertTriangle, CheckCircle, Activity } from 'lucide-react';
import BoundingBoxOverlay from '../components/BoundingBoxOverlay';
import clsx from 'clsx';

const LiveCamera = () => {
    const videoRef = useRef(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [autoDetect, setAutoDetect] = useState(false);
    const [result, setResult] = useState(null);
    const [fps, setFps] = useState(0);

    // Refs to break stale closures in the detect loop
    const isStreamingRef = useRef(false);
    const autoDetectRef = useRef(false);

    // Keep refs in sync with state
    useEffect(() => { isStreamingRef.current = isStreaming; }, [isStreaming]);
    useEffect(() => { autoDetectRef.current = autoDetect; }, [autoDetect]);

    // RequestAnimationFrame ID
    const rafId = useRef(null);
    const lastFrameTime = useRef(0);
    const frameCount = useRef(0);
    const lastFpsTime = useRef(Date.now());

    // Dimensions for overlay scaling
    const [dimensions, setDimensions] = useState({ displayedW: 0, displayedH: 0, naturalW: 0, naturalH: 0 });

    const updateDimensions = useCallback(() => {
        if (videoRef.current) {
            setDimensions({
                displayedW: videoRef.current.clientWidth,
                displayedH: videoRef.current.clientHeight,
                naturalW: videoRef.current.videoWidth,
                naturalH: videoRef.current.videoHeight
            });
        }
    }, [videoRef]);

    useEffect(() => {
        window.addEventListener('resize', updateDimensions);
        return () => window.removeEventListener('resize', updateDimensions);
    }, [updateDimensions]);

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current.play();
                    updateDimensions();
                    setIsStreaming(true);
                };
            }
        } catch (err) {
            alert("Could not access camera: " + err.message);
        }
    };

    const stopCamera = () => {
        if (rafId.current) cancelAnimationFrame(rafId.current);
        if (videoRef.current && videoRef.current.srcObject) {
            videoRef.current.srcObject.getTracks().forEach(t => t.stop());
            videoRef.current.srcObject = null;
        }
        setIsStreaming(false);
        setAutoDetect(false);
        setResult(null);
    };

    const detectLoop = async (timestamp) => {
        // Use refs instead of state to avoid stale closures
        if (!isStreamingRef.current || !autoDetectRef.current || !videoRef.current) return;

        // Throttle to ~10 FPS (100ms) for stability
        if (timestamp - lastFrameTime.current < 100) {
            rafId.current = requestAnimationFrame(detectLoop);
            return;
        }

        lastFrameTime.current = timestamp;

        // Sync dimensions check
        if (videoRef.current.videoWidth !== dimensions.naturalW) {
            updateDimensions();
        }

        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async (blob) => {
            if (!blob) return;
            try {
                const file = new File([blob], "frame.jpg", { type: "image/jpeg" });
                const res = await detectMedia(file, 'frame');
                setResult(res);

                // Metrics
                frameCount.current++;
                const now = Date.now();
                if (now - lastFpsTime.current >= 1000) {
                    setFps(frameCount.current);
                    frameCount.current = 0;
                    lastFpsTime.current = now;
                }

                // Continue Loop if still active (check refs, not state)
                if (isStreamingRef.current && autoDetectRef.current) {
                    rafId.current = requestAnimationFrame(detectLoop);
                }

            } catch (err) {
                console.error("Frame detection failed", err);
                // Retry even on fail (check refs)
                if (isStreamingRef.current && autoDetectRef.current) {
                    rafId.current = requestAnimationFrame(detectLoop);
                }
            }
        }, 'image/jpeg', 0.6); // Lower quality for speed
    };

    const manualCapture = () => {
        detectLoop(performance.now());
    };

    // Toggle logic
    useEffect(() => {
        if (isStreaming && autoDetect) {
            rafId.current = requestAnimationFrame(detectLoop);
        } else {
            if (rafId.current) cancelAnimationFrame(rafId.current);
        }
        return () => {
            if (rafId.current) cancelAnimationFrame(rafId.current);
        };
    }, [isStreaming, autoDetect]);

    return (
        <div className="h-full flex flex-col p-6 animate-fade-in">
            <header className="mb-6 flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold mb-1 text-white">Live Monitor</h1>
                    <p className="text-gray-400">AI-Powered Real-time Surveillance.</p>
                </div>

                {result && (
                    <div className={clsx("px-6 py-3 rounded-xl border flex items-center gap-4 transition-all duration-300",
                        (result.status || result.safety_status) === 'UNSAFE'
                            ? "bg-red-500/10 border-red-500/30 text-red-500 shadow-[0_0_20px_rgba(239,68,68,0.2)]"
                            : "bg-green-500/10 border-green-500/30 text-green-500 shadow-[0_0_20px_rgba(16,185,129,0.2)]"
                    )}>
                        {(result.status || result.safety_status) === 'UNSAFE' ? <AlertTriangle size={24} /> : <CheckCircle size={24} />}
                        <div className="flex flex-col">
                            <span className="font-bold text-lg tracking-wide">{result.status || result.safety_status}</span>
                            <span className="text-xs opacity-80 uppercase tracking-wider">{result.activity}</span>
                        </div>
                    </div>
                )}
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 flex-1 min-h-0">
                <div className="lg:col-span-3 bg-black rounded-2xl overflow-hidden relative flex items-center justify-center border border-dark-700 shadow-2xl">
                    {!isStreaming && (
                        <div className="text-gray-600 flex flex-col items-center">
                            <div className="bg-dark-800 p-6 rounded-full mb-4">
                                <Camera size={48} className="opacity-50" />
                            </div>
                            <p className="text-lg">Camera Feed Offline</p>
                        </div>
                    )}

                    <video
                        ref={videoRef}
                        className={clsx("absolute w-full h-full object-contain", !isStreaming && "hidden")}
                        muted
                        playsInline
                    />

                    {isStreaming && result && result.detections && (
                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                            <div style={{ width: dimensions.displayedW, height: dimensions.displayedH, position: 'relative' }}>
                                <BoundingBoxOverlay
                                    detections={result.detections}
                                    mediaWidth={dimensions.naturalW}
                                    mediaHeight={dimensions.naturalH}
                                    displayedWidth={dimensions.displayedW}
                                    displayedHeight={dimensions.displayedH}
                                />
                            </div>
                        </div>
                    )}
                </div>

                <div className="lg:col-span-1 space-y-4 overflow-y-auto">
                    <div className="bg-dark-800 border border-dark-700 rounded-2xl p-6 shadow-lg">
                        <h3 className="font-bold text-gray-200 border-b border-dark-700 pb-3 mb-4 flex items-center gap-2">
                            Controls
                        </h3>
                        <div className="space-y-3">
                            {!isStreaming ? (
                                <button onClick={startCamera} className="w-full flex items-center justify-center gap-2 bg-primary-600 hover:bg-primary-500 text-white py-3 rounded-xl font-bold transition-all hover:scale-[1.02]">
                                    <Camera size={20} /> Start Camera
                                </button>
                            ) : (
                                <button onClick={stopCamera} className="w-full flex items-center justify-center gap-2 bg-red-600 hover:bg-red-500 text-white py-3 rounded-xl font-bold transition-all hover:scale-[1.02]">
                                    <StopCircle size={20} /> Stop Feed
                                </button>
                            )}

                            <div className="flex items-center justify-between bg-dark-900/50 p-1 rounded-xl border border-dark-700 mt-2">
                                <button
                                    onClick={() => setAutoDetect(!autoDetect)}
                                    disabled={!isStreaming}
                                    className={clsx("flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-semibold transition-all",
                                        autoDetect
                                            ? "bg-primary-500 text-white shadow-lg"
                                            : "text-gray-400 hover:text-white"
                                    )}
                                >
                                    <Zap size={16} className={autoDetect ? "fill-white" : ""} />
                                    {autoDetect ? "Detection ON" : "Off"}
                                </button>
                            </div>

                            <button
                                onClick={manualCapture}
                                disabled={!isStreaming || autoDetect}
                                className="w-full flex items-center justify-center gap-2 bg-dark-700 hover:bg-dark-600 text-gray-300 py-3 rounded-xl font-medium transition-colors disabled:opacity-30 text-sm"
                            >
                                <RefreshCw size={16} /> Single Capture
                            </button>
                        </div>
                    </div>

                    {result ? (
                        <div className="bg-dark-800 border border-dark-700 rounded-2xl p-6 shadow-lg animate-fade-in-data">
                            <h3 className="font-bold text-gray-200 border-b border-dark-700 pb-3 mb-4 flex items-center gap-2">
                                <Activity size={18} className="text-primary-500" /> Analysis
                            </h3>

                            <div className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div className="bg-dark-900 p-3 rounded-lg border border-dark-700">
                                        <div className="text-xs text-gray-500 uppercase font-bold mb-1">FPS</div>
                                        <div className="text-xl font-mono text-white">{fps}</div>
                                    </div>
                                    <div className="bg-dark-900 p-3 rounded-lg border border-dark-700">
                                        <div className="text-xs text-gray-500 uppercase font-bold mb-1">Conf</div>
                                        <div className="text-xl font-mono text-white">{(result.confidence * 100).toFixed(0)}%</div>
                                    </div>
                                </div>

                                <div>
                                    <div className="text-xs text-gray-400 mb-2 font-semibold">DETECTED</div>
                                    <div className="flex flex-wrap gap-2">
                                        {result.detections.length > 0 ? result.detections.map((d, i) => (
                                            <span key={i} className={clsx("text-xs px-2 py-1 rounded border flex items-center gap-1 font-medium",
                                                d.type === 'unsafe' ? "bg-red-500/20 text-red-400 border-red-500/30" : "bg-blue-500/20 text-blue-400 border-blue-500/30"
                                            )}>
                                                {d.label}
                                            </span>
                                        )) : <span className="text-gray-600 italic text-sm">None</span>}
                                    </div>
                                </div>
                            </div>
                        </div>
                    ) : null}
                </div>
            </div>
        </div>
    );
};

export default LiveCamera;
