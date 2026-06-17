import { useState, useRef, useEffect, useCallback } from 'react';
import { detectMedia } from '../utils/api';
import { addToHistory } from '../utils/storage';
import { Upload as UploadIcon, X, CheckCircle, AlertTriangle, PlayCircle, PauseCircle, Activity } from 'lucide-react';
import BoundingBoxOverlay from '../components/BoundingBoxOverlay';
import clsx from 'clsx';

const UploadDetection = () => {
    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    // Video State
    const videoRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const playbackInterval = useRef(null);

    // Dimensions
    const containerRef = useRef(null);
    const [dimensions, setDimensions] = useState({ displayedW: 0, displayedH: 0, naturalW: 0, naturalH: 0 });

    const handleFileChange = (e) => {
        const selected = e.target.files[0];
        if (!selected) return;

        setFile(selected);
        setPreviewUrl(URL.createObjectURL(selected));

        // Reset state
        setResult(null);
        setError(null);
        setIsPlaying(false);
        if (playbackInterval.current) clearInterval(playbackInterval.current);
    };

    const handleClear = () => {
        setFile(null);
        setPreviewUrl(null);
        setResult(null);
        setError(null);
        setDimensions({ displayedW: 0, displayedH: 0, naturalW: 0, naturalH: 0 });
        if (playbackInterval.current) clearInterval(playbackInterval.current);
    };

    const updateDimensionsImage = (e) => {
        setDimensions({
            displayedW: e.target.width,
            displayedH: e.target.height,
            naturalW: e.target.naturalWidth,
            naturalH: e.target.naturalHeight
        });
    };

    // Called on video metadata load AND updates during playback if resize happens
    const updateDimensionsVideo = useCallback(() => {
        if (!videoRef.current) return;
        setDimensions({
            displayedW: videoRef.current.clientWidth,
            displayedH: videoRef.current.clientHeight,
            naturalW: videoRef.current.videoWidth,
            naturalH: videoRef.current.videoHeight
        });
    }, []);

    // Static Analysis (Button Click)
    const runStaticDetection = async () => {
        if (!file) return;
        setLoading(true);
        setError(null);

        try {
            // mode='file' hits /detect (Uploads + Analysis)
            const res = await detectMedia(file, 'file');
            setResult(res);
            addToHistory({ ...res, source: "Upload" });
        } catch (err) {
            setError(err.message || "Detection failed");
        } finally {
            setLoading(false);
        }
    };

    // Live Playback Analysis (Frame loop)
    const captureVideoFrame = async () => {
        if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) return;

        try {
            updateDimensionsVideo();

            const canvas = document.createElement('canvas');
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoRef.current, 0, 0);

            canvas.toBlob(async (blob) => {
                if (!blob) return;

                // Hit /detect_frame for realtime feedback
                const fileBlob = new File([blob], "frame.jpg", { type: "image/jpeg" });
                const res = await detectMedia(fileBlob, 'frame');

                // Update Overlay only
                setResult(res);
            }, 'image/jpeg', 0.8);

        } catch (e) {
            console.error(e);
        }
    };

    const handleVideoPlay = () => {
        setIsPlaying(true);
        // Start Analysis Loop (3 FPS -> 330ms)
        playbackInterval.current = setInterval(captureVideoFrame, 330);
    };

    const handleVideoPause = () => {
        setIsPlaying(false);
        if (playbackInterval.current) clearInterval(playbackInterval.current);
    };

    useEffect(() => {
        return () => {
            if (playbackInterval.current) clearInterval(playbackInterval.current);
        };
    }, []);

    const isVideo = file?.type.startsWith('video');

    return (
        <div className="h-full flex flex-col p-6 animate-fade-in">
            <header className="mb-6 flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold mb-1 text-white">Upload Analysis</h1>
                    <p className="text-gray-400">Deep forensic analysis of images and videos.</p>
                </div>
                {result && (
                    <div className={clsx("px-4 py-2 rounded-xl border flex items-center gap-2",
                        result.safety_status === 'UNSAFE'
                            ? "bg-red-500/10 border-red-500/20 text-red-500"
                            : "bg-green-500/10 border-green-500/20 text-green-500"
                    )}>
                        {result.safety_status === 'UNSAFE' ? <AlertTriangle size={20} /> : <CheckCircle size={20} />}
                        <span className="font-bold uppercase">{result.safety_status}</span>
                    </div>
                )}
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 flex-1 min-h-0">

                {/* Left Panel: Upload & Controls */}
                <div className="lg:col-span-1 space-y-6">
                    {!file ? (
                        <div className="border-2 border-dashed border-dark-600 hover:border-primary-500 rounded-2xl p-12 text-center transition-all cursor-pointer bg-dark-800/50 hover:bg-dark-800 group"
                            onClick={() => document.getElementById('file-upload').click()}
                        >
                            <div className="bg-dark-700 p-4 rounded-full inline-block mb-4 group-hover:scale-110 transition-transform">
                                <UploadIcon size={32} className="text-gray-300" />
                            </div>
                            <p className="text-lg font-medium text-white">Click to upload media</p>
                            <p className="text-sm text-gray-500 mt-2">MP4 Video or JPG/PNG Image</p>
                            <input
                                id="file-upload"
                                type="file"
                                className="hidden"
                                accept="image/*,video/*"
                                onChange={handleFileChange}
                            />
                        </div>
                    ) : (
                        <div className="bg-dark-800 border border-dark-700 rounded-2xl p-6 shadow-lg">
                            <div className="flex justify-between items-center mb-6">
                                <div className="flex items-center gap-3 overflow-hidden">
                                    <div className="w-10 h-10 bg-primary-900/50 rounded flex items-center justify-center text-primary-400">
                                        {isVideo ? "VID" : "IMG"}
                                    </div>
                                    <span className="font-medium truncate text-gray-200">{file.name}</span>
                                </div>
                                <button onClick={handleClear} className="text-gray-400 hover:text-red-400 p-2 hover:bg-dark-700 rounded-lg transition-colors">
                                    <X size={20} />
                                </button>
                            </div>

                            {/* Actions */}
                            <div className="space-y-3">
                                {isVideo ? (
                                    <div className="text-sm text-gray-400 bg-dark-900/50 p-4 rounded-lg border border-dark-700">
                                        <div className="flex gap-2 items-center mb-2 text-primary-400">
                                            <PlayCircle size={16} />
                                            <span className="font-bold">Interactive Mode</span>
                                        </div>
                                        <p>Play the video to run real-time detection on frames.</p>
                                    </div>
                                ) : (
                                    <button
                                        onClick={runStaticDetection}
                                        disabled={loading}
                                        className={clsx("w-full py-3.5 rounded-xl font-bold text-white transition-all shadow-lg hover:shadow-primary-600/20",
                                            loading ? "bg-dark-600 cursor-not-allowed" : "bg-primary-600 hover:bg-primary-500"
                                        )}
                                    >
                                        {loading ? "Analyzing..." : "Run Static Analysis"}
                                    </button>
                                )}
                            </div>

                            {error && (
                                <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-xl flex gap-2">
                                    <AlertTriangle size={16} className="shrink-0 mt-0.5" />
                                    {error}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Results Details */}
                    {result && result.detections && (
                        <div className="bg-dark-800 border border-dark-700 rounded-2xl p-6 space-y-5 animate-fade-in-up shadow-lg">
                            <h3 className="font-bold text-gray-200 border-b border-dark-700 pb-3 flex items-center gap-2">
                                <Activity size={18} /> Diagnostics
                            </h3>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-dark-900 p-3 rounded-lg border border-dark-700">
                                    <div className="text-xs text-gray-500 uppercase font-bold mb-1">Activity</div>
                                    <div className="font-medium text-lg capitalize text-white truncate">{result.activity || "None"}</div>
                                </div>
                                <div className="bg-dark-900 p-3 rounded-lg border border-dark-700">
                                    <div className="text-xs text-gray-500 uppercase font-bold mb-1">Confidence</div>
                                    <div className="font-medium text-lg text-white">{(result.confidence * 100).toFixed(1)}%</div>
                                </div>
                            </div>

                            <div>
                                <div className="text-xs text-gray-400 uppercase font-bold mb-2">Detected Objects</div>
                                <div className="max-h-48 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
                                    {result.detections.length === 0 ? (
                                        <div className="text-sm text-gray-500 italic p-2">No threats or persons detected.</div>
                                    ) : (
                                        result.detections.map((d, i) => (
                                            <div key={i} className="flex justify-between items-center text-sm bg-dark-900/80 p-3 rounded-lg border border-dark-700">
                                                <span className="font-medium text-gray-300">{d.label}</span>
                                                <span className={clsx("text-xs font-mono px-2 py-0.5 rounded",
                                                    d.confidence > 0.8 ? "bg-green-500/20 text-green-400" : "bg-yellow-500/20 text-yellow-500"
                                                )}>
                                                    {(d.confidence * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                        ))
                                    )}
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Panel: Viewing Area */}
                <div className="lg:col-span-2 bg-black rounded-2xl border border-dark-700 flex items-center justify-center relative overflow-hidden shadow-2xl min-h-[500px]">
                    {previewUrl ? (
                        <div className="relative inline-block" ref={containerRef}>
                            {/* Media */}
                            {isVideo ? (
                                <video
                                    ref={videoRef}
                                    src={previewUrl}
                                    controls
                                    className="max-h-[700px] w-auto max-w-full"
                                    onLoadedMetadata={updateDimensionsVideo}
                                    onPlay={handleVideoPlay}
                                    onPause={handleVideoPause}
                                    onEnded={handleVideoPause}
                                />
                            ) : (
                                <img
                                    src={previewUrl}
                                    alt="Analysis Target"
                                    className="max-h-[700px] w-auto max-w-full"
                                    onLoad={updateDimensionsImage}
                                />
                            )}

                            {/* Overlay */}
                            {result && result.detections && (
                                <div style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: dimensions.displayedW,
                                    height: dimensions.displayedH,
                                    pointerEvents: 'none'
                                }}>
                                    <BoundingBoxOverlay
                                        detections={result.detections}
                                        mediaWidth={dimensions.naturalW}
                                        mediaHeight={dimensions.naturalH}
                                        displayedWidth={dimensions.displayedW}
                                        displayedHeight={dimensions.displayedH}
                                    />
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="text-center text-gray-600">
                            <UploadIcon size={48} className="mx-auto mb-4 opacity-20" />
                            <p className="text-lg">No media loaded</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default UploadDetection;
