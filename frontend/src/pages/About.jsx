import { ShieldCheck, Target, Lock, Brain } from 'lucide-react';
import { Link } from 'react-router-dom';

const About = () => {
    return (
        <div className="max-w-4xl mx-auto py-12 px-4">
            <div className="text-center mb-16">
                <div className="inline-flex items-center justify-center w-20 h-20 bg-primary-500/10 rounded-full mb-6">
                    <ShieldCheck size={40} className="text-primary-500" />
                </div>
                <h1 className="text-4xl font-extrabold text-white mb-4">About VisionSafe</h1>
                <p className="text-xl text-gray-400">Enterprise-grade AI Safety Monitoring & Anomaly Detection.</p>
            </div>

            <div className="grid md:grid-cols-3 gap-8 mb-16">
                <div className="bg-dark-800 p-6 rounded-xl border border-dark-700">
                    <div className="w-12 h-12 bg-blue-500/10 rounded-lg flex items-center justify-center text-blue-500 mb-4">
                        <Brain size={24} />
                    </div>
                    <h3 className="text-lg font-bold text-white mb-2">Hybrid AI Engine</h3>
                    <p className="text-gray-400 text-sm leading-relaxed">
                        Combines YOLOv8 object detection with Pose Estimation to understand complex human activities and environmental hazards simultaneously.
                    </p>
                </div>
                <div className="bg-dark-800 p-6 rounded-xl border border-dark-700">
                    <div className="w-12 h-12 bg-red-500/10 rounded-lg flex items-center justify-center text-red-500 mb-4">
                        <Target size={24} />
                    </div>
                    <h3 className="text-lg font-bold text-white mb-2">Real-time Threat Detection</h3>
                    <p className="text-gray-400 text-sm leading-relaxed">
                        Instantly identifies unsafe conditions such as fire, fighting, unauthorised vehicles, or falls with sub-second latency.
                    </p>
                </div>
                <div className="bg-dark-800 p-6 rounded-xl border border-dark-700">
                    <div className="w-12 h-12 bg-green-500/10 rounded-lg flex items-center justify-center text-green-500 mb-4">
                        <Lock size={24} />
                    </div>
                    <h3 className="text-lg font-bold text-white mb-2">Privacy First</h3>
                    <p className="text-gray-400 text-sm leading-relaxed">
                        Edge-compatible architecture ensures data stays local. No cloud processing required, ensuring compliance with strict security protocols.
                    </p>
                </div>
            </div>

            <div className="bg-dark-800 rounded-2xl p-8 border border-dark-700 mb-12">
                <h2 className="text-2xl font-bold mb-6">How It Works</h2>
                <div className="space-y-6">
                    <div className="flex gap-4">
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-dark-700 flex items-center justify-center text-sm font-bold">1</div>
                        <div>
                            <h4 className="font-bold text-lg mb-1">Input Stream Analysis</h4>
                            <p className="text-gray-400">Video feeds from CCTV or webcams are ingested frame-by-frame into the local inference engine.</p>
                        </div>
                    </div>
                    <div className="flex gap-4">
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-dark-700 flex items-center justify-center text-sm font-bold">2</div>
                        <div>
                            <h4 className="font-bold text-lg mb-1">Hybrid Processing</h4>
                            <p className="text-gray-400">The frame is split into two parallel pipelines: Object Detection for hazards and Pose Estimation for human behavior understanding.</p>
                        </div>
                    </div>
                    <div className="flex gap-4">
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-dark-700 flex items-center justify-center text-sm font-bold">3</div>
                        <div>
                            <h4 className="font-bold text-lg mb-1">Alert Generation</h4>
                            <p className="text-gray-400">Detections are fused to determine context. If "Unsafe" criteria are met, an immediate alert is logged and displayed on the dashboard.</p>
                        </div>
                    </div>
                </div>
            </div>

            <div className="text-center">
                <Link to="/login" className="text-primary-500 hover:text-white font-medium transition-colors">
                    Back to Login
                </Link>
            </div>
        </div>
    );
};

export default About;
