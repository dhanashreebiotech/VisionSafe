import { useState, useEffect } from 'react';
import { checkHealth, getApiUrl } from '../utils/api';
import { getHistory } from '../utils/storage';
import { AlertTriangle, CheckCircle, Server, Activity, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import clsx from 'clsx';

const Dashboard = () => {
    const [backendStatus, setBackendStatus] = useState('checking'); // checking, online, offline
    const [history, setHistory] = useState([]);

    useEffect(() => {
        const init = async () => {
            setHistory(getHistory());
            const health = await checkHealth();
            setBackendStatus(health && health.status === 'ok' ? 'online' : 'offline');
        };
        init();

        // Poll status
        const interval = setInterval(async () => {
            const health = await checkHealth();
            setBackendStatus(health && health.status === 'ok' ? 'online' : 'offline');
        }, 5000);
        return () => clearInterval(interval);
    }, []);

    const recentDetections = history.slice(0, 5);
    const totalDetections = history.length;
    const unsafeCount = history.filter(h => h.safety_status === 'UNSAFE').length;

    return (
        <div>
            <header className="mb-8">
                <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
                <p className="text-gray-400">System Overview and Status</p>
            </header>

            {/* Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div className="bg-dark-800 border border-dark-700 rounded-xl p-6 flex items-center justify-between">
                    <div>
                        <div className="text-gray-400 text-sm font-medium mb-1">Backend Connectivity</div>
                        <div className={clsx("text-lg font-bold flex items-center gap-2",
                            backendStatus === 'online' ? "text-green-500" : "text-red-500"
                        )}>
                            <Server size={20} />
                            {backendStatus === 'online' ? "Connected" : "Disconnected"}
                        </div>
                        <div className="text-xs text-gray-500 mt-1">{getApiUrl()}</div>
                    </div>
                </div>

                <div className="bg-dark-800 border border-dark-700 rounded-xl p-6 flex items-center justify-between">
                    <div>
                        <div className="text-gray-400 text-sm font-medium mb-1">Total Detections</div>
                        <div className="text-3xl font-bold text-white">{totalDetections}</div>
                        <div className="text-xs text-gray-500 mt-1">Since last clear</div>
                    </div>
                    <div className="p-3 bg-primary-500/10 rounded-lg text-primary-500">
                        <Activity size={24} />
                    </div>
                </div>

                <div className="bg-dark-800 border border-dark-700 rounded-xl p-6 flex items-center justify-between">
                    <div>
                        <div className="text-gray-400 text-sm font-medium mb-1">Safety Alerts</div>
                        <div className="text-3xl font-bold text-red-500">{unsafeCount}</div>
                        <div className="text-xs text-gray-500 mt-1">Incidents recorded</div>
                    </div>
                    <div className="p-3 bg-red-500/10 rounded-lg text-red-500">
                        <AlertTriangle size={24} />
                    </div>
                </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-dark-800 border border-dark-700 rounded-xl overflow-hidden">
                <div className="px-6 py-4 border-b border-dark-700 flex justify-between items-center">
                    <h3 className="font-bold text-lg">Recent Detections</h3>
                    <Link to="/history" className="text-primary-500 text-sm hover:underline flex items-center gap-1">
                        View All <ArrowRight size={16} />
                    </Link>
                </div>
                {recentDetections.length === 0 ? (
                    <div className="p-8 text-center text-gray-500">
                        No detections recorded yet. Start by Uploading or using Camera.
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full text-left text-sm text-gray-400">
                            <thead className="bg-dark-900 border-b border-dark-700 uppercase">
                                <tr>
                                    <th className="px-6 py-3 font-medium">Time</th>
                                    <th className="px-6 py-3 font-medium">Activity</th>
                                    <th className="px-6 py-3 font-medium">Status</th>
                                    <th className="px-6 py-3 font-medium">Confidence</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-dark-700">
                                {recentDetections.map((row) => (
                                    <tr key={row.id} className="hover:bg-dark-700/50">
                                        <td className="px-6 py-4 text-white">
                                            {new Date(row.timestamp).toLocaleTimeString()}
                                        </td>
                                        <td className="px-6 py-4 text-white font-medium capitalize">
                                            {row.activity || '-'}
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={clsx(
                                                "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border",
                                                row.safety_status === 'UNSAFE'
                                                    ? "bg-red-500/10 text-red-500 border-red-500/20"
                                                    : "bg-green-500/10 text-green-500 border-green-500/20"
                                            )}>
                                                {row.safety_status === 'UNSAFE' ? <AlertTriangle size={12} /> : <CheckCircle size={12} />}
                                                {row.safety_status}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            {row.confidence ? (row.confidence * 100).toFixed(1) + '%' : '-'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-2 gap-6 mt-8">
                <Link to="/upload" className="block p-6 bg-dark-800 border border-dark-700 rounded-xl hover:border-primary-500 transition-colors group">
                    <h3 className="text-xl font-bold mb-2 group-hover:text-primary-500">Upload Image/Video</h3>
                    <p className="text-gray-400 text-sm">Run detection on existing files.</p>
                </Link>
                <Link to="/camera" className="block p-6 bg-dark-800 border border-dark-700 rounded-xl hover:border-primary-500 transition-colors group">
                    <h3 className="text-xl font-bold mb-2 group-hover:text-primary-500">Live Camera Feed</h3>
                    <p className="text-gray-400 text-sm">Real-time monitoring using webcam.</p>
                </Link>
            </div>
        </div>
    );
};

export default Dashboard;
