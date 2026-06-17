import { useState, useEffect } from 'react';
import { getApiUrl, setApiUrl } from '../utils/api';
import { Save, AlertCircle } from 'lucide-react';

const Settings = () => {
    const [url, setUrl] = useState('');
    const [interval, setIntervalVal] = useState(1000);
    const [saved, setSaved] = useState(false);

    useEffect(() => {
        setUrl(getApiUrl());
        const savedInterval = localStorage.getItem('visionsafe_interval');
        if (savedInterval) setIntervalVal(Number(savedInterval));
    }, []);

    const handleSave = (e) => {
        e.preventDefault();
        setApiUrl(url);
        localStorage.setItem('visionsafe_interval', interval);
        setSaved(true);
        setTimeout(() => setSaved(false), 3000);
    };

    return (
        <div className="max-w-2xl">
            <header className="mb-8">
                <h1 className="text-3xl font-bold mb-1">System Settings</h1>
                <p className="text-gray-400">Configure application behavior.</p>
            </header>

            <form onSubmit={handleSave} className="bg-dark-800 border border-dark-700 rounded-xl p-6 space-y-8">

                {/* Connection Settings */}
                <div className="space-y-4">
                    <h3 className="text-lg font-bold flex items-center gap-2 pb-2 border-b border-dark-700">
                        Connectivity
                    </h3>

                    <div>
                        <label className="block text-sm font-medium text-gray-400 mb-2">Backend API URL</label>
                        <input
                            type="url"
                            required
                            className="w-full bg-dark-900 border border-dark-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary-500 transition-colors"
                            placeholder="http://127.0.0.1:8000"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                        />
                        <p className="text-xs text-gray-500 mt-2 flex items-center gap-1">
                            <AlertCircle size={14} />
                            Must include protocol (http://)
                        </p>
                    </div>
                </div>

                {/* Performance Settings */}
                <div className="space-y-4">
                    <h3 className="text-lg font-bold flex items-center gap-2 pb-2 border-b border-dark-700">
                        Performance
                    </h3>

                    <div>
                        <label className="block text-sm font-medium text-gray-400 mb-2">Auto-Detection Interval (ms)</label>
                        <input
                            type="number"
                            min="100"
                            max="10000"
                            step="100"
                            required
                            className="w-full bg-dark-900 border border-dark-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary-500 transition-colors"
                            value={interval}
                            onChange={(e) => setIntervalVal(e.target.value)}
                        />
                        <p className="text-xs text-gray-500 mt-2">
                            Faster interval increases load. Recommended: 1000ms.
                        </p>
                    </div>
                </div>

                <div className="pt-4 flex items-center gap-4">
                    <button
                        type="submit"
                        className="bg-primary-600 hover:bg-primary-500 text-white font-bold py-3 px-8 rounded-lg transition-colors flex items-center gap-2"
                    >
                        <Save size={20} /> Save Changes
                    </button>
                    {saved && (
                        <span className="text-green-500 font-medium animate-fade-in">Settings Saved!</span>
                    )}
                </div>
            </form>
        </div>
    );
};

export default Settings;
