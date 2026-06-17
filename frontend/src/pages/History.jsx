import { useState, useEffect } from 'react';
import { getHistory, clearHistory } from '../utils/storage';
import { Trash2, Download, AlertTriangle, CheckCircle, Search } from 'lucide-react';
import clsx from 'clsx';

const History = () => {
    const [data, setData] = useState([]);
    const [filter, setFilter] = useState('');

    const loadHistory = () => {
        setData(getHistory());
    };

    useEffect(() => {
        loadHistory();
    }, []);

    const handleClear = () => {
        if (confirm("Are you sure you want to clear all history? This cannot be undone.")) {
            clearHistory();
            loadHistory();
        }
    };

    const handleExport = () => {
        const jsonString = `data:text/json;chatset=utf-8,${encodeURIComponent(JSON.stringify(data, null, 2))}`;
        const link = document.createElement("a");
        link.href = jsonString;
        link.download = `visionsafe_history_${Date.now()}.json`;
        link.click();
    };

    const filteredData = data.filter(d =>
    (d.activity?.toLowerCase().includes(filter.toLowerCase()) ||
        d.safety_status?.toLowerCase().includes(filter.toLowerCase()) ||
        d.source?.toLowerCase().includes(filter.toLowerCase()))
    );

    return (
        <div>
            <header className="mb-8 flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-bold mb-1">Detection History</h1>
                    <p className="text-gray-400">Log of all analyzed events.</p>
                </div>
                <div className="flex gap-3">
                    <button onClick={handleExport} className="flex items-center gap-2 px-4 py-2 bg-dark-800 hover:bg-dark-700 border border-dark-700 rounded-lg transition-colors">
                        <Download size={18} /> Export
                    </button>
                    <button onClick={handleClear} className="flex items-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20 rounded-lg transition-colors">
                        <Trash2 size={18} /> Clear
                    </button>
                </div>
            </header>

            {/* Filter */}
            <div className="mb-6 relative max-w-md">
                <Search className="absolute left-3 top-3 text-gray-500" size={20} />
                <input
                    type="text"
                    placeholder="Search activity, status, or source..."
                    className="w-full bg-dark-800 border border-dark-700 rounded-lg pl-10 pr-4 py-3 text-white focus:outline-none focus:border-primary-500 transition-colors"
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                />
            </div>

            <div className="bg-dark-800 border border-dark-700 rounded-xl overflow-hidden shadow-lg">
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm text-gray-400">
                        <thead className="bg-dark-900 border-b border-dark-700 uppercase">
                            <tr>
                                <th className="px-6 py-4 font-medium">Timestamp</th>
                                <th className="px-6 py-4 font-medium">Source</th>
                                <th className="px-6 py-4 font-medium">Activity</th>
                                <th className="px-6 py-4 font-medium">Status</th>
                                <th className="px-6 py-4 font-medium">Confidence</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-dark-700">
                            {filteredData.length === 0 ? (
                                <tr>
                                    <td colSpan="5" className="px-6 py-12 text-center text-gray-500">
                                        No records found.
                                    </td>
                                </tr>
                            ) : (
                                filteredData.map((row) => (
                                    <tr key={row.id} className="hover:bg-dark-700/50 transition-colors">
                                        <td className="px-6 py-4 text-white">
                                            {new Date(row.timestamp).toLocaleString()}
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className="bg-dark-900 border border-dark-700 px-2 py-1 rounded text-xs text-gray-400 font-medium">
                                                {row.source}
                                            </span>
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
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
            <div className="mt-4 text-xs text-center text-gray-500">
                Showing {filteredData.length} records
            </div>
        </div>
    );
};

export default History;
