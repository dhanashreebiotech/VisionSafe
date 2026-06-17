import { useNavigate, useLocation, Link } from 'react-router-dom';
import { clearAuth } from '../utils/auth';
import {
    LayoutDashboard,
    Upload,
    Camera,
    History,
    Settings,
    Info,
    LogOut,
    ShieldCheck
} from 'lucide-react';
import clsx from 'clsx';

const SidebarItem = ({ to, icon: Icon, label, active }) => (
    <Link
        to={to}
        className={clsx(
            "flex items-center gap-3 px-4 py-3 rounded-lg transition-colors font-medium",
            active ? "bg-primary-600 text-white" : "text-gray-400 hover:bg-dark-800 hover:text-white"
        )}
    >
        <Icon size={20} />
        <span>{label}</span>
    </Link>
);

const AppLayout = ({ children }) => {
    const navigate = useNavigate();
    const location = useLocation();

    const handleLogout = () => {
        clearAuth();
        navigate('/login');
    };

    const routes = [
        { path: '/', label: 'Dashboard', icon: LayoutDashboard },
        { path: '/upload', label: 'Upload Detection', icon: Upload },
        { path: '/camera', label: 'Live Camera', icon: Camera },
        { path: '/history', label: 'History', icon: History },
        { path: '/settings', label: 'Settings', icon: Settings },
    ];

    return (
        <div className="flex h-screen bg-dark-900 text-white overflow-hidden">
            {/* Sidebar */}
            <aside className="w-64 bg-dark-900 border-r border-dark-800 flex flex-col">
                <div className="p-6 flex items-center gap-3 border-b border-dark-800">
                    <ShieldCheck size={32} className="text-primary-500" />
                    <div>
                        <h1 className="font-bold text-xl tracking-tight">VisionSafe</h1>
                        <p className="text-xs text-gray-400">Enterprise AI Safety</p>
                    </div>
                </div>

                <nav className="flex-1 p-4 space-y-1">
                    {routes.map(r => (
                        <SidebarItem
                            key={r.path}
                            to={r.path}
                            icon={r.icon}
                            label={r.label}
                            active={location.pathname === r.path}
                        />
                    ))}

                    <div className="pt-4 mt-4 border-t border-dark-800">
                        <SidebarItem
                            to="/about"
                            icon={Info}
                            label="About VisionSafe"
                            active={location.pathname === '/about'}
                        />
                    </div>
                </nav>

                <div className="p-4 border-t border-dark-800">
                    <button
                        onClick={handleLogout}
                        className="flex items-center gap-3 px-4 py-3 w-full text-left text-red-400 hover:bg-dark-800 rounded-lg transition-colors"
                    >
                        <LogOut size={20} />
                        <span>Sign Out</span>
                    </button>
                    <div className="mt-4 text-xs text-center text-dark-400">
                        v2.0.0 (Build 2026)
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 overflow-auto bg-dark-900 relative">
                <div className="max-w-7xl mx-auto p-8">
                    {children}
                </div>
            </main>
        </div>
    );
};

export default AppLayout;
