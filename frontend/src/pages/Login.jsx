import { useNavigate } from 'react-router-dom';
import { setAuth } from '../utils/auth';
import { ShieldCheck, Eye, EyeOff } from 'lucide-react';
import { useState } from 'react';

const Login = () => {
    const navigate = useNavigate();
    const [showPass, setShowPass] = useState(false);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleLogin = (e) => {
        e.preventDefault();
        setAuth(); // logic handled here
        navigate('/');
    };

    return (
        <div className="min-h-screen bg-dark-900 flex items-center justify-center p-4">
            <div className="bg-dark-800 border border-dark-700 rounded-2xl p-8 w-full max-w-md shadow-2xl">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-500/10 rounded-full mb-4">
                        <ShieldCheck size={32} className="text-primary-500" />
                    </div>
                    <h1 className="text-2xl font-bold text-white mb-2">Welcome to VisionSafe</h1>
                    <p className="text-gray-400">Enterprise AI Safety Monitoring</p>
                </div>

                <form onSubmit={handleLogin} className="space-y-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-400 mb-2">Email Address</label>
                        <input
                            type="email"
                            required
                            className="w-full bg-dark-900 border border-dark-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary-500 transition-colors"
                            placeholder="admin@visionsafe.ai"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-400 mb-2">Password</label>
                        <div className="relative">
                            <input
                                type={showPass ? "text" : "password"}
                                required
                                className="w-full bg-dark-900 border border-dark-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-primary-500 transition-colors"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                            />
                            <button
                                type="button"
                                onClick={() => setShowPass(!showPass)}
                                className="absolute right-3 top-3.5 text-gray-500 hover:text-white"
                            >
                                {showPass ? <EyeOff size={20} /> : <Eye size={20} />}
                            </button>
                        </div>
                    </div>

                    <div className="flex items-center justify-between text-sm">
                        <label className="flex items-center text-gray-400 gap-2 cursor-pointer">
                            <input type="checkbox" className="rounded border-dark-700 bg-dark-900 text-primary-500" />
                            Remember me
                        </label>
                        <a href="#" className="text-primary-500 hover:underline">Forgot password?</a>
                    </div>

                    <button
                        type="submit"
                        className="w-full bg-primary-600 hover:bg-primary-500 text-white font-semibold py-3 rounded-lg transition-colors"
                    >
                        Sign In
                    </button>

                    <button
                        type="button"
                        onClick={handleLogin}
                        className="w-full bg-dark-700 hover:bg-dark-600 text-gray-300 font-medium py-3 rounded-lg transition-colors border border-dark-600"
                    >
                        Continue as Demo User
                    </button>
                </form>

                <div className="mt-8 text-center text-sm text-gray-500">
                    &copy; 2026 VisionSafe Inc. All rights reserved.
                </div>
            </div>
        </div>
    );
};

export default Login;
