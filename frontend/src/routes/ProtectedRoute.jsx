import { Navigate, Outlet } from 'react-router-dom';
import { getAuth } from '../utils/auth';

const ProtectedRoute = () => {
    const isAuth = getAuth();
    return isAuth ? <Outlet /> : <Navigate to="/login" replace />;
};

export default ProtectedRoute;
