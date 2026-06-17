export const setAuth = () => localStorage.setItem("visionsafe_auth", "true");
export const getAuth = () => localStorage.getItem("visionsafe_auth") === "true";
export const clearAuth = () => localStorage.removeItem("visionsafe_auth");
